//! State management for Cortex
//!
//! Provides:
//! - Checkpointing: Save and restore complete runtime state
//! - Branching: Fork execution for parallel exploration
//! - Persistence: Optional disk-backed state

mod checkpoint;

pub use checkpoint::{Branch, Checkpoint, CheckpointManager};

use crate::inference::EngineState;
use crate::memory::MemoryState;
use crate::{CortexError, Message, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Complete runtime state that can be checkpointed
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeState {
    /// Unique checkpoint ID
    pub id: String,

    /// Human-readable name
    pub name: Option<String>,

    /// Conversation history
    pub messages: Vec<Message>,

    /// Memory state
    pub memory: MemoryState,

    /// Engine state (for model context)
    pub engine_state: EngineState,

    /// Creation timestamp
    pub created_at: u64,

    /// Custom metadata
    pub metadata: std::collections::HashMap<String, String>,
}

impl RuntimeState {
    /// Create new runtime state
    pub fn new(
        messages: Vec<Message>,
        memory: MemoryState,
        engine_state: EngineState,
    ) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            name: None,
            messages,
            memory,
            engine_state,
            created_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            metadata: Default::default(),
        }
    }

    /// Create with a name
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Save to file
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let data =
            bincode::serialize(self).map_err(|e| CortexError::Serialization(e.to_string()))?;
        std::fs::write(path.as_ref(), data)?;
        Ok(())
    }

    /// Load from file
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let data = std::fs::read(path.as_ref())?;
        let state: Self =
            bincode::deserialize(&data).map_err(|e| CortexError::Serialization(e.to_string()))?;
        Ok(state)
    }
}

/// State store for managing checkpoints
pub struct StateStore {
    /// In-memory checkpoints
    checkpoints: std::collections::HashMap<String, RuntimeState>,

    /// Persistence directory
    persist_dir: Option<std::path::PathBuf>,

    /// Maximum checkpoints to keep
    max_checkpoints: usize,

    /// Checkpoint IDs in order (for LRU eviction)
    checkpoint_order: Vec<String>,
}

impl StateStore {
    /// Create new state store
    pub fn new(persist_dir: Option<std::path::PathBuf>, max_checkpoints: usize) -> Self {
        Self {
            checkpoints: std::collections::HashMap::new(),
            persist_dir,
            max_checkpoints,
            checkpoint_order: Vec::new(),
        }
    }

    /// Save a checkpoint
    pub fn save(&mut self, state: RuntimeState) -> Result<String> {
        let id = state.id.clone();

        // Persist if enabled
        if let Some(dir) = &self.persist_dir {
            std::fs::create_dir_all(dir)?;
            let path = dir.join(format!("{}.ckpt", &id));
            state.save(&path)?;
        }

        // Store in memory
        self.checkpoints.insert(id.clone(), state);
        self.checkpoint_order.push(id.clone());

        // Evict oldest if over limit
        while self.checkpoints.len() > self.max_checkpoints {
            if let Some(oldest_id) = self.checkpoint_order.first().cloned() {
                self.checkpoints.remove(&oldest_id);
                self.checkpoint_order.remove(0);

                // Remove from disk too
                if let Some(dir) = &self.persist_dir {
                    let path = dir.join(format!("{}.ckpt", &oldest_id));
                    let _ = std::fs::remove_file(path);
                }
            }
        }

        Ok(id)
    }

    /// Load a checkpoint
    pub fn load(&self, id: &str) -> Result<RuntimeState> {
        // Try memory first
        if let Some(state) = self.checkpoints.get(id) {
            return Ok(state.clone());
        }

        // Try disk
        if let Some(dir) = &self.persist_dir {
            let path = dir.join(format!("{}.ckpt", id));
            if path.exists() {
                return RuntimeState::load(&path);
            }
        }

        Err(CortexError::InvalidCheckpoint(format!(
            "Checkpoint not found: {}",
            id
        )))
    }

    /// Delete a checkpoint
    pub fn delete(&mut self, id: &str) -> bool {
        let removed = self.checkpoints.remove(id).is_some();
        self.checkpoint_order.retain(|i| i != id);

        if let Some(dir) = &self.persist_dir {
            let path = dir.join(format!("{}.ckpt", id));
            let _ = std::fs::remove_file(path);
        }

        removed
    }

    /// List all checkpoint IDs
    pub fn list(&self) -> Vec<&str> {
        self.checkpoint_order.iter().map(|s| s.as_str()).collect()
    }

    /// Get checkpoint count
    pub fn len(&self) -> usize {
        self.checkpoints.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.checkpoints.is_empty()
    }
}
