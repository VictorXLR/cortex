//! Memory subsystem for Cortex
//!
//! Provides built-in vector storage with:
//! - Key-value storage with vector embeddings
//! - Similarity search
//! - Optional disk persistence

mod vector;

pub use vector::VectorStore;

use crate::config::MemoryConfig;
use crate::{CortexError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// Memory entry with embedding and metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEntry {
    /// Unique key
    pub key: String,
    /// Text content
    pub content: String,
    /// Vector embedding
    pub embedding: Vec<f32>,
    /// Optional metadata
    pub metadata: HashMap<String, String>,
    /// Timestamp (unix epoch)
    pub created_at: u64,
}

/// Search result from memory
#[derive(Debug, Clone)]
pub struct SearchResult {
    /// The memory entry
    pub entry: MemoryEntry,
    /// Similarity score (0.0 - 1.0)
    pub score: f32,
}

/// Memory interface
///
/// This is the main interface for memory operations.
/// It wraps a vector store and provides high-level operations.
pub struct Memory {
    store: VectorStore,
    config: MemoryConfig,
}

impl Memory {
    /// Create new memory with config
    pub fn new(config: MemoryConfig) -> Self {
        let store = VectorStore::new(config.embedding_dim, config.max_entries);
        Self { store, config }
    }

    /// Load memory from disk
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let data = std::fs::read(path.as_ref())?;
        let state: MemoryState =
            bincode::deserialize(&data).map_err(|e| CortexError::Serialization(e.to_string()))?;

        let mut store = VectorStore::new(state.embedding_dim, state.max_entries);
        for entry in state.entries {
            store.insert(entry);
        }

        Ok(Self {
            store,
            config: MemoryConfig {
                embedding_dim: state.embedding_dim,
                max_entries: state.max_entries,
                persist_path: Some(path.as_ref().to_path_buf()),
                ..Default::default()
            },
        })
    }

    /// Write to memory
    ///
    /// If the key exists, it will be updated.
    pub fn write(&mut self, key: impl Into<String>, content: impl Into<String>, embedding: Vec<f32>) -> Result<()> {
        let key = key.into();
        let content = content.into();

        if embedding.len() != self.config.embedding_dim {
            return Err(CortexError::Memory(format!(
                "Embedding dimension mismatch: expected {}, got {}",
                self.config.embedding_dim,
                embedding.len()
            )));
        }

        let entry = MemoryEntry {
            key: key.clone(),
            content,
            embedding,
            metadata: HashMap::new(),
            created_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        };

        // Remove existing entry with same key
        self.store.remove(&key);
        self.store.insert(entry);

        Ok(())
    }

    /// Write with metadata
    pub fn write_with_metadata(
        &mut self,
        key: impl Into<String>,
        content: impl Into<String>,
        embedding: Vec<f32>,
        metadata: HashMap<String, String>,
    ) -> Result<()> {
        let key = key.into();
        let content = content.into();

        if embedding.len() != self.config.embedding_dim {
            return Err(CortexError::Memory(format!(
                "Embedding dimension mismatch: expected {}, got {}",
                self.config.embedding_dim,
                embedding.len()
            )));
        }

        let entry = MemoryEntry {
            key: key.clone(),
            content,
            embedding,
            metadata,
            created_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        };

        self.store.remove(&key);
        self.store.insert(entry);

        Ok(())
    }

    /// Read by key
    pub fn read(&self, key: &str) -> Option<&MemoryEntry> {
        self.store.get(key)
    }

    /// Delete by key
    pub fn delete(&mut self, key: &str) -> bool {
        self.store.remove(key)
    }

    /// Search by similarity
    pub fn search(&self, query_embedding: &[f32], k: usize) -> Vec<SearchResult> {
        self.store
            .search(query_embedding, k)
            .into_iter()
            .filter(|r| r.score >= self.config.similarity_threshold)
            .collect()
    }

    /// Search with custom threshold
    pub fn search_with_threshold(
        &self,
        query_embedding: &[f32],
        k: usize,
        threshold: f32,
    ) -> Vec<SearchResult> {
        self.store
            .search(query_embedding, k)
            .into_iter()
            .filter(|r| r.score >= threshold)
            .collect()
    }

    /// Get all entries
    pub fn entries(&self) -> Vec<&MemoryEntry> {
        self.store.entries()
    }

    /// Get number of entries
    pub fn len(&self) -> usize {
        self.store.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.store.is_empty()
    }

    /// Clear all entries
    pub fn clear(&mut self) {
        self.store.clear();
    }

    /// Persist to disk
    pub fn persist(&self, path: impl AsRef<Path>) -> Result<()> {
        let state = MemoryState {
            embedding_dim: self.config.embedding_dim,
            max_entries: self.config.max_entries,
            entries: self.store.entries().into_iter().cloned().collect(),
        };

        let data =
            bincode::serialize(&state).map_err(|e| CortexError::Serialization(e.to_string()))?;

        std::fs::write(path.as_ref(), data)?;
        Ok(())
    }

    /// Get serializable state
    pub fn get_state(&self) -> MemoryState {
        MemoryState {
            embedding_dim: self.config.embedding_dim,
            max_entries: self.config.max_entries,
            entries: self.store.entries().into_iter().cloned().collect(),
        }
    }

    /// Restore from state
    pub fn set_state(&mut self, state: MemoryState) {
        self.store = VectorStore::new(state.embedding_dim, state.max_entries);
        for entry in state.entries {
            self.store.insert(entry);
        }
    }
}

/// Serializable memory state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryState {
    pub embedding_dim: usize,
    pub max_entries: usize,
    pub entries: Vec<MemoryEntry>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_embedding(dim: usize, seed: f32) -> Vec<f32> {
        (0..dim).map(|i| (i as f32 * seed).sin()).collect()
    }

    #[test]
    fn test_write_read() {
        let config = MemoryConfig {
            embedding_dim: 64,
            ..Default::default()
        };
        let mut mem = Memory::new(config);

        let emb = make_embedding(64, 1.0);
        mem.write("test", "Hello world", emb).unwrap();

        let entry = mem.read("test").unwrap();
        assert_eq!(entry.content, "Hello world");
    }

    #[test]
    fn test_search() {
        let config = MemoryConfig {
            embedding_dim: 64,
            similarity_threshold: 0.0,
            ..Default::default()
        };
        let mut mem = Memory::new(config);

        // Add some entries
        for i in 0..10 {
            let emb = make_embedding(64, i as f32);
            mem.write(format!("entry_{}", i), format!("Content {}", i), emb)
                .unwrap();
        }

        // Search with similar embedding
        let query = make_embedding(64, 5.0);
        let results = mem.search(&query, 3);

        assert_eq!(results.len(), 3);
        assert_eq!(results[0].entry.key, "entry_5"); // Should be exact match
    }
}
