//! Checkpoint and branching primitives

use super::RuntimeState;

/// A checkpoint handle
///
/// This is a lightweight reference to a saved state.
#[derive(Debug, Clone)]
pub struct Checkpoint {
    /// Checkpoint ID
    pub id: String,
    /// Optional name
    pub name: Option<String>,
    /// Creation timestamp
    pub created_at: u64,
}

impl Checkpoint {
    /// Create from runtime state
    pub fn from_state(state: &RuntimeState) -> Self {
        Self {
            id: state.id.clone(),
            name: state.name.clone(),
            created_at: state.created_at,
        }
    }
}

/// A branch of execution
///
/// Branches are independent copies of the runtime state
/// that can evolve separately and optionally merge back.
pub struct Branch {
    /// Branch ID
    pub id: String,
    /// Parent checkpoint ID
    pub parent_id: String,
    /// Branch state
    state: RuntimeState,
}

impl Branch {
    /// Create a new branch from a checkpoint
    pub fn new(parent_id: String, state: RuntimeState) -> Self {
        let id = uuid::Uuid::new_v4().to_string();
        Self {
            id,
            parent_id,
            state,
        }
    }

    /// Get the branch's state
    pub fn state(&self) -> &RuntimeState {
        &self.state
    }

    /// Get mutable state
    pub fn state_mut(&mut self) -> &mut RuntimeState {
        &mut self.state
    }

    /// Take ownership of state (consumes branch)
    pub fn into_state(self) -> RuntimeState {
        self.state
    }
}

/// Manages checkpoints for a runtime
pub struct CheckpointManager {
    /// All checkpoints
    checkpoints: Vec<Checkpoint>,
    /// Maximum checkpoints to retain
    max_checkpoints: usize,
}

impl CheckpointManager {
    /// Create new checkpoint manager
    pub fn new(max_checkpoints: usize) -> Self {
        Self {
            checkpoints: Vec::new(),
            max_checkpoints,
        }
    }

    /// Record a checkpoint
    pub fn record(&mut self, checkpoint: Checkpoint) {
        self.checkpoints.push(checkpoint);

        // Evict oldest if over limit
        while self.checkpoints.len() > self.max_checkpoints {
            self.checkpoints.remove(0);
        }
    }

    /// Get most recent checkpoint
    pub fn latest(&self) -> Option<&Checkpoint> {
        self.checkpoints.last()
    }

    /// Get checkpoint by ID
    pub fn get(&self, id: &str) -> Option<&Checkpoint> {
        self.checkpoints.iter().find(|c| c.id == id)
    }

    /// List all checkpoints
    pub fn list(&self) -> &[Checkpoint] {
        &self.checkpoints
    }

    /// Clear all checkpoints
    pub fn clear(&mut self) {
        self.checkpoints.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::EngineState;
    use crate::memory::MemoryState;

    fn make_state() -> RuntimeState {
        RuntimeState::new(
            vec![],
            MemoryState {
                embedding_dim: 64,
                max_entries: 100,
                entries: vec![],
            },
            EngineState::default(),
        )
    }

    #[test]
    fn test_checkpoint_manager() {
        let mut manager = CheckpointManager::new(3);

        for _ in 0..5 {
            let state = make_state();
            let checkpoint = Checkpoint::from_state(&state);
            manager.record(checkpoint);
        }

        // Should only have 3 checkpoints
        assert_eq!(manager.list().len(), 3);
    }

    #[test]
    fn test_branch() {
        let state = make_state();
        let checkpoint = Checkpoint::from_state(&state);
        let branch = Branch::new(checkpoint.id.clone(), state);

        assert_eq!(branch.parent_id, checkpoint.id);
    }
}
