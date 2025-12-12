//! Configuration for Cortex runtime

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Main configuration for the Cortex runtime
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CortexConfig {
    /// Path to the model file (GGUF format)
    pub model_path: PathBuf,

    /// Number of GPU layers to offload (0 = CPU only)
    pub n_gpu_layers: u32,

    /// Context size (number of tokens)
    pub n_ctx: u32,

    /// Batch size for prompt processing
    pub n_batch: u32,

    /// Number of threads for CPU inference
    pub n_threads: u32,

    /// Memory configuration
    pub memory: MemoryConfig,

    /// State persistence configuration
    pub state: StateConfig,

    /// Generation defaults
    pub generation: GenerationConfig,
}

impl Default for CortexConfig {
    fn default() -> Self {
        Self {
            model_path: PathBuf::new(),
            n_gpu_layers: 0,
            n_ctx: 4096,
            n_batch: 512,
            n_threads: num_cpus::get() as u32,
            memory: MemoryConfig::default(),
            state: StateConfig::default(),
            generation: GenerationConfig::default(),
        }
    }
}

impl CortexConfig {
    /// Create config for a specific model path
    pub fn for_model(path: impl Into<PathBuf>) -> Self {
        Self {
            model_path: path.into(),
            ..Default::default()
        }
    }

    /// Set number of GPU layers
    pub fn with_gpu_layers(mut self, n: u32) -> Self {
        self.n_gpu_layers = n;
        self
    }

    /// Set context size
    pub fn with_context_size(mut self, n: u32) -> Self {
        self.n_ctx = n;
        self
    }

    /// Set state directory for persistence
    pub fn with_state_dir(mut self, path: impl Into<PathBuf>) -> Self {
        self.state.directory = Some(path.into());
        self
    }

    /// Enable memory persistence
    pub fn with_memory_persistence(mut self, path: impl Into<PathBuf>) -> Self {
        self.memory.persist_path = Some(path.into());
        self
    }
}

/// Configuration for the memory subsystem
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryConfig {
    /// Embedding dimension (must match model)
    pub embedding_dim: usize,

    /// Maximum number of memory entries
    pub max_entries: usize,

    /// Path to persist memory (None = in-memory only)
    pub persist_path: Option<PathBuf>,

    /// Number of results for similarity search
    pub default_search_k: usize,

    /// Similarity threshold (0.0 - 1.0)
    pub similarity_threshold: f32,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            embedding_dim: 4096, // Common for 7B/8B models
            max_entries: 100_000,
            persist_path: None,
            default_search_k: 5,
            similarity_threshold: 0.7,
        }
    }
}

/// Configuration for state management
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateConfig {
    /// Directory for state persistence (None = no persistence)
    pub directory: Option<PathBuf>,

    /// Maximum number of checkpoints to keep
    pub max_checkpoints: usize,

    /// Auto-checkpoint interval (in messages, 0 = disabled)
    pub auto_checkpoint_interval: usize,
}

impl Default for StateConfig {
    fn default() -> Self {
        Self {
            directory: None,
            max_checkpoints: 100,
            auto_checkpoint_interval: 0,
        }
    }
}

/// Configuration for text generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationConfig {
    /// Maximum tokens to generate
    pub max_tokens: u32,

    /// Temperature (0.0 = deterministic, higher = more random)
    pub temperature: f32,

    /// Top-p (nucleus) sampling
    pub top_p: f32,

    /// Top-k sampling (0 = disabled)
    pub top_k: u32,

    /// Repetition penalty
    pub repeat_penalty: f32,

    /// Stop sequences
    pub stop: Vec<String>,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_tokens: 1024,
            temperature: 0.7,
            top_p: 0.9,
            top_k: 40,
            repeat_penalty: 1.1,
            stop: vec![],
        }
    }
}

impl GenerationConfig {
    pub fn deterministic() -> Self {
        Self {
            temperature: 0.0,
            top_p: 1.0,
            top_k: 1,
            ..Default::default()
        }
    }

    pub fn creative() -> Self {
        Self {
            temperature: 1.0,
            top_p: 0.95,
            top_k: 0,
            ..Default::default()
        }
    }

    pub fn with_max_tokens(mut self, n: u32) -> Self {
        self.max_tokens = n;
        self
    }

    pub fn with_temperature(mut self, t: f32) -> Self {
        self.temperature = t;
        self
    }

    pub fn with_stop(mut self, stop: Vec<String>) -> Self {
        self.stop = stop;
        self
    }
}

