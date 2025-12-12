//! # Cortex: The AI Runtime
//!
//! Memory and state as primitives, not infrastructure.
//!
//! ```rust,ignore
//! use cortex::{Cortex, Message};
//!
//! let mut ctx = Cortex::new();
//! ctx.remember("fact", "The sky is blue")?;
//! let snap = ctx.checkpoint()?;
//! let response = ctx.chat(&[Message::user("What color is the sky?")])?;
//! ```
//!
//! No Pinecone. No Redis. No LangChain. One binary. Just run.

pub mod config;
pub mod inference;
pub mod memory;
pub mod runtime;
pub mod session;
pub mod state;

// Re-exports for convenience
pub use config::{CortexConfig, GenerationConfig};
pub use inference::{CandleLLM, ChatTemplate, Embedder, EngineState, StubEngine, TextEngine};
pub use memory::Memory;
pub use runtime::Cortex;
pub use session::Session;
pub use state::{Branch, Checkpoint};

/// Message role in a conversation
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

/// A chat message
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: String,
    pub name: Option<String>,
}

impl Message {
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: Role::System,
            content: content.into(),
            name: None,
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: content.into(),
            name: None,
        }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: Role::Assistant,
            content: content.into(),
            name: None,
        }
    }

    pub fn tool(content: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            role: Role::Tool,
            content: content.into(),
            name: Some(name.into()),
        }
    }
}

/// Result type for Cortex operations
pub type Result<T> = std::result::Result<T, CortexError>;

/// Errors that can occur in Cortex operations
#[derive(Debug, thiserror::Error)]
pub enum CortexError {
    #[error("Failed to load model: {0}")]
    ModelLoad(String),

    #[error("Inference error: {0}")]
    Inference(String),

    #[error("Memory error: {0}")]
    Memory(String),

    #[error("State error: {0}")]
    State(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    Serialization(String),

    #[error("Invalid checkpoint: {0}")]
    InvalidCheckpoint(String),

    #[error("Tool error: {0}")]
    Tool(String),

    #[error("Configuration error: {0}")]
    Config(String),
}
