//! High-level Session API
//!
//! Sessions provide automatic state persistence and a simpler interface.
//!
//! # Example
//!
//! ```rust,ignore
//! use cortex::Session;
//!
//! // Create or resume a session
//! let mut session = Session::new("user_123")?;
//!
//! // Chat (state automatically saved)
//! let response = session.chat("Hello!")?;
//!
//! // Later, in a new process...
//! let mut session = Session::new("user_123")?;
//! // Automatically restored!
//! ```

use crate::config::GenerationConfig;
use crate::inference::{EngineState, StubEngine, TextEngine};
use crate::runtime::Cortex;
use crate::state::RuntimeState;
use crate::{Message, Result};

use std::path::PathBuf;

/// A persistent session with automatic state management
pub struct Session {
    /// Underlying runtime
    runtime: Cortex,

    /// Session ID
    session_id: String,

    /// Session directory
    session_dir: PathBuf,

    /// Auto-save on every message
    auto_save: bool,
}

impl Session {
    /// Create or resume a session with stub engine
    pub fn new(session_id: impl Into<String>) -> Result<Self> {
        Self::with_engine(session_id, StubEngine::new())
    }

    /// Create or resume a session with custom engine
    pub fn with_engine<E: TextEngine + 'static>(
        session_id: impl Into<String>,
        engine: E,
    ) -> Result<Self> {
        let session_id = session_id.into();
        let session_dir = default_session_dir(&session_id);

        // Create session directory
        std::fs::create_dir_all(&session_dir)?;

        // Create runtime with engine
        let mut runtime = Cortex::with_engine(engine);

        // Try to restore existing session
        let state_path = session_dir.join("session.state");
        if state_path.exists() {
            if let Ok(state) = RuntimeState::load(&state_path) {
                runtime.memory.set_state(state.memory);
                // Note: Can't restore messages directly, but memory is restored
            }
        }

        Ok(Self {
            runtime,
            session_id,
            session_dir,
            auto_save: true,
        })
    }

    /// Disable auto-save
    pub fn without_auto_save(mut self) -> Self {
        self.auto_save = false;
        self
    }

    /// Get session ID
    pub fn id(&self) -> &str {
        &self.session_id
    }

    /// Chat with the session
    pub fn chat(&mut self, message: impl Into<String>) -> Result<String> {
        let response = self.runtime.chat(&[Message::user(message)])?;

        if self.auto_save {
            self.save()?;
        }

        Ok(response)
    }

    /// Chat with custom generation config
    pub fn chat_with_config(
        &mut self,
        message: impl Into<String>,
        config: &GenerationConfig,
    ) -> Result<String> {
        let response = self
            .runtime
            .chat_with_config(&[Message::user(message)], config)?;

        if self.auto_save {
            self.save()?;
        }

        Ok(response)
    }

    /// Chat with streaming
    pub fn chat_streaming(
        &mut self,
        message: impl Into<String>,
        callback: &mut dyn FnMut(&str) -> bool,
    ) -> Result<String> {
        let config = self.runtime.config().generation.clone();
        let response = self
            .runtime
            .chat_streaming(&[Message::user(message)], &config, callback)?;

        if self.auto_save {
            self.save()?;
        }

        Ok(response)
    }

    /// Add a system message
    pub fn set_system(&mut self, message: impl Into<String>) {
        self.runtime.clear_messages();
        let _ = self.runtime.chat(&[Message::system(message)]);
    }

    /// Remember something
    pub fn remember(&mut self, key: impl Into<String>, value: impl Into<String>) -> Result<()> {
        self.runtime.remember(key, value)?;
        if self.auto_save {
            self.save()?;
        }
        Ok(())
    }

    /// Recall from memory
    pub fn recall(&self, query: &str, k: usize) -> Result<Vec<String>> {
        self.runtime.recall(query, k)
    }

    /// Save session state
    pub fn save(&self) -> Result<()> {
        let state = RuntimeState::new(
            self.runtime.messages().to_vec(),
            self.runtime.memory.get_state(),
            EngineState::default(),
        );

        let state_path = self.session_dir.join("session.state");
        state.save(&state_path)?;

        // Also save memory separately for easier access
        let memory_path = self.session_dir.join("memory.bin");
        self.runtime.memory.persist(&memory_path)?;

        Ok(())
    }

    /// Clear the session (delete all state)
    pub fn clear(&mut self) -> Result<()> {
        self.runtime.clear_messages();
        self.runtime.memory.clear();

        // Delete state files
        let state_path = self.session_dir.join("session.state");
        let _ = std::fs::remove_file(state_path);

        let memory_path = self.session_dir.join("memory.bin");
        let _ = std::fs::remove_file(memory_path);

        Ok(())
    }

    /// Get conversation history
    pub fn messages(&self) -> &[Message] {
        self.runtime.messages()
    }

    /// Get underlying runtime for advanced operations
    pub fn runtime(&self) -> &Cortex {
        &self.runtime
    }

    /// Get mutable runtime
    pub fn runtime_mut(&mut self) -> &mut Cortex {
        &mut self.runtime
    }
}

/// Get default session directory
fn default_session_dir(session_id: &str) -> PathBuf {
    let base = dirs::data_local_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("cortex")
        .join("sessions");

    base.join(session_id)
}

/// List all sessions in the default directory
pub fn list_sessions() -> Result<Vec<String>> {
    let base = dirs::data_local_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("cortex")
        .join("sessions");

    if !base.exists() {
        return Ok(vec![]);
    }

    let mut sessions = Vec::new();
    for entry in std::fs::read_dir(base)? {
        let entry = entry?;
        if entry.file_type()?.is_dir() {
            if let Some(name) = entry.file_name().to_str() {
                sessions.push(name.to_string());
            }
        }
    }

    Ok(sessions)
}

/// Delete a session
pub fn delete_session(session_id: &str) -> Result<()> {
    let session_dir = default_session_dir(session_id);
    if session_dir.exists() {
        std::fs::remove_dir_all(session_dir)?;
    }
    Ok(())
}
