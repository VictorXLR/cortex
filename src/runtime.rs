//! Core Cortex runtime
//!
//! The runtime layer that provides memory, state, and execution primitives.

use crate::config::{CortexConfig, GenerationConfig};
use crate::inference::{format_chat_prompt, CandleLLM, ChatTemplate, Embedder, StubEngine, TextEngine};
use crate::memory::Memory;
use crate::state::{Branch, Checkpoint, CheckpointManager, RuntimeState, StateStore};
use crate::{Message, Result};

use std::path::Path;

/// The Cortex runtime
///
/// Provides memory and state primitives for AI applications.
/// Inference backends plug into this runtime.
///
/// # Example
///
/// ```rust,ignore
/// use cortex::{Cortex, Message};
///
/// // Create with stub engine (for testing)
/// let mut ctx = Cortex::new();
///
/// // Or with a custom engine
/// let mut ctx = Cortex::with_engine(my_llama_engine);
///
/// // Use memory
/// ctx.remember("user_pref", "likes jazz")?;
///
/// // Use state
/// let snap = ctx.checkpoint()?;
/// ctx.chat(&[Message::user("Hello")])?;
/// ctx.restore(&snap)?;
/// ```
pub struct Cortex {
    /// Configuration
    config: CortexConfig,

    /// Text engine (boxed for dynamic dispatch)
    engine: Box<dyn TextEngine>,

    /// Dedicated embedding model (for semantic search)
    embedder: Option<Embedder>,

    /// Memory subsystem
    pub memory: Memory,

    /// State store for checkpoints
    state_store: StateStore,

    /// Checkpoint manager
    checkpoint_manager: CheckpointManager,

    /// Conversation history
    messages: Vec<Message>,

    /// Chat template to use
    chat_template: ChatTemplate,
}

impl Cortex {
    /// Create a new runtime with stub engine
    ///
    /// Useful for testing or when you'll set the engine later.
    pub fn new() -> Self {
        Self::with_engine(StubEngine::new())
    }

    /// Create runtime with a custom text engine
    pub fn with_engine<E: TextEngine + 'static>(engine: E) -> Self {
        let config = CortexConfig::default();
        let memory = Memory::new(config.memory.clone());
        let state_store = StateStore::new(
            config.state.directory.clone(),
            config.state.max_checkpoints,
        );
        let checkpoint_manager = CheckpointManager::new(config.state.max_checkpoints);

        Self {
            config,
            engine: Box::new(engine),
            embedder: None,
            memory,
            state_store,
            checkpoint_manager,
            messages: Vec::new(),
            chat_template: ChatTemplate::default(),
        }
    }

    /// Create runtime with config and engine
    pub fn with_config_and_engine<E: TextEngine + 'static>(
        config: CortexConfig,
        engine: E,
    ) -> Self {
        let memory = Memory::new(config.memory.clone());
        let state_store = StateStore::new(
            config.state.directory.clone(),
            config.state.max_checkpoints,
        );
        let checkpoint_manager = CheckpointManager::new(config.state.max_checkpoints);

        Self {
            config,
            engine: Box::new(engine),
            memory,
            state_store,
            checkpoint_manager,
            messages: Vec::new(),
            chat_template: ChatTemplate::default(),
        }
    }

    /// Load a model from a GGUF file
    ///
    /// Uses CandleLLM for inference with quantized models.
    pub fn load(model_path: impl AsRef<Path>) -> Result<Self> {
        let config = CortexConfig::for_model(model_path.as_ref());
        let engine = CandleLLM::load(model_path)?;
        Ok(Self::with_config_and_engine(config, engine))
    }

    /// Set the chat template
    pub fn with_template(mut self, template: ChatTemplate) -> Self {
        self.chat_template = template;
        self
    }

    // ==================== Generation ====================

    /// Generate a completion for raw text
    pub fn generate(&mut self, prompt: &str) -> Result<String> {
        self.generate_with_config(prompt, &self.config.generation.clone())
    }

    /// Generate with custom config
    pub fn generate_with_config(
        &mut self,
        prompt: &str,
        config: &GenerationConfig,
    ) -> Result<String> {
        self.engine.generate(prompt, config)
    }

    /// Generate with streaming
    pub fn generate_streaming(
        &mut self,
        prompt: &str,
        config: &GenerationConfig,
        callback: &mut dyn FnMut(&str) -> bool,
    ) -> Result<String> {
        self.engine.generate_streaming(prompt, config, callback)
    }

    /// Chat with message history
    pub fn chat(&mut self, messages: &[Message]) -> Result<String> {
        self.chat_with_config(messages, &self.config.generation.clone())
    }

    /// Chat with custom config
    pub fn chat_with_config(
        &mut self,
        messages: &[Message],
        config: &GenerationConfig,
    ) -> Result<String> {
        // Add new messages to history
        self.messages.extend(messages.iter().cloned());

        // Format prompt
        let prompt = format_chat_prompt(&self.messages, self.chat_template);

        // Generate response
        let response = self.engine.generate(&prompt, config)?;

        // Add assistant response to history
        self.messages.push(Message::assistant(&response));

        Ok(response)
    }

    /// Chat with streaming
    pub fn chat_streaming(
        &mut self,
        messages: &[Message],
        config: &GenerationConfig,
        callback: &mut dyn FnMut(&str) -> bool,
    ) -> Result<String> {
        self.messages.extend(messages.iter().cloned());
        let prompt = format_chat_prompt(&self.messages, self.chat_template);
        let response = self.engine.generate_streaming(&prompt, config, callback)?;
        self.messages.push(Message::assistant(&response));
        Ok(response)
    }

    /// Get conversation history
    pub fn messages(&self) -> &[Message] {
        &self.messages
    }

    /// Clear conversation history
    pub fn clear_messages(&mut self) {
        self.messages.clear();
        self.engine.clear();
    }

    // ==================== Memory ====================

    /// Write to memory with auto-embedding
    pub fn remember(&mut self, key: impl Into<String>, content: impl Into<String>) -> Result<()> {
        let content = content.into();
        let embedding = self.engine.embed(&content)?;
        self.memory.write(key, content, embedding)
    }

    /// Search memory by text query
    pub fn recall(&self, query: &str, k: usize) -> Result<Vec<String>> {
        let query_embedding = self.engine.embed(query)?;
        let results = self.memory.search(&query_embedding, k);
        Ok(results.into_iter().map(|r| r.entry.content).collect())
    }

    // ==================== State ====================

    /// Create a checkpoint of current state
    pub fn checkpoint(&mut self) -> Result<Checkpoint> {
        let state = RuntimeState::new(
            self.messages.clone(),
            self.memory.get_state(),
            self.engine.get_state()?,
        );

        let checkpoint = Checkpoint::from_state(&state);
        self.state_store.save(state)?;
        self.checkpoint_manager.record(checkpoint.clone());

        Ok(checkpoint)
    }

    /// Create a named checkpoint
    pub fn checkpoint_named(&mut self, name: impl Into<String>) -> Result<Checkpoint> {
        let state = RuntimeState::new(
            self.messages.clone(),
            self.memory.get_state(),
            self.engine.get_state()?,
        )
        .with_name(name);

        let checkpoint = Checkpoint::from_state(&state);
        self.state_store.save(state)?;
        self.checkpoint_manager.record(checkpoint.clone());

        Ok(checkpoint)
    }

    /// Restore from a checkpoint
    pub fn restore(&mut self, checkpoint: &Checkpoint) -> Result<()> {
        let state = self.state_store.load(&checkpoint.id)?;

        self.messages = state.messages;
        self.memory.set_state(state.memory);
        self.engine.set_state(&state.engine_state)?;

        Ok(())
    }

    /// Restore from checkpoint ID
    pub fn restore_id(&mut self, id: &str) -> Result<()> {
        let state = self.state_store.load(id)?;

        self.messages = state.messages;
        self.memory.set_state(state.memory);
        self.engine.set_state(&state.engine_state)?;

        Ok(())
    }

    /// Create a branch from current state
    pub fn branch(&mut self) -> Result<Branch> {
        let checkpoint = self.checkpoint()?;
        let state = self.state_store.load(&checkpoint.id)?;
        Ok(Branch::new(checkpoint.id, state))
    }

    /// Get the latest checkpoint
    pub fn latest_checkpoint(&self) -> Option<&Checkpoint> {
        self.checkpoint_manager.latest()
    }

    /// List all checkpoints
    pub fn checkpoints(&self) -> &[Checkpoint] {
        self.checkpoint_manager.list()
    }

    // ==================== Info ====================

    /// Get context window size
    pub fn context_size(&self) -> usize {
        self.engine.context_size()
    }

    /// Get context tokens currently used
    pub fn context_used(&self) -> usize {
        self.engine.context_used()
    }

    /// Get embedding dimension
    pub fn embedding_dim(&self) -> usize {
        self.engine.embedding_dim()
    }

    /// Get config
    pub fn config(&self) -> &CortexConfig {
        &self.config
    }
}

impl Default for Cortex {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_roundtrip() {
        let mut ctx = Cortex::new();
        ctx.remember("fact", "The sky is blue").unwrap();

        // Debug: check memory length
        assert_eq!(ctx.memory.len(), 1);
        
        // Debug: check if we can read directly
        let entry = ctx.memory.read("fact").unwrap();
        assert!(entry.content.contains("blue"));
        
        // Debug: check embeddings directly
        let stored_embedding = &ctx.memory.read("fact").unwrap().embedding;
        let query_embedding = ctx.engine.embed("What color is the sky?").unwrap();
        
        // Check first few values
        println!("Stored first 5: {:?}", &stored_embedding[..5]);
        println!("Query first 5: {:?}", &query_embedding[..5]);
        
        // Test cosine similarity directly
        let dot: f32 = query_embedding.iter().zip(stored_embedding.iter()).map(|(x, y)| x * y).sum();
        let norm_query: f32 = query_embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_stored: f32 = stored_embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        let similarity = if norm_query == 0.0 || norm_stored == 0.0 {
            0.0
        } else {
            dot / (norm_query * norm_stored)
        };
        println!("Direct similarity: {}", similarity);
        
        let results = ctx.memory.search_with_threshold(&query_embedding, 1, 0.0);
        
        // Debug: print actual results
        println!("Results: {}", results.len());
        for r in &results {
            println!("Score: {}, Content: {}", r.score, r.entry.content);
        }
        
        assert_eq!(results.len(), 1);
        assert!(results[0].entry.content.contains("blue"));
    }

    #[test]
    fn test_checkpoint_restore() {
        let mut ctx = Cortex::new();

        ctx.remember("before", "original value").unwrap();
        let snap = ctx.checkpoint().unwrap();

        ctx.remember("after", "new value").unwrap();
        assert_eq!(ctx.memory.len(), 2);

        ctx.restore(&snap).unwrap();
        assert_eq!(ctx.memory.len(), 1);
    }

    #[test]
    fn test_chat() {
        let mut ctx = Cortex::new();
        let response = ctx.chat(&[Message::user("Hello")]).unwrap();
        assert!(!response.is_empty());
        assert_eq!(ctx.messages().len(), 2); // user + assistant
    }
}
