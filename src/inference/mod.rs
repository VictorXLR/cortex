//! Inference engine abstraction
//!
//! Provides trait-based interfaces for different model types:
//! - `TextEngine` for LLMs (generation, embeddings)
//! - `Embedder` for semantic embeddings (separate BERT model)
//! - `ImageEngine` for diffusion models (future)
//!
//! The Candle backend provides pure-Rust implementations.

mod candle_llm;
mod embedder;

pub use candle_llm::CandleLLM;
pub use embedder::Embedder;

use crate::config::GenerationConfig;
use crate::Result;

/// Engine state for checkpointing
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct EngineState {
    /// Serialized KV cache or other state
    pub data: Vec<u8>,
    /// Number of tokens in context
    pub n_tokens: usize,
    /// Engine identifier
    pub engine_id: String,
}

impl Default for EngineState {
    fn default() -> Self {
        Self {
            data: vec![],
            n_tokens: 0,
            engine_id: "none".to_string(),
        }
    }
}

/// Text generation engine trait (LLMs)
///
/// Implement this for language models that can:
/// - Generate text completions
/// - Produce embeddings
/// - Manage KV cache state
pub trait TextEngine: Send {
    /// Get the model's embedding dimension
    fn embedding_dim(&self) -> usize;

    /// Get the context size (max tokens)
    fn context_size(&self) -> usize;

    /// Get embedding for text (for memory/RAG)
    fn embed(&self, text: &str) -> Result<Vec<f32>>;

    /// Generate text completion
    fn generate(&mut self, prompt: &str, config: &GenerationConfig) -> Result<String>;

    /// Generate with streaming callback
    fn generate_streaming(
        &mut self,
        prompt: &str,
        config: &GenerationConfig,
        callback: &mut dyn FnMut(&str) -> bool,
    ) -> Result<String>;

    /// Get current state for checkpointing
    fn get_state(&self) -> Result<EngineState>;

    /// Restore from checkpoint state
    fn set_state(&mut self, state: &EngineState) -> Result<()>;

    /// Clear context/KV cache
    fn clear(&mut self);

    /// Get number of tokens currently in context
    fn context_used(&self) -> usize;
}

/// Chat message formatting
#[derive(Debug, Clone, Copy, Default)]
pub enum ChatTemplate {
    #[default]
    Llama3,
    ChatML,
    Phi3,
    Gemma,
    Raw,
}

/// Format a chat conversation into a prompt string
pub fn format_chat_prompt(messages: &[crate::Message], template: ChatTemplate) -> String {
    match template {
        ChatTemplate::Llama3 => format_llama3(messages),
        ChatTemplate::ChatML => format_chatml(messages),
        ChatTemplate::Phi3 => format_phi3(messages),
        ChatTemplate::Gemma => format_gemma(messages),
        ChatTemplate::Raw => format_raw(messages),
    }
}

fn format_llama3(messages: &[crate::Message]) -> String {
    let mut prompt = String::from("<|begin_of_text|>");
    for msg in messages {
        let role = match msg.role {
            crate::Role::System => "system",
            crate::Role::User => "user",
            crate::Role::Assistant => "assistant",
            crate::Role::Tool => "tool",
        };
        prompt.push_str(&format!(
            "<|start_header_id|>{}<|end_header_id|>\n\n{}<|eot_id|>",
            role, msg.content
        ));
    }
    prompt.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
    prompt
}

fn format_chatml(messages: &[crate::Message]) -> String {
    let mut prompt = String::new();
    for msg in messages {
        let role = match msg.role {
            crate::Role::System => "system",
            crate::Role::User => "user",
            crate::Role::Assistant => "assistant",
            crate::Role::Tool => "tool",
        };
        prompt.push_str(&format!("<|im_start|>{}\n{}<|im_end|>\n", role, msg.content));
    }
    prompt.push_str("<|im_start|>assistant\n");
    prompt
}

fn format_phi3(messages: &[crate::Message]) -> String {
    let mut prompt = String::new();
    for msg in messages {
        match msg.role {
            crate::Role::System => {
                prompt.push_str(&format!("<|system|>\n{}<|end|>\n", msg.content));
            }
            crate::Role::User => {
                prompt.push_str(&format!("<|user|>\n{}<|end|>\n", msg.content));
            }
            crate::Role::Assistant => {
                prompt.push_str(&format!("<|assistant|>\n{}<|end|>\n", msg.content));
            }
            crate::Role::Tool => {
                prompt.push_str(&format!("<|tool|>\n{}<|end|>\n", msg.content));
            }
        }
    }
    prompt.push_str("<|assistant|>\n");
    prompt
}

fn format_gemma(messages: &[crate::Message]) -> String {
    let mut prompt = String::new();
    for msg in messages {
        match msg.role {
            crate::Role::User => {
                prompt.push_str(&format!("<start_of_turn>user\n{}<end_of_turn>\n", msg.content));
            }
            crate::Role::Assistant => {
                prompt.push_str(&format!("<start_of_turn>model\n{}<end_of_turn>\n", msg.content));
            }
            _ => {
                prompt.push_str(&format!("<start_of_turn>user\n{}<end_of_turn>\n", msg.content));
            }
        }
    }
    prompt.push_str("<start_of_turn>model\n");
    prompt
}

fn format_raw(messages: &[crate::Message]) -> String {
    messages
        .iter()
        .map(|m| m.content.clone())
        .collect::<Vec<_>>()
        .join("\n")
}

// ============================================================================
// Stub Engine (for testing)
// ============================================================================

/// Stub engine for testing without a real model
pub struct StubEngine {
    embedding_dim: usize,
    context_size: usize,
    context_used: usize,
    response_prefix: String,
}

impl StubEngine {
    pub fn new() -> Self {
        Self {
            embedding_dim: 4096,
            context_size: 8192,
            context_used: 0,
            response_prefix: "".to_string(),
        }
    }

    pub fn with_response_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.response_prefix = prefix.into();
        self
    }
}

impl Default for StubEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl TextEngine for StubEngine {
    fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }

    fn context_size(&self) -> usize {
        self.context_size
    }

    fn embed(&self, text: &str) -> Result<Vec<f32>> {
        // Bag-of-words style embedding: each word hashes to positions in the vector
        // This gives texts with similar words similar embeddings (for testing)
        let mut embedding = vec![0.0f32; self.embedding_dim];

        // Normalize text and split into words
        let text_lower = text.to_lowercase();
        let words: Vec<&str> = text_lower
            .split(|c: char| !c.is_alphanumeric())
            .filter(|w| w.len() > 1)
            .collect();

        // Each word contributes to multiple positions based on its hash
        for word in &words {
            let hash = word.bytes().fold(0u64, |acc, b| {
                acc.wrapping_mul(31).wrapping_add(b as u64)
            });

            // Activate multiple positions per word (sparse features)
            for i in 0..8 {
                let pos = ((hash.wrapping_add(i * 7919)) as usize) % self.embedding_dim;
                embedding[pos] += 1.0;
            }
        }

        // Normalize to unit length
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in &mut embedding {
                *x /= norm;
            }
        }

        Ok(embedding)
    }

    fn generate(&mut self, prompt: &str, config: &GenerationConfig) -> Result<String> {
        self.generate_streaming(prompt, config, &mut |_| true)
    }

    fn generate_streaming(
        &mut self,
        prompt: &str,
        config: &GenerationConfig,
        callback: &mut dyn FnMut(&str) -> bool,
    ) -> Result<String> {
        let response = format!(
            "{}[Stub response for: \"{}\", temp={}, max={}]",
            self.response_prefix,
            prompt.chars().take(30).collect::<String>(),
            config.temperature,
            config.max_tokens
        );

        for word in response.split_inclusive(' ') {
            if !callback(word) {
                break;
            }
        }

        self.context_used += prompt.len() / 4 + response.len() / 4;
        Ok(response)
    }

    fn get_state(&self) -> Result<EngineState> {
        Ok(EngineState {
            data: bincode::serialize(&self.context_used).unwrap_or_default(),
            n_tokens: self.context_used,
            engine_id: "stub".to_string(),
        })
    }

    fn set_state(&mut self, state: &EngineState) -> Result<()> {
        self.context_used = state.n_tokens;
        Ok(())
    }

    fn clear(&mut self) {
        self.context_used = 0;
    }

    fn context_used(&self) -> usize {
        self.context_used
    }
}
