//! Candle-based LLM inference engine with GGUF support
//!
//! Supports loading quantized GGUF models (llama.cpp format).
//! Works with Llama, Mistral, Phi, Qwen, and other architectures.

use crate::config::GenerationConfig;
use crate::{CortexError, Result};
use candle_core::{quantized::gguf_file, Device, Tensor};
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::quantized_llama::ModelWeights;
use std::path::Path;
use tokenizers::Tokenizer;

use super::{EngineState, TextEngine};

/// Candle-based LLM engine supporting GGUF quantized models
pub struct CandleLLM {
    model: ModelWeights,
    tokenizer: Tokenizer,
    device: Device,
    /// Tokens in current context
    tokens: Vec<u32>,
    /// EOS token ID
    eos_token_id: u32,
    /// Context size
    context_size: usize,
    /// Hidden size for embeddings
    hidden_size: usize,
}

// Safety: CandleLLM is Send when used from single thread context
unsafe impl Send for CandleLLM {}

impl CandleLLM {
    /// Load a GGUF model from file
    pub fn load(model_path: impl AsRef<Path>) -> Result<Self> {
        let model_path = model_path.as_ref();

        println!("Loading model from {:?}...", model_path);

        // Determine device
        let device = Self::get_device()?;
        println!("Using device: {:?}", device);

        // Load GGUF file
        let mut file = std::fs::File::open(model_path)
            .map_err(|e| CortexError::ModelLoad(format!("Failed to open model: {}", e)))?;

        let gguf = gguf_file::Content::read(&mut file)
            .map_err(|e| CortexError::ModelLoad(format!("Failed to read GGUF: {}", e)))?;

        // Extract metadata
        let context_size = Self::get_metadata_u32(&gguf, "llama.context_length")
            .or_else(|| Self::get_metadata_u32(&gguf, "context_length"))
            .unwrap_or(4096) as usize;

        let hidden_size = Self::get_metadata_u32(&gguf, "llama.embedding_length")
            .or_else(|| Self::get_metadata_u32(&gguf, "embedding_length"))
            .unwrap_or(4096) as usize;

        // Get EOS token
        let eos_token_id = Self::get_metadata_u32(&gguf, "tokenizer.ggml.eos_token_id")
            .unwrap_or(2);

        println!("Context size: {}, Hidden size: {}", context_size, hidden_size);

        // Load model weights
        let model = ModelWeights::from_gguf(gguf, &mut file, &device)
            .map_err(|e| CortexError::ModelLoad(format!("Failed to load weights: {}", e)))?;

        // Try to load tokenizer from same directory or HF cache
        let tokenizer = Self::load_tokenizer(model_path)?;

        println!("Model loaded successfully!");

        Ok(Self {
            model,
            tokenizer,
            device,
            tokens: Vec::new(),
            eos_token_id,
            context_size,
            hidden_size,
        })
    }

    fn get_device() -> Result<Device> {
        // Try Metal first (Mac)
        #[cfg(feature = "metal")]
        {
            match Device::new_metal(0) {
                Ok(device) => return Ok(device),
                Err(e) => eprintln!("Metal unavailable: {}, falling back to CPU", e),
            }
        }

        // Try CUDA
        #[cfg(feature = "cuda")]
        {
            match Device::new_cuda(0) {
                Ok(device) => return Ok(device),
                Err(e) => eprintln!("CUDA unavailable: {}, falling back to CPU", e),
            }
        }

        Ok(Device::Cpu)
    }

    fn get_metadata_u32(gguf: &gguf_file::Content, key: &str) -> Option<u32> {
        gguf.metadata.get(key).and_then(|v| {
            match v {
                gguf_file::Value::U32(n) => Some(*n),
                gguf_file::Value::I32(n) => Some(*n as u32),
                gguf_file::Value::U64(n) => Some(*n as u32),
                gguf_file::Value::I64(n) => Some(*n as u32),
                _ => None,
            }
        })
    }

    fn load_tokenizer(model_path: &Path) -> Result<Tokenizer> {
        // Try to find tokenizer in same directory
        let dir = model_path.parent().unwrap_or(Path::new("."));
        let tokenizer_path = dir.join("tokenizer.json");

        if tokenizer_path.exists() {
            return Tokenizer::from_file(&tokenizer_path)
                .map_err(|e| CortexError::ModelLoad(format!("Failed to load tokenizer: {}", e)));
        }

        // Try to detect model type and download tokenizer from HF
        let filename = model_path.file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("")
            .to_lowercase();

        // Use public tokenizers that don't require auth
        let model_id = if filename.contains("llama-3") {
            // Use NousResearch's open Llama 3 tokenizer
            "NousResearch/Hermes-3-Llama-3.1-8B"
        } else if filename.contains("llama") {
            "NousResearch/Llama-2-7b-hf"
        } else if filename.contains("mistral") {
            "mistralai/Mistral-7B-v0.1"
        } else if filename.contains("phi") {
            "microsoft/phi-2"
        } else if filename.contains("qwen") {
            "Qwen/Qwen2-0.5B"
        } else if filename.contains("gemma") {
            "google/gemma-2b"
        } else {
            // Default to open Llama tokenizer
            "NousResearch/Llama-2-7b-hf"
        };

        println!("Downloading tokenizer from {}...", model_id);
        Self::download_tokenizer(model_id)
    }

    fn download_tokenizer(model_id: &str) -> Result<Tokenizer> {
        // Try direct HTTP download
        let url = format!(
            "https://huggingface.co/{}/resolve/main/tokenizer.json",
            model_id
        );

        println!("Fetching tokenizer from {}", url);

        let response = ureq::get(&url)
            .call()
            .map_err(|e| CortexError::ModelLoad(format!("Failed to download tokenizer: {}", e)))?;

        let json: serde_json::Value = response.into_json()
            .map_err(|e| CortexError::ModelLoad(format!("Failed to parse tokenizer JSON: {}", e)))?;

        // Save to cache
        let cache_dir = dirs::cache_dir()
            .unwrap_or_else(|| std::path::PathBuf::from("."))
            .join("cortex")
            .join("tokenizers");
        std::fs::create_dir_all(&cache_dir).ok();

        let cache_path = cache_dir.join(format!("{}.json", model_id.replace('/', "_")));
        std::fs::write(&cache_path, serde_json::to_string(&json).unwrap_or_default()).ok();

        Tokenizer::from_bytes(serde_json::to_vec(&json).unwrap_or_default())
            .map_err(|e| CortexError::ModelLoad(format!("Failed to load tokenizer: {}", e)))
    }

    fn tokenize(&self, text: &str) -> Result<Vec<u32>> {
        let encoding = self.tokenizer.encode(text, true)
            .map_err(|e| CortexError::Inference(format!("Tokenization failed: {}", e)))?;
        Ok(encoding.get_ids().to_vec())
    }

    fn decode(&self, tokens: &[u32]) -> Result<String> {
        self.tokenizer.decode(tokens, true)
            .map_err(|e| CortexError::Inference(format!("Decoding failed: {}", e)))
    }

    fn forward(&mut self, tokens: &[u32], pos: usize) -> Result<Tensor> {
        // Create 1D tensor and add batch dimension for [batch, seq_len]
        let input = Tensor::new(tokens, &self.device)
            .map_err(|e| CortexError::Inference(e.to_string()))?
            .unsqueeze(0)  // Add batch dimension: [seq_len] -> [1, seq_len]
            .map_err(|e| CortexError::Inference(e.to_string()))?;

        self.model.forward(&input, pos)
            .map_err(|e| CortexError::Inference(e.to_string()))
    }

    fn sample(&self, logits: &Tensor, config: &GenerationConfig) -> Result<u32> {
        // Output is [batch, seq_len, vocab_size], we want last token's logits
        let dims = logits.dims();
        let logits = match dims.len() {
            3 => {
                // [batch, seq, vocab] -> get [vocab] for last token
                let seq_len = dims[1];
                logits.get(0)  // Remove batch dim
                    .map_err(|e| CortexError::Inference(e.to_string()))?
                    .get(seq_len - 1)  // Get last seq position
                    .map_err(|e| CortexError::Inference(e.to_string()))?
            }
            2 => {
                // [seq, vocab] -> get last position
                let seq_len = dims[0];
                logits.get(seq_len - 1)
                    .map_err(|e| CortexError::Inference(e.to_string()))?
            }
            1 => logits.clone(),  // Already [vocab]
            _ => return Err(CortexError::Inference(format!("Unexpected logits shape: {:?}", dims))),
        };

        let mut processor = LogitsProcessor::new(
            rand::random(),
            Some(config.temperature as f64),
            Some(config.top_p as f64),
        );

        processor.sample(&logits)
            .map_err(|e| CortexError::Inference(e.to_string()))
    }
}

impl TextEngine for CandleLLM {
    fn embedding_dim(&self) -> usize {
        self.hidden_size
    }

    fn context_size(&self) -> usize {
        self.context_size
    }

    fn embed(&self, text: &str) -> Result<Vec<f32>> {
        // Hash-based embedding for now
        // TODO: Proper embedding via model forward pass
        let tokens = self.tokenize(text)?;
        let hash = tokens.iter().fold(0u64, |acc, &t| {
            acc.wrapping_add(t as u64).wrapping_mul(31)
        });

        let embedding: Vec<f32> = (0..self.hidden_size)
            .map(|i| {
                let seed = hash.wrapping_add(i as u64);
                ((seed % 10000) as f32 / 10000.0) - 0.5
            })
            .collect();

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
        // Tokenize prompt
        let prompt_tokens = self.tokenize(prompt)?;
        let prompt_len = prompt_tokens.len();

        // Clear previous context and set new tokens
        self.clear();
        self.tokens = prompt_tokens.clone();

        // Process prompt tokens one by one to build KV cache
        let mut logits = Tensor::new(&[0f32], &self.device)
            .map_err(|e| CortexError::Inference(e.to_string()))?;

        for (pos, &token) in prompt_tokens.iter().enumerate() {
            logits = self.forward(&[token], pos)?;
        }

        // Generate tokens
        let mut output_tokens = Vec::new();
        let mut output_text = String::new();

        for i in 0..config.max_tokens {
            let next_token = self.sample(&logits, config)?;

            if next_token == self.eos_token_id {
                break;
            }

            output_tokens.push(next_token);
            self.tokens.push(next_token);

            // Decode incrementally
            let new_text = self.decode(&output_tokens)?;
            let delta = if new_text.len() > output_text.len() {
                &new_text[output_text.len()..]
            } else {
                ""
            };

            if !delta.is_empty() {
                if !callback(delta) {
                    break;
                }
                output_text = new_text;
            }

            // Check stop sequences
            let mut should_stop = false;
            for stop in &config.stop {
                if output_text.ends_with(stop) {
                    should_stop = true;
                    break;
                }
            }
            if should_stop {
                break;
            }

            // Forward next token
            let pos = prompt_len + i as usize;
            logits = self.forward(&[next_token], pos)?;
        }

        Ok(output_text)
    }

    fn get_state(&self) -> Result<EngineState> {
        let data = bincode::serialize(&self.tokens)
            .map_err(|e| CortexError::State(e.to_string()))?;

        Ok(EngineState {
            data,
            n_tokens: self.tokens.len(),
            engine_id: "candle".to_string(),
        })
    }

    fn set_state(&mut self, state: &EngineState) -> Result<()> {
        if state.engine_id != "candle" && state.engine_id != "none" {
            return Err(CortexError::State(format!(
                "Cannot restore state from engine '{}'", state.engine_id
            )));
        }

        if !state.data.is_empty() {
            self.tokens = bincode::deserialize(&state.data)
                .map_err(|e| CortexError::State(e.to_string()))?;
        } else {
            self.tokens.clear();
        }

        Ok(())
    }

    fn clear(&mut self) {
        self.tokens.clear();
        // Note: ModelWeights doesn't expose clear_kv_cache directly
        // We rely on position tracking to overwrite old cache entries
    }

    fn context_used(&self) -> usize {
        self.tokens.len()
    }
}
