//! Dedicated embedding model for semantic similarity
//!
//! Uses a small BERT-based model (all-MiniLM-L6-v2) for high-quality
//! sentence embeddings. This is separate from the main LLM.

use crate::{CortexError, Result};
use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config as BertConfig, DTYPE};
use std::path::PathBuf;
use tokenizers::{PaddingParams, Tokenizer, TruncationParams};

/// Embedding model for semantic similarity search
pub struct Embedder {
    model: BertModel,
    tokenizer: Tokenizer,
    device: Device,
    dim: usize,
}

impl Embedder {
    /// Load the default embedding model (all-MiniLM-L6-v2)
    pub fn load_default() -> Result<Self> {
        Self::load("sentence-transformers/all-MiniLM-L6-v2")
    }

    /// Load an embedding model from HuggingFace
    pub fn load(model_id: &str) -> Result<Self> {
        println!("Loading embedding model: {}...", model_id);

        let device = Self::get_device()?;
        let (model_path, tokenizer_path, config_path) = Self::download_model(model_id)?;

        // Load config
        let config_str = std::fs::read_to_string(&config_path)
            .map_err(|e| CortexError::ModelLoad(format!("Failed to read config: {}", e)))?;
        let config: BertConfig = serde_json::from_str(&config_str)
            .map_err(|e| CortexError::ModelLoad(format!("Failed to parse config: {}", e)))?;

        let dim = config.hidden_size;

        // Load model weights
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[model_path], DTYPE, &device)
                .map_err(|e| CortexError::ModelLoad(format!("Failed to load weights: {}", e)))?
        };

        let model = BertModel::load(vb, &config)
            .map_err(|e| CortexError::ModelLoad(format!("Failed to build model: {}", e)))?;

        // Load tokenizer
        let mut tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| CortexError::ModelLoad(format!("Failed to load tokenizer: {}", e)))?;

        // Configure tokenizer for batch processing
        let padding = PaddingParams {
            strategy: tokenizers::PaddingStrategy::BatchLongest,
            ..Default::default()
        };
        tokenizer.with_padding(Some(padding));

        let truncation = TruncationParams {
            max_length: 512,
            ..Default::default()
        };
        tokenizer
            .with_truncation(Some(truncation))
            .map_err(|e| CortexError::ModelLoad(format!("Failed to set truncation: {}", e)))?;

        println!("Embedding model loaded! (dim={})", dim);

        Ok(Self {
            model,
            tokenizer,
            device,
            dim,
        })
    }

    fn get_device() -> Result<Device> {
        #[cfg(feature = "metal")]
        {
            if let Ok(device) = Device::new_metal(0) {
                return Ok(device);
            }
        }

        #[cfg(feature = "cuda")]
        {
            if let Ok(device) = Device::new_cuda(0) {
                return Ok(device);
            }
        }

        Ok(Device::Cpu)
    }

    fn download_model(model_id: &str) -> Result<(PathBuf, PathBuf, PathBuf)> {
        let api = hf_hub::api::sync::Api::new()
            .map_err(|e| CortexError::ModelLoad(format!("Failed to create HF API: {}", e)))?;

        let repo = api.model(model_id.to_string());

        let model_path = repo
            .get("model.safetensors")
            .map_err(|e| CortexError::ModelLoad(format!("Failed to download model: {}", e)))?;

        let tokenizer_path = repo
            .get("tokenizer.json")
            .map_err(|e| CortexError::ModelLoad(format!("Failed to download tokenizer: {}", e)))?;

        let config_path = repo
            .get("config.json")
            .map_err(|e| CortexError::ModelLoad(format!("Failed to download config: {}", e)))?;

        Ok((model_path, tokenizer_path, config_path))
    }

    /// Get the embedding dimension
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Embed a single text
    pub fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let embeddings = self.embed_batch(&[text])?;
        Ok(embeddings.into_iter().next().unwrap())
    }

    /// Embed multiple texts efficiently
    pub fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(vec![]);
        }

        // Tokenize all texts
        let encodings = self
            .tokenizer
            .encode_batch(texts.to_vec(), true)
            .map_err(|e| CortexError::Inference(format!("Tokenization failed: {}", e)))?;

        let batch_size = encodings.len();
        let seq_len = encodings[0].get_ids().len();

        // Build input tensors
        let mut input_ids = Vec::with_capacity(batch_size * seq_len);
        let mut attention_mask = Vec::with_capacity(batch_size * seq_len);
        let mut token_type_ids = Vec::with_capacity(batch_size * seq_len);

        for encoding in &encodings {
            input_ids.extend(encoding.get_ids().iter().map(|&id| id as i64));
            attention_mask.extend(encoding.get_attention_mask().iter().map(|&m| m as i64));
            token_type_ids.extend(encoding.get_type_ids().iter().map(|&t| t as i64));
        }

        let input_ids = Tensor::from_vec(input_ids, (batch_size, seq_len), &self.device)
            .map_err(|e| CortexError::Inference(e.to_string()))?;
        let attention_mask = Tensor::from_vec(attention_mask, (batch_size, seq_len), &self.device)
            .map_err(|e| CortexError::Inference(e.to_string()))?;
        let token_type_ids = Tensor::from_vec(token_type_ids, (batch_size, seq_len), &self.device)
            .map_err(|e| CortexError::Inference(e.to_string()))?;

        // Forward pass
        let output = self
            .model
            .forward(&input_ids, &token_type_ids, Some(&attention_mask))
            .map_err(|e| CortexError::Inference(format!("Forward pass failed: {}", e)))?;

        // Mean pooling with attention mask
        let embeddings = self.mean_pooling(&output, &attention_mask)?;

        // L2 normalize
        let embeddings = self.normalize(&embeddings)?;

        // Convert to Vec<Vec<f32>>
        let embeddings: Vec<Vec<f32>> = embeddings
            .to_vec2()
            .map_err(|e| CortexError::Inference(e.to_string()))?;

        Ok(embeddings)
    }

    fn mean_pooling(&self, hidden_states: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        // hidden_states: [batch, seq, hidden]
        // attention_mask: [batch, seq]

        // Expand attention mask to match hidden states
        let mask = attention_mask
            .unsqueeze(2)
            .map_err(|e| CortexError::Inference(e.to_string()))?
            .to_dtype(hidden_states.dtype())
            .map_err(|e| CortexError::Inference(e.to_string()))?;

        // Mask and sum
        let masked = hidden_states
            .broadcast_mul(&mask)
            .map_err(|e| CortexError::Inference(e.to_string()))?;

        let summed = masked
            .sum(1)
            .map_err(|e| CortexError::Inference(e.to_string()))?;

        // Count non-padding tokens
        let counts = mask
            .sum(1)
            .map_err(|e| CortexError::Inference(e.to_string()))?
            .clamp(1e-9, f64::MAX)
            .map_err(|e| CortexError::Inference(e.to_string()))?;

        // Mean
        summed
            .broadcast_div(&counts)
            .map_err(|e| CortexError::Inference(e.to_string()))
    }

    fn normalize(&self, embeddings: &Tensor) -> Result<Tensor> {
        let norms = embeddings
            .sqr()
            .map_err(|e| CortexError::Inference(e.to_string()))?
            .sum_keepdim(1)
            .map_err(|e| CortexError::Inference(e.to_string()))?
            .sqrt()
            .map_err(|e| CortexError::Inference(e.to_string()))?
            .clamp(1e-12, f64::MAX)
            .map_err(|e| CortexError::Inference(e.to_string()))?;

        embeddings
            .broadcast_div(&norms)
            .map_err(|e| CortexError::Inference(e.to_string()))
    }
}

// Safety: Embedder is Send when used from single thread context
unsafe impl Send for Embedder {}
unsafe impl Sync for Embedder {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore] // Requires model download
    fn test_embed() {
        let embedder = Embedder::load_default().unwrap();

        let emb1 = embedder.embed("The cat sat on the mat").unwrap();
        let emb2 = embedder.embed("A feline rested on the rug").unwrap();
        let emb3 = embedder.embed("Python is a programming language").unwrap();

        // Similar sentences should have higher similarity
        let sim_12: f32 = emb1.iter().zip(&emb2).map(|(a, b)| a * b).sum();
        let sim_13: f32 = emb1.iter().zip(&emb3).map(|(a, b)| a * b).sum();

        println!("cat/feline similarity: {}", sim_12);
        println!("cat/python similarity: {}", sim_13);

        assert!(sim_12 > sim_13, "Similar sentences should have higher similarity");
    }
}
