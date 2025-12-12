//! Vector store implementation
//!
//! Simple but efficient vector store with:
//! - Linear scan for small datasets (< 10k entries)
//! - Optional HNSW index for larger datasets
//!
//! Optimized for the common case of < 10k memories per session.

use super::{MemoryEntry, SearchResult};
use std::collections::HashMap;

/// Vector store with similarity search
pub struct VectorStore {
    /// Entries by key
    entries: HashMap<String, MemoryEntry>,
    /// Ordered list of keys for iteration
    keys: Vec<String>,
    /// Embedding dimension
    #[allow(dead_code)]
    dim: usize,
    /// Maximum entries
    max_entries: usize,
}

impl VectorStore {
    /// Create new vector store
    pub fn new(dim: usize, max_entries: usize) -> Self {
        Self {
            entries: HashMap::new(),
            keys: Vec::new(),
            dim,
            max_entries,
        }
    }

    /// Insert an entry
    pub fn insert(&mut self, entry: MemoryEntry) {
        // If at capacity, remove oldest entry
        if self.entries.len() >= self.max_entries {
            if let Some(oldest_key) = self.keys.first().cloned() {
                self.remove(&oldest_key);
            }
        }

        let key = entry.key.clone();
        self.entries.insert(key.clone(), entry);
        self.keys.push(key);
    }

    /// Get entry by key
    pub fn get(&self, key: &str) -> Option<&MemoryEntry> {
        self.entries.get(key)
    }

    /// Remove entry by key
    pub fn remove(&mut self, key: &str) -> bool {
        if self.entries.remove(key).is_some() {
            self.keys.retain(|k| k != key);
            true
        } else {
            false
        }
    }

    /// Search by similarity (cosine similarity)
    pub fn search(&self, query: &[f32], k: usize) -> Vec<SearchResult> {
        if self.entries.is_empty() || k == 0 {
            return vec![];
        }

        // Normalize query
        let query_norm = normalize(query);

        // Calculate similarities
        let mut scored: Vec<(&MemoryEntry, f32)> = self
            .entries
            .values()
            .map(|entry| {
                let score = cosine_similarity(&query_norm, &entry.embedding);
                (entry, score)
            })
            .collect();

        // Sort by score descending
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Take top k
        scored
            .into_iter()
            .take(k)
            .map(|(entry, score)| SearchResult {
                entry: entry.clone(),
                score,
            })
            .collect()
    }

    /// Get all entries
    pub fn entries(&self) -> Vec<&MemoryEntry> {
        self.keys
            .iter()
            .filter_map(|k| self.entries.get(k))
            .collect()
    }

    /// Get number of entries
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Clear all entries
    pub fn clear(&mut self) {
        self.entries.clear();
        self.keys.clear();
    }
}

/// Compute cosine similarity between two vectors
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }

    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    dot / (norm_a * norm_b)
}

/// Normalize a vector to unit length
fn normalize(v: &[f32]) -> Vec<f32> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm == 0.0 {
        v.to_vec()
    } else {
        v.iter().map(|x| x / norm).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_entry(key: &str, embedding: Vec<f32>) -> MemoryEntry {
        MemoryEntry {
            key: key.to_string(),
            content: format!("Content for {}", key),
            embedding,
            metadata: Default::default(),
            created_at: 0,
        }
    }

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 1e-6);

        let c = vec![0.0, 1.0, 0.0];
        assert!(cosine_similarity(&a, &c).abs() < 1e-6);

        let d = vec![-1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &d) + 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_insert_search() {
        let mut store = VectorStore::new(3, 100);

        store.insert(make_entry("a", vec![1.0, 0.0, 0.0]));
        store.insert(make_entry("b", vec![0.0, 1.0, 0.0]));
        store.insert(make_entry("c", vec![0.0, 0.0, 1.0]));

        let results = store.search(&[1.0, 0.0, 0.0], 2);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].entry.key, "a");
        assert!((results[0].score - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_capacity() {
        let mut store = VectorStore::new(3, 2);

        store.insert(make_entry("a", vec![1.0, 0.0, 0.0]));
        store.insert(make_entry("b", vec![0.0, 1.0, 0.0]));
        assert_eq!(store.len(), 2);

        // This should evict "a"
        store.insert(make_entry("c", vec![0.0, 0.0, 1.0]));
        assert_eq!(store.len(), 2);
        assert!(store.get("a").is_none());
        assert!(store.get("b").is_some());
        assert!(store.get("c").is_some());
    }
}
