use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum EmbeddingError {
    #[error("empty text provided")]
    EmptyText,
    #[error("provider error: {0}")]
    ProviderError(String),
    #[error("model not found: {0}")]
    ModelNotFound(String),
    #[error("invalid dimension: {0}")]
    InvalidDimension(usize),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Embedding {
    pub vector: Vec<f32>,
    pub model: String,
    pub dimension: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingRequest {
    pub texts: Vec<String>,
    pub model: Option<String>,
    pub normalize: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingResponse {
    pub embeddings: Vec<Embedding>,
    pub model: String,
    pub usage: Usage,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Usage {
    pub tokens: usize,
}

pub trait EmbeddingProvider: Send + Sync {
    fn embed(&self, texts: &[String]) -> Result<Vec<Embedding>, EmbeddingError>;
    fn dimension(&self) -> usize;
    fn name(&self) -> &str;
}

pub struct EmbeddingEngine {
    provider: Box<dyn EmbeddingProvider>,
    dimension: usize,
}

impl EmbeddingEngine {
    pub fn new(provider: Box<dyn EmbeddingProvider>) -> Self {
        let dimension = provider.dimension();
        Self {
            provider,
            dimension,
        }
    }

    pub fn embed(&self, texts: &[String]) -> Result<Vec<Embedding>, EmbeddingError> {
        self.provider.embed(texts)
    }

    pub fn embed_single(&self, text: &str) -> Result<Embedding, EmbeddingError> {
        self.provider
            .embed(&[text.to_string()])?
            .into_iter()
            .next()
            .ok_or(EmbeddingError::EmptyText)
    }

    pub fn compute_similarity(a: &Embedding, b: &Embedding) -> f32 {
        if a.dimension != b.dimension {
            return 0.0;
        }

        let dot_product: f32 = a
            .vector
            .iter()
            .zip(b.vector.iter())
            .map(|(x, y)| x * y)
            .sum();

        let norm_a: f32 = a.vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.vector.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            return 0.0;
        }

        dot_product / (norm_a * norm_b)
    }

    pub fn dimension(&self) -> usize {
        self.dimension
    }

    pub fn name(&self) -> &str {
        self.provider.name()
    }
}

pub struct MockEmbeddingProvider {
    dimension: usize,
}

impl MockEmbeddingProvider {
    pub fn new(dimension: usize) -> Self {
        Self { dimension }
    }
}

impl EmbeddingProvider for MockEmbeddingProvider {
    fn embed(&self, texts: &[String]) -> Result<Vec<Embedding>, EmbeddingError> {
        Ok(texts
            .iter()
            .enumerate()
            .map(|(i, text)| {
                let mut vector = vec![0.0; self.dimension];
                let hash = simple_hash(text);
                for j in 0..self.dimension {
                    vector[j] = ((hash + i * j) % 1000) as f32 / 1000.0;
                }
                Embedding {
                    vector,
                    model: "mock".to_string(),
                    dimension: self.dimension,
                }
            })
            .collect())
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn name(&self) -> &str {
        "mock"
    }
}

fn simple_hash(s: &str) -> usize {
    s.bytes()
        .fold(0, |acc, b| acc.wrapping_add(b as usize).wrapping_mul(31))
}

pub fn create_default_engine() -> EmbeddingEngine {
    EmbeddingEngine::new(Box::new(MockEmbeddingProvider::new(384)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedding_engine() {
        let engine = create_default_engine();
        let result = engine.embed_single("Hello world");
        assert!(result.is_ok());
        assert_eq!(result.unwrap().dimension, 384);
    }

    #[test]
    fn test_similarity() {
        let a = Embedding {
            vector: vec![1.0, 0.0, 0.0],
            model: "test".to_string(),
            dimension: 3,
        };
        let b = Embedding {
            vector: vec![1.0, 0.0, 0.0],
            model: "test".to_string(),
            dimension: 3,
        };
        let c = Embedding {
            vector: vec![0.0, 1.0, 0.0],
            model: "test".to_string(),
            dimension: 3,
        };

        assert!((EmbeddingEngine::compute_similarity(&a, &b) - 1.0).abs() < 0.001);
        assert!((EmbeddingEngine::compute_similarity(&a, &c) - 0.0).abs() < 0.001);
    }
}