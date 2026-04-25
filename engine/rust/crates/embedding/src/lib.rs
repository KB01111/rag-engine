use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use thiserror::Error;

const DEFAULT_FASTEMBED_MODEL: &str = "sentence-transformers/all-MiniLM-L6-v2";
const DEFAULT_MOCK_MODEL: &str = "mock-384";
const MOCK_VERSION: &str = "mock-v1";
const FASTEMBED_VERSION: &str = "fastembed-5.13.3";

#[derive(Error, Debug)]
pub enum EmbeddingError {
    #[error("empty text provided")]
    EmptyText,
    #[error("unsupported embedding provider: {0}")]
    UnsupportedProvider(String),
    #[error("provider error: {0}")]
    ProviderError(String),
    #[error("model not found: {0}")]
    ModelNotFound(String),
    #[error("embedding downloads are disabled and cache is missing or empty: {0}")]
    DownloadDisabled(String),
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingConfig {
    pub provider: String,
    pub model: String,
    pub cache_dir: Option<PathBuf>,
    pub allow_download: bool,
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            provider: "fastembed".to_string(),
            model: DEFAULT_FASTEMBED_MODEL.to_string(),
            cache_dir: None,
            allow_download: true,
        }
    }
}

impl EmbeddingConfig {
    pub fn mock() -> Self {
        Self {
            provider: "mock".to_string(),
            model: DEFAULT_MOCK_MODEL.to_string(),
            cache_dir: None,
            allow_download: false,
        }
    }
}

pub trait EmbeddingProvider: Send + Sync {
    fn embed(&self, texts: &[String]) -> Result<Vec<Embedding>, EmbeddingError>;
    fn dimension(&self) -> usize;
    fn name(&self) -> &str;
    fn model(&self) -> &str;
    fn version(&self) -> &str;

    fn fingerprint(&self) -> String {
        format!(
            "{}:{}:{}:{}",
            self.name(),
            self.model(),
            self.dimension(),
            self.version()
        )
    }
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

    pub fn model(&self) -> &str {
        self.provider.model()
    }

    pub fn version(&self) -> &str {
        self.provider.version()
    }

    pub fn fingerprint(&self) -> String {
        self.provider.fingerprint()
    }
}

pub struct MockEmbeddingProvider {
    dimension: usize,
    model: String,
}

impl MockEmbeddingProvider {
    pub fn new(dimension: usize) -> Self {
        Self {
            dimension,
            model: format!("mock-{dimension}"),
        }
    }

    pub fn with_model(dimension: usize, model: impl Into<String>) -> Self {
        Self {
            dimension,
            model: model.into(),
        }
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
                    model: self.model.clone(),
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

    fn model(&self) -> &str {
        &self.model
    }

    fn version(&self) -> &str {
        MOCK_VERSION
    }
}

fn simple_hash(s: &str) -> usize {
    s.bytes()
        .fold(0, |acc, b| acc.wrapping_add(b as usize).wrapping_mul(31))
}

pub fn create_default_engine() -> EmbeddingEngine {
    EmbeddingEngine::new(Box::new(MockEmbeddingProvider::new(384)))
}

pub fn create_engine(config: EmbeddingConfig) -> Result<EmbeddingEngine, EmbeddingError> {
    match normalize_provider(&config.provider).as_str() {
        "mock" => Ok(EmbeddingEngine::new(Box::new(
            MockEmbeddingProvider::with_model(384, normalize_mock_model(&config.model)),
        ))),
        "fastembed" => create_fastembed_engine(config),
        other => Err(EmbeddingError::UnsupportedProvider(other.to_string())),
    }
}

fn normalize_provider(provider: &str) -> String {
    match provider.trim().to_ascii_lowercase().as_str() {
        "" => "fastembed".to_string(),
        "fast-embed" | "fast_embed" => "fastembed".to_string(),
        other => other.to_string(),
    }
}

fn normalize_mock_model(model: &str) -> String {
    let model = model.trim();
    if model.is_empty() {
        DEFAULT_MOCK_MODEL.to_string()
    } else {
        model.to_string()
    }
}

#[cfg(feature = "fastembed-provider")]
fn create_fastembed_engine(config: EmbeddingConfig) -> Result<EmbeddingEngine, EmbeddingError> {
    Ok(EmbeddingEngine::new(Box::new(FastEmbedProvider::new(
        config,
    )?)))
}

#[cfg(not(feature = "fastembed-provider"))]
fn create_fastembed_engine(_config: EmbeddingConfig) -> Result<EmbeddingEngine, EmbeddingError> {
    Err(EmbeddingError::UnsupportedProvider(
        "fastembed requires the fastembed-provider feature".to_string(),
    ))
}

#[cfg(feature = "fastembed-provider")]
mod fastembed_provider {
    use std::str::FromStr;
    use std::sync::Mutex;

    use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};

    use super::{
        Embedding, EmbeddingConfig, EmbeddingError, EmbeddingProvider, DEFAULT_FASTEMBED_MODEL,
        FASTEMBED_VERSION,
    };

    pub struct FastEmbedProvider {
        inner: Mutex<TextEmbedding>,
        model: String,
        dimension: usize,
    }

    impl FastEmbedProvider {
        pub fn new(config: EmbeddingConfig) -> Result<Self, EmbeddingError> {
            let model = parse_fastembed_model(&config.model)?;
            if !config.allow_download {
                ensure_cache_available(&config)?;
            }

            let mut options = InitOptions::new(model.clone());
            if let Some(cache_dir) = config.cache_dir.clone() {
                options.cache_dir = cache_dir;
            }
            options.show_download_progress = config.allow_download;

            let inner = TextEmbedding::try_new(options)
                .map_err(|err| EmbeddingError::ProviderError(err.to_string()))?;
            let dimension = model_dimension(&model);
            if dimension == 0 {
                return Err(EmbeddingError::InvalidDimension(dimension));
            }
            Ok(Self {
                inner: Mutex::new(inner),
                model: canonical_model_name(&model).to_string(),
                dimension,
            })
        }
    }

    impl EmbeddingProvider for FastEmbedProvider {
        fn embed(&self, texts: &[String]) -> Result<Vec<Embedding>, EmbeddingError> {
            let mut inner = self
                .inner
                .lock()
                .map_err(|err| EmbeddingError::ProviderError(err.to_string()))?;
            let embeddings = inner
                .embed(texts, None)
                .map_err(|err| EmbeddingError::ProviderError(err.to_string()))?;
            Ok(embeddings
                .into_iter()
                .map(|vector| Embedding {
                    dimension: vector.len(),
                    vector,
                    model: self.model.clone(),
                })
                .collect())
        }

        fn dimension(&self) -> usize {
            self.dimension
        }

        fn name(&self) -> &str {
            "fastembed"
        }

        fn model(&self) -> &str {
            &self.model
        }

        fn version(&self) -> &str {
            FASTEMBED_VERSION
        }
    }

    fn ensure_cache_available(config: &EmbeddingConfig) -> Result<(), EmbeddingError> {
        let Some(cache_dir) = &config.cache_dir else {
            return Err(EmbeddingError::DownloadDisabled(
                "<unset embedding cache dir>".to_string(),
            ));
        };
        if !cache_dir.exists() {
            return Err(EmbeddingError::DownloadDisabled(
                cache_dir.display().to_string(),
            ));
        }
        let mut entries = std::fs::read_dir(cache_dir)
            .map_err(|err| EmbeddingError::ProviderError(err.to_string()))?;
        if entries.next().is_none() {
            return Err(EmbeddingError::DownloadDisabled(
                cache_dir.display().to_string(),
            ));
        }
        Ok(())
    }

    pub fn parse_fastembed_model(model: &str) -> Result<EmbeddingModel, EmbeddingError> {
        match model.trim().to_ascii_lowercase().as_str() {
            "" | "all-minilm-l6-v2" | "sentence-transformers/all-minilm-l6-v2" => {
                Ok(EmbeddingModel::AllMiniLML6V2)
            }
            _ => EmbeddingModel::from_str(model.trim())
                .map_err(|_| EmbeddingError::ModelNotFound(model.to_string())),
        }
    }

    pub fn canonical_model_name(model: &EmbeddingModel) -> &'static str {
        match model {
            EmbeddingModel::AllMiniLML6V2 => DEFAULT_FASTEMBED_MODEL,
            _ => "fastembed-custom",
        }
    }

    pub fn model_dimension(model: &EmbeddingModel) -> usize {
        match model {
            EmbeddingModel::AllMiniLML6V2 => 384,
            _ => 0,
        }
    }
}

#[cfg(feature = "fastembed-provider")]
use fastembed_provider::FastEmbedProvider;

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

    #[test]
    fn test_mock_engine_metadata() {
        let engine = create_engine(EmbeddingConfig::mock()).unwrap();

        assert_eq!(engine.name(), "mock");
        assert_eq!(engine.model(), DEFAULT_MOCK_MODEL);
        assert_eq!(engine.dimension(), 384);
        assert_eq!(
            engine.fingerprint(),
            "mock:mock-384:384:mock-v1".to_string()
        );
    }

    #[test]
    fn test_unknown_provider_fails() {
        let result = create_engine(EmbeddingConfig {
            provider: "remote".to_string(),
            ..EmbeddingConfig::mock()
        });

        assert!(matches!(
            result,
            Err(EmbeddingError::UnsupportedProvider(_))
        ));
    }

    #[cfg(feature = "fastembed-provider")]
    #[test]
    fn test_fastembed_default_model_metadata_without_download() {
        let model =
            fastembed_provider::parse_fastembed_model("sentence-transformers/all-MiniLM-L6-v2")
                .unwrap();

        assert_eq!(
            fastembed_provider::canonical_model_name(&model),
            DEFAULT_FASTEMBED_MODEL
        );
        assert_eq!(fastembed_provider::model_dimension(&model), 384);
    }
}
