use std::collections::HashMap;
use std::sync::Arc;

use anyhow::Result as AnyhowResult;
use chrono::Utc;
use lazy_static::lazy_static;
use log::{debug, error, info, warn};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use validator::Validate;

use chunking::ChunkingConfig;
use embedding::{create_default_engine, EmbeddingEngine};
use storage::{DocumentStore, InMemoryVectorStore, VectorStore};

#[derive(Error, Debug)]
pub enum RagError {
    #[error("document not found: {0}")]
    DocumentNotFound(String),
    #[error("embedding error: {0}")]
    EmbeddingError(String),
    #[error("storage error: {0}")]
    StorageError(String),
    #[error("invalid config: {0}")]
    InvalidConfig(String),
    #[error("validation error: {0}")]
    ValidationError(String),
}

#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
#[validate(schema(function = "validate_content"))]
pub struct UpsertRequest {
    #[validate(length(min = 1, max = 1000))]
    pub document_id: Option<String>,
    #[validate(length(min = 1, max = 1_000_000))]
    pub content: String,
    pub metadata: HashMap<String, String>,
    pub title: Option<String>,
}

fn validate_content(s: &UpsertRequest) -> Result<(), validator::ValidationError> {
    if s.content.trim().is_empty() {
        return Err(validator::ValidationError::new("empty_content"));
    }
    Ok(())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpsertResponse {
    pub document_id: String,
    pub chunks_indexed: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct SearchRequest {
    #[validate(length(min = 1, max = 10000))]
    pub query: String,
    #[validate(range(min = 1, max = 1000))]
    pub top_k: usize,
    pub filters: Option<HashMap<String, String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResponse {
    pub results: Vec<SearchHit>,
    pub query_time_ms: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchHit {
    pub document_id: String,
    pub chunk_text: String,
    pub score: f32,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RagStatus {
    pub document_count: usize,
    pub chunk_count: usize,
    pub embedding_model: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentInfo {
    pub id: String,
    pub title: String,
    pub chunk_count: usize,
    pub created_at: i64,
    pub updated_at: i64,
}

#[derive(Debug, Clone)]
pub struct RagConfig {
    pub chunk_size: usize,
    pub chunk_overlap: usize,
    pub top_k: usize,
}

impl Default for RagConfig {
    fn default() -> Self {
        Self {
            chunk_size: 512,
            chunk_overlap: 50,
            top_k: 10,
        }
    }
}

lazy_static! {
    static ref DEFAULT_ENGINE: Arc<EmbeddingEngine> = Arc::new(create_default_engine());
}

pub struct RagEngine {
    vector_store: Arc<dyn VectorStore>,
    document_store: Arc<DocumentStore>,
    embedding_engine: Arc<EmbeddingEngine>,
    chunking_config: ChunkingConfig,
    top_k: usize,
}

impl RagEngine {
    pub fn new(config: Option<RagConfig>) -> Self {
        let config = config.unwrap_or_default();

        info!("Creating RAG Engine with chunk_size={}", config.chunk_size);

        Self {
            vector_store: Arc::new(InMemoryVectorStore::new()),
            document_store: Arc::new(DocumentStore::new()),
            embedding_engine: DEFAULT_ENGINE.clone(),
            chunking_config: ChunkingConfig {
                chunk_size: config.chunk_size,
                chunk_overlap: config.chunk_overlap,
                min_chunk_size: 50,
            },
            top_k: config.top_k,
        }
    }

    pub fn with_stores(
        vector_store: Arc<dyn VectorStore>,
        document_store: Arc<DocumentStore>,
    ) -> Self {
        Self {
            vector_store,
            document_store,
            embedding_engine: DEFAULT_ENGINE.clone(),
            chunking_config: ChunkingConfig::default(),
            top_k: 10,
        }
    }

    pub fn upsert(&self, request: UpsertRequest) -> AnyhowResult<UpsertResponse> {
        request
            .validate()
            .map_err(|e| anyhow::anyhow!("validation failed: {}", e))?;

        debug!("Upserting document: {:?}", request.document_id);

        let document_id = request
            .document_id
            .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
        let now = Utc::now().timestamp();

        let chunks = chunking::chunk_text(&request.content, &self.chunking_config)
            .map_err(|e| RagError::InvalidConfig(e.to_string()))?;

        if chunks.is_empty() {
            warn!("No chunks generated from content");
        }

        let chunk_texts: Vec<String> = chunks.iter().map(|c| c.text.clone()).collect();

        let embeddings = self
            .embedding_engine
            .embed(&chunk_texts, None)
            .map_err(|e| RagError::EmbeddingError(e.to_string()))?;

        let mut vector_records = Vec::with_capacity(chunks.len());

        for (i, chunk) in chunks.iter().enumerate() {
            vector_records.push(storage::VectorRecord {
                id: uuid::Uuid::new_v4().to_string(),
                document_id: document_id.clone(),
                chunk_text: chunk.text.clone(),
                vector: embeddings[i].vector.clone(),
                metadata: request.metadata.clone(),
                created_at: now,
            });
        }

        self.vector_store
            .upsert(vector_records)
            .map_err(|e| RagError::StorageError(e.to_string()))?;

        let title = request.title.unwrap_or_else(|| document_id.clone());

        let doc = storage::DocumentRecord {
            id: document_id.clone(),
            title,
            content: request.content,
            metadata: request.metadata,
            created_at: now,
            updated_at: now,
        };

        if let Err(e) = self.document_store.insert(doc) {
            if let storage::StorageError::AlreadyExists(_) = e {
                debug!("Document {} already exists, updating", document_id);
            } else {
                error!("Failed to insert document: {}", e);
                return Err(RagError::StorageError(e.to_string()).into());
            }
        }

        info!(
            "Indexed {} chunks for document {}",
            chunks.len(),
            document_id
        );

        Ok(UpsertResponse {
            document_id,
            chunks_indexed: chunks.len(),
        })
    }

    pub fn delete(&self, document_id: &str) -> AnyhowResult<()> {
        debug!("Deleting document: {}", document_id);

        self.vector_store
            .delete(document_id)
            .map_err(|e| RagError::StorageError(e.to_string()))?;

        info!("Deleted document: {}", document_id);
        Ok(())
    }

    pub fn search(&self, request: SearchRequest) -> AnyhowResult<SearchResponse> {
        request
            .validate()
            .map_err(|e| anyhow::anyhow!("validation failed: {}", e))?;

        debug!("Searching: {}", request.query);

        let start = std::time::Instant::now();

        let query_embedding = self
            .embedding_engine
            .embed_single(&request.query, None)
            .map_err(|e| RagError::EmbeddingError(e.to_string()))?;

        let top_k = if request.top_k > 0 {
            request.top_k
        } else {
            self.top_k
        };

        let results = self
            .vector_store
            .search(&query_embedding.vector, top_k, request.filters.as_ref())
            .map_err(|e| RagError::StorageError(e.to_string()))?;

        let hits: Vec<SearchHit> = results
            .into_iter()
            .map(|r| SearchHit {
                document_id: r.record.document_id,
                chunk_text: r.record.chunk_text,
                score: r.score,
                metadata: r.record.metadata,
            })
            .collect();

        let query_time_ms = start.elapsed().as_secs_f64() * 1000.0;

        debug!(
            "Search completed in {:.2}ms, found {} results",
            query_time_ms,
            hits.len()
        );

        Ok(SearchResponse {
            results: hits,
            query_time_ms,
        })
    }

    pub fn get_status(&self) -> RagStatus {
        RagStatus {
            document_count: self.document_store.count(),
            chunk_count: self.vector_store.count(),
            embedding_model: "default".to_string(),
        }
    }

    pub fn list_documents(&self) -> Vec<DocumentInfo> {
        self.document_store
            .list()
            .into_iter()
            .map(|doc| {
                let chunk_count = self
                    .vector_store
                    .search(
                        &vec![0.0; 384],
                        usize::MAX,
                        Some(&HashMap::from([(
                            "document_id".to_string(),
                            doc.id.clone(),
                        )])),
                    )
                    .map(|r| r.len())
                    .unwrap_or(0);

                DocumentInfo {
                    id: doc.id,
                    title: doc.title,
                    chunk_count,
                    created_at: doc.created_at,
                    updated_at: doc.updated_at,
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rag_upsert() {
        let engine = RagEngine::new(None);

        let request = UpsertRequest {
            document_id: None,
            content: "This is a test document with some content for testing.".to_string(),
            metadata: HashMap::new(),
            title: Some("Test Doc".to_string()),
        };

        let result = engine.upsert(request).unwrap();
        assert!(!result.document_id.is_empty());
        assert!(result.chunks_indexed > 0);
    }

    #[test]
    fn test_rag_search() {
        let engine = RagEngine::new(None);

        engine
            .upsert(UpsertRequest {
                document_id: Some("doc1".to_string()),
                content: "The quick brown fox jumps over the lazy dog.".to_string(),
                metadata: HashMap::new(),
                title: None,
            })
            .unwrap();

        let response = engine
            .search(SearchRequest {
                query: "fox dog".to_string(),
                top_k: 5,
                filters: None,
            })
            .unwrap();

        assert!(!response.results.is_empty());
    }

    #[test]
    fn test_rag_delete() {
        let engine = RagEngine::new(None);

        let doc_id = "delete_test".to_string();
        engine
            .upsert(UpsertRequest {
                document_id: Some(doc_id.clone()),
                content: "Content to delete".to_string(),
                metadata: HashMap::new(),
                title: None,
            })
            .unwrap();

        engine.delete(&doc_id).unwrap();

        let status = engine.get_status();
        assert_eq!(status.chunk_count, 0);
    }
}
