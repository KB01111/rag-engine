use std::collections::HashMap;

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum StorageError {
    #[error("not found: {0}")]
    NotFound(String),
    #[error("already exists: {0}")]
    AlreadyExists(String),
    #[error("invalid data: {0}")]
    InvalidData(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorRecord {
    pub id: String,
    pub document_id: String,
    pub chunk_text: String,
    pub vector: Vec<f32>,
    pub metadata: HashMap<String, String>,
    pub created_at: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentRecord {
    pub id: String,
    pub title: String,
    pub content: String,
    pub metadata: HashMap<String, String>,
    pub created_at: i64,
    pub updated_at: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub record: VectorRecord,
    pub score: f32,
}

pub trait VectorStore: Send + Sync {
    fn upsert(&self, records: Vec<VectorRecord>) -> Result<(), StorageError>;
    fn delete(&self, document_id: &str) -> Result<(), StorageError>;
    fn search(
        &self,
        query_vector: &[f32],
        top_k: usize,
        filters: Option<&HashMap<String, String>>,
    ) -> Result<Vec<SearchResult>, StorageError>;
    fn count(&self) -> usize;
}

pub struct InMemoryVectorStore {
    vectors: RwLock<HashMap<String, VectorRecord>>,
    documents: RwLock<HashMap<String, DocumentRecord>>,
}

impl InMemoryVectorStore {
    pub fn new() -> Self {
        Self {
            vectors: RwLock::new(HashMap::new()),
            documents: RwLock::new(HashMap::new()),
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            vectors: RwLock::new(HashMap::with_capacity(capacity)),
            documents: RwLock::new(HashMap::new()),
        }
    }

    pub fn stats(&self) -> StorageStats {
        let vectors = self.vectors.read();
        let docs = self.documents.read();

        let total_size: usize = vectors
            .values()
            .map(|r| r.vector.len() * std::mem::size_of::<f32>())
            .sum();

        StorageStats {
            document_count: docs.len(),
            chunk_count: vectors.len(),
            index_size_bytes: total_size,
        }
    }
}

impl Default for InMemoryVectorStore {
    fn default() -> Self {
        Self::new()
    }
}

impl VectorStore for InMemoryVectorStore {
    fn upsert(&self, records: Vec<VectorRecord>) -> Result<(), StorageError> {
        let mut vectors = self.vectors.write();
        for record in records {
            vectors.insert(record.id.clone(), record);
        }
        Ok(())
    }

    fn delete(&self, document_id: &str) -> Result<(), StorageError> {
        let mut vectors = self.vectors.write();
        vectors.retain(|_, v| v.document_id != document_id);

        let mut docs = self.documents.write();
        docs.remove(document_id);

        Ok(())
    }

    fn search(
        &self,
        query_vector: &[f32],
        top_k: usize,
        filters: Option<&HashMap<String, String>>,
    ) -> Result<Vec<SearchResult>, StorageError> {
        let vectors = self.vectors.read();

        let mut results: Vec<SearchResult> = vectors
            .values()
            .filter(|record| {
                if let Some(f) = filters {
                    f.iter()
                        .all(|(k, v)| record.metadata.get(k) == Some(&v.to_string()))
                } else {
                    true
                }
            })
            .map(|record| {
                let score = cosine_similarity(query_vector, &record.vector);
                SearchResult {
                    record: record.clone(),
                    score,
                }
            })
            .collect();

        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        if results.len() > top_k {
            results.truncate(top_k);
        }

        Ok(results)
    }

    fn count(&self) -> usize {
        self.vectors.read().len()
    }
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
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

#[derive(Debug, Clone, Default)]
pub struct StorageStats {
    pub document_count: usize,
    pub chunk_count: usize,
    pub index_size_bytes: usize,
}

pub struct DocumentStore {
    documents: RwLock<HashMap<String, DocumentRecord>>,
}

impl DocumentStore {
    pub fn new() -> Self {
        Self {
            documents: RwLock::new(HashMap::new()),
        }
    }

    pub fn insert(&self, doc: DocumentRecord) -> Result<(), StorageError> {
        let mut docs = self.documents.write();
        if docs.contains_key(&doc.id) {
            return Err(StorageError::AlreadyExists(doc.id));
        }
        docs.insert(doc.id.clone(), doc);
        Ok(())
    }

    pub fn update(&self, doc: DocumentRecord) -> Result<(), StorageError> {
        let mut docs = self.documents.write();
        if !docs.contains_key(&doc.id) {
            return Err(StorageError::NotFound(doc.id));
        }
        docs.insert(doc.id.clone(), doc);
        Ok(())
    }

    pub fn get(&self, id: &str) -> Result<DocumentRecord, StorageError> {
        let docs = self.documents.read();
        docs.get(id)
            .cloned()
            .ok_or_else(|| StorageError::NotFound(id.to_string()))
    }

    pub fn delete(&self, id: &str) -> Result<(), StorageError> {
        let mut docs = self.documents.write();
        if docs.remove(id).is_none() {
            return Err(StorageError::NotFound(id.to_string()));
        }
        Ok(())
    }

    pub fn list(&self) -> Vec<DocumentRecord> {
        let docs = self.documents.read();
        docs.values().cloned().collect()
    }

    pub fn count(&self) -> usize {
        self.documents.read().len()
    }
}

impl Default for DocumentStore {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector_store() {
        let store = InMemoryVectorStore::new();

        let record = VectorRecord {
            id: "test-1".to_string(),
            document_id: "doc1".to_string(),
            chunk_text: "test chunk".to_string(),
            vector: vec![0.1, 0.2, 0.3],
            metadata: HashMap::new(),
            created_at: 0,
        };

        store.upsert(vec![record.clone()]).unwrap();
        assert_eq!(store.count(), 1);

        let results = store.search(&[0.1, 0.2, 0.3], 10, None).unwrap();
        assert!(!results.is_empty());
        assert!((results[0].score - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_document_store() {
        let store = DocumentStore::new();

        let doc = DocumentRecord {
            id: "doc1".to_string(),
            title: "Test".to_string(),
            content: "Content".to_string(),
            metadata: HashMap::new(),
            created_at: 0,
            updated_at: 0,
        };

        store.insert(doc.clone()).unwrap();
        assert_eq!(store.count(), 1);

        let retrieved = store.get("doc1").unwrap();
        assert_eq!(retrieved.title, "Test");
    }
}
