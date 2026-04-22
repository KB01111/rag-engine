use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::types::Float32Type;
use arrow_array::{
    Array, BooleanArray, FixedSizeListArray, Float32Array, Int64Array, RecordBatch, StringArray,
};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use futures::TryStreamExt;
use lancedb::index::{scalar::FtsIndexBuilder, Index};
use lancedb::query::ExecutableQuery;
use lancedb::{connect, Connection, Table};
use parking_lot::RwLock as ParkingRwLock;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokio::sync::RwLock as AsyncRwLock;

#[derive(Error, Debug)]
pub enum StorageError {
    #[error("not found: {0}")]
    NotFound(String),
    #[error("already exists: {0}")]
    AlreadyExists(String),
    #[error("invalid data: {0}")]
    InvalidData(String),
    #[error("backend error: {0}")]
    Backend(String),
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

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum SearchMode {
    Auto,
    Vector,
    Fts,
    Hybrid,
}

#[derive(Debug, Clone)]
pub struct ChunkSearchQuery {
    pub vector: Vec<f32>,
    pub text_query: Option<String>,
    pub top_k: usize,
    pub filters: HashMap<String, String>,
    pub mode: SearchMode,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelRecord {
    pub id: String,
    pub name: String,
    pub path: String,
    pub backend: String,
    pub status: String,
    pub metadata_json: String,
    pub size_bytes: i64,
    pub updated_at: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingRunRecord {
    pub id: String,
    pub name: String,
    pub model_id: String,
    pub dataset_path: String,
    pub status: String,
    pub progress: f32,
    pub error: String,
    pub backend: String,
    pub artifact_dir: String,
    pub config_json: String,
    pub started_at: i64,
    pub completed_at: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingLogRecord {
    pub run_id: String,
    pub level: String,
    pub message: String,
    pub fields_json: String,
    pub timestamp: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MCPConnectionRecord {
    pub connection_id: String,
    pub server_url: String,
    pub server_name: String,
    pub connected: bool,
    /// WARNING: This field stores authentication credentials.
    /// TODO: Replace with encrypted storage or secret reference (e.g., auth_ref or encrypted_auth)
    /// to avoid persisting plaintext secrets.
    pub auth_json: String,
    pub tools_json: String,
    pub updated_at: i64,
}

#[derive(Debug, Clone)]
pub struct EngineStore {
    uri: String,
    gate: Arc<AsyncRwLock<()>>,
}

impl EngineStore {
    pub fn new(uri: impl Into<String>) -> Self {
        Self {
            uri: uri.into(),
            gate: Arc::new(AsyncRwLock::new(())),
        }
    }

    pub async fn upsert_document(
        &self,
        document: DocumentRecord,
        chunks: Vec<VectorRecord>,
    ) -> Result<(), StorageError> {
        let _guard = self.gate.write().await;
        let db = self.connect().await?;
        let documents = self.ensure_documents_table(&db).await?;
        let document_id = document.id.clone();

        delete_where(
            &documents,
            &format!("id = '{}'", escape_literal(&document_id)),
        )
        .await?;
        add_record_batch(&documents, document_batch(&[document])?).await?;

        // Only create chunks table and perform chunk operations when we have actual chunks
        if !chunks.is_empty() {
            let vector_dim = chunks[0].vector.len();
            let chunks_table = self.ensure_chunks_table(&db, vector_dim).await?;

            delete_where(
                &chunks_table,
                &format!("document_id = '{}'", escape_literal(&document_id)),
            )
            .await?;
            add_record_batch(&chunks_table, chunk_batch(&chunks)?).await?;
            // self.ensure_chunk_indexes(&chunks_table).await?;
        } else if let Some(chunks_table) = self.open_table(&db, "chunks").await? {
            // If chunks table exists but we have no chunks, still delete old chunks for this document
            delete_where(
                &chunks_table,
                &format!("document_id = '{}'", escape_literal(&document_id)),
            )
            .await?;
        }
        Ok(())
    }

    pub async fn delete_document(&self, document_id: &str) -> Result<(), StorageError> {
        let _guard = self.gate.write().await;
        let db = self.connect().await?;
        if let Some(table) = self.open_table(&db, "documents").await? {
            delete_where(&table, &format!("id = '{}'", escape_literal(document_id))).await?;
        }
        if let Some(table) = self.open_table(&db, "chunks").await? {
            delete_where(
                &table,
                &format!("document_id = '{}'", escape_literal(document_id)),
            )
            .await?;
        }
        Ok(())
    }

    pub async fn list_documents(&self) -> Result<Vec<DocumentRecord>, StorageError> {
        let _guard = self.gate.read().await;
        let db = self.connect().await?;
        let Some(table) = self.open_table(&db, "documents").await? else {
            return Ok(Vec::new());
        };
        let batches = collect_batches(&table).await?;
        decode_documents(&batches)
    }

    pub async fn search_chunks(
        &self,
        query: ChunkSearchQuery,
    ) -> Result<Vec<SearchResult>, StorageError> {
        let _guard = self.gate.read().await;
        let db = self.connect().await?;
        let Some(table) = self.open_table(&db, "chunks").await? else {
            return Ok(Vec::new());
        };

        let batches = collect_batches(&table).await?;
        let resolved_mode = resolve_mode(&query);
        let text_query = query.text_query.unwrap_or_default();
        let mut results = decode_chunks(&batches)?
            .into_iter()
            .filter(|record| matches_filters(record, &query.filters))
            .map(|record| {
                let vector_score = if query.vector.is_empty() {
                    0.0
                } else {
                    cosine_similarity(&query.vector, &record.vector)
                };
                let text_score = if text_query.is_empty() {
                    0.0
                } else {
                    lexical_score(&text_query, &record.chunk_text)
                };
                let score = match resolved_mode {
                    SearchMode::Vector => vector_score,
                    SearchMode::Fts => text_score,
                    SearchMode::Hybrid | SearchMode::Auto => {
                        if vector_score == 0.0 {
                            text_score
                        } else if text_score == 0.0 {
                            vector_score
                        } else {
                            (vector_score * 0.7) + (text_score * 0.3)
                        }
                    }
                };
                SearchResult { record, score }
            })
            .filter(|result| result.score > 0.0)
            .collect::<Vec<_>>();

        results.sort_by(|left, right| {
            right
                .score
                .partial_cmp(&left.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(query.top_k.max(1));
        Ok(results)
    }

    pub async fn list_chunks(&self) -> Result<Vec<VectorRecord>, StorageError> {
        let _guard = self.gate.read().await;
        let db = self.connect().await?;
        let Some(table) = self.open_table(&db, "chunks").await? else {
            return Ok(Vec::new());
        };
        let batches = collect_batches(&table).await?;
        decode_chunks(&batches)
    }

    pub async fn upsert_model(&self, model: ModelRecord) -> Result<(), StorageError> {
        let _guard = self.gate.write().await;
        let db = self.connect().await?;
        let table = self.ensure_models_table(&db).await?;
        // TODO: Replace delete+add with atomic merge_insert:
        // table.merge_insert(&["id"]).when_matched_update_all().when_not_matched_insert_all().execute(batch)
        delete_where(&table, &format!("id = '{}'", escape_literal(&model.id))).await?;
        add_record_batch(&table, model_batch(&[model])?).await
    }

    pub async fn list_models(&self) -> Result<Vec<ModelRecord>, StorageError> {
        let _guard = self.gate.read().await;
        let db = self.connect().await?;
        let Some(table) = self.open_table(&db, "models").await? else {
            return Ok(Vec::new());
        };
        let batches = collect_batches(&table).await?;
        decode_models(&batches)
    }

    pub async fn upsert_training_run(&self, run: TrainingRunRecord) -> Result<(), StorageError> {
        let _guard = self.gate.write().await;
        let db = self.connect().await?;
        let table = self.ensure_training_runs_table(&db).await?;
        // TODO: Replace delete+add with atomic merge_insert:
        // table.merge_insert(&["id"]).when_matched_update_all().when_not_matched_insert_all().execute(batch)
        delete_where(&table, &format!("id = '{}'", escape_literal(&run.id))).await?;
        add_record_batch(&table, training_run_batch(&[run])?).await
    }

    pub async fn list_training_runs(&self) -> Result<Vec<TrainingRunRecord>, StorageError> {
        let _guard = self.gate.read().await;
        let db = self.connect().await?;
        let Some(table) = self.open_table(&db, "training_runs").await? else {
            return Ok(Vec::new());
        };
        let batches = collect_batches(&table).await?;
        decode_training_runs(&batches)
    }

    pub async fn append_training_logs(
        &self,
        logs: Vec<TrainingLogRecord>,
    ) -> Result<(), StorageError> {
        if logs.is_empty() {
            return Ok(());
        }
        let _guard = self.gate.write().await;
        let db = self.connect().await?;
        let table = self.ensure_training_logs_table(&db).await?;
        add_record_batch(&table, training_log_batch(&logs)?).await
    }

    pub async fn list_training_logs(
        &self,
        run_id: &str,
    ) -> Result<Vec<TrainingLogRecord>, StorageError> {
        let _guard = self.gate.read().await;
        let db = self.connect().await?;
        let Some(table) = self.open_table(&db, "training_logs").await? else {
            return Ok(Vec::new());
        };
        let batches = collect_batches(&table).await?;
        let mut logs = decode_training_logs(&batches)?;
        logs.retain(|log| log.run_id == run_id);
        logs.sort_by_key(|log| log.timestamp);
        Ok(logs)
    }

    pub async fn upsert_mcp_connection(
        &self,
        connection: MCPConnectionRecord,
    ) -> Result<(), StorageError> {
        let _guard = self.gate.write().await;
        let db = self.connect().await?;
        let table = self.ensure_mcp_connections_table(&db).await?;
        // TODO: Replace delete+add with atomic merge_insert:
        // table.merge_insert(&["connection_id"]).when_matched_update_all().when_not_matched_insert_all().execute(batch)
        delete_where(
            &table,
            &format!(
                "connection_id = '{}'",
                escape_literal(&connection.connection_id)
            ),
        )
        .await?;
        add_record_batch(&table, mcp_connection_batch(&[connection])?).await
    }

    pub async fn list_mcp_connections(&self) -> Result<Vec<MCPConnectionRecord>, StorageError> {
        let _guard = self.gate.read().await;
        let db = self.connect().await?;
        let Some(table) = self.open_table(&db, "mcp_connections").await? else {
            return Ok(Vec::new());
        };
        let batches = collect_batches(&table).await?;
        decode_mcp_connections(&batches)
    }

    pub async fn delete_mcp_connection(&self, connection_id: &str) -> Result<(), StorageError> {
        let _guard = self.gate.write().await;
        let db = self.connect().await?;
        let Some(table) = self.open_table(&db, "mcp_connections").await? else {
            return Ok(());
        };
        delete_where(
            &table,
            &format!("connection_id = '{}'", escape_literal(connection_id)),
        )
        .await
    }

    async fn connect(&self) -> Result<Connection, StorageError> {
        connect(&self.uri)
            .execute()
            .await
            .map_err(|error| StorageError::Backend(error.to_string()))
    }

    async fn ensure_documents_table(&self, db: &Connection) -> Result<Table, StorageError> {
        if let Some(table) = self.open_table(db, "documents").await? {
            return Ok(table);
        }
        db.create_table("documents", RecordBatch::new_empty(document_schema()))
            .execute()
            .await
            .map_err(|error| StorageError::Backend(error.to_string()))?;
        self.open_table(db, "documents").await?.ok_or_else(|| {
            StorageError::Backend("documents table missing after creation".to_string())
        })
    }

    async fn ensure_chunks_table(
        &self,
        db: &Connection,
        vector_dimension: usize,
    ) -> Result<Table, StorageError> {
        if let Some(table) = self.open_table(db, "chunks").await? {
            return Ok(table);
        }
        db.create_table(
            "chunks",
            RecordBatch::new_empty(chunk_schema(vector_dimension)),
        )
        .execute()
        .await
        .map_err(|error| StorageError::Backend(error.to_string()))?;
        self.open_table(db, "chunks")
            .await?
            .ok_or_else(|| StorageError::Backend("chunks table missing after creation".to_string()))
    }

    async fn ensure_models_table(&self, db: &Connection) -> Result<Table, StorageError> {
        if let Some(table) = self.open_table(db, "models").await? {
            return Ok(table);
        }
        db.create_table("models", RecordBatch::new_empty(model_schema()))
            .execute()
            .await
            .map_err(|error| StorageError::Backend(error.to_string()))?;
        self.open_table(db, "models")
            .await?
            .ok_or_else(|| StorageError::Backend("models table missing after creation".to_string()))
    }

    async fn ensure_training_runs_table(&self, db: &Connection) -> Result<Table, StorageError> {
        if let Some(table) = self.open_table(db, "training_runs").await? {
            return Ok(table);
        }
        db.create_table(
            "training_runs",
            RecordBatch::new_empty(training_run_schema()),
        )
        .execute()
        .await
        .map_err(|error| StorageError::Backend(error.to_string()))?;
        self.open_table(db, "training_runs").await?.ok_or_else(|| {
            StorageError::Backend("training_runs table missing after creation".to_string())
        })
    }

    async fn ensure_training_logs_table(&self, db: &Connection) -> Result<Table, StorageError> {
        if let Some(table) = self.open_table(db, "training_logs").await? {
            return Ok(table);
        }
        db.create_table(
            "training_logs",
            RecordBatch::new_empty(training_log_schema()),
        )
        .execute()
        .await
        .map_err(|error| StorageError::Backend(error.to_string()))?;
        self.open_table(db, "training_logs").await?.ok_or_else(|| {
            StorageError::Backend("training_logs table missing after creation".to_string())
        })
    }

    async fn ensure_mcp_connections_table(&self, db: &Connection) -> Result<Table, StorageError> {
        if let Some(table) = self.open_table(db, "mcp_connections").await? {
            return Ok(table);
        }
        db.create_table(
            "mcp_connections",
            RecordBatch::new_empty(mcp_connection_schema()),
        )
        .execute()
        .await
        .map_err(|error| StorageError::Backend(error.to_string()))?;
        self.open_table(db, "mcp_connections")
            .await?
            .ok_or_else(|| {
                StorageError::Backend("mcp_connections table missing after creation".to_string())
            })
    }

    async fn ensure_chunk_indexes(&self, table: &Table) -> Result<(), StorageError> {
        let vector_index_result = table.create_index(&["vector"], Index::Auto).execute().await;
        if let Err(error) = vector_index_result {
            let message = error.to_string();
            if !message.contains("Not enough rows to train PQ") {
                return Err(StorageError::Backend(message));
            }
        }
        table
            .create_index(&["chunk_text"], Index::FTS(FtsIndexBuilder::default()))
            .execute()
            .await
            .map_err(|error| StorageError::Backend(error.to_string()))?;
        Ok(())
    }

    async fn open_table(&self, db: &Connection, name: &str) -> Result<Option<Table>, StorageError> {
        match db.open_table(name).execute().await {
            Ok(table) => Ok(Some(table)),
            Err(error) => {
                let message = error.to_string();
                if message.contains("not found") || message.contains("does not exist") {
                    Ok(None)
                } else {
                    Err(StorageError::Backend(message))
                }
            }
        }
    }
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
    vectors: ParkingRwLock<HashMap<String, VectorRecord>>,
    documents: ParkingRwLock<HashMap<String, DocumentRecord>>,
}

impl InMemoryVectorStore {
    pub fn new() -> Self {
        Self {
            vectors: ParkingRwLock::new(HashMap::new()),
            documents: ParkingRwLock::new(HashMap::new()),
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            vectors: ParkingRwLock::new(HashMap::with_capacity(capacity)),
            documents: ParkingRwLock::new(HashMap::new()),
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
    documents: ParkingRwLock<HashMap<String, DocumentRecord>>,
}

impl DocumentStore {
    pub fn new() -> Self {
        Self {
            documents: ParkingRwLock::new(HashMap::new()),
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

async fn add_record_batch(table: &Table, batch: RecordBatch) -> Result<(), StorageError> {
    table
        .add(batch)
        .execute()
        .await
        .map(|_| ())
        .map_err(|error| StorageError::Backend(error.to_string()))
}

async fn delete_where(table: &Table, predicate: &str) -> Result<(), StorageError> {
    table
        .delete(predicate)
        .await
        .map(|_| ())
        .map_err(|error| StorageError::Backend(error.to_string()))
}

async fn collect_batches(table: &Table) -> Result<Vec<RecordBatch>, StorageError> {
    table
        .query()
        .execute()
        .await
        .map_err(|error| StorageError::Backend(error.to_string()))?
        .try_collect::<Vec<_>>()
        .await
        .map_err(|error| StorageError::Backend(error.to_string()))
}

fn document_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("title", DataType::Utf8, false),
        Field::new("content", DataType::Utf8, false),
        Field::new("metadata_json", DataType::Utf8, false),
        Field::new("created_at", DataType::Int64, false),
        Field::new("updated_at", DataType::Int64, false),
    ]))
}

fn chunk_schema(vector_dimension: usize) -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("document_id", DataType::Utf8, false),
        Field::new("chunk_text", DataType::Utf8, false),
        Field::new(
            "vector",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                vector_dimension as i32,
            ),
            false,
        ),
        Field::new("metadata_json", DataType::Utf8, false),
        Field::new("created_at", DataType::Int64, false),
    ]))
}

fn model_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("name", DataType::Utf8, false),
        Field::new("path", DataType::Utf8, false),
        Field::new("backend", DataType::Utf8, false),
        Field::new("status", DataType::Utf8, false),
        Field::new("metadata_json", DataType::Utf8, false),
        Field::new("size_bytes", DataType::Int64, false),
        Field::new("updated_at", DataType::Int64, false),
    ]))
}

fn training_run_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("name", DataType::Utf8, false),
        Field::new("model_id", DataType::Utf8, false),
        Field::new("dataset_path", DataType::Utf8, false),
        Field::new("status", DataType::Utf8, false),
        Field::new("progress", DataType::Float32, false),
        Field::new("error", DataType::Utf8, false),
        Field::new("backend", DataType::Utf8, false),
        Field::new("artifact_dir", DataType::Utf8, false),
        Field::new("config_json", DataType::Utf8, false),
        Field::new("started_at", DataType::Int64, false),
        Field::new("completed_at", DataType::Int64, false),
    ]))
}

fn training_log_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("run_id", DataType::Utf8, false),
        Field::new("level", DataType::Utf8, false),
        Field::new("message", DataType::Utf8, false),
        Field::new("fields_json", DataType::Utf8, false),
        Field::new("timestamp", DataType::Int64, false),
    ]))
}

fn mcp_connection_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("connection_id", DataType::Utf8, false),
        Field::new("server_url", DataType::Utf8, false),
        Field::new("server_name", DataType::Utf8, false),
        Field::new("connected", DataType::Boolean, false),
        Field::new("auth_json", DataType::Utf8, false),
        Field::new("tools_json", DataType::Utf8, false),
        Field::new("updated_at", DataType::Int64, false),
    ]))
}

fn document_batch(records: &[DocumentRecord]) -> Result<RecordBatch, StorageError> {
    RecordBatch::try_new(
        document_schema(),
        vec![
            Arc::new(StringArray::from(
                records
                    .iter()
                    .map(|record| record.id.as_str())
                    .collect::<Vec<_>>(),
            )),
            Arc::new(StringArray::from(
                records
                    .iter()
                    .map(|record| record.title.as_str())
                    .collect::<Vec<_>>(),
            )),
            Arc::new(StringArray::from(
                records
                    .iter()
                    .map(|record| record.content.as_str())
                    .collect::<Vec<_>>(),
            )),
            Arc::new(StringArray::from(
                records
                    .iter()
                    .map(|record| serde_json::to_string(&record.metadata).unwrap())
                    .collect::<Vec<_>>(),
            )),
            Arc::new(Int64Array::from(
                records
                    .iter()
                    .map(|record| record.created_at)
                    .collect::<Vec<_>>(),
            )),
            Arc::new(Int64Array::from(
                records
                    .iter()
                    .map(|record| record.updated_at)
                    .collect::<Vec<_>>(),
            )),
        ],
    )
    .map_err(|error| StorageError::InvalidData(error.to_string()))
}

fn chunk_batch(records: &[VectorRecord]) -> Result<RecordBatch, StorageError> {
    if records.is_empty() {
        return Err(StorageError::InvalidData(
            "chunk_batch requires at least one record".to_string(),
        ));
    }

    let vector_dimension = records[0].vector.len();

    // Validate all records have the same vector dimension
    for (index, record) in records.iter().enumerate() {
        if record.vector.len() != vector_dimension {
            return Err(StorageError::InvalidData(format!(
                "vector dimension mismatch at index {}: expected {}, got {}",
                index,
                vector_dimension,
                record.vector.len()
            )));
        }
    }

    let vector_values = records
        .iter()
        .flat_map(|record| record.vector.iter().copied().map(Some))
        .collect::<Vec<_>>();
    let list_array = FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
        records
            .iter()
            .map(|record| Some(record.vector.iter().copied().map(Some))),
        vector_dimension as i32,
    );

    let _ = vector_values;

    RecordBatch::try_new(
        chunk_schema(vector_dimension),
        vec![
            Arc::new(StringArray::from(
                records
                    .iter()
                    .map(|record| record.id.as_str())
                    .collect::<Vec<_>>(),
            )),
            Arc::new(StringArray::from(
                records
                    .iter()
                    .map(|record| record.document_id.as_str())
                    .collect::<Vec<_>>(),
            )),
            Arc::new(StringArray::from(
                records
                    .iter()
                    .map(|record| record.chunk_text.as_str())
                    .collect::<Vec<_>>(),
            )),
            Arc::new(list_array),
            Arc::new(StringArray::from(
                records
                    .iter()
                    .map(|record| serde_json::to_string(&record.metadata).unwrap())
                    .collect::<Vec<_>>(),
            )),
            Arc::new(Int64Array::from(
                records
                    .iter()
                    .map(|record| record.created_at)
                    .collect::<Vec<_>>(),
            )),
        ],
    )
    .map_err(|error| StorageError::InvalidData(error.to_string()))
}

fn model_batch(records: &[ModelRecord]) -> Result<RecordBatch, StorageError> {
    RecordBatch::try_new(
        model_schema(),
        vec![
            Arc::new(StringArray::from(
                records
                    .iter()
                    .map(|record| record.id.as_str())
                    .collect::<Vec<_>>(),
            )),
            Arc::new(StringArray::from(
                records
                    .iter()
                    .map(|record| record.name.as_str())
                    .collect::<Vec<_>>(),
            )),
            Arc::new(StringArray::from(
                records
                    .iter()
                    .map(|record| record.path.as_str())
                    .collect::<Vec<_>>(),
            )),
            Arc::new(StringArray::from(
                records
                    .iter()
                    .map(|record| record.backend.as_str())
                    .collect::<Vec<_>>(),
            )),
            Arc::new(StringArray::from(
                records
                    .iter()
                    .map(|record| record.status.as_str())
                    .collect::<Vec<_>>(),
            )),
            Arc::new(StringArray::from(
                records
                    .iter()
                    .map(|record| record.metadata_json.as_str())
                    .collect::<Vec<_>>(),
            )),
            Arc::new(Int64Array::from(
                records
                    .iter()
                    .map(|record| record.size_bytes)
                    .collect::<Vec<_>>(),
            )),
            Arc::new(Int64Array::from(
                records
                    .iter()
                    .map(|record| record.updated_at)
                    .collect::<Vec<_>>(),
            )),
        ],
    )
    .map_err(|error| StorageError::InvalidData(error.to_string()))
}

fn training_run_batch(records: &[TrainingRunRecord]) -> Result<RecordBatch, StorageError> {
    RecordBatch::try_new(
        training_run_schema(),
        vec![
            Arc::new(StringArray::from(
                records
                    .iter()
                    .map(|record| record.id.as_str())
                    .collect::<Vec<_>>(),
            )),
            Arc::new(StringArray::from(
                records
                    .iter()
                    .map(|record| record.name.as_str())
                    .collect::<Vec<_>>(),
            )),
            Arc::new(StringArray::from(
                records
                    .iter()
                    .map(|record| record.model_id.as_str())
                    .collect::<Vec<_>>(),
            )),
            Arc::new(StringArray::from(
                records
                    .iter()
                    .map(|record| record.dataset_path.as_str())
                    .collect::<Vec<_>>(),
            )),
            Arc::new(StringArray::from(
                records
                    .iter()
                    .map(|record| record.status.as_str())
                    .collect::<Vec<_>>(),
            )),
            Arc::new(Float32Array::from(
                records
                    .iter()
                    .map(|record| record.progress)
                    .collect::<Vec<_>>(),
            )),
            Arc::new(StringArray::from(
                records
                    .iter()
                    .map(|record| record.error.as_str())
                    .collect::<Vec<_>>(),
            )),
            Arc::new(StringArray::from(
                records
                    .iter()
                    .map(|record| record.backend.as_str())
                    .collect::<Vec<_>>(),
            )),
            Arc::new(StringArray::from(
                records
                    .iter()
                    .map(|record| record.artifact_dir.as_str())
                    .collect::<Vec<_>>(),
            )),
            Arc::new(StringArray::from(
                records
                    .iter()
                    .map(|record| record.config_json.as_str())
                    .collect::<Vec<_>>(),
            )),
            Arc::new(Int64Array::from(
                records
                    .iter()
                    .map(|record| record.started_at)
                    .collect::<Vec<_>>(),
            )),
            Arc::new(Int64Array::from(
                records
                    .iter()
                    .map(|record| record.completed_at)
                    .collect::<Vec<_>>(),
            )),
        ],
    )
    .map_err(|error| StorageError::InvalidData(error.to_string()))
}

fn training_log_batch(records: &[TrainingLogRecord]) -> Result<RecordBatch, StorageError> {
    RecordBatch::try_new(
        training_log_schema(),
        vec![
            Arc::new(StringArray::from(
                records
                    .iter()
                    .map(|record| record.run_id.as_str())
                    .collect::<Vec<_>>(),
            )),
            Arc::new(StringArray::from(
                records
                    .iter()
                    .map(|record| record.level.as_str())
                    .collect::<Vec<_>>(),
            )),
            Arc::new(StringArray::from(
                records
                    .iter()
                    .map(|record| record.message.as_str())
                    .collect::<Vec<_>>(),
            )),
            Arc::new(StringArray::from(
                records
                    .iter()
                    .map(|record| record.fields_json.as_str())
                    .collect::<Vec<_>>(),
            )),
            Arc::new(Int64Array::from(
                records
                    .iter()
                    .map(|record| record.timestamp)
                    .collect::<Vec<_>>(),
            )),
        ],
    )
    .map_err(|error| StorageError::InvalidData(error.to_string()))
}

fn mcp_connection_batch(records: &[MCPConnectionRecord]) -> Result<RecordBatch, StorageError> {
    RecordBatch::try_new(
        mcp_connection_schema(),
        vec![
            Arc::new(StringArray::from(
                records
                    .iter()
                    .map(|record| record.connection_id.as_str())
                    .collect::<Vec<_>>(),
            )),
            Arc::new(StringArray::from(
                records
                    .iter()
                    .map(|record| record.server_url.as_str())
                    .collect::<Vec<_>>(),
            )),
            Arc::new(StringArray::from(
                records
                    .iter()
                    .map(|record| record.server_name.as_str())
                    .collect::<Vec<_>>(),
            )),
            Arc::new(BooleanArray::from(
                records
                    .iter()
                    .map(|record| record.connected)
                    .collect::<Vec<_>>(),
            )),
            Arc::new(StringArray::from(
                records
                    .iter()
                    .map(|record| record.auth_json.as_str())
                    .collect::<Vec<_>>(),
            )),
            Arc::new(StringArray::from(
                records
                    .iter()
                    .map(|record| record.tools_json.as_str())
                    .collect::<Vec<_>>(),
            )),
            Arc::new(Int64Array::from(
                records
                    .iter()
                    .map(|record| record.updated_at)
                    .collect::<Vec<_>>(),
            )),
        ],
    )
    .map_err(|error| StorageError::InvalidData(error.to_string()))
}

fn decode_documents(batches: &[RecordBatch]) -> Result<Vec<DocumentRecord>, StorageError> {
    let mut documents = Vec::new();
    for batch in batches {
        let ids = string_column(batch, "id")?;
        let titles = string_column(batch, "title")?;
        let contents = string_column(batch, "content")?;
        let metadata = string_column(batch, "metadata_json")?;
        let created_at = int64_column(batch, "created_at")?;
        let updated_at = int64_column(batch, "updated_at")?;

        for row in 0..batch.num_rows() {
            documents.push(DocumentRecord {
                id: ids.value(row).to_string(),
                title: titles.value(row).to_string(),
                content: contents.value(row).to_string(),
                metadata: serde_json::from_str(metadata.value(row))
                    .map_err(|error| StorageError::InvalidData(error.to_string()))?,
                created_at: created_at.value(row),
                updated_at: updated_at.value(row),
            });
        }
    }
    Ok(documents)
}

fn decode_chunks(batches: &[RecordBatch]) -> Result<Vec<VectorRecord>, StorageError> {
    let mut chunks = Vec::new();
    for batch in batches {
        let ids = string_column(batch, "id")?;
        let document_ids = string_column(batch, "document_id")?;
        let texts = string_column(batch, "chunk_text")?;
        let metadata = string_column(batch, "metadata_json")?;
        let created_at = int64_column(batch, "created_at")?;
        let vectors = fixed_size_list_column(batch, "vector")?;

        for row in 0..batch.num_rows() {
            chunks.push(VectorRecord {
                id: ids.value(row).to_string(),
                document_id: document_ids.value(row).to_string(),
                chunk_text: texts.value(row).to_string(),
                vector: fixed_size_list_value(vectors, row)?,
                metadata: serde_json::from_str(metadata.value(row))
                    .map_err(|error| StorageError::InvalidData(error.to_string()))?,
                created_at: created_at.value(row),
            });
        }
    }
    Ok(chunks)
}

fn decode_models(batches: &[RecordBatch]) -> Result<Vec<ModelRecord>, StorageError> {
    let mut models = Vec::new();
    for batch in batches {
        let ids = string_column(batch, "id")?;
        let names = string_column(batch, "name")?;
        let paths = string_column(batch, "path")?;
        let backends = string_column(batch, "backend")?;
        let statuses = string_column(batch, "status")?;
        let metadata = string_column(batch, "metadata_json")?;
        let size_bytes = int64_column(batch, "size_bytes")?;
        let updated_at = int64_column(batch, "updated_at")?;

        for row in 0..batch.num_rows() {
            models.push(ModelRecord {
                id: ids.value(row).to_string(),
                name: names.value(row).to_string(),
                path: paths.value(row).to_string(),
                backend: backends.value(row).to_string(),
                status: statuses.value(row).to_string(),
                metadata_json: metadata.value(row).to_string(),
                size_bytes: size_bytes.value(row),
                updated_at: updated_at.value(row),
            });
        }
    }
    Ok(models)
}

fn decode_training_runs(batches: &[RecordBatch]) -> Result<Vec<TrainingRunRecord>, StorageError> {
    let mut runs = Vec::new();
    for batch in batches {
        let ids = string_column(batch, "id")?;
        let names = string_column(batch, "name")?;
        let model_ids = string_column(batch, "model_id")?;
        let datasets = string_column(batch, "dataset_path")?;
        let statuses = string_column(batch, "status")?;
        let progress = float32_column(batch, "progress")?;
        let errors = string_column(batch, "error")?;
        let backends = string_column(batch, "backend")?;
        let artifact_dirs = string_column(batch, "artifact_dir")?;
        let config_json = string_column(batch, "config_json")?;
        let started_at = int64_column(batch, "started_at")?;
        let completed_at = int64_column(batch, "completed_at")?;

        for row in 0..batch.num_rows() {
            runs.push(TrainingRunRecord {
                id: ids.value(row).to_string(),
                name: names.value(row).to_string(),
                model_id: model_ids.value(row).to_string(),
                dataset_path: datasets.value(row).to_string(),
                status: statuses.value(row).to_string(),
                progress: progress.value(row),
                error: errors.value(row).to_string(),
                backend: backends.value(row).to_string(),
                artifact_dir: artifact_dirs.value(row).to_string(),
                config_json: config_json.value(row).to_string(),
                started_at: started_at.value(row),
                completed_at: completed_at.value(row),
            });
        }
    }
    Ok(runs)
}

fn decode_training_logs(batches: &[RecordBatch]) -> Result<Vec<TrainingLogRecord>, StorageError> {
    let mut logs = Vec::new();
    for batch in batches {
        let run_ids = string_column(batch, "run_id")?;
        let levels = string_column(batch, "level")?;
        let messages = string_column(batch, "message")?;
        let fields_json = string_column(batch, "fields_json")?;
        let timestamps = int64_column(batch, "timestamp")?;

        for row in 0..batch.num_rows() {
            logs.push(TrainingLogRecord {
                run_id: run_ids.value(row).to_string(),
                level: levels.value(row).to_string(),
                message: messages.value(row).to_string(),
                fields_json: fields_json.value(row).to_string(),
                timestamp: timestamps.value(row),
            });
        }
    }
    Ok(logs)
}

fn decode_mcp_connections(
    batches: &[RecordBatch],
) -> Result<Vec<MCPConnectionRecord>, StorageError> {
    let mut connections = Vec::new();
    for batch in batches {
        let connection_ids = string_column(batch, "connection_id")?;
        let server_urls = string_column(batch, "server_url")?;
        let server_names = string_column(batch, "server_name")?;
        let connected = boolean_column(batch, "connected")?;
        let auth_json = string_column(batch, "auth_json")?;
        let tools_json = string_column(batch, "tools_json")?;
        let updated_at = int64_column(batch, "updated_at")?;

        for row in 0..batch.num_rows() {
            connections.push(MCPConnectionRecord {
                connection_id: connection_ids.value(row).to_string(),
                server_url: server_urls.value(row).to_string(),
                server_name: server_names.value(row).to_string(),
                connected: connected.value(row),
                auth_json: auth_json.value(row).to_string(),
                tools_json: tools_json.value(row).to_string(),
                updated_at: updated_at.value(row),
            });
        }
    }
    Ok(connections)
}

fn string_column<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a StringArray, StorageError> {
    batch
        .column_by_name(name)
        .and_then(|column| column.as_any().downcast_ref::<StringArray>())
        .ok_or_else(|| StorageError::InvalidData(format!("missing string column {name}")))
}

fn int64_column<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a Int64Array, StorageError> {
    batch
        .column_by_name(name)
        .and_then(|column| column.as_any().downcast_ref::<Int64Array>())
        .ok_or_else(|| StorageError::InvalidData(format!("missing int64 column {name}")))
}

fn boolean_column<'a>(
    batch: &'a RecordBatch,
    name: &str,
) -> Result<&'a BooleanArray, StorageError> {
    batch
        .column_by_name(name)
        .and_then(|column| column.as_any().downcast_ref::<BooleanArray>())
        .ok_or_else(|| StorageError::InvalidData(format!("missing bool column {name}")))
}

fn float32_column<'a>(
    batch: &'a RecordBatch,
    name: &str,
) -> Result<&'a Float32Array, StorageError> {
    batch
        .column_by_name(name)
        .and_then(|column| column.as_any().downcast_ref::<Float32Array>())
        .ok_or_else(|| StorageError::InvalidData(format!("missing float32 column {name}")))
}

fn fixed_size_list_column<'a>(
    batch: &'a RecordBatch,
    name: &str,
) -> Result<&'a FixedSizeListArray, StorageError> {
    batch
        .column_by_name(name)
        .and_then(|column| column.as_any().downcast_ref::<FixedSizeListArray>())
        .ok_or_else(|| StorageError::InvalidData(format!("missing vector column {name}")))
}

fn fixed_size_list_value(
    array: &FixedSizeListArray,
    index: usize,
) -> Result<Vec<f32>, StorageError> {
    let values = array.value(index);
    let floats = values
        .as_any()
        .downcast_ref::<Float32Array>()
        .ok_or_else(|| StorageError::InvalidData("vector column was not Float32".to_string()))?;

    Ok((0..floats.len()).map(|idx| floats.value(idx)).collect())
}

fn resolve_mode(query: &ChunkSearchQuery) -> SearchMode {
    match query.mode {
        SearchMode::Auto => {
            if query.text_query.as_deref().unwrap_or("").trim().is_empty() {
                SearchMode::Vector
            } else {
                SearchMode::Hybrid
            }
        }
        _ => query.mode.clone(),
    }
}

fn matches_filters(record: &VectorRecord, filters: &HashMap<String, String>) -> bool {
    filters
        .iter()
        .all(|(key, expected)| record.metadata.get(key) == Some(expected))
}

fn lexical_score(query: &str, text: &str) -> f32 {
    let words = query
        .split_whitespace()
        .map(|value| value.to_ascii_lowercase())
        .collect::<Vec<_>>();
    if words.is_empty() {
        return 0.0;
    }

    let haystack = text.to_ascii_lowercase();
    let matches = words
        .iter()
        .filter(|word| haystack.contains(word.as_str()))
        .count();
    matches as f32 / words.len() as f32
}

fn escape_literal(value: &str) -> String {
    value.replace('\'', "''")
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

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

    #[tokio::test]
    async fn test_engine_store_round_trip_document_and_chunk_search() {
        let dir = tempdir().unwrap();
        let store = EngineStore::new(dir.path().to_string_lossy().to_string());

        store
            .upsert_document(
                DocumentRecord {
                    id: "doc-1".to_string(),
                    title: "Test".to_string(),
                    content: "hello vector world".to_string(),
                    metadata: HashMap::from([
                        ("source_db".to_string(), "turso".to_string()),
                        ("external_id".to_string(), "entity-1".to_string()),
                    ]),
                    created_at: 1,
                    updated_at: 2,
                },
                vec![VectorRecord {
                    id: "chunk-1".to_string(),
                    document_id: "doc-1".to_string(),
                    chunk_text: "hello vector world".to_string(),
                    vector: vec![1.0, 0.0, 0.0, 0.0],
                    metadata: HashMap::from([
                        ("source_db".to_string(), "turso".to_string()),
                        ("external_id".to_string(), "entity-1".to_string()),
                    ]),
                    created_at: 1,
                }],
            )
            .await
            .unwrap();

        let docs = store.list_documents().await.unwrap();
        assert_eq!(docs.len(), 1);

        let results = store
            .search_chunks(ChunkSearchQuery {
                vector: vec![1.0, 0.0, 0.0, 0.0],
                text_query: Some("hello vector".to_string()),
                top_k: 5,
                filters: HashMap::from([("source_db".to_string(), "turso".to_string())]),
                mode: SearchMode::Hybrid,
            })
            .await
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].record.document_id, "doc-1");

        store.delete_document("doc-1").await.unwrap();
        assert!(store.list_documents().await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn test_engine_store_persists_training_runs_and_logs() {
        let dir = tempdir().unwrap();
        let store = EngineStore::new(dir.path().to_string_lossy().to_string());

        store
            .upsert_training_run(TrainingRunRecord {
                id: "run-1".to_string(),
                name: "demo".to_string(),
                model_id: "llama".to_string(),
                dataset_path: "dataset.jsonl".to_string(),
                status: "running".to_string(),
                progress: 0.4,
                error: String::new(),
                backend: "llama.cpp".to_string(),
                artifact_dir: "artifacts/run-1".to_string(),
                config_json: "{}".to_string(),
                started_at: 10,
                completed_at: 0,
            })
            .await
            .unwrap();
        store
            .append_training_logs(vec![TrainingLogRecord {
                run_id: "run-1".to_string(),
                level: "info".to_string(),
                message: "started".to_string(),
                fields_json: "{}".to_string(),
                timestamp: 11,
            }])
            .await
            .unwrap();

        let runs = store.list_training_runs().await.unwrap();
        assert_eq!(runs.len(), 1);
        assert_eq!(runs[0].backend, "llama.cpp");

        let logs = store.list_training_logs("run-1").await.unwrap();
        assert_eq!(logs.len(), 1);
        assert_eq!(logs[0].message, "started");
    }

    #[tokio::test]
    async fn test_engine_store_persists_mcp_connections() {
        let dir = tempdir().unwrap();
        let store = EngineStore::new(dir.path().to_string_lossy().to_string());

        store
            .upsert_mcp_connection(MCPConnectionRecord {
                connection_id: "conn-1".to_string(),
                server_url: "http://localhost:3000".to_string(),
                server_name: "Local MCP".to_string(),
                connected: true,
                auth_json: "{\"token\":\"secret\"}".to_string(),
                tools_json: "[{\"name\":\"search_files\"}]".to_string(),
                updated_at: 12,
            })
            .await
            .unwrap();

        let connections = store.list_mcp_connections().await.unwrap();
        assert_eq!(connections.len(), 1);
        assert_eq!(connections[0].server_name, "Local MCP");
    }
}
