use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Component, Path, PathBuf};
use std::sync::Arc;

use arrow_array::{
    types::Float32Type, Array, FixedSizeListArray, Float32Array, Int64Array, RecordBatch,
    StringArray,
};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use chrono::Utc;
use dashmap::DashMap;
use futures::TryStreamExt;
use lancedb::query::{ExecutableQuery, QueryBase};
use lancedb::{connect, Table};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use sqlx::sqlite::{SqliteConnectOptions, SqliteJournalMode, SqlitePoolOptions};
use sqlx::{Row, SqlitePool};
use tokio::fs;
use tokio::io::AsyncWriteExt;
use tokio::sync::{Mutex, OnceCell};
use uuid::Uuid;

use crate::bridge::{BridgeError, OpenVikingBridgeClient, OpenVikingBridgeConfig};
use crate::graph::{GraphFactRecord, GraphNode, GraphNodeKind, GraphRelationKind, GraphStore};
use crate::memory::{DragonflyConfig, WorkingMemoryProvider};
use crate::model::{
    DeleteOutcome, FileEntry, LayeredContent, ResourceLayer, ResourceSummary,
    ResourceUpsertRequest, SearchHit, SearchRequest, SessionEntry, SessionEventRequest,
    StatusResponse, UpsertOutcome, WorkspaceSyncOutcome,
};
use crate::uri::{UriError, VikingNamespace, VikingUri};

const EMBEDDING_DIM: usize = 384;
const VECTOR_TABLE_NAME: &str = "resource_chunks";

#[derive(Debug, thiserror::Error)]
pub enum ContextError {
    #[error("invalid request: {0}")]
    InvalidRequest(String),
    #[error("not found: {0}")]
    NotFound(String),
    #[error("conflict: {0}")]
    Conflict(String),
    #[error("uri error: {0}")]
    Uri(#[from] UriError),
    #[error("sqlx error: {0}")]
    Sqlx(#[from] sqlx::Error),
    #[error("lancedb error: {0}")]
    LanceDb(#[from] lancedb::Error),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("bridge error: {0}")]
    Bridge(#[from] BridgeError),
    #[error("serde error: {0}")]
    Serde(#[from] serde_json::Error),
}

impl axum::response::IntoResponse for ContextError {
    fn into_response(self) -> axum::response::Response {
        let message = self.to_string();
        let status = match self {
            ContextError::NotFound(_) => axum::http::StatusCode::NOT_FOUND,
            ContextError::Conflict(_) => axum::http::StatusCode::CONFLICT,
            ContextError::InvalidRequest(_) | ContextError::Uri(_) => {
                axum::http::StatusCode::BAD_REQUEST
            }
            _ => axum::http::StatusCode::INTERNAL_SERVER_ERROR,
        };
        let body = serde_json::json!({ "error": message });
        (status, axum::Json(body)).into_response()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManagedRoot {
    pub name: String,
    pub path: PathBuf,
}

impl ManagedRoot {
    pub fn new(name: impl Into<String>, path: impl Into<PathBuf>) -> Result<Self, ContextError> {
        let path = path.into();
        std::fs::create_dir_all(&path)?;
        let path = path.canonicalize()?;
        Ok(Self {
            name: name.into(),
            path,
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextConfig {
    pub data_dir: PathBuf,
    pub roots: Vec<ManagedRoot>,
    pub bridge: Option<OpenVikingBridgeConfig>,
    pub dragonfly: Option<DragonflyConfig>,
}

impl ContextConfig {
    /// Create a ContextConfig that uses the given data directory and managed roots.
    ///
    /// The returned configuration has no bridge or dragonfly client configured.
    ///
    /// # Examples
    ///
    /// ```
    /// let roots = vec![ManagedRoot::new("workspace", "/path").unwrap()];
    /// let cfg = ContextConfig::default_in("/data", roots.clone());
    /// assert_eq!(cfg.data_dir, std::path::PathBuf::from("/data"));
    /// assert_eq!(cfg.roots, roots);
    /// ```
    pub fn default_in(dir: impl Into<PathBuf>, roots: Vec<ManagedRoot>) -> Self {
        Self {
            data_dir: dir.into(),
            roots,
            bridge: None,
            dragonfly: None,
        }
    }
}

impl Default for ContextConfig {
    /// Create a ContextConfig with standard default settings.
    ///
    /// The default configuration uses "./context-data" as the data directory, no managed
    /// roots, and no configured OpenViking bridge or Dragonfly working-memory store.
    ///
    /// # Examples
    ///
    /// ```
    /// let cfg = ContextConfig::default();
    /// assert_eq!(cfg.data_dir, std::path::PathBuf::from("./context-data"));
    /// assert!(cfg.roots.is_empty());
    /// assert!(cfg.bridge.is_none());
    /// assert!(cfg.dragonfly.is_none());
    /// ```
    fn default() -> Self {
        Self {
            data_dir: PathBuf::from("./context-data"),
            roots: Vec::new(),
            bridge: None,
            dragonfly: None,
        }
    }
}

#[derive(Debug, Clone)]
struct ResourceRow {
    resource_id: String,
    uri: String,
    title: String,
    root: String,
    path: String,
    source_layer: String,
    metadata_json: String,
    content_hash: String,
    updated_at: i64,
}

#[derive(Debug, Clone)]
struct ChunkRow {
    chunk_id: String,
    resource_id: String,
    layer: String,
    ordinal: i64,
    content: String,
    content_hash: String,
    metadata_json: String,
    updated_at: i64,
}

#[derive(Debug, Clone)]
struct ScoredChunk {
    chunk_id: String,
    resource_id: String,
    layer: String,
    content: String,
    score: f32,
    metadata_json: String,
}

#[derive(Clone)]
pub struct ContextEngine {
    inner: Arc<ContextInner>,
}

struct ContextInner {
    config: ContextConfig,
    pool: SqlitePool,
    vector_table: Arc<Table>,
    bridge: OnceCell<Option<OpenVikingBridgeClient>>,
    file_locks: DashMap<String, Arc<Mutex<()>>>,
}

impl ContextEngine {
    pub async fn open(config: ContextConfig) -> Result<Self, ContextError> {
        fs::create_dir_all(&config.data_dir).await?;
        let sqlite_path = config.data_dir.join("context.sqlite");
        let lancedb_path = config.data_dir.join("lancedb");
        fs::create_dir_all(&lancedb_path).await?;

        let sqlite_options = SqliteConnectOptions::new()
            .filename(&sqlite_path)
            .create_if_missing(true)
            .journal_mode(SqliteJournalMode::Wal);
        let pool = SqlitePoolOptions::new()
            .max_connections(4)
            .connect_with(sqlite_options)
            .await?;

        Self::init_sqlite(&pool).await?;

        let db = connect(lancedb_path.to_string_lossy().as_ref())
            .execute()
            .await?;
        let vector_table = Arc::new(Self::open_or_create_vector_table(&db).await?);

        Ok(Self {
            inner: Arc::new(ContextInner {
                config,
                pool,
                vector_table,
                bridge: OnceCell::new(),
                file_locks: DashMap::new(),
            }),
        })
    }

    pub fn placeholder() -> Self {
        let base = std::env::temp_dir().join(format!("context-engine-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&base).expect("temp dir");
        let root = ManagedRoot::new("workspace", base.join("workspace")).expect("root");
        let config = ContextConfig::default_in(base.join("data"), vec![root]);

        match tokio::runtime::Handle::try_current() {
            Ok(handle) => handle.block_on(Self::open(config)).expect("engine"),
            Err(_) => tokio::runtime::Runtime::new()
                .expect("runtime")
                .block_on(Self::open(config))
                .expect("engine"),
        }
    }

    /// Creates the SQLite schema required by the context engine.
    ///
    /// This initializes the persistent tables and FTS5 virtual table used by the engine:
    /// `resources`, `resource_chunks`, `resource_chunks_fts`, `sessions`, `memories`,
    /// `graph_nodes`, and `graph_edges`.
    ///
    /// # Examples
    ///
    /// ```
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let pool = sqlx::SqlitePool::connect(":memory:").await?;
    /// // `init_sqlite` returns a `Result<(), ContextError>`; adapt error handling as needed.
    /// let _ = crate::init_sqlite(&pool).await;
    /// # Ok(()) }
    /// ```
    async fn init_sqlite(pool: &SqlitePool) -> Result<(), ContextError> {
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS resources (
                resource_id TEXT PRIMARY KEY,
                uri TEXT NOT NULL UNIQUE,
                title TEXT NOT NULL,
                root TEXT NOT NULL,
                path TEXT NOT NULL,
                source_layer TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                updated_at INTEGER NOT NULL
            )
            "#,
        )
        .execute(pool)
        .await?;

        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS resource_chunks (
                chunk_id TEXT PRIMARY KEY,
                resource_id TEXT NOT NULL,
                layer TEXT NOT NULL,
                ordinal INTEGER NOT NULL,
                content TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                updated_at INTEGER NOT NULL
            )
            "#,
        )
        .execute(pool)
        .await?;

        sqlx::query(
            r#"
            CREATE VIRTUAL TABLE IF NOT EXISTS resource_chunks_fts
            USING fts5(chunk_id UNINDEXED, resource_id UNINDEXED, layer UNINDEXED, content)
            "#,
        )
        .execute(pool)
        .await?;

        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                created_at INTEGER NOT NULL
            )
            "#,
        )
        .execute(pool)
        .await?;

        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS memories (
                uri TEXT PRIMARY KEY,
                namespace TEXT NOT NULL,
                content TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                updated_at INTEGER NOT NULL
            )
            "#,
        )
        .execute(pool)
        .await?;

        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS graph_nodes (
                node_id TEXT PRIMARY KEY,
                kind TEXT NOT NULL,
                name TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                updated_at INTEGER NOT NULL
            )
            "#,
        )
        .execute(pool)
        .await?;

        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS graph_edges (
                edge_id TEXT PRIMARY KEY,
                subject_id TEXT NOT NULL,
                relation TEXT NOT NULL,
                object_id TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                resource_uri TEXT,
                session_id TEXT,
                updated_at INTEGER NOT NULL,
                UNIQUE(subject_id, relation, object_id)
            )
            "#,
        )
        .execute(pool)
        .await?;

        Ok(())
    }

    async fn open_or_create_vector_table(db: &lancedb::Connection) -> Result<Table, ContextError> {
        match db.open_table(VECTOR_TABLE_NAME).execute().await {
            Ok(table) => Ok(table),
            Err(_) => {
                let schema = Self::vector_schema();
                let table = db
                    .create_empty_table(VECTOR_TABLE_NAME, schema)
                    .execute()
                    .await?;
                Ok(table)
            }
        }
    }

    fn vector_schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("chunk_id", DataType::Utf8, false),
            Field::new("resource_id", DataType::Utf8, false),
            Field::new("layer", DataType::Utf8, false),
            Field::new("ordinal", DataType::Int64, false),
            Field::new("content", DataType::Utf8, false),
            Field::new(
                "vector",
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float32, true)),
                    EMBEDDING_DIM as i32,
                ),
                false,
            ),
            Field::new("metadata_json", DataType::Utf8, false),
            Field::new("updated_at", DataType::Int64, false),
        ]))
    }

    fn root_by_name(&self, name: &str) -> Result<&ManagedRoot, ContextError> {
        self.inner
            .config
            .roots
            .iter()
            .find(|root| root.name == name)
            .ok_or_else(|| ContextError::NotFound(format!("managed root {name}")))
    }

    fn normalize_relative_path(path: impl AsRef<Path>) -> Result<PathBuf, ContextError> {
        let mut normalized = PathBuf::new();
        for component in path.as_ref().components() {
            match component {
                Component::Normal(part) => normalized.push(part),
                Component::CurDir => {}
                Component::ParentDir | Component::RootDir | Component::Prefix(_) => {
                    return Err(ContextError::InvalidRequest(
                        "path escapes managed root".to_string(),
                    ));
                }
            }
        }

        Ok(normalized)
    }

    async fn read_root_file(
        &self,
        root: &str,
        path: impl AsRef<Path>,
    ) -> Result<String, ContextError> {
        let resolved = self.resolve_managed_path(root, path.as_ref())?;
        Ok(fs::read_to_string(resolved).await?)
    }

    fn resolve_managed_path(
        &self,
        root: &str,
        path: impl AsRef<Path>,
    ) -> Result<PathBuf, ContextError> {
        let root = self.root_by_name(root)?;
        let relative = Self::normalize_relative_path(path)?;
        Ok(root.path.join(relative))
    }

    fn relative_managed_path(&self, root: &str, path: &Path) -> Result<String, ContextError> {
        let root = self.root_by_name(root)?;
        Ok(path
            .strip_prefix(&root.path)
            .map_err(|_| ContextError::InvalidRequest("path is outside managed root".to_string()))?
            .to_string_lossy()
            .replace('\\', "/"))
    }

    fn resource_uri_for_path(&self, root: &str, relative_path: &str) -> String {
        VikingUri::resource(root, relative_path).to_string()
    }

    fn acquire_file_lock(&self, root: &str, path: &Path) -> Arc<Mutex<()>> {
        let lock_key = format!("{}:{}", root, path.display());
        self.inner
            .file_locks
            .entry(lock_key)
            .or_insert_with(|| Arc::new(Mutex::new(())))
            .clone()
    }

    async fn index_managed_file(
        &self,
        root: &str,
        path: impl AsRef<Path>,
        previous_uri: Option<String>,
    ) -> Result<UpsertOutcome, ContextError> {
        let normalized = Self::normalize_relative_path(path)?;
        let content = self.read_root_file(root, &normalized).await?;
        let relative_path = normalized.to_string_lossy().replace('\\', "/");
        let title = Path::new(&relative_path)
            .file_name()
            .and_then(|part| part.to_str())
            .unwrap_or("resource")
            .to_string();
        let mut metadata = BTreeMap::new();
        metadata.insert("root".to_string(), root.to_string());
        metadata.insert("path".to_string(), relative_path.clone());
        metadata.insert("kind".to_string(), "file".to_string());

        self.upsert_resource(ResourceUpsertRequest {
            uri: self.resource_uri_for_path(root, &relative_path),
            title: Some(title),
            content,
            layer: ResourceLayer::L2,
            metadata,
            previous_uri,
        })
        .await
    }

    pub async fn read_file(
        &self,
        root: &str,
        path: impl AsRef<Path>,
    ) -> Result<String, ContextError> {
        self.read_root_file(root, path).await
    }

    pub async fn write_file(
        &self,
        root: &str,
        path: impl AsRef<Path>,
        content: &str,
        version: Option<i64>,
    ) -> Result<i64, ContextError> {
        let normalized = Self::normalize_relative_path(path)?;
        let lock = self.acquire_file_lock(root, &normalized);
        let _guard = lock.lock().await;

        let resolved = self.resolve_managed_path(root, &normalized)?;
        if let Some(parent) = resolved.parent() {
            fs::create_dir_all(parent).await?;
        }

        if let Some(expected) = version {
            if let Ok(current) = self.file_version(root, &normalized).await {
                if current != expected {
                    return Err(ContextError::Conflict("file version mismatch".to_string()));
                }
            }
        }

        let mut file = fs::File::create(&resolved).await?;
        file.write_all(content.as_bytes()).await?;
        file.flush().await?;
        let version = self.file_version(root, &normalized).await?;
        let _ = self.index_managed_file(root, &normalized, None).await?;
        Ok(version)
    }

    pub async fn file_version(
        &self,
        root: &str,
        path: impl AsRef<Path>,
    ) -> Result<i64, ContextError> {
        let resolved = self.resolve_managed_path(root, path.as_ref())?;
        let metadata = fs::metadata(resolved).await?;
        let modified = metadata.modified()?;
        Ok(modified
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(|_| {
                ContextError::InvalidRequest("file timestamp is before unix epoch".into())
            })?
            .as_millis() as i64)
    }

    pub async fn list_files(
        &self,
        root: &str,
        path: impl AsRef<Path>,
    ) -> Result<Vec<FileEntry>, ContextError> {
        let start = self.resolve_managed_path(root, path.as_ref())?;
        let mut entries = Vec::new();
        let mut stack = vec![start];

        while let Some(current) = stack.pop() {
            if current.is_dir() {
                let mut dir = fs::read_dir(&current).await?;
                while let Some(entry) = dir.next_entry().await? {
                    let metadata = entry.metadata().await?;
                    let path = entry.path();
                    let relative = path
                        .strip_prefix(&self.root_by_name(root)?.path)
                        .unwrap()
                        .to_string_lossy()
                        .replace('\\', "/");
                    let version = metadata
                        .modified()
                        .ok()
                        .and_then(|m| m.duration_since(std::time::UNIX_EPOCH).ok())
                        .map(|d| d.as_millis() as i64);
                    entries.push(FileEntry {
                        name: entry.file_name().to_string_lossy().to_string(),
                        path: relative,
                        is_dir: metadata.is_dir(),
                        size_bytes: metadata.len(),
                        version,
                    });
                    if metadata.is_dir() {
                        stack.push(path);
                    }
                }
            }
        }

        entries.sort_by(|a, b| a.path.cmp(&b.path));
        Ok(entries)
    }

    pub async fn delete_file(
        &self,
        root: &str,
        path: impl AsRef<Path>,
        version: Option<i64>,
    ) -> Result<bool, ContextError> {
        let normalized = Self::normalize_relative_path(path)?;
        let lock = self.acquire_file_lock(root, &normalized);
        let _guard = lock.lock().await;

        let resolved = self.resolve_managed_path(root, &normalized)?;
        let metadata = match fs::metadata(&resolved).await {
            Ok(metadata) => metadata,
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(false),
            Err(err) => return Err(ContextError::Io(err)),
        };

        if let Some(expected) = version {
            let current = self.file_version(root, &normalized).await?;
            if current != expected {
                return Err(ContextError::Conflict("file version mismatch".to_string()));
            }
        }

        let relative_path = normalized.to_string_lossy().replace('\\', "/");
        if metadata.is_dir() {
            fs::remove_dir_all(&resolved).await?;
            let resources = self
                .load_resources_for_scope(root, Some(&relative_path))
                .await?;
            for resource in resources {
                let _ = self.delete_resource(&resource.uri).await?;
            }
        } else {
            fs::remove_file(&resolved).await?;
            let uri = self.resource_uri_for_path(root, &relative_path);
            let _ = self.delete_resource(&uri).await?;
        }

        Ok(true)
    }

    pub async fn move_file(
        &self,
        root: &str,
        from_path: impl AsRef<Path>,
        to_path: impl AsRef<Path>,
        version: Option<i64>,
    ) -> Result<i64, ContextError> {
        let from_normalized = Self::normalize_relative_path(from_path)?;
        let to_normalized = Self::normalize_relative_path(to_path)?;

        let from_lock = self.acquire_file_lock(root, &from_normalized);
        let to_lock = self.acquire_file_lock(root, &to_normalized);
        let _from_guard = from_lock.lock().await;
        let _to_guard = to_lock.lock().await;

        let from_resolved = self.resolve_managed_path(root, &from_normalized)?;
        let to_resolved = self.resolve_managed_path(root, &to_normalized)?;

        let metadata = fs::metadata(&from_resolved)
            .await
            .map_err(|err| match err.kind() {
                std::io::ErrorKind::NotFound => {
                    ContextError::NotFound(format!("managed path {}", from_normalized.display()))
                }
                _ => ContextError::Io(err),
            })?;

        if let Some(expected) = version {
            let current = self.file_version(root, &from_normalized).await?;
            if current != expected {
                return Err(ContextError::Conflict("file version mismatch".to_string()));
            }
        }

        if let Some(parent) = to_resolved.parent() {
            fs::create_dir_all(parent).await?;
        }
        fs::rename(&from_resolved, &to_resolved).await?;

        let from_relative = from_normalized.to_string_lossy().replace('\\', "/");
        let to_relative = to_normalized.to_string_lossy().replace('\\', "/");

        if metadata.is_dir() {
            let moved_resources = self
                .load_resources_for_scope(root, Some(&from_relative))
                .await?;
            for resource in moved_resources {
                let suffix = Path::new(&resource.path)
                    .strip_prefix(Path::new(&from_relative))
                    .ok()
                    .and_then(|p| p.to_str())
                    .unwrap_or("")
                    .trim_start_matches('/')
                    .to_string();
                let next_relative = if suffix.is_empty() {
                    to_relative.clone()
                } else {
                    format!("{to_relative}/{suffix}")
                };
                let _ = self
                    .index_managed_file(
                        root,
                        PathBuf::from(&next_relative),
                        Some(resource.uri.clone()),
                    )
                    .await?;
            }
        } else {
            let previous_uri = Some(self.resource_uri_for_path(root, &from_relative));
            let _ = self
                .index_managed_file(root, &to_normalized, previous_uri)
                .await?;
        }

        self.file_version(root, &to_normalized).await.or(Ok(0))
    }

    pub async fn sync_workspace(
        &self,
        root: &str,
        path: Option<PathBuf>,
    ) -> Result<WorkspaceSyncOutcome, ContextError> {
        const MAX_FILE_SIZE: u64 = 1_048_576; // 1MB

        let prefix_path = match path {
            Some(path) => Some(Self::normalize_relative_path(path)?),
            None => None,
        };
        let start = match &prefix_path {
            Some(prefix) => self.resolve_managed_path(root, prefix)?,
            None => self.resolve_managed_path(root, PathBuf::new())?,
        };

        let prefix_string = prefix_path
            .as_ref()
            .map(|prefix| prefix.to_string_lossy().replace('\\', "/"))
            .filter(|prefix| !prefix.is_empty());

        let mut indexed_resources = 0;
        let mut reindexed_resources = 0;
        let mut skipped_files = 0;
        let mut seen_uris = HashSet::new();

        // Use the ignore crate to walk files while respecting .gitignore
        let walker = ignore::WalkBuilder::new(&start)
            .hidden(false)
            .git_ignore(true)
            .git_global(false)
            .git_exclude(false)
            .filter_entry(|entry| {
                let file_name = entry.file_name().to_string_lossy();
                // Exclude common build/dependency directories
                !matches!(
                    file_name.as_ref(),
                    ".git" | "target" | "node_modules" | "dist" | "build" | ".venv"
                )
            })
            .build();

        for result in walker {
            let entry = match result {
                Ok(entry) => entry,
                Err(_) => continue,
            };

            let current = entry.path();
            let metadata = match entry.metadata() {
                Ok(m) => m,
                Err(_) => continue,
            };

            if metadata.is_dir() {
                continue;
            }

            // Skip files that are too large
            if metadata.len() > MAX_FILE_SIZE {
                skipped_files += 1;
                continue;
            }

            let relative_path = match self.relative_managed_path(root, current) {
                Ok(p) => p,
                Err(_) => continue,
            };

            // Read the content once
            let content = match fs::read_to_string(current).await {
                Ok(content) => content,
                Err(err) if err.kind() == std::io::ErrorKind::InvalidData => {
                    skipped_files += 1;
                    continue;
                }
                Err(_) => {
                    skipped_files += 1;
                    continue;
                }
            };

            // Index using the already-read content
            let normalized = match Self::normalize_relative_path(&relative_path) {
                Ok(p) => p,
                Err(_) => continue,
            };
            let relative_path_str = normalized.to_string_lossy().replace('\\', "/");
            let title = Path::new(&relative_path_str)
                .file_name()
                .and_then(|part| part.to_str())
                .unwrap_or("resource")
                .to_string();
            let mut metadata_map = BTreeMap::new();
            metadata_map.insert("root".to_string(), root.to_string());
            metadata_map.insert("path".to_string(), relative_path_str.clone());
            metadata_map.insert("kind".to_string(), "file".to_string());

            let outcome = self
                .upsert_resource(ResourceUpsertRequest {
                    uri: self.resource_uri_for_path(root, &relative_path_str),
                    title: Some(title),
                    content,
                    layer: ResourceLayer::L2,
                    metadata: metadata_map,
                    previous_uri: None,
                })
                .await?;

            indexed_resources += 1;
            if outcome.reindexed_chunks > 0 {
                reindexed_resources += 1;
            }
            seen_uris.insert(self.resource_uri_for_path(root, &relative_path_str));
        }

        let existing = self
            .load_resources_for_scope(root, prefix_string.as_deref())
            .await?;
        let mut deleted_resources = 0;
        for resource in existing {
            if !seen_uris.contains(&resource.uri) {
                let _ = self.delete_resource(&resource.uri).await?;
                deleted_resources += 1;
            }
        }

        Ok(WorkspaceSyncOutcome {
            root: root.to_string(),
            prefix: prefix_string,
            indexed_resources,
            reindexed_resources,
            deleted_resources,
            skipped_files,
        })
    }

    /// Upserts a resource and its derived chunks into the engine, updating the resources table, chunk rows, FTS index, vector table, and persisted graph as needed.
    ///
    /// Returns an `UpsertOutcome` summarizing the stored resource and chunk changes.
    ///
    /// # Examples
    ///
    /// ```
    /// # use tokio::runtime::Runtime;
    /// # use crate::{ContextEngine, ResourceUpsertRequest};
    /// # let rt = Runtime::new().unwrap();
    /// rt.block_on(async {
    ///     let engine = ContextEngine::placeholder();
    ///     let req = ResourceUpsertRequest {
    ///         uri: "viking://resources/workspace/example.txt".to_string(),
    ///         content: "example content".to_string(),
    ///         metadata: serde_json::json!({}),
    ///         title: None,
    ///         layer: crate::Layer::L2,
    ///         previous_uri: None,
    ///     };
    ///     let outcome = engine.upsert_resource(req).await.unwrap();
    ///     assert!(outcome.chunks_indexed >= 0);
    /// });
    /// ```
    pub async fn upsert_resource(
        &self,
        request: ResourceUpsertRequest,
    ) -> Result<UpsertOutcome, ContextError> {
        let resource_uri = VikingUri::parse(&request.uri)?;
        if resource_uri.namespace() != &VikingNamespace::Resources {
            return Err(ContextError::InvalidRequest(
                "only viking://resources/... URIs can be indexed here".to_string(),
            ));
        }
        let root_name = resource_uri
            .resource_root()
            .ok_or_else(|| ContextError::InvalidRequest("resource URI missing root".to_string()))?;
        let resource_path = resource_uri
            .resource_path()
            .ok_or_else(|| ContextError::InvalidRequest("resource URI missing path".to_string()))?;
        let _ = self.root_by_name(root_name)?;
        let title = request.title.unwrap_or_else(|| {
            Path::new(&resource_path)
                .file_stem()
                .or_else(|| Path::new(&resource_path).file_name())
                .and_then(|part| part.to_str())
                .unwrap_or("resource")
                .to_string()
        });

        let resource_id = self
            .resolve_resource_id(&request.uri, request.previous_uri.as_deref())
            .await?;
        let now = Utc::now().timestamp_millis();
        let layered = LayeredContent::from_full_text(&request.content);
        let content_hash = hash_text(&request.content);
        let metadata_json = serde_json::to_string(&request.metadata)?;

        let resource_row = ResourceRow {
            resource_id: resource_id.clone(),
            uri: request.uri.clone(),
            title: title.clone(),
            root: root_name.to_string(),
            path: resource_path.clone(),
            source_layer: request.layer.as_str().to_string(),
            metadata_json: metadata_json.clone(),
            content_hash: content_hash.clone(),
            updated_at: now,
        };

        let chunks = self.build_chunks(&resource_id, &request.metadata, &layered);
        let existing = self.load_chunks_for_resource(&resource_id).await?;
        let existing_ids: HashSet<String> =
            existing.iter().map(|row| row.chunk_id.clone()).collect();
        let new_ids: HashSet<String> = chunks.iter().map(|row| row.chunk_id.clone()).collect();
        let reused_ids: HashSet<String> = existing_ids.intersection(&new_ids).cloned().collect();
        let deleted_ids: Vec<String> = existing_ids.difference(&new_ids).cloned().collect();
        let added_chunks: Vec<ChunkRow> = chunks
            .iter()
            .filter(|chunk| !existing_ids.contains(&chunk.chunk_id))
            .cloned()
            .collect();

        let mut tx = self.inner.pool.begin().await?;

        // Check if there's a conflicting resource at the target URI with a different resource_id
        // If so, delete it to avoid UNIQUE constraint violation
        let conflicting_resource: Option<(String,)> = sqlx::query_as(
            r#"
            SELECT resource_id FROM resources
            WHERE uri = ?1 AND resource_id != ?2
            "#,
        )
        .bind(&resource_row.uri)
        .bind(&resource_row.resource_id)
        .fetch_optional(&mut *tx)
        .await?;

        if let Some((conflicting_id,)) = conflicting_resource {
            // Delete the conflicting resource and its dependent rows
            sqlx::query("DELETE FROM resource_chunks WHERE resource_id = ?1")
                .bind(&conflicting_id)
                .execute(&mut *tx)
                .await?;
            sqlx::query("DELETE FROM resource_chunks_fts WHERE resource_id = ?1")
                .bind(&conflicting_id)
                .execute(&mut *tx)
                .await?;
            sqlx::query("DELETE FROM resources WHERE resource_id = ?1")
                .bind(&conflicting_id)
                .execute(&mut *tx)
                .await?;
        }

        sqlx::query(
            r#"
            INSERT INTO resources (resource_id, uri, title, root, path, source_layer, metadata_json, content_hash, updated_at)
            VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)
            ON CONFLICT(resource_id) DO UPDATE SET
                uri = excluded.uri,
                title = excluded.title,
                root = excluded.root,
                path = excluded.path,
                source_layer = excluded.source_layer,
                metadata_json = excluded.metadata_json,
                content_hash = excluded.content_hash,
                updated_at = excluded.updated_at
            "#,
        )
        .bind(&resource_row.resource_id)
        .bind(&resource_row.uri)
        .bind(&resource_row.title)
        .bind(&resource_row.root)
        .bind(&resource_row.path)
        .bind(&resource_row.source_layer)
        .bind(&resource_row.metadata_json)
        .bind(&resource_row.content_hash)
        .bind(resource_row.updated_at)
        .execute(&mut *tx)
        .await?;

        for deleted_id in &deleted_ids {
            sqlx::query("DELETE FROM resource_chunks WHERE chunk_id = ?1")
                .bind(deleted_id)
                .execute(&mut *tx)
                .await?;
            sqlx::query("DELETE FROM resource_chunks_fts WHERE chunk_id = ?1")
                .bind(deleted_id)
                .execute(&mut *tx)
                .await?;
        }

        for chunk in &added_chunks {
            sqlx::query(
                r#"
                INSERT INTO resource_chunks (chunk_id, resource_id, layer, ordinal, content, content_hash, metadata_json, updated_at)
                VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)
                ON CONFLICT(chunk_id) DO UPDATE SET
                    resource_id = excluded.resource_id,
                    layer = excluded.layer,
                    ordinal = excluded.ordinal,
                    content = excluded.content,
                    content_hash = excluded.content_hash,
                    metadata_json = excluded.metadata_json,
                    updated_at = excluded.updated_at
                "#,
            )
            .bind(&chunk.chunk_id)
            .bind(&chunk.resource_id)
            .bind(&chunk.layer)
            .bind(chunk.ordinal)
            .bind(&chunk.content)
            .bind(&chunk.content_hash)
            .bind(&chunk.metadata_json)
            .bind(chunk.updated_at)
            .execute(&mut *tx)
            .await?;

            sqlx::query(
                r#"
                INSERT INTO resource_chunks_fts (chunk_id, resource_id, layer, content)
                VALUES (?1, ?2, ?3, ?4)
                "#,
            )
            .bind(&chunk.chunk_id)
            .bind(&chunk.resource_id)
            .bind(&chunk.layer)
            .bind(&chunk.content)
            .execute(&mut *tx)
            .await?;
        }

        tx.commit().await?;

        self.sync_vector_table(resource_row.clone(), &chunks, &deleted_ids, &added_chunks)
            .await?;
        self.persist_graph_from_metadata(
            &request.metadata,
            &request.uri,
            request.previous_uri.as_deref(),
            now,
        )
        .await?;

        let chunks_indexed = chunks.len() as i32;
        let reused_chunks = reused_ids.len() as i32;
        let reindexed_chunks = added_chunks.len() as i32;

        Ok(UpsertOutcome {
            resource: ResourceSummary {
                uri: resource_row.uri,
                title: resource_row.title,
                layer: resource_row.source_layer,
                metadata: request.metadata,
            },
            chunks_indexed,
            reused_chunks,
            reindexed_chunks,
        })
    }

    async fn resolve_resource_id(
        &self,
        current_uri: &str,
        previous_uri: Option<&str>,
    ) -> Result<String, ContextError> {
        if let Some(previous_uri) = previous_uri {
            if let Some(existing) = self.lookup_resource_by_uri(previous_uri).await? {
                return Ok(existing.resource_id);
            }
        }

        if let Some(existing) = self.lookup_resource_by_uri(current_uri).await? {
            return Ok(existing.resource_id);
        }

        Ok(Uuid::new_v4().to_string())
    }

    async fn sync_vector_table(
        &self,
        resource_row: ResourceRow,
        existing_chunks: &[ChunkRow],
        deleted_ids: &[String],
        added_chunks: &[ChunkRow],
    ) -> Result<(), ContextError> {
        if deleted_ids.is_empty() && added_chunks.is_empty() {
            return Ok(());
        }

        for deleted_id in deleted_ids {
            self.inner
                .vector_table
                .delete(&format!("chunk_id = '{}'", escape_sql(deleted_id)))
                .await?;
        }

        if added_chunks.is_empty() {
            return Ok(());
        }

        let schema = Self::vector_schema();
        let vectors: Vec<RecordBatch> = added_chunks
            .iter()
            .map(|chunk| {
                let embedding = embed_text(&chunk.content);
                let vector_values: Vec<Option<f32>> = embedding.into_iter().map(Some).collect();
                let vector_array = FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
                    std::iter::once(Some(vector_values)),
                    EMBEDDING_DIM as i32,
                );
                RecordBatch::try_new(
                    schema.clone(),
                    vec![
                        Arc::new(StringArray::from(vec![chunk.chunk_id.clone()])),
                        Arc::new(StringArray::from(vec![chunk.resource_id.clone()])),
                        Arc::new(StringArray::from(vec![chunk.layer.clone()])),
                        Arc::new(Int64Array::from(vec![chunk.ordinal])),
                        Arc::new(StringArray::from(vec![chunk.content.clone()])),
                        Arc::new(vector_array),
                        Arc::new(StringArray::from(vec![chunk.metadata_json.clone()])),
                        Arc::new(Int64Array::from(vec![chunk.updated_at])),
                    ],
                )
                .expect("valid record batch")
            })
            .collect();

        self.inner.vector_table.add(vectors).execute().await?;

        let _ = resource_row;
        let _ = existing_chunks;
        Ok(())
    }

    fn build_chunks(
        &self,
        resource_id: &str,
        metadata: &BTreeMap<String, String>,
        layered: &LayeredContent,
    ) -> Vec<ChunkRow> {
        let mut rows = Vec::new();
        let now = Utc::now().timestamp_millis();

        for layer in [ResourceLayer::L0, ResourceLayer::L1, ResourceLayer::L2] {
            let text = layered.layer(layer);
            if text.trim().is_empty() {
                continue;
            }
            for (ordinal, chunk) in split_into_chunks(text).into_iter().enumerate() {
                rows.push(ChunkRow {
                    chunk_id: chunk_id(resource_id, layer, &chunk),
                    resource_id: resource_id.to_string(),
                    layer: layer.as_str().to_string(),
                    ordinal: ordinal as i64,
                    content: chunk,
                    content_hash: hash_text(text),
                    metadata_json: serde_json::to_string(metadata)
                        .unwrap_or_else(|_| "{}".to_string()),
                    updated_at: now,
                });
            }
        }

        rows
    }

    async fn load_chunks_for_resource(
        &self,
        resource_id: &str,
    ) -> Result<Vec<ChunkRow>, ContextError> {
        let rows = sqlx::query(
            r#"
            SELECT chunk_id, resource_id, layer, ordinal, content, content_hash, metadata_json, updated_at
            FROM resource_chunks
            WHERE resource_id = ?1
            ORDER BY layer, ordinal
            "#,
        )
        .bind(resource_id)
        .fetch_all(&self.inner.pool)
        .await?;

        Ok(rows
            .into_iter()
            .map(|row| ChunkRow {
                chunk_id: row.get("chunk_id"),
                resource_id: row.get("resource_id"),
                layer: row.get("layer"),
                ordinal: row.get::<i64, _>("ordinal"),
                content: row.get("content"),
                content_hash: row.get("content_hash"),
                metadata_json: row.get("metadata_json"),
                updated_at: row.get::<i64, _>("updated_at"),
            })
            .collect())
    }

    async fn lookup_resource_by_uri(&self, uri: &str) -> Result<Option<ResourceRow>, ContextError> {
        let row = sqlx::query(
            r#"
            SELECT resource_id, uri, title, root, path, source_layer, metadata_json, content_hash, updated_at
            FROM resources
            WHERE uri = ?1
            "#,
        )
        .bind(uri)
        .fetch_optional(&self.inner.pool)
        .await?;

        Ok(row.map(|row| ResourceRow {
            resource_id: row.get("resource_id"),
            uri: row.get("uri"),
            title: row.get("title"),
            root: row.get("root"),
            path: row.get("path"),
            source_layer: row.get("source_layer"),
            metadata_json: row.get("metadata_json"),
            content_hash: row.get("content_hash"),
            updated_at: row.get::<i64, _>("updated_at"),
        }))
    }

    async fn lookup_resource_by_id(
        &self,
        resource_id: &str,
    ) -> Result<Option<ResourceRow>, ContextError> {
        let row = sqlx::query(
            r#"
            SELECT resource_id, uri, title, root, path, source_layer, metadata_json, content_hash, updated_at
            FROM resources
            WHERE resource_id = ?1
            "#,
        )
        .bind(resource_id)
        .fetch_optional(&self.inner.pool)
        .await?;

        Ok(row.map(|row| ResourceRow {
            resource_id: row.get("resource_id"),
            uri: row.get("uri"),
            title: row.get("title"),
            root: row.get("root"),
            path: row.get("path"),
            source_layer: row.get("source_layer"),
            metadata_json: row.get("metadata_json"),
            content_hash: row.get("content_hash"),
            updated_at: row.get::<i64, _>("updated_at"),
        }))
    }

    async fn load_resources_for_scope(
        &self,
        root: &str,
        prefix: Option<&str>,
    ) -> Result<Vec<ResourceRow>, ContextError> {
        let rows = if let Some(prefix) = prefix {
            let prefix = prefix.trim_matches('/');
            sqlx::query(
                r#"
                SELECT resource_id, uri, title, root, path, source_layer, metadata_json, content_hash, updated_at
                FROM resources
                WHERE root = ?1 AND (path = ?2 OR path LIKE ?3)
                ORDER BY path ASC
                "#,
            )
            .bind(root)
            .bind(prefix)
            .bind(format!("{prefix}/%"))
            .fetch_all(&self.inner.pool)
            .await?
        } else {
            sqlx::query(
                r#"
                SELECT resource_id, uri, title, root, path, source_layer, metadata_json, content_hash, updated_at
                FROM resources
                WHERE root = ?1
                ORDER BY path ASC
                "#,
            )
            .bind(root)
            .fetch_all(&self.inner.pool)
            .await?
        };

        Ok(rows
            .into_iter()
            .map(|row| ResourceRow {
                resource_id: row.get("resource_id"),
                uri: row.get("uri"),
                title: row.get("title"),
                root: row.get("root"),
                path: row.get("path"),
                source_layer: row.get("source_layer"),
                metadata_json: row.get("metadata_json"),
                content_hash: row.get("content_hash"),
                updated_at: row.get::<i64, _>("updated_at"),
            })
            .collect())
    }

    /// Deletes the resource identified by `uri` along with its chunk rows, FTS entries, graph edges, and vector records.
    ///
    /// If no resource exists for the provided `uri`, no modifications are performed and the function reports that nothing was deleted.
    /// Database deletions are performed inside a transaction; vector rows in the LanceDB table are removed after the transaction commits.
    ///
    /// # Returns
    ///
    /// `DeleteOutcome { deleted: true }` when a resource was found and deleted, `DeleteOutcome { deleted: false }` when no resource matched the URI.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # async fn example(engine: &crate::ContextEngine) -> Result<(), crate::ContextError> {
    /// let outcome = engine.delete_resource("viking://resources/workspace/notes.txt").await?;
    /// assert!(outcome.deleted || !outcome.deleted);
    /// # Ok(()) }
    /// ```
    pub async fn delete_resource(&self, uri: &str) -> Result<DeleteOutcome, ContextError> {
        let Some(resource) = self.lookup_resource_by_uri(uri).await? else {
            return Ok(DeleteOutcome { deleted: false });
        };

        let mut tx = self.inner.pool.begin().await?;
        sqlx::query("DELETE FROM resource_chunks WHERE resource_id = ?1")
            .bind(&resource.resource_id)
            .execute(&mut *tx)
            .await?;
        sqlx::query("DELETE FROM resource_chunks_fts WHERE resource_id = ?1")
            .bind(&resource.resource_id)
            .execute(&mut *tx)
            .await?;
        sqlx::query("DELETE FROM resources WHERE resource_id = ?1")
            .bind(&resource.resource_id)
            .execute(&mut *tx)
            .await?;
        sqlx::query("DELETE FROM graph_edges WHERE resource_uri = ?1")
            .bind(uri)
            .execute(&mut *tx)
            .await?;
        tx.commit().await?;

        self.inner
            .vector_table
            .delete(&format!(
                "resource_id = '{}'",
                escape_sql(&resource.resource_id)
            ))
            .await?;

        Ok(DeleteOutcome { deleted: true })
    }

    pub async fn list_resources(&self) -> Result<Vec<ResourceSummary>, ContextError> {
        let rows = sqlx::query(
            r#"
            SELECT uri, title, source_layer, metadata_json
            FROM resources
            ORDER BY updated_at DESC
            "#,
        )
        .fetch_all(&self.inner.pool)
        .await?;

        let mut summaries = Vec::with_capacity(rows.len());
        for row in rows {
            summaries.push(ResourceSummary {
                uri: row.get("uri"),
                title: row.get("title"),
                layer: row.get("source_layer"),
                metadata: serde_json::from_str(&row.get::<String, _>("metadata_json"))?,
            });
        }
        Ok(summaries)
    }

    pub async fn search_context(
        &self,
        request: SearchRequest,
    ) -> Result<Vec<SearchHit>, ContextError> {
        let top_k = request.top_k.unwrap_or(10).max(1);
        let mut candidates: HashMap<String, ScoredChunk> = HashMap::new();
        let normalized_query = normalize_search_query(&request.query);

        let lexical_candidates = self
            .search_lexical(&normalized_query, top_k * 3)
            .await
            .unwrap_or_default();
        for candidate in lexical_candidates {
            candidates
                .entry(candidate.chunk_id.clone())
                .and_modify(|existing| {
                    if candidate.score > existing.score {
                        *existing = candidate.clone();
                    }
                })
                .or_insert(candidate);
        }

        if request.rerank.unwrap_or(true) {
            let dense_candidates = self.search_dense(&request.query, top_k * 3).await?;
            for candidate in dense_candidates {
                candidates
                    .entry(candidate.chunk_id.clone())
                    .and_modify(|existing| {
                        if candidate.score > existing.score {
                            *existing = candidate.clone();
                        }
                    })
                    .or_insert(candidate);
            }
        }

        let mut joined = Vec::new();
        for candidate in candidates.into_values() {
            let Some(resource) = self.lookup_resource_by_id(&candidate.resource_id).await? else {
                continue;
            };

            if let Some(scope_uri) = &request.scope_uri {
                if !resource.uri.starts_with(scope_uri) {
                    continue;
                }
            }

            if let Some(layer) = request.layer {
                if candidate.layer != layer.as_str() {
                    continue;
                }
            }

            let metadata = merge_metadata(&resource.metadata_json, &candidate.metadata_json)?;
            if !filters_match(request.filters.as_ref(), &metadata) {
                continue;
            }

            joined.push(SearchHit {
                uri: resource.uri,
                document_id: resource.resource_id,
                chunk_text: candidate.content,
                score: candidate.score,
                metadata,
                layer: candidate.layer,
            });
        }

        joined.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        joined.truncate(top_k);
        Ok(joined)
    }

    async fn search_lexical(
        &self,
        query: &str,
        limit: usize,
    ) -> Result<Vec<ScoredChunk>, ContextError> {
        let sql = r#"
            SELECT chunk_id, resource_id, layer, content, bm25(resource_chunks_fts) AS score
            FROM resource_chunks_fts
            WHERE resource_chunks_fts MATCH ?1
            ORDER BY score
            LIMIT ?2
        "#;

        let rows = sqlx::query(sql)
            .bind(query)
            .bind(limit as i64)
            .fetch_all(&self.inner.pool)
            .await?;

        Ok(rows
            .into_iter()
            .map(|row| {
                let bm25_score = row.get::<f64, _>("score") as f32;
                ScoredChunk {
                    chunk_id: row.get("chunk_id"),
                    resource_id: row.get("resource_id"),
                    layer: row.get("layer"),
                    content: row.get("content"),
                    score: lexical_score(-bm25_score),
                    metadata_json: "{}".to_string(),
                }
            })
            .collect())
    }

    async fn search_dense(
        &self,
        query: &str,
        limit: usize,
    ) -> Result<Vec<ScoredChunk>, ContextError> {
        let embedding = embed_text(query);
        let query_results = self
            .inner
            .vector_table
            .query()
            .nearest_to(embedding)?
            .limit(limit)
            .execute()
            .await?
            .try_collect::<Vec<RecordBatch>>()
            .await?;

        let mut rows = Vec::new();
        for batch in query_results {
            rows.extend(parse_vector_batch(batch)?);
        }
        Ok(rows)
    }

    pub async fn list_sessions(&self, session_id: &str) -> Result<Vec<SessionEntry>, ContextError> {
        let rows = sqlx::query(
            r#"
            SELECT session_id, role, content, metadata_json, created_at
            FROM sessions
            WHERE session_id = ?1
            ORDER BY created_at ASC, id ASC
            "#,
        )
        .bind(session_id)
        .fetch_all(&self.inner.pool)
        .await?;

        Ok(rows
            .into_iter()
            .map(|row| SessionEntry {
                session_id: row.get("session_id"),
                role: row.get("role"),
                content: row.get("content"),
                metadata: serde_json::from_str(&row.get::<String, _>("metadata_json"))
                    .unwrap_or_default(),
                created_at: row.get::<i64, _>("created_at"),
            })
            .collect())
    }

    /// Appends a session event to storage and updates the working-memory cache when configured.
    ///
    /// Inserts a session event using the provided `SessionEventRequest` and returns a `SessionEntry`
    /// populated with the stored fields and the insertion timestamp. When a Dragonfly working-memory
    /// cache is configured, the engine refreshes that cache for the session after insertion.
    ///
    /// # Returns
    ///
    /// `Ok(SessionEntry)` containing the stored event with `created_at` set to the current
    /// millisecond timestamp, or `Err(ContextError)` if persistence or cache refresh fails.
    ///
    /// # Examples
    ///
    /// ```
    /// # use chrono::Utc;
    /// # use tokio_test::block_on;
    /// # async fn _example(engine: &crate::ContextEngine) {
    /// let req = crate::SessionEventRequest {
    ///     session_id: "sess-123".to_string(),
    ///     role: "user".to_string(),
    ///     content: "Hello".to_string(),
    ///     metadata: serde_json::json!({}),
    /// };
    /// let entry = engine.append_session(req).await.unwrap();
    /// assert_eq!(entry.session_id, "sess-123");
    /// # }
    /// ```
    pub async fn append_session(
        &self,
        request: SessionEventRequest,
    ) -> Result<SessionEntry, ContextError> {
        let created_at = Utc::now().timestamp_millis();
        let metadata_json = serde_json::to_string(&request.metadata)?;
        sqlx::query(
            r#"
            INSERT INTO sessions (session_id, role, content, metadata_json, created_at)
            VALUES (?1, ?2, ?3, ?4, ?5)
            "#,
        )
        .bind(&request.session_id)
        .bind(&request.role)
        .bind(&request.content)
        .bind(&metadata_json)
        .bind(created_at)
        .execute(&self.inner.pool)
        .await?;

        self.refresh_working_memory_cache(&request.session_id)
            .await?;

        Ok(SessionEntry {
            session_id: request.session_id,
            role: request.role,
            content: request.content,
            metadata: request.metadata,
            created_at,
        })
    }

    /// Retrieve recent session entries for a session, preferring the Dragonfly working-memory cache when enabled and the requested limit fits within the configured recent window.
    ///
    /// If a Dragonfly cache is configured and `limit` is less than or equal to the cache's `recent_window`, cached entries are returned when present; otherwise the function loads entries from SQLite.
    ///
    /// # Returns
    ///
    /// A `Vec<SessionEntry>` containing up to `limit` entries in chronological order (oldest first).
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // Assuming `engine: ContextEngine` is available:
    /// let entries = engine.recent_session_entries("session-id", 10).await.unwrap();
    /// ```
    pub async fn recent_session_entries(
        &self,
        session_id: &str,
        limit: usize,
    ) -> Result<Vec<SessionEntry>, ContextError> {
        let limit = limit.max(1);
        if let Some(window) = self
            .inner
            .config
            .dragonfly
            .as_ref()
            .map(|config| config.recent_window.max(1))
        {
            if limit <= window {
                if let Some(mut entries) = self.load_working_memory_cache(session_id).await? {
                    if entries.len() > limit {
                        entries = entries.split_off(entries.len() - limit);
                    }
                    return Ok(entries);
                }
            }
        }
        self.load_recent_session_entries_from_sql(session_id, limit)
            .await
    }

    /// Searches graph nodes and edges for a text fragment and returns matching graph fact records.
    ///
    /// The query is matched against subject/object `node_id`, `name`, edge `relation`, and edge `metadata_json` (SQL `LIKE '%query%'`).
    /// Results are ordered by edge `updated_at` descending and limited to `top_k`.
    ///
    /// # Examples
    ///
    /// ```
    /// # async fn run_example(engine: &crate::ContextEngine) -> Result<(), crate::ContextError> {
    /// let facts = engine.graph_facts("alice", 5).await?;
    /// for f in facts {
    ///     println!("{} - {} -> {}", f.subject.name, f.relation.as_str(), f.object.name);
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub async fn graph_facts(
        &self,
        query: &str,
        top_k: usize,
    ) -> Result<Vec<GraphFactRecord>, ContextError> {
        let like = format!("%{}%", query.trim());
        let rows = sqlx::query(
            r#"
            SELECT
                e.edge_id,
                e.subject_id,
                e.relation,
                e.object_id,
                e.metadata_json,
                e.resource_uri,
                e.session_id,
                e.updated_at,
                s.kind AS subject_kind,
                s.name AS subject_name,
                s.metadata_json AS subject_metadata_json,
                o.kind AS object_kind,
                o.name AS object_name,
                o.metadata_json AS object_metadata_json
            FROM graph_edges e
            JOIN graph_nodes s ON s.node_id = e.subject_id
            JOIN graph_nodes o ON o.node_id = e.object_id
            WHERE
                s.node_id LIKE ?1 OR
                s.name LIKE ?1 OR
                o.node_id LIKE ?1 OR
                o.name LIKE ?1 OR
                e.relation LIKE ?1 OR
                e.metadata_json LIKE ?1
            ORDER BY e.updated_at DESC
            LIMIT ?2
            "#,
        )
        .bind(&like)
        .bind(top_k as i64)
        .fetch_all(&self.inner.pool)
        .await?;

        let mut facts = Vec::with_capacity(rows.len());
        for row in rows {
            let metadata_json = row.get::<String, _>("metadata_json");
            let subject_metadata_json = row.get::<String, _>("subject_metadata_json");
            let object_metadata_json = row.get::<String, _>("object_metadata_json");
            facts.push(GraphFactRecord {
                edge_id: row.get("edge_id"),
                subject: GraphNode {
                    id: row.get("subject_id"),
                    name: row.get("subject_name"),
                    kind: GraphNodeKind::from_metadata(&row.get::<String, _>("subject_kind")),
                    metadata: serde_json::from_str(&subject_metadata_json)?,
                },
                relation: GraphRelationKind::from_metadata(&row.get::<String, _>("relation")),
                object: GraphNode {
                    id: row.get("object_id"),
                    name: row.get("object_name"),
                    kind: GraphNodeKind::from_metadata(&row.get::<String, _>("object_kind")),
                    metadata: serde_json::from_str(&object_metadata_json)?,
                },
                metadata: serde_json::from_str(&metadata_json)?,
                resource_uri: row.get("resource_uri"),
                session_id: row.get("session_id"),
                updated_at: row.get("updated_at"),
            });
        }
        Ok(facts)
    }

    /// Loads recent session entries for a session from the SQLite `sessions` table.
    ///
    /// Returns the most recent `limit` entries for `session_id`, ordered chronologically
    /// (oldest first). The query selects rows ordered by `created_at DESC, id DESC` and then
    /// reverses them before returning so callers receive entries in ascending time order.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # async fn example(engine: &crate::ContextEngine) -> Result<(), crate::ContextError> {
    /// let entries = engine.load_recent_session_entries_from_sql("session-123", 20).await?;
    /// for entry in entries {
    ///     println!("{}: {}", entry.created_at, entry.content);
    /// }
    /// # Ok(()) }
    /// ```
    async fn load_recent_session_entries_from_sql(
        &self,
        session_id: &str,
        limit: usize,
    ) -> Result<Vec<SessionEntry>, ContextError> {
        let rows = sqlx::query(
            r#"
            SELECT session_id, role, content, metadata_json, created_at
            FROM sessions
            WHERE session_id = ?1
            ORDER BY created_at DESC, id DESC
            LIMIT ?2
            "#,
        )
        .bind(session_id)
        .bind(limit as i64)
        .fetch_all(&self.inner.pool)
        .await?;

        let mut entries = rows
            .into_iter()
            .map(|row| SessionEntry {
                session_id: row.get("session_id"),
                role: row.get("role"),
                content: row.get("content"),
                metadata: serde_json::from_str(&row.get::<String, _>("metadata_json"))
                    .unwrap_or_default(),
                created_at: row.get::<i64, _>("created_at"),
            })
            .collect::<Vec<_>>();
        entries.reverse();
        Ok(entries)
    }

    /// Refreshes the Dragonfly-backed working memory cache for a session by storing recent session entries in the `memories` table.
    ///
    /// If Dragonfly is not configured for this engine, this method does nothing. Otherwise, it loads the most recent session entries (bounded by the configured `recent_window`) and upserts a JSON-encoded cache record keyed by the working-memory URI and namespace derived from the Dragonfly configuration.
    ///
    /// Returns an error if URI generation, JSON serialization, or the database upsert fails.
    ///
    /// # Examples
    ///
    /// ```
    /// # tokio_test::block_on(async {
    /// let engine = ContextEngine::placeholder();
    /// // No-op if Dragonfly isn't configured; otherwise updates the working memory cache.
    /// engine.refresh_working_memory_cache("session-123").await.unwrap();
    /// # });
    /// ```
    async fn refresh_working_memory_cache(&self, session_id: &str) -> Result<(), ContextError> {
        let Some(config) = self.inner.config.dragonfly.as_ref() else {
            return Ok(());
        };

        let entries = self
            .load_recent_session_entries_from_sql(session_id, config.recent_window.max(1))
            .await?;
        let uri = working_memory_cache_uri(session_id)?;
        let metadata_json = serde_json::to_string(&BTreeMap::from([
            ("provider".to_string(), "dragonfly".to_string()),
            ("session_id".to_string(), session_id.to_string()),
            (
                "recent_window".to_string(),
                config.recent_window.max(1).to_string(),
            ),
        ]))?;

        sqlx::query(
            r#"
            INSERT INTO memories (uri, namespace, content, metadata_json, updated_at)
            VALUES (?1, ?2, ?3, ?4, ?5)
            ON CONFLICT(uri) DO UPDATE SET
                namespace = excluded.namespace,
                content = excluded.content,
                metadata_json = excluded.metadata_json,
                updated_at = excluded.updated_at
            "#,
        )
        .bind(&uri)
        .bind(&config.key_prefix)
        .bind(serde_json::to_string(&entries)?)
        .bind(metadata_json)
        .bind(Utc::now().timestamp_millis())
        .execute(&self.inner.pool)
        .await?;

        Ok(())
    }

    /// Loads the working-memory cache for a session if a Dragonfly cache is configured.
    ///
    /// Returns `Ok(Some(entries))` when a cached memory record exists for `session_id`,
    /// `Ok(None)` when Dragonfly is not configured or no cache record is found, and
    /// `Err(ContextError)` on SQL or deserialization failures.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # async fn example(engine: &crate::ContextEngine) -> Result<(), crate::ContextError> {
    /// let session_id = "session-123";
    /// let cached = engine.load_working_memory_cache(session_id).await?;
    /// if let Some(entries) = cached {
    ///     // use entries...
    /// }
    /// # Ok(())
    /// # }
    /// ```
    async fn load_working_memory_cache(
        &self,
        session_id: &str,
    ) -> Result<Option<Vec<SessionEntry>>, ContextError> {
        let Some(config) = self.inner.config.dragonfly.as_ref() else {
            return Ok(None);
        };
        let uri = working_memory_cache_uri(session_id)?;
        let row = sqlx::query(
            r#"
            SELECT content
            FROM memories
            WHERE uri = ?1 AND namespace = ?2
            "#,
        )
        .bind(uri)
        .bind(&config.key_prefix)
        .fetch_optional(&self.inner.pool)
        .await?;

        row.map(|row| serde_json::from_str(&row.get::<String, _>("content")))
            .transpose()
            .map_err(ContextError::from)
    }

    /// Persist graph nodes and a graph edge derived from resource metadata into the graph tables.
    ///
    /// This will remove any existing edges associated with `resource_uri` (and `previous_uri` if
    /// provided and different), and then, when `metadata["kind"]` equals `"graph"` (case-insensitive)
    /// and both `subject_id` and `object_id` are present and non-empty, upsert the corresponding
    /// subject and object nodes and a graph edge linking them. The inserted edge's metadata will
    /// include `kind = "graph"` and `resource_uri`, and will also record `previous_resource_uri` when
    /// `previous_uri` is supplied.
    ///
    /// @param metadata  Resource metadata map used to derive nodes and edge fields (expects keys like
    ///                  `kind`, `subject_id`, `object_id`, `subject_name`, `object_name`, `relation`,
    ///                  and optional `subject_*/object_*` prefixed node metadata).
    /// @param resource_uri  The URI of the resource that produced the metadata; stored on the edge.
    /// @param previous_uri  Optional prior resource URI to record on the edge and to delete prior edges
    ///                      (ignored if equal to `resource_uri`).
    /// @param updated_at  Millisecond Unix timestamp to record as the nodes' and edge's update time.
    ///
    /// # Returns
    ///
    /// `Ok(())` when graph rows were persisted or when the metadata does not describe a graph; `Err` on
    /// database or serialization errors.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::collections::BTreeMap;
    ///
    /// // Prepare metadata describing a graph fact
    /// let mut metadata = BTreeMap::new();
    /// metadata.insert("kind".to_string(), "graph".to_string());
    /// metadata.insert("subject_id".to_string(), "node:a".to_string());
    /// metadata.insert("object_id".to_string(), "node:b".to_string());
    /// metadata.insert("relation".to_string(), "RELATED_TO".to_string());
    ///
    /// // Example usage (obtain a ContextEngine instance appropriate for your application):
    /// // let engine: ContextEngine = /* ... */ ;
    /// // let _ = tokio::runtime::Handle::current()
    /// //     .block_on(engine.persist_graph_from_metadata(&metadata, "viking://resources/1", None, 1670000000000));
    /// ```
    async fn persist_graph_from_metadata(
        &self,
        metadata: &BTreeMap<String, String>,
        resource_uri: &str,
        previous_uri: Option<&str>,
        updated_at: i64,
    ) -> Result<(), ContextError> {
        let delete_current = sqlx::query("DELETE FROM graph_edges WHERE resource_uri = ?1")
            .bind(resource_uri)
            .execute(&self.inner.pool);
        let delete_previous = async {
            if let Some(previous_uri) = previous_uri.filter(|uri| *uri != resource_uri) {
                sqlx::query("DELETE FROM graph_edges WHERE resource_uri = ?1")
                    .bind(previous_uri)
                    .execute(&self.inner.pool)
                    .await?;
            }
            Ok::<(), sqlx::Error>(())
        };
        let (current_delete, previous_delete) = tokio::join!(delete_current, delete_previous);
        current_delete?;
        previous_delete?;

        if metadata
            .get("kind")
            .map(|value| !value.eq_ignore_ascii_case("graph"))
            .unwrap_or(true)
        {
            return Ok(());
        }

        let Some(subject_id) = metadata
            .get("subject_id")
            .map(|value| value.trim())
            .filter(|value| !value.is_empty())
        else {
            return Ok(());
        };
        let Some(object_id) = metadata
            .get("object_id")
            .map(|value| value.trim())
            .filter(|value| !value.is_empty())
        else {
            return Ok(());
        };

        let relation = GraphRelationKind::from_metadata(
            metadata
                .get("relation")
                .map(String::as_str)
                .unwrap_or("RELATED_TO"),
        );
        let subject = GraphNode {
            id: subject_id.to_string(),
            name: metadata
                .get("subject_name")
                .cloned()
                .filter(|value| !value.trim().is_empty())
                .unwrap_or_else(|| subject_id.to_string()),
            kind: GraphNodeKind::from_metadata(
                metadata
                    .get("subject_type")
                    .map(String::as_str)
                    .unwrap_or("generic"),
            ),
            metadata: collect_graph_node_metadata(metadata, "subject_"),
        };
        let object = GraphNode {
            id: object_id.to_string(),
            name: metadata
                .get("object_name")
                .cloned()
                .filter(|value| !value.trim().is_empty())
                .unwrap_or_else(|| object_id.to_string()),
            kind: GraphNodeKind::from_metadata(
                metadata
                    .get("object_type")
                    .map(String::as_str)
                    .unwrap_or("generic"),
            ),
            metadata: collect_graph_node_metadata(metadata, "object_"),
        };

        self.upsert_graph_node(&subject, updated_at).await?;
        self.upsert_graph_node(&object, updated_at).await?;

        let mut edge_metadata = metadata.clone();
        edge_metadata.insert("kind".to_string(), "graph".to_string());
        edge_metadata
            .entry("resource_uri".to_string())
            .or_insert_with(|| resource_uri.to_string());
        if let Some(previous_uri) = previous_uri {
            edge_metadata
                .entry("previous_resource_uri".to_string())
                .or_insert_with(|| previous_uri.to_string());
        }

        self.upsert_graph_edge(&GraphEdgeRow {
            edge_id: graph_edge_id(&subject.id, relation.as_str(), &object.id),
            subject_id: subject.id,
            relation: relation.as_str().to_string(),
            object_id: object.id,
            metadata_json: serde_json::to_string(&edge_metadata)?,
            resource_uri: Some(resource_uri.to_string()),
            session_id: metadata.get("session_id").cloned(),
            updated_at,
        })
        .await
    }

    /// Inserts or updates a graph node record in the `graph_nodes` table.
    ///
    /// Ensures a row exists for `node.id` with the node's `kind`, `name`, `metadata_json` (serialized from `node.metadata`), and `updated_at`; if a row with the same `node_id` already exists it is replaced with the provided values.
    ///
    /// # Examples
    ///
    /// ```
    /// # async fn example(engine: &ContextEngine) -> Result<(), ContextError> {
    /// let node = GraphNode {
    ///     id: "node-123".to_string(),
    ///     kind: "person".to_string(),
    ///     name: "Alice".to_string(),
    ///     metadata: serde_json::json!({ "role": "author" }),
    /// };
    /// engine.upsert_graph_node(&node, 1_700_000_000_000_i64).await?;
    /// # Ok(())
    /// # }
    /// ```
    async fn upsert_graph_node(
        &self,
        node: &GraphNode,
        updated_at: i64,
    ) -> Result<(), ContextError> {
        sqlx::query(
            r#"
            INSERT INTO graph_nodes (node_id, kind, name, metadata_json, updated_at)
            VALUES (?1, ?2, ?3, ?4, ?5)
            ON CONFLICT(node_id) DO UPDATE SET
                kind = excluded.kind,
                name = excluded.name,
                metadata_json = excluded.metadata_json,
                updated_at = excluded.updated_at
            "#,
        )
        .bind(&node.id)
        .bind(node.kind.as_str())
        .bind(&node.name)
        .bind(serde_json::to_string(&node.metadata)?)
        .bind(updated_at)
        .execute(&self.inner.pool)
        .await?;

        Ok(())
    }

    /// Inserts or updates a graph edge record in the `graph_edges` table.
    ///
    /// If a row with the same `edge_id` already exists, the existing row is replaced
    /// with the values from `edge`.
    ///
    /// # Errors
    ///
    /// Returns a `ContextError` if the database operation fails.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # async fn example(engine: &crate::ContextInner) -> Result<(), crate::ContextError> {
    /// let edge = crate::GraphEdgeRow {
    ///     edge_id: "edge1".into(),
    ///     subject_id: "subject1".into(),
    ///     relation: "RELATED_TO".into(),
    ///     object_id: "object1".into(),
    ///     metadata_json: "{}".into(),
    ///     resource_uri: "viking://resources/root/path".into(),
    ///     session_id: None,
    ///     updated_at: 0,
    /// };
    /// engine.upsert_graph_edge(&edge).await?;
    /// # Ok(())
    /// # }
    /// ```
    async fn upsert_graph_edge(&self, edge: &GraphEdgeRow) -> Result<(), ContextError> {
        sqlx::query(
            r#"
            INSERT INTO graph_edges (edge_id, subject_id, relation, object_id, metadata_json, resource_uri, session_id, updated_at)
            VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)
            ON CONFLICT(edge_id) DO UPDATE SET
                subject_id = excluded.subject_id,
                relation = excluded.relation,
                object_id = excluded.object_id,
                metadata_json = excluded.metadata_json,
                resource_uri = excluded.resource_uri,
                session_id = excluded.session_id,
                updated_at = excluded.updated_at
            "#,
        )
        .bind(&edge.edge_id)
        .bind(&edge.subject_id)
        .bind(&edge.relation)
        .bind(&edge.object_id)
        .bind(&edge.metadata_json)
        .bind(&edge.resource_uri)
        .bind(&edge.session_id)
        .bind(edge.updated_at)
        .execute(&self.inner.pool)
        .await?;

        Ok(())
    }

    /// Provides runtime status and storage metrics for the engine.
    ///
    /// The returned `StatusResponse` includes counts of indexed documents and chunks,
    /// an estimated index size in bytes, the embedding model identifier, readiness,
    /// and a list of managed roots.
    ///
    /// # Examples
    ///
    /// ```
    /// # use tokio;
    /// # async fn _run() {
    /// let engine = ContextEngine::placeholder();
    /// let status = engine.status().await.unwrap();
    /// assert!(status.ready);
    /// assert!(status.document_count >= 0);
    /// # }
    /// ```
    pub async fn status(&self) -> Result<StatusResponse, ContextError> {
        let document_count = sqlx::query_scalar::<_, i64>("SELECT COUNT(*) FROM resources")
            .fetch_one(&self.inner.pool)
            .await?;
        let chunk_count = sqlx::query_scalar::<_, i64>("SELECT COUNT(*) FROM resource_chunks")
            .fetch_one(&self.inner.pool)
            .await?;
        let index_size_bytes = chunk_count * EMBEDDING_DIM as i64 * 4;
        let managed_roots = self
            .inner
            .config
            .roots
            .iter()
            .map(|root| format!("{}:{}", root.name, root.path.display()))
            .collect();

        Ok(StatusResponse {
            document_count,
            chunk_count,
            index_size_bytes,
            embedding_model: "hash-embedder-384".to_string(),
            ready: true,
            managed_roots,
        })
    }

    pub async fn managed_roots(&self) -> Vec<ManagedRoot> {
        self.inner.config.roots.clone()
    }

    /// Lazily initializes and returns the configured OpenViking bridge client, if one is configured.
    ///
    /// # Examples
    ///
    /// ```
    /// // Demonstrates calling `bridge` in an async context.
    /// // Replace `your_crate::ContextEngine` with the actual path if needed.
    /// # use your_crate::ContextEngine;
    /// #[tokio::main]
    /// async fn main() {
    ///     let engine = ContextEngine::placeholder();
    ///     let result = engine.bridge().await.unwrap();
    ///     // `result` is `Some(client)` when a bridge is configured, otherwise `None`.
    ///     assert!(result.is_none() || result.is_some());
    /// }
    /// ```
    ///
    /// @returns `Ok(Some(client))` with the configured `OpenVikingBridgeClient`, `Ok(None)` when no bridge is configured.
    pub async fn bridge(&self) -> Result<Option<OpenVikingBridgeClient>, ContextError> {
        if self.inner.bridge.get().is_none() {
            let bridge = self
                .inner
                .config
                .bridge
                .clone()
                .map(OpenVikingBridgeClient::new);
            let _ = self.inner.bridge.set(bridge);
        }
        Ok(self.inner.bridge.get().cloned().flatten())
    }
}

impl GraphStore for ContextEngine {
    /// Ensures the subject and object graph nodes exist, then upserts the edge described by `fact`.
    ///
    /// On success the graph will contain the two nodes and the relation edge encoded in `fact`.
    ///
    /// Returns `Ok(())` on success, `Err(ContextError)` if any database, serialization, or I/O error occurs.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// // `engine` implements the GraphStore trait
    /// let fact = GraphFactRecord { /* populate required fields */ };
    /// futures::executor::block_on(engine.upsert_fact(fact)).unwrap();
    /// ```
    fn upsert_fact<'a>(
        &'a self,
        fact: GraphFactRecord,
    ) -> crate::graph::ProviderFuture<'a, Result<(), ContextError>> {
        Box::pin(async move {
            self.upsert_graph_node(&fact.subject, fact.updated_at)
                .await?;
            self.upsert_graph_node(&fact.object, fact.updated_at)
                .await?;
            self.upsert_graph_edge(&GraphEdgeRow {
                edge_id: fact.edge_id,
                subject_id: fact.subject.id,
                relation: fact.relation.as_str().to_string(),
                object_id: fact.object.id,
                metadata_json: serde_json::to_string(&fact.metadata)?,
                resource_uri: fact.resource_uri,
                session_id: fact.session_id,
                updated_at: fact.updated_at,
            })
            .await
        })
    }

    /// Query graph facts that match a text query and return up to `limit` results.
    
    ///
    
    /// The returned future resolves to `Ok(Vec<GraphFactRecord>)` when the query succeeds,
    
    /// or a `ContextError` on failure.
    
    ///
    
    /// # Examples
    
    ///
    
    /// ```no_run
    
    /// # use std::sync::Arc;
    
    /// # use futures::executor::block_on;
    
    /// # use my_crate::ContextEngine; // replace with the actual path
    
    /// # async fn _example(engine: Arc<ContextEngine>) {
    
    /// let fut = engine.related_facts("alice", 10);
    
    /// let result = fut.await;
    
    /// match result {
    
    ///     Ok(facts) => println!("found {} facts", facts.len()),
    
    ///     Err(e) => eprintln!("error: {:?}", e),
    
    /// }
    
    /// # }
    
    /// ```
    fn related_facts<'a>(
        &'a self,
        query: &'a str,
        limit: usize,
    ) -> crate::graph::ProviderFuture<'a, Result<Vec<GraphFactRecord>, ContextError>> {
        Box::pin(async move { self.graph_facts(query, limit).await })
    }
}

impl WorkingMemoryProvider for ContextEngine {
    /// Appends a session entry to the engine's session store and refreshes the working-memory cache if configured.
    ///
    /// The provided `SessionEntry` is converted into an internal session event and persisted; when a Dragonfly
    /// working-memory provider is configured the in-memory cache is also updated.
    ///
    /// # Returns
    ///
    /// `Ok(())` on success, `Err(ContextError)` on failure.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// async fn example(engine: &ContextEngine) {
    ///     let entry = SessionEntry {
    ///         session_id: "session-123".to_string(),
    ///         role: "user".to_string(),
    ///         content: "Hello".to_string(),
    ///         metadata: serde_json::json!({}),
    ///     };
    ///     engine.append_session_entry(entry).await.unwrap();
    /// }
    /// ```
    fn append_session_entry<'a>(
        &'a self,
        entry: SessionEntry,
    ) -> crate::memory::ProviderFuture<'a, Result<(), ContextError>> {
        Box::pin(async move {
            self.append_session(SessionEventRequest {
                session_id: entry.session_id,
                role: entry.role,
                content: entry.content,
                metadata: entry.metadata,
            })
            .await?;
            Ok(())
        })
    }

    /// Provide recent session entries for a session, using the engine's working-memory cache when available.
    ///
    /// Returns up to `limit` recent `SessionEntry` items for `session_id` in chronological order, or an `ContextError` on failure.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // Async context required:
    /// let engine = ContextEngine::placeholder();
    /// let entries = futures::executor::block_on(engine.recent_entries("session-1", 10)).unwrap();
    /// assert!(entries.len() <= 10);
    /// ```
    fn recent_entries<'a>(
        &'a self,
        session_id: &'a str,
        limit: usize,
    ) -> crate::memory::ProviderFuture<'a, Result<Vec<SessionEntry>, ContextError>> {
        Box::pin(async move { self.recent_session_entries(session_id, limit).await })
    }
}

/// Convert a LanceDB/Arrow `RecordBatch` produced by a vector nearest-neighbors query
/// into a vector of `ScoredChunk` records.
///
/// Parses the following columns from `batch`:
/// - `chunk_id`, `resource_id`, `layer`, `content` (required string columns)
/// - `metadata_json` (optional string column, defaults to `"{}"` per row)
/// - `_distance` (optional Float32 column; when present each row's `score` is computed as `1.0 / (1.0 + distance)`, otherwise `score` is `1.0`)
///
/// # Examples
///
/// ```ignore
/// // Given a RecordBatch `batch` returned from a LanceDB nearest-to query:
/// let scored_chunks = parse_vector_batch(batch).expect("valid batch");
/// for chunk in scored_chunks {
///     println!("{} (score={})", chunk.chunk_id, chunk.score);
/// }
/// ```
fn parse_vector_batch(batch: RecordBatch) -> Result<Vec<ScoredChunk>, ContextError> {
    let chunk_ids = column_str(&batch, "chunk_id")?;
    let resource_ids = column_str(&batch, "resource_id")?;
    let layers = column_str(&batch, "layer")?;
    let contents = column_str(&batch, "content")?;
    let metadata_json = batch
        .column_by_name("metadata_json")
        .and_then(|array| array.as_any().downcast_ref::<StringArray>())
        .map(|array| {
            (0..array.len())
                .map(|index| array.value(index).to_string())
                .collect::<Vec<_>>()
        })
        .unwrap_or_else(|| vec!["{}".to_string(); batch.num_rows()]);

    let scores = batch
        .column_by_name("_distance")
        .and_then(|array| array.as_any().downcast_ref::<Float32Array>())
        .map(|array| {
            (0..array.len())
                .map(|index| 1.0 / (1.0 + array.value(index)))
                .collect::<Vec<f32>>()
        })
        .unwrap_or_else(|| vec![1.0; batch.num_rows()]);

    let mut rows = Vec::new();
    for index in 0..batch.num_rows() {
        rows.push(ScoredChunk {
            chunk_id: chunk_ids[index].clone(),
            resource_id: resource_ids[index].clone(),
            layer: layers[index].clone(),
            content: contents[index].clone(),
            score: scores[index],
            metadata_json: metadata_json[index].clone(),
        });
    }
    Ok(rows)
}

fn column_str(batch: &RecordBatch, name: &str) -> Result<Vec<String>, ContextError> {
    let array = batch
        .column_by_name(name)
        .ok_or_else(|| ContextError::InvalidRequest(format!("missing column {name}")))?;
    let array = array
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| ContextError::InvalidRequest(format!("column {name} is not string")))?;
    Ok((0..array.len())
        .map(|index| array.value(index).to_string())
        .collect())
}

fn filters_match(
    filters: Option<&BTreeMap<String, String>>,
    metadata: &BTreeMap<String, String>,
) -> bool {
    filters
        .map(|filters| {
            filters
                .iter()
                .all(|(key, value)| metadata.get(key) == Some(value))
        })
        .unwrap_or(true)
}

/// Merge two JSON-encoded metadata objects into a single map, preferring keys from the resource metadata.
///
/// Parses `resource_json` and `chunk_json` as `Map<String, String>` and returns a new `BTreeMap` containing
/// all keys from `resource_json` plus any keys from `chunk_json` that are not present in `resource_json`.
///
/// # Returns
///
/// `Ok(BTreeMap<String, String>)` with the merged metadata; `Err(ContextError)` if either input is invalid JSON or cannot be deserialized into a `Map<String, String>`.
///
/// # Examples
///
/// ```
/// let resource_json = r#"{"title":"Doc","author":"Alice"}"#;
/// let chunk_json = r#"{"author":"Bob","chunk":"intro"}"#;
/// let merged = merge_metadata(resource_json, chunk_json).unwrap();
/// assert_eq!(merged.get("title").map(String::as_str), Some("Doc"));
/// // resource value wins when keys conflict
/// assert_eq!(merged.get("author").map(String::as_str), Some("Alice"));
/// // chunk-only key is preserved
/// assert_eq!(merged.get("chunk").map(String::as_str), Some("intro"));
/// ```
fn merge_metadata(
    resource_json: &str,
    chunk_json: &str,
) -> Result<BTreeMap<String, String>, ContextError> {
    let mut merged: BTreeMap<String, String> = serde_json::from_str(resource_json)?;
    let chunk: BTreeMap<String, String> = serde_json::from_str(chunk_json)?;
    for (key, value) in chunk {
        merged.entry(key).or_insert(value);
    }
    Ok(merged)
}

/// Extracts metadata entries with the given prefix, returning a map of the keys with the prefix removed.
///
/// Keys equal to `id`, `name`, or `type` (after removing the prefix) are omitted.
///
/// # Examples
///
/// ```
/// use std::collections::BTreeMap;
///
/// let mut src = BTreeMap::new();
/// src.insert("subject_id".to_string(), "s1".to_string());
/// src.insert("subject_name".to_string(), "Subject".to_string());
/// src.insert("subject_role".to_string(), "actor".to_string());
/// src.insert("other".to_string(), "x".to_string());
///
/// let out = collect_graph_node_metadata(&src, "subject_");
/// assert_eq!(out.get("role").map(|s| s.as_str()), Some("actor"));
/// assert!(out.get("id").is_none());
/// assert!(out.get("name").is_none());
/// ```
fn collect_graph_node_metadata(
    metadata: &BTreeMap<String, String>,
    prefix: &str,
) -> BTreeMap<String, String> {
    metadata
        .iter()
        .filter_map(|(key, value)| {
            key.strip_prefix(prefix).and_then(|stripped| {
                if matches!(stripped, "id" | "name" | "type") {
                    None
                } else {
                    Some((stripped.to_string(), value.clone()))
                }
            })
        })
        .collect()
}

/// Normalizes a search query by trimming and replacing any sequence of whitespace with a single space.
///
/// This produces a compact query suitable for lexical matching or embedding.
///
/// # Examples
///
/// ```
/// let q = "  quick   brown\tfox\njumps  ";
/// assert_eq!(normalize_search_query(q), "quick brown fox jumps");
/// ```
fn normalize_search_query(query: &str) -> String {
    query
        .split_whitespace()
        .filter(|token| !token.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
}

fn split_into_chunks(text: &str) -> Vec<String> {
    let normalized = text.trim().replace("\r\n", "\n");
    if normalized.is_empty() {
        return Vec::new();
    }

    let paragraphs: Vec<&str> = normalized
        .split("\n\n")
        .map(|part| part.trim())
        .filter(|part| !part.is_empty())
        .collect();

    if paragraphs.is_empty() {
        return vec![normalized];
    }

    let mut chunks = Vec::new();
    for paragraph in paragraphs {
        let chars: Vec<char> = paragraph.chars().collect();
        if chars.len() <= 900 {
            chunks.push(paragraph.to_string());
        } else {
            let mut start = 0;
            while start < chars.len() {
                let end = (start + 900).min(chars.len());
                chunks.push(chars[start..end].iter().collect());
                if end == chars.len() {
                    break;
                }
                start = end.saturating_sub(64);
            }
        }
    }
    chunks
}

fn hash_text(text: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(text.as_bytes());
    hex::encode(hasher.finalize())
}

fn chunk_id(resource_id: &str, layer: ResourceLayer, text: &str) -> String {
    hash_text(&format!("{resource_id}|{}|{text}", layer.as_str()))
}

fn lexical_score(score: f32) -> f32 {
    1.0 / (1.0 + score.abs())
}

/// Escapes single quotes for embedding a string in an SQL string literal by doubling them.
///
/// Converts each `'` to `''`.
///
/// # Examples
///
/// ```
/// let s = "O'Reilly";
/// assert_eq!(escape_sql(s), "O''Reilly");
/// ```
fn escape_sql(value: &str) -> String {
    value.replace('\'', "''")
}

/// Creates a deterministic edge identifier from subject, relation, and object.
///
/// The identifier is the hex-encoded SHA-256 hash of the string
/// "{subject_id}|{relation}|{object_id}".
///
/// # Examples
///
/// ```
/// let a = graph_edge_id("node:alice", "FRIENDS_WITH", "node:bob");
/// let b = graph_edge_id("node:alice", "FRIENDS_WITH", "node:bob");
/// assert_eq!(a, b);
/// let c = graph_edge_id("node:bob", "FRIENDS_WITH", "node:alice");
/// assert_ne!(a, c);
/// ```
fn graph_edge_id(subject_id: &str, relation: &str, object_id: &str) -> String {
    hash_text(&format!("{subject_id}|{relation}|{object_id}"))
}

/// Constructs the Viking URI string for a session's working-memory record.
///
/// # Errors
///
/// Returns a `ContextError` if the underlying `VikingUri::user_memory` fails to construct a valid URI.
///
/// # Examples
///
/// ```
/// let uri = working_memory_cache_uri("session-123").expect("should build URI");
/// assert!(uri.contains("working-memory/session-123"));
/// ```
fn working_memory_cache_uri(session_id: &str) -> Result<String, ContextError> {
    VikingUri::user_memory(format!("working-memory/{session_id}"))
        .map(|uri| uri.to_string())
        .map_err(ContextError::from)
}

/// Produces a deterministic fixed-size embedding vector for the given text.
///
/// The resulting vector is L2-normalized and deterministic for the same input text and
/// embedding dimension. Token-level contributions are aggregated into a fixed-size
/// float vector so that similar texts produce related embeddings.
///
/// # Examples
///
/// ```
/// let v = embed_text("hello world");
/// assert_eq!(v.len(), EMBEDDING_DIM);
/// let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
/// assert!((norm - 1.0).abs() < 1e-5);
/// ```
fn embed_text(text: &str) -> Vec<f32> {
    let mut vector = vec![0.0f32; EMBEDDING_DIM];
    for token in tokenize(text) {
        let hash = hash_token(&token);
        let index = (hash as usize) % EMBEDDING_DIM;
        vector[index] += 1.0;
        vector[(index + 1) % EMBEDDING_DIM] += 0.5;
        vector[(index + 31) % EMBEDDING_DIM] += 0.25;
    }

    let norm = vector.iter().map(|value| value * value).sum::<f32>().sqrt();
    if norm > 0.0 {
        for value in &mut vector {
            *value /= norm;
        }
    }
    vector
}

fn tokenize(text: &str) -> Vec<String> {
    text.split(|character: char| !character.is_alphanumeric())
        .map(|token| token.to_lowercase())
        .filter(|token| !token.is_empty())
        .collect()
}

/// Produces a stable 64-bit hash for a token using SHA-256.
///
/// The returned value is the first 8 bytes of the token's SHA-256 digest interpreted as a
/// little-endian `u64`.
///
/// # Examples
///
/// ```
/// let a = hash_token("example");
/// let b = hash_token("example");
/// assert_eq!(a, b);
/// ```
fn hash_token(token: &str) -> u64 {
    let mut hasher = Sha256::new();
    hasher.update(token.as_bytes());
    let digest = hasher.finalize();
    u64::from_le_bytes(digest[0..8].try_into().unwrap())
}

#[derive(Debug, Clone)]
struct GraphEdgeRow {
    edge_id: String,
    subject_id: String,
    relation: String,
    object_id: String,
    metadata_json: String,
    resource_uri: Option<String>,
    session_id: Option<String>,
    updated_at: i64,
}
