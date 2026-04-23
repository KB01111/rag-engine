use std::collections::BTreeMap;
use std::net::SocketAddr;
use std::path::PathBuf;

use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use axum::routing::{get, post};
use axum::Json;
use axum::Router;
use serde::{Deserialize, Serialize};

use context_engine::OpenVikingBridgeConfig;
use context_engine::{
    ContextConfig, ContextEngine, ContextError, FileEntry, ManagedRoot, ResourceLayer,
    ResourceSummary, ResourceUpsertRequest, SearchHit, SearchRequest, SessionEntry,
    SessionEventRequest, StatusResponse, UpsertOutcome, WorkspaceSyncOutcome,
};

#[derive(Clone)]
pub struct AppState {
    pub engine: ContextEngine,
}

#[derive(Debug, Clone, Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub ready: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ResourcesResponse {
    pub resources: Vec<ResourceSummary>,
}

#[derive(Debug, Clone, Serialize)]
pub struct UpsertResourceResponse {
    pub resource: ResourceSummary,
    pub chunks_indexed: i32,
}

#[derive(Debug, Clone, Serialize)]
pub struct SearchResponse {
    pub results: Vec<SearchHit>,
    pub query_time_ms: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct FileReadResponse {
    pub path: String,
    pub content: String,
    pub version: i64,
}

#[derive(Debug, Clone, Serialize)]
pub struct FileWriteResponse {
    pub path: String,
    pub version: i64,
}

#[derive(Debug, Clone, Serialize)]
pub struct FileMoveResponse {
    pub from_path: String,
    pub to_path: String,
    pub version: i64,
}

#[derive(Debug, Clone, Serialize)]
pub struct FileDeleteResponse {
    pub path: String,
    pub deleted: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct FilesListResponse {
    pub entries: Vec<FileEntry>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SessionHistoryResponse {
    pub session_id: String,
    pub entries: Vec<SessionEntry>,
}

#[derive(Debug, Clone, Serialize)]
pub struct WorkspaceSyncResponse {
    pub root: String,
    pub prefix: Option<String>,
    pub indexed_resources: i32,
    pub reindexed_resources: i32,
    pub deleted_resources: i32,
    pub skipped_files: i32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct HttpResourceUpsertRequest {
    pub uri: String,
    pub title: Option<String>,
    pub content: String,
    pub layer: ResourceLayer,
    #[serde(default)]
    pub metadata: BTreeMap<String, String>,
    #[serde(default)]
    pub previous_uri: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct HttpSearchRequest {
    pub query: String,
    pub scope_uri: Option<String>,
    pub top_k: Option<usize>,
    pub filters: Option<BTreeMap<String, String>>,
    pub layer: Option<ResourceLayer>,
    pub rerank: Option<bool>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct HttpFileRequest {
    pub root: String,
    #[serde(default)]
    pub path: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct HttpFileWriteRequest {
    pub root: String,
    pub path: String,
    pub content: String,
    pub version: Option<i64>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct HttpFileDeleteRequest {
    pub root: String,
    pub path: String,
    pub version: Option<i64>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct HttpFileMoveRequest {
    pub root: String,
    pub from_path: String,
    pub to_path: String,
    pub version: Option<i64>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct HttpWorkspaceSyncRequest {
    pub root: String,
    #[serde(default)]
    pub path: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct HttpSessionAppendRequest {
    pub session_id: String,
    pub role: String,
    pub content: String,
    #[serde(default)]
    pub metadata: BTreeMap<String, String>,
}

pub fn router(engine: ContextEngine) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/v1/status", get(status))
        .route(
            "/v1/resources",
            get(list_resources)
                .post(upsert_resource)
                .delete(delete_resource),
        )
        .route("/v1/search", post(search_context))
        .route("/v1/workspaces/sync", post(sync_workspace))
        .route("/v1/files/list", post(list_files))
        .route("/v1/files/read", post(read_file))
        .route("/v1/files/write", post(write_file))
        .route("/v1/files/delete", post(delete_file))
        .route("/v1/files/move", post(move_file))
        .route("/v1/sessions/append", post(append_session))
        .route("/v1/sessions/:id", get(get_session))
        .with_state(AppState { engine })
}

pub async fn serve(addr: SocketAddr, engine: ContextEngine) -> anyhow::Result<()> {
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, router(engine)).await?;
    Ok(())
}

async fn health(State(state): State<AppState>) -> Json<HealthResponse> {
    let ready = state.engine.status().await.is_ok();
    Json(HealthResponse {
        status: if ready {
            "ok".to_string()
        } else {
            "degraded".to_string()
        },
        ready,
        message: if ready {
            None
        } else {
            Some("engine not ready".to_string())
        },
    })
}

async fn status(State(state): State<AppState>) -> Result<Json<StatusResponse>, ContextError> {
    Ok(Json(state.engine.status().await?))
}

async fn list_resources(
    State(state): State<AppState>,
) -> Result<Json<ResourcesResponse>, ContextError> {
    Ok(Json(ResourcesResponse {
        resources: state.engine.list_resources().await?,
    }))
}

async fn upsert_resource(
    State(state): State<AppState>,
    Json(request): Json<HttpResourceUpsertRequest>,
) -> Result<Json<UpsertResourceResponse>, ContextError> {
    let outcome: UpsertOutcome = state
        .engine
        .upsert_resource(ResourceUpsertRequest {
            uri: request.uri,
            title: request.title,
            content: request.content,
            layer: request.layer,
            metadata: request.metadata,
            previous_uri: request.previous_uri,
        })
        .await?;

    Ok(Json(UpsertResourceResponse {
        resource: outcome.resource,
        chunks_indexed: outcome.chunks_indexed,
    }))
}

async fn delete_resource(
    State(state): State<AppState>,
    Query(query): Query<BTreeMap<String, String>>,
) -> Result<StatusCode, ContextError> {
    let uri = query
        .get("uri")
        .cloned()
        .ok_or_else(|| ContextError::InvalidRequest("missing uri".to_string()))?;
    let _ = state.engine.delete_resource(&uri).await?;
    Ok(StatusCode::NO_CONTENT)
}

async fn search_context(
    State(state): State<AppState>,
    Json(request): Json<HttpSearchRequest>,
) -> Result<Json<SearchResponse>, ContextError> {
    let start = std::time::Instant::now();
    let results = state
        .engine
        .search_context(SearchRequest {
            query: request.query,
            scope_uri: request.scope_uri,
            top_k: request.top_k,
            filters: request.filters,
            layer: request.layer,
            rerank: request.rerank,
        })
        .await?;

    Ok(Json(SearchResponse {
        results,
        query_time_ms: start.elapsed().as_secs_f64() * 1000.0,
    }))
}

async fn sync_workspace(
    State(state): State<AppState>,
    Json(request): Json<HttpWorkspaceSyncRequest>,
) -> Result<Json<WorkspaceSyncResponse>, ContextError> {
    let outcome: WorkspaceSyncOutcome = state
        .engine
        .sync_workspace(&request.root, request.path.map(PathBuf::from))
        .await?;
    Ok(Json(WorkspaceSyncResponse {
        root: outcome.root,
        prefix: outcome.prefix,
        indexed_resources: outcome.indexed_resources,
        reindexed_resources: outcome.reindexed_resources,
        deleted_resources: outcome.deleted_resources,
        skipped_files: outcome.skipped_files,
    }))
}

async fn list_files(
    State(state): State<AppState>,
    Json(request): Json<HttpFileRequest>,
) -> Result<Json<FilesListResponse>, ContextError> {
    Ok(Json(FilesListResponse {
        entries: state
            .engine
            .list_files(&request.root, PathBuf::from(request.path))
            .await?,
    }))
}

async fn read_file(
    State(state): State<AppState>,
    Json(request): Json<HttpFileRequest>,
) -> Result<Json<FileReadResponse>, ContextError> {
    let path = PathBuf::from(&request.path);
    let content = state.engine.read_file(&request.root, path.clone()).await?;
    let version = state
        .engine
        .file_version(&request.root, path)
        .await
        .unwrap_or(0);
    Ok(Json(FileReadResponse {
        path: request.path,
        content,
        version,
    }))
}

async fn write_file(
    State(state): State<AppState>,
    Json(request): Json<HttpFileWriteRequest>,
) -> Result<Json<FileWriteResponse>, ContextError> {
    let version = state
        .engine
        .write_file(
            &request.root,
            PathBuf::from(&request.path),
            &request.content,
            request.version,
        )
        .await?;
    Ok(Json(FileWriteResponse {
        path: request.path,
        version,
    }))
}

async fn delete_file(
    State(state): State<AppState>,
    Json(request): Json<HttpFileDeleteRequest>,
) -> Result<Json<FileDeleteResponse>, ContextError> {
    let deleted = state
        .engine
        .delete_file(&request.root, PathBuf::from(&request.path), request.version)
        .await?;
    Ok(Json(FileDeleteResponse {
        path: request.path,
        deleted,
    }))
}

async fn move_file(
    State(state): State<AppState>,
    Json(request): Json<HttpFileMoveRequest>,
) -> Result<Json<FileMoveResponse>, ContextError> {
    let version = state
        .engine
        .move_file(
            &request.root,
            PathBuf::from(&request.from_path),
            PathBuf::from(&request.to_path),
            request.version,
        )
        .await?;
    Ok(Json(FileMoveResponse {
        from_path: request.from_path,
        to_path: request.to_path,
        version,
    }))
}

async fn append_session(
    State(state): State<AppState>,
    Json(request): Json<HttpSessionAppendRequest>,
) -> Result<Json<SessionHistoryResponse>, ContextError> {
    let session_id = request.session_id.clone();
    state
        .engine
        .append_session(SessionEventRequest {
            session_id: session_id.clone(),
            role: request.role,
            content: request.content,
            metadata: request.metadata,
        })
        .await?;
    let entries = state.engine.list_sessions(&session_id).await?;
    Ok(Json(SessionHistoryResponse {
        session_id,
        entries,
    }))
}

async fn get_session(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<SessionHistoryResponse>, ContextError> {
    Ok(Json(SessionHistoryResponse {
        session_id: id.clone(),
        entries: state.engine.list_sessions(&id).await?,
    }))
}

pub async fn engine_from_env() -> anyhow::Result<ContextEngine> {
    let data_dir =
        std::env::var("CONTEXT_DATA_DIR").unwrap_or_else(|_| "./context-data".to_string());
    let roots_env = std::env::var("CONTEXT_ROOTS").unwrap_or_else(|_| "workspace=.".to_string());
    let roots = parse_roots(&roots_env)?;
    let bridge = std::env::var("CONTEXT_OPENVIKING_URL")
        .ok()
        .map(|base_url| {
            let mut bridge = OpenVikingBridgeConfig::new(base_url);
            bridge.token = std::env::var("CONTEXT_OPENVIKING_API_KEY").ok();
            if let Ok(import_path) = std::env::var("CONTEXT_OPENVIKING_IMPORT_PATH") {
                bridge.import_path = import_path;
            }
            if let Ok(sync_path) = std::env::var("CONTEXT_OPENVIKING_SYNC_PATH") {
                bridge.sync_path = sync_path;
            }
            if let Ok(find_path) = std::env::var("CONTEXT_OPENVIKING_FIND_PATH") {
                bridge.find_path = find_path;
            }
            if let Ok(read_path) = std::env::var("CONTEXT_OPENVIKING_READ_PATH") {
                bridge.read_path = read_path;
            }
            bridge
        });
    let config = ContextConfig {
        data_dir: PathBuf::from(data_dir),
        roots,
        bridge,
    };
    Ok(ContextEngine::open(config).await?)
}

fn parse_roots(value: &str) -> anyhow::Result<Vec<ManagedRoot>> {
    let mut roots = Vec::new();
    for entry in value.split(';').filter(|part| !part.trim().is_empty()) {
        if let Some((name, path)) = entry.split_once('=') {
            roots.push(ManagedRoot::new(name.trim(), PathBuf::from(path.trim()))?);
            continue;
        }

        let default_name = if roots.is_empty() {
            "workspace".to_string()
        } else {
            format!("workspace{}", roots.len() + 1)
        };
        roots.push(ManagedRoot::new(default_name, PathBuf::from(entry.trim()))?);
    }
    Ok(roots)
}
