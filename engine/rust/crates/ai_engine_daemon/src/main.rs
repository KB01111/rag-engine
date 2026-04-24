use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::Arc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use chunking::ChunkingConfig;
use context_engine::{
    ContextConfig, ContextEngine, DragonflyConfig, GraphFactRecord, ManagedRoot,
    OpenVikingBridgeConfig, ResourceLayer, ResourceUpsertRequest,
    SearchRequest as ContextSearchRequestModel, SessionEventRequest,
};
use embedding::{create_default_engine, EmbeddingEngine};
use prost_types::{value, Struct, Timestamp, Value};
use runtime_engine::RuntimeEngine;
use serde::{Deserialize, Serialize};
use storage::{
    ChunkSearchQuery, DocumentRecord, EngineStore, MCPConnectionRecord, ModelRecord, SearchMode,
    VectorRecord,
};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tonic::{transport::Server, Request, Response, Status};
use tonic_health::server::health_reporter;
use training_engine::TrainingEngine;
use uuid::Uuid;

pub mod engine {
    tonic::include_proto!("engine");
}

use engine::context_server::{Context as ContextRpc, ContextServer};
use engine::mcp_server::{Mcp, McpServer};
use engine::rag_server::{Rag, RagServer};
use engine::runtime_server::{Runtime, RuntimeServer};
use engine::training_server::{Training, TrainingServer};
use engine::{
    Artifact, ArtifactList, CallToolRequest, CallToolResponse, CancelRequest, DeleteRequest,
    DisconnectRequest, DocumentInfo, DocumentList, InferenceRequest, InferenceResponse,
    LoadModelRequest, LogsRequest, McpConnection, McpConnectionRequest, ModelInfo, ModelList,
    RagStatus, RuntimeStatus, SearchRequest, SearchResponse, SearchResult, SystemResources, Tool,
    ToolList, ToolParameter, TrainingLog, TrainingRun, TrainingRunList, TrainingRunRequest,
    UpsertRequest, UpsertResponse,
};

const MAX_TOP_K: i32 = 1000;
const MAX_CONTEXT_TOP_K: usize = 100;

type InferenceStream =
    Pin<Box<dyn tokio_stream::Stream<Item = Result<InferenceResponse, Status>> + Send>>;
type TrainingLogStream =
    Pin<Box<dyn tokio_stream::Stream<Item = Result<TrainingLog, Status>> + Send>>;

#[derive(Clone)]
struct AppState {
    store: EngineStore,
    runtime: RuntimeEngine,
    training: TrainingEngine,
    embedding: Arc<EmbeddingEngine>,
    embedding_provider_name: String,
    chunking: ChunkingConfig,
}

#[derive(Clone)]
struct EngineService {
    state: AppState,
}

#[derive(Clone)]
struct ContextGrpcService {
    engine: ContextEngine,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct StoredTool {
    name: String,
    description: String,
    #[serde(default)]
    parameters: Vec<StoredToolParameter>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct StoredToolParameter {
    name: String,
    #[serde(rename = "type")]
    parameter_type: String,
    required: bool,
    description: String,
}

#[derive(Clone)]
struct ActiveConnection {
    record: MCPConnectionRecord,
    tools: Vec<Tool>,
}

/// Starts the AI Engine daemon gRPC server configured from environment variables.
///
/// Reads runtime configuration from environment (addresses, storage/model paths, training dir,
/// and context engine settings), constructs shared application state and services, registers
/// health reporting, and serves the Runtime, Rag, Context, Training, and MCP gRPC services
/// until shutdown.
///
/// Returns `Ok(())` if the server shuts down cleanly, or an error with context if startup or
/// runtime initialization fails.
///
/// # Examples
///
/// ```ignore
/// // Start the daemon (example marked `ignore` to avoid running the server during tests).
/// // In a real deployment this is invoked as the program entry point.
/// crate::main().await.unwrap();
/// ```
#[tokio::main]
async fn main() -> Result<()> {
    let addr =
        std::env::var("AI_ENGINE_DAEMON_ADDR").unwrap_or_else(|_| "127.0.0.1:50061".to_string());
    let lancedb_uri =
        std::env::var("AI_ENGINE_LANCEDB_URI").unwrap_or_else(|_| ".ai-engine/lancedb".to_string());
    let models_path =
        std::env::var("AI_ENGINE_MODELS_PATH").unwrap_or_else(|_| ".ai-engine/models".to_string());
    let llama_cli =
        std::env::var("AI_ENGINE_LLAMA_CLI").unwrap_or_else(|_| "llama-cli".to_string());
    let training_dir = std::env::var("AI_ENGINE_TRAINING_DIR")
        .unwrap_or_else(|_| ".ai-engine/training".to_string());
    let backend = "llama.cpp".to_string();

    let store = EngineStore::new(lancedb_uri);
    let embedding_engine = Arc::new(create_default_engine());
    let embedding_provider_name = embedding_engine.name().to_string();
    let context_service = ContextGrpcService {
        engine: open_context_engine_from_env().await?,
    };
    let service = EngineService {
        state: AppState {
            store: store.clone(),
            runtime: RuntimeEngine::new(store.clone(), models_path, backend.clone(), llama_cli),
            training: TrainingEngine::new(store.clone(), training_dir, backend),
            embedding: embedding_engine,
            embedding_provider_name,
            chunking: ChunkingConfig::default(),
        },
    };

    let (health_reporter, health_service) = health_reporter();
    health_reporter
        .set_serving::<RuntimeServer<EngineService>>()
        .await;
    health_reporter
        .set_serving::<RagServer<EngineService>>()
        .await;
    health_reporter
        .set_serving::<ContextServer<ContextGrpcService>>()
        .await;
    health_reporter
        .set_serving::<TrainingServer<EngineService>>()
        .await;
    health_reporter
        .set_serving::<McpServer<EngineService>>()
        .await;

    Server::builder()
        .add_service(health_service)
        .add_service(RuntimeServer::new(service.clone()))
        .add_service(RagServer::new(service.clone()))
        .add_service(ContextServer::new(context_service))
        .add_service(TrainingServer::new(service.clone()))
        .add_service(McpServer::new(service))
        .serve(addr.parse()?)
        .await
        .context("serve daemon")?;

    Ok(())
}

#[tonic::async_trait]
impl Runtime for EngineService {
    type StreamInferenceStream = InferenceStream;

    async fn get_status(&self, _request: Request<()>) -> Result<Response<RuntimeStatus>, Status> {
        let models = self
            .state
            .runtime
            .list_models()
            .await
            .map_err(internal_status)?;
        let resources = self.state.runtime.system_resources();
        let loaded = models
            .into_iter()
            .filter(|model| model.status == "loaded")
            .map(model_info)
            .collect::<Vec<_>>();

        Ok(Response::new(RuntimeStatus {
            version: "1.0.0".to_string(),
            loaded_models: loaded,
            resources: Some(SystemResources {
                cpu_percent: resources.cpu_percent as f64,
                memory_used_bytes: resources.memory_used_bytes,
                memory_total_bytes: resources.memory_total_bytes,
            }),
            healthy: true,
        }))
    }

    async fn list_models(&self, _request: Request<()>) -> Result<Response<ModelList>, Status> {
        let models = self
            .state
            .runtime
            .list_models()
            .await
            .map_err(internal_status)?
            .into_iter()
            .map(model_info)
            .collect();
        Ok(Response::new(ModelList { models }))
    }

    async fn load_model(
        &self,
        request: Request<LoadModelRequest>,
    ) -> Result<Response<ModelInfo>, Status> {
        let info = self
            .state
            .runtime
            .load_model(&request.into_inner().model_id)
            .await
            .map_err(load_model_status)?;
        Ok(Response::new(model_info(info)))
    }

    async fn unload_model(
        &self,
        request: Request<engine::UnloadModelRequest>,
    ) -> Result<Response<()>, Status> {
        self.state
            .runtime
            .unload_model(&request.into_inner().model_id)
            .await
            .map_err(load_model_status)?;
        Ok(Response::new(()))
    }

    async fn stream_inference(
        &self,
        request: Request<tonic::Streaming<InferenceRequest>>,
    ) -> Result<Response<Self::StreamInferenceStream>, Status> {
        let mut stream = request.into_inner();
        let first = stream
            .message()
            .await?
            .ok_or_else(|| Status::invalid_argument("missing inference request"))?;
        if first.model_id.trim().is_empty() {
            return Err(Status::invalid_argument("model_id is required"));
        }

        let chunks = self
            .state
            .runtime
            .stream_inference(&first.model_id, &first.prompt, &first.parameters)
            .await
            .map_err(inference_status)?;

        Ok(Response::new(Box::pin(bridge_inference_chunks(chunks))))
    }
}

#[tonic::async_trait]
impl Rag for EngineService {
    async fn upsert_document(
        &self,
        request: Request<UpsertRequest>,
    ) -> Result<Response<UpsertResponse>, Status> {
        let request = request.into_inner();
        if request.content.trim().is_empty() {
            return Err(Status::invalid_argument("content is required"));
        }
        let document_id = if request.document_id.is_empty() {
            Uuid::new_v4().to_string()
        } else {
            request.document_id
        };
        let title = request
            .metadata
            .get("title")
            .cloned()
            .unwrap_or_else(|| document_id.clone());
        let now = now();

        let chunks = chunking::chunk_text(&request.content, &self.state.chunking)
            .map_err(internal_status)?;
        let chunk_texts = chunks
            .iter()
            .map(|chunk| chunk.text.clone())
            .collect::<Vec<_>>();
        let embeddings = self
            .state
            .embedding
            .embed(&chunk_texts)
            .map_err(internal_status)?;

        // Verify embeddings and chunks have matching lengths
        if embeddings.len() != chunks.len() {
            return Err(Status::internal(format!(
                "embedding count mismatch for document {}: got {} embeddings for {} chunks",
                document_id,
                embeddings.len(),
                chunks.len()
            )));
        }

        let vector_records = chunks
            .iter()
            .enumerate()
            .map(|(index, chunk)| VectorRecord {
                id: format!("{document_id}-chunk-{index}"),
                document_id: document_id.clone(),
                chunk_text: chunk.text.clone(),
                vector: embeddings[index].vector.clone(),
                metadata: request.metadata.clone(),
                created_at: now,
            })
            .collect::<Vec<_>>();

        self.state
            .store
            .upsert_document(
                DocumentRecord {
                    id: document_id.clone(),
                    title,
                    content: request.content,
                    metadata: request.metadata,
                    created_at: now,
                    updated_at: now,
                },
                vector_records,
            )
            .await
            .map_err(internal_status)?;

        Ok(Response::new(UpsertResponse {
            document_id,
            chunks_indexed: chunks.len() as i32,
        }))
    }

    async fn delete_document(
        &self,
        request: Request<DeleteRequest>,
    ) -> Result<Response<()>, Status> {
        self.state
            .store
            .delete_document(&request.into_inner().document_id)
            .await
            .map_err(internal_status)?;
        Ok(Response::new(()))
    }

    async fn search(
        &self,
        request: Request<SearchRequest>,
    ) -> Result<Response<SearchResponse>, Status> {
        let request = request.into_inner();
        if request.query.trim().is_empty() {
            return Err(Status::invalid_argument("query is required"));
        }
        let started = Instant::now();
        let query_embedding = self
            .state
            .embedding
            .embed_single(&request.query)
            .map_err(internal_status)?;
        let top_k = if request.top_k <= 0 {
            10
        } else {
            std::cmp::min(request.top_k, MAX_TOP_K) as usize
        };

        let results = self
            .state
            .store
            .search_chunks(ChunkSearchQuery {
                vector: query_embedding.vector,
                text_query: Some(request.query),
                top_k,
                filters: request.filters,
                mode: SearchMode::Hybrid,
            })
            .await
            .map_err(internal_status)?;

        Ok(Response::new(SearchResponse {
            results: results
                .into_iter()
                .map(|result| SearchResult {
                    document_id: result.record.document_id,
                    chunk_text: result.record.chunk_text,
                    score: result.score,
                    metadata: result.record.metadata,
                })
                .collect(),
            query_time_ms: started.elapsed().as_secs_f64() * 1000.0,
        }))
    }

    async fn get_rag_status(&self, _request: Request<()>) -> Result<Response<RagStatus>, Status> {
        let documents = self
            .state
            .store
            .list_documents()
            .await
            .map_err(internal_status)?;
        let chunks = self
            .state
            .store
            .list_chunks()
            .await
            .map_err(internal_status)?;
        let index_size_bytes = chunks
            .iter()
            .map(|chunk| (chunk.vector.len() * std::mem::size_of::<f32>()) as i64)
            .sum();

        Ok(Response::new(RagStatus {
            document_count: documents.len() as i64,
            chunk_count: chunks.len() as i64,
            index_size_bytes,
            embedding_model: self.state.embedding_provider_name.clone(),
        }))
    }

    async fn list_documents(
        &self,
        _request: Request<()>,
    ) -> Result<Response<DocumentList>, Status> {
        let documents = self
            .state
            .store
            .list_documents()
            .await
            .map_err(internal_status)?;
        let chunks = self
            .state
            .store
            .list_chunks()
            .await
            .map_err(internal_status)?;

        let infos = documents
            .into_iter()
            .map(|document| {
                let chunk_count = chunks
                    .iter()
                    .filter(|chunk| chunk.document_id == document.id)
                    .count() as i64;
                DocumentInfo {
                    id: document.id,
                    title: document.title,
                    chunk_count,
                    created_at: Some(timestamp(document.created_at)),
                    updated_at: Some(timestamp(document.updated_at)),
                }
            })
            .collect();
        Ok(Response::new(DocumentList { documents: infos }))
    }
}

#[tonic::async_trait]
impl Training for EngineService {
    type StreamLogsStream = TrainingLogStream;

    async fn start_run(
        &self,
        request: Request<TrainingRunRequest>,
    ) -> Result<Response<TrainingRun>, Status> {
        let request = request.into_inner();
        let config_json = serde_json::to_string(&request.config).map_err(internal_status)?;
        let run = self
            .state
            .training
            .start_run(
                &request.name,
                &request.model_id,
                &request.dataset_path,
                &config_json,
            )
            .await
            .map_err(internal_status)?;
        Ok(Response::new(training_run(run)))
    }

    async fn cancel_run(&self, request: Request<CancelRequest>) -> Result<Response<()>, Status> {
        self.state
            .training
            .cancel_run(&request.into_inner().run_id)
            .await
            .map_err(not_found_status)?;
        Ok(Response::new(()))
    }

    async fn list_runs(&self, _request: Request<()>) -> Result<Response<TrainingRunList>, Status> {
        let runs = self
            .state
            .training
            .list_runs()
            .await
            .map_err(internal_status)?
            .into_iter()
            .map(training_run)
            .collect();
        Ok(Response::new(TrainingRunList { runs }))
    }

    async fn list_artifacts(
        &self,
        request: Request<engine::ArtifactsRequest>,
    ) -> Result<Response<ArtifactList>, Status> {
        let artifacts = self
            .state
            .training
            .list_artifacts(&request.into_inner().run_id)
            .await
            .map_err(not_found_status)?;

        let mut items = Vec::new();
        for path in artifacts {
            let metadata = tokio::fs::metadata(&path).await.ok();
            if let Some(meta) = metadata {
                let created_timestamp = meta
                    .created()
                    .or_else(|_| meta.modified())
                    .ok()
                    .and_then(|time| {
                        time.duration_since(std::time::UNIX_EPOCH)
                            .ok()
                            .map(|d| d.as_secs() as i64)
                    })
                    .unwrap_or_else(now);

                if let Some(name) = path.file_name() {
                    items.push(Artifact {
                        name: name.to_string_lossy().to_string(),
                        path: path.to_string_lossy().to_string(),
                        size_bytes: meta.len() as i64,
                        created_at: Some(timestamp(created_timestamp)),
                    });
                }
            }
        }
        Ok(Response::new(ArtifactList { artifacts: items }))
    }

    async fn stream_logs(
        &self,
        request: Request<LogsRequest>,
    ) -> Result<Response<Self::StreamLogsStream>, Status> {
        let request = request.into_inner();
        let training = self.state.training.clone();

        let stream = async_stream::try_stream! {
            let mut sent = 0usize;
            loop {
                let logs = training.list_logs(&request.run_id).await.map_err(internal_status)?;
                while sent < logs.len() {
                    let entry = &logs[sent];
                    sent += 1;
                    yield TrainingLog {
                        run_id: entry.run_id.clone(),
                        level: entry.level.clone(),
                        message: entry.message.clone(),
                        timestamp: Some(timestamp(entry.timestamp)),
                        fields: HashMap::new(),
                    };
                }

                if !request.follow {
                    break;
                }

                let runs = training.list_runs().await.map_err(internal_status)?;
                let done = runs
                    .iter()
                    .find(|run| run.id == request.run_id)
                    .map(|run| matches!(run.status.as_str(), "completed" | "failed" | "cancelled"))
                    .unwrap_or(true);
                if done {
                    break;
                }
                tokio::time::sleep(tokio::time::Duration::from_millis(250)).await;
            }
        };

        Ok(Response::new(Box::pin(stream)))
    }
}

#[tonic::async_trait]
impl Mcp for EngineService {
    async fn connect(
        &self,
        request: Request<McpConnectionRequest>,
    ) -> Result<Response<McpConnection>, Status> {
        let request = request.into_inner();
        let connection_id = format!("conn-{}", stable_id(&request.server_url));
        let auth_json = serde_json::to_string(&request.auth).map_err(internal_status)?;
        let server_name = server_name_from_url(&request.server_url);
        let tools = builtin_tools();
        let tools_json = serde_json::to_string(&tools.iter().map(stored_tool).collect::<Vec<_>>())
            .map_err(internal_status)?;

        self.state
            .store
            .upsert_mcp_connection(MCPConnectionRecord {
                connection_id: connection_id.clone(),
                server_url: request.server_url.clone(),
                server_name: server_name.clone(),
                connected: true,
                auth_json,
                tools_json,
                updated_at: now(),
            })
            .await
            .map_err(internal_status)?;

        Ok(Response::new(McpConnection {
            connection_id,
            connected: true,
            server_name,
        }))
    }

    async fn disconnect(
        &self,
        request: Request<DisconnectRequest>,
    ) -> Result<Response<()>, Status> {
        self.state
            .store
            .delete_mcp_connection(&request.into_inner().connection_id)
            .await
            .map_err(internal_status)?;
        Ok(Response::new(()))
    }

    async fn list_tools(
        &self,
        request: Request<McpConnectionRequest>,
    ) -> Result<Response<ToolList>, Status> {
        let request = request.into_inner();
        let Some(connection) = self
            .find_connection_by_url(&request.server_url)
            .await
            .map_err(internal_status)?
        else {
            return Err(Status::not_found("connection not found"));
        };

        Ok(Response::new(ToolList {
            tools: connection.tools,
        }))
    }

    async fn call_tool(
        &self,
        request: Request<CallToolRequest>,
    ) -> Result<Response<CallToolResponse>, Status> {
        let request = request.into_inner();
        let Some(connection) = self
            .find_connection_by_id(&request.connection_id)
            .await
            .map_err(internal_status)?
        else {
            return Ok(Response::new(CallToolResponse {
                success: false,
                result: None,
                error: format!("connection not found: {}", request.connection_id),
            }));
        };

        if !connection
            .tools
            .iter()
            .any(|tool| tool.name == request.tool_name)
        {
            return Ok(Response::new(CallToolResponse {
                success: false,
                result: None,
                error: format!("tool not found: {}", request.tool_name),
            }));
        }

        let result = match request.tool_name.as_str() {
            "mcp.describe_connection" => Struct {
                fields: BTreeMap::from([
                    (
                        "connection_id".to_string(),
                        string_value(&connection.record.connection_id),
                    ),
                    (
                        "server_url".to_string(),
                        string_value(&connection.record.server_url),
                    ),
                    (
                        "server_name".to_string(),
                        string_value(&connection.record.server_name),
                    ),
                    ("status".to_string(), string_value("connected")),
                ]),
            },
            "mcp.echo" => {
                let mut fields = BTreeMap::from([
                    (
                        "connection_id".to_string(),
                        string_value(&connection.record.connection_id),
                    ),
                    ("tool".to_string(), string_value(&request.tool_name)),
                ]);
                if let Some(arguments) = request.arguments {
                    fields.insert(
                        "arguments".to_string(),
                        Value {
                            kind: Some(value::Kind::StructValue(arguments)),
                        },
                    );
                }
                Struct { fields }
            }
            _ => {
                return Ok(Response::new(CallToolResponse {
                    success: false,
                    result: None,
                    error: format!("tool {} is staged and not executable", request.tool_name),
                }));
            }
        };

        Ok(Response::new(CallToolResponse {
            success: true,
            result: Some(result),
            error: String::new(),
        }))
    }
}

impl EngineService {
    async fn find_connection_by_url(&self, server_url: &str) -> Result<Option<ActiveConnection>> {
        let connection = self
            .state
            .store
            .list_mcp_connections()
            .await?
            .into_iter()
            .find(|connection| connection.server_url == server_url);
        Ok(connection.map(to_active_connection))
    }

    async fn find_connection_by_id(&self, connection_id: &str) -> Result<Option<ActiveConnection>> {
        let connection = self
            .state
            .store
            .list_mcp_connections()
            .await?
            .into_iter()
            .find(|connection| connection.connection_id == connection_id);
        Ok(connection.map(to_active_connection))
    }
}

fn model_info(model: ModelRecord) -> ModelInfo {
    ModelInfo {
        id: model.id,
        name: model.name,
        path: model.path,
        size_bytes: model.size_bytes,
        loaded: model.status == "loaded",
        metadata: HashMap::from([
            ("backend".to_string(), model.backend),
            ("status".to_string(), model.status),
        ]),
    }
}

/// Convert a storage-layer `TrainingRunRecord` into a gRPC `TrainingRun` message.
///
/// The resulting `TrainingRun` mirrors the record's fields; `started_at` is always set,
/// and `completed_at` is `None` when the record's `completed_at` timestamp is zero.
///
/// # Examples
///
/// ```
/// // construct a storage::TrainingRunRecord (fields shown for illustration)
/// let record = storage::TrainingRunRecord {
///     id: "run-1".to_string(),
///     name: "test".to_string(),
///     status: "running".to_string(),
///     started_at: 1_700_000_000,
///     completed_at: 0,
///     progress: 42,
///     error: "".to_string(),
/// };
/// let proto = training_run(record);
/// assert_eq!(proto.id, "run-1");
/// assert!(proto.started_at.is_some());
/// assert!(proto.completed_at.is_none());
/// ```
fn training_run(run: storage::TrainingRunRecord) -> TrainingRun {
    TrainingRun {
        id: run.id,
        name: run.name,
        status: run.status,
        started_at: Some(timestamp(run.started_at)),
        completed_at: if run.completed_at == 0 {
            None
        } else {
            Some(timestamp(run.completed_at))
        },
        progress: run.progress,
        error: run.error,
    }
}

/// Convert a context engine resource summary into a protobuf `ContextResource`.
///
/// The returned `ContextResource` preserves the URI, title, layer, and copies the metadata map.
///
/// # Examples
///
/// ```
/// let src = context_engine::ResourceSummary {
///     uri: "urn:example:1".into(),
///     title: "Example".into(),
///     layer: 1,
///     metadata: std::collections::HashMap::from([("k".into(), "v".into())]),
/// };
/// let out = context_resource(src);
/// assert_eq!(out.uri, "urn:example:1");
/// assert_eq!(out.title, "Example");
/// assert_eq!(out.layer, 1);
/// assert_eq!(out.metadata.get("k").map(|v| v.as_str()), Some("v"));
/// ```
fn context_resource(resource: context_engine::ResourceSummary) -> engine::ContextResource {
    engine::ContextResource {
        uri: resource.uri,
        title: resource.title,
        layer: resource.layer,
        metadata: resource.metadata.into_iter().collect(),
    }
}

/// Convert a context engine `SearchHit` into an `engine::ContextSearchResult`.
///
/// The returned `ContextSearchResult` contains the same URI, document ID, chunk text,
/// score, and layer as the input, with `metadata` converted into a protobuf map.
///
/// # Examples
///
/// ```
/// let hit = context_engine::SearchHit {
///     uri: "file://a".to_string(),
///     document_id: "doc1".to_string(),
///     chunk_text: "text".to_string(),
///     score: 0.9,
///     metadata: vec![("k".to_string(), "v".to_string())],
///     layer: 1,
/// };
/// let proto = context_search_result(hit);
/// assert_eq!(proto.uri, "file://a");
/// assert_eq!(proto.metadata.get("k").map(|v| v.as_str()), Some("v"));
/// ```
fn context_search_result(result: context_engine::SearchHit) -> engine::ContextSearchResult {
    engine::ContextSearchResult {
        uri: result.uri,
        document_id: result.document_id,
        chunk_text: result.chunk_text,
        score: result.score,
        metadata: result.metadata.into_iter().collect(),
        layer: result.layer,
    }
}

/// Convert a graph fact record into a context search result enriched with graph-specific metadata.
///
/// The returned `ContextSearchResult` contains a human-readable `chunk_text` of the form
/// "<subject> <relation> <object>", a deterministic `uri` (falls back to `viking://graph/{edge_id}` if the fact has no resource URI),
/// a `score` computed as `(1.0 - rank * 0.01)` with a minimum of `0.1`, and metadata populated with graph fields
/// such as `subject_id`, `subject_type`, `object_id`, `object_type`, `relation`, and `kind = "graph"`.
///
/// # Parameters
///
/// - `fact`: the graph fact record to convert; its metadata map is preserved and augmented.
/// - `rank`: zero-based rank used to compute the result score; smaller ranks yield higher scores.
///
/// # Returns
///
/// `engine::ContextSearchResult` representing the graph fact with enriched metadata and a clamped score.
///
/// # Examples
///
/// ```
/// use std::collections::HashMap;
///
/// // Construct a minimal GraphFactRecord-like value for illustration.
/// let fact = GraphFactRecord {
///     edge_id: "edge-1".to_string(),
///     subject: GraphNode { id: "s1".to_string(), name: "Alice".to_string(), kind: GraphNodeKind::Person },
///     object: GraphNode { id: "o1".to_string(), name: "Wonderland".to_string(), kind: GraphNodeKind::Place },
///     relation: "visited".to_string(),
///     metadata: {
///         let mut m = HashMap::new();
///         m.insert("source".to_string(), "import".to_string());
///         m
///     },
///     resource_uri: None,
///     session_id: None,
/// };
///
/// let res = context_graph_search_result(fact, 2);
/// assert_eq!(res.chunk_text, "Alice visited Wonderland");
/// assert_eq!(res.document_id, "edge-1");
/// assert!(res.score <= 1.0 && res.score >= 0.1);
/// ```
fn context_graph_search_result(fact: GraphFactRecord, rank: usize) -> engine::ContextSearchResult {
    let edge_id = fact.edge_id.clone();
    let subject_name = fact.subject.name.clone();
    let object_name = fact.object.name.clone();
    let relation = fact.relation.as_str().to_string();
    let mut metadata = fact.metadata;
    metadata.insert("kind".to_string(), "graph".to_string());
    metadata.insert("subject_id".to_string(), fact.subject.id.clone());
    metadata.insert(
        "subject_type".to_string(),
        fact.subject.kind.as_str().to_string(),
    );
    metadata.insert("subject_name".to_string(), fact.subject.name.clone());
    metadata.insert("relation".to_string(), fact.relation.as_str().to_string());
    metadata.insert("object_id".to_string(), fact.object.id.clone());
    metadata.insert(
        "object_type".to_string(),
        fact.object.kind.as_str().to_string(),
    );
    metadata.insert("object_name".to_string(), fact.object.name.clone());
    if let Some(session_id) = &fact.session_id {
        metadata.insert("session_id".to_string(), session_id.clone());
    }

    engine::ContextSearchResult {
        uri: fact
            .resource_uri
            .unwrap_or_else(|| format!("viking://graph/{edge_id}")),
        document_id: edge_id,
        chunk_text: format!("{subject_name} {relation} {object_name}"),
        score: (1.0 - (rank as f32 * 0.01)).max(0.1),
        metadata: metadata.into_iter().collect(),
        layer: ResourceLayer::L1.as_str().to_string(),
    }
}

/// Converts a context engine `FileEntry` into the gRPC `ContextFileEntry` DTO.
///
/// The returned value preserves the entry's name, path, directory flag and version,
/// and converts `size_bytes` to an `i64`.
///
/// # Examples
///
/// ```
/// let entry = context_engine::FileEntry {
///     name: "notes.txt".into(),
///     path: "workspace/notes.txt".into(),
///     is_dir: false,
///     size_bytes: 1024,
///     version: 3,
/// };
/// let proto = context_file_entry(entry);
/// assert_eq!(proto.name, "notes.txt");
/// assert_eq!(proto.path, "workspace/notes.txt");
/// assert!(!proto.is_dir);
/// assert_eq!(proto.size_bytes, 1024i64);
/// assert_eq!(proto.version, 3);
/// ```
fn context_file_entry(entry: context_engine::FileEntry) -> engine::ContextFileEntry {
    engine::ContextFileEntry {
        name: entry.name,
        path: entry.path,
        is_dir: entry.is_dir,
        size_bytes: entry.size_bytes as i64,
        version: entry.version,
    }
}

/// Convert a context engine `SessionEntry` into a protobuf `ContextSessionEntry`.
///
/// The returned `ContextSessionEntry` contains the same `session_id`, `role`,
/// `content`, and `metadata` as the source entry, and `created_at` converted
/// from the entry's millisecond timestamp.
///
/// # Examples
///
/// ```
/// use std::collections::HashMap;
/// let mut meta = HashMap::new();
/// meta.insert("key".to_string(), "value".to_string());
/// let src = context_engine::SessionEntry {
///     session_id: "s1".to_string(),
///     role: "user".to_string(),
///     content: "hello".to_string(),
///     metadata: meta.clone(),
///     created_at: 1_650_000_000_123, // millis
/// };
/// let out = crate::context_session_entry(src);
/// assert_eq!(out.session_id, "s1");
/// assert_eq!(out.role, "user");
/// assert_eq!(out.content, "hello");
/// assert_eq!(out.metadata.get("key").map(|v| v.as_str()), Some("value"));
/// assert!(out.created_at.is_some());
/// ```
fn context_session_entry(entry: context_engine::SessionEntry) -> engine::ContextSessionEntry {
    engine::ContextSessionEntry {
        session_id: entry.session_id,
        role: entry.role,
        content: entry.content,
        metadata: entry.metadata.into_iter().collect(),
        created_at: Some(timestamp_millis(entry.created_at)),
    }
}

/// Map a numeric protobuf layer identifier to the corresponding `ResourceLayer`.
///
/// The mapping is: `1` -> `L0`, `2` -> `L1`, any other value -> `L2`.
///
/// # Parameters
///
/// - `layer`: Protobuf integer representing a resource layer.
///
/// # Returns
///
/// The corresponding `ResourceLayer` variant.
///
/// # Examples
///
/// ```
/// assert_eq!(resource_layer_from_proto(1), ResourceLayer::L0);
/// assert_eq!(resource_layer_from_proto(2), ResourceLayer::L1);
/// assert_eq!(resource_layer_from_proto(0), ResourceLayer::L2);
/// ```
fn resource_layer_from_proto(layer: i32) -> ResourceLayer {
    match layer {
        1 => ResourceLayer::L0,
        2 => ResourceLayer::L1,
        _ => ResourceLayer::L2,
    }
}

/// Maps a numeric layer index to a `ResourceLayer` variant.
///
/// Returns the corresponding `ResourceLayer` for indices 1 → `L0`, 2 → `L1`, 3 → `L2`, or `None` for any other value.
///
/// # Examples
///
/// ```
/// assert_eq!(context_search_layer(1), Some(ResourceLayer::L0));
/// assert_eq!(context_search_layer(2), Some(ResourceLayer::L1));
/// assert_eq!(context_search_layer(3), Some(ResourceLayer::L2));
/// assert_eq!(context_search_layer(0), None);
/// ```
fn context_search_layer(layer: i32) -> Option<ResourceLayer> {
    match layer {
        1 => Some(ResourceLayer::L0),
        2 => Some(ResourceLayer::L1),
        3 => Some(ResourceLayer::L2),
        _ => None,
    }
}

/// Creates a `Timestamp` representing the given number of seconds since the Unix epoch with zero nanoseconds.
///
/// The returned `Timestamp`'s `seconds` field is set to `seconds` and `nanos` is set to `0`.
///
/// # Examples
///
/// ```
/// let ts = timestamp(1_700_000_000);
/// assert_eq!(ts.seconds, 1_700_000_000);
/// assert_eq!(ts.nanos, 0);
/// ```
fn timestamp(seconds: i64) -> Timestamp {
    Timestamp { seconds, nanos: 0 }
}

/// Convert milliseconds since the Unix epoch into a `Timestamp` with correctly computed
/// seconds and nanoseconds.
///
/// This uses euclidean division so values before the epoch (negative milliseconds)
/// yield a `seconds`/`nanos` pair that conforms to the `Timestamp` representation
/// (nanos is always non-negative and less than 1_000_000_000).
///
/// # Examples
///
/// ```
/// let t = timestamp_millis(1500);
/// assert_eq!(t.seconds, 1);
/// assert_eq!(t.nanos, 500_000_000);
///
/// let t_neg = timestamp_millis(-500);
/// // -500 ms => seconds = -1, nanos = 500_000_000
/// assert_eq!(t_neg.seconds, -1);
/// assert_eq!(t_neg.nanos, 500_000_000);
/// ```
fn timestamp_millis(millis: i64) -> Timestamp {
    let seconds = millis.div_euclid(1000);
    let nanos = (millis.rem_euclid(1000) * 1_000_000) as i32;
    Timestamp { seconds, nanos }
}

/// Constructs the built-in MCP tool definitions bundled with the daemon.
///
/// Returns the tools advertised for in-process/demo MCP connections and used as a
/// fallback when persisted tool metadata cannot be decoded.
///
/// # Examples
///
/// ```
/// let tools = builtin_tools();
/// assert!(!tools.is_empty());
/// ```
fn builtin_tools() -> Vec<Tool> {
    vec![
        Tool {
            name: "mcp.describe_connection".to_string(),
            description: "Return the stored metadata for the active MCP connection.".to_string(),
            parameters: vec![ToolParameter {
                name: "verbose".to_string(),
                r#type: "boolean".to_string(),
                required: false,
                description: "Include any additional connection metadata when available."
                    .to_string(),
            }],
        },
        Tool {
            name: "mcp.echo".to_string(),
            description: "Echo a structured payload to verify end-to-end MCP execution wiring."
                .to_string(),
            parameters: vec![ToolParameter {
                name: "payload".to_string(),
                r#type: "object".to_string(),
                required: false,
                description: "Arbitrary JSON-compatible payload to echo back.".to_string(),
            }],
        },
    ]
}

fn to_proto_tool(tool: StoredTool) -> Tool {
    Tool {
        name: tool.name,
        description: tool.description,
        parameters: tool
            .parameters
            .into_iter()
            .map(|parameter| ToolParameter {
                name: parameter.name,
                r#type: parameter.parameter_type,
                required: parameter.required,
                description: parameter.description,
            })
            .collect(),
    }
}

fn stable_id(value: &str) -> String {
    let mut hash = 1469598103934665603u64;
    for byte in value.as_bytes() {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(1099511628211);
    }
    hash.to_string()
}

fn string_value(value: &str) -> Value {
    Value {
        kind: Some(value::Kind::StringValue(value.to_string())),
    }
}

fn internal_status(error: impl std::fmt::Display) -> Status {
    Status::internal(error.to_string())
}

fn load_model_status(error: impl std::fmt::Display) -> Status {
    let message = error.to_string();
    if message.contains("model not found") {
        Status::not_found(message)
    } else if message.contains("backend command not found") {
        Status::failed_precondition(message)
    } else {
        Status::internal(message)
    }
}

fn inference_status(error: impl std::fmt::Display) -> Status {
    let message = error.to_string();
    if message.contains("prompt is required") {
        Status::invalid_argument(message)
    } else if message.contains("model not found") {
        Status::not_found(message)
    } else if message.contains("model not loaded") || message.contains("backend command not found")
    {
        Status::failed_precondition(message)
    } else {
        Status::internal(message)
    }
}

fn not_found_status(error: impl std::fmt::Display) -> Status {
    Status::not_found(error.to_string())
}

/// Get current time as seconds since the UNIX epoch.
///
/// If the system clock is earlier than the UNIX epoch, returns `0`.
///
/// # Examples
///
/// ```
/// let t = now();
/// assert!(t >= 0);
/// ```
fn now() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::{Path, PathBuf};

    use tempfile::tempdir;
    use tokio_stream::StreamExt;
    use tonic::Code;

    use super::*;

    #[tokio::test]
    async fn load_model_returns_not_found_for_unknown_model() {
        let service = test_service();
        let err = service
            .load_model(Request::new(LoadModelRequest {
                model_id: "missing.gguf".to_string(),
                options: HashMap::new(),
            }))
            .await
            .unwrap_err();

        assert_eq!(err.code(), Code::NotFound);
    }

    #[tokio::test]
    async fn upsert_document_rejects_empty_content() {
        let service = test_service();
        let err = service
            .upsert_document(Request::new(UpsertRequest {
                document_id: String::new(),
                content: String::new(),
                metadata: HashMap::new(),
            }))
            .await
            .unwrap_err();

        assert_eq!(err.code(), Code::InvalidArgument);
    }

    #[tokio::test]
    async fn search_rejects_empty_query() {
        let service = test_service();
        let err = service
            .search(Request::new(SearchRequest {
                query: String::new(),
                top_k: 5,
                filters: HashMap::new(),
            }))
            .await
            .unwrap_err();

        assert_eq!(err.code(), Code::InvalidArgument);
    }

    #[tokio::test]
    async fn stream_inference_emits_completion_when_backend_returns_no_tokens() {
        let (sender, receiver) = mpsc::channel(1);
        drop(sender);

        let mut stream = bridge_inference_chunks(receiver);
        let response = stream.next().await.unwrap().unwrap();

        assert!(response.complete);
        assert!(response.token.is_empty());
        assert!(stream.next().await.is_none());
    }

    fn test_service() -> EngineService {
        let tempdir = tempdir().unwrap();
        let models_dir = tempdir.path().join("models");
        let training_dir = tempdir.path().join("training");
        let db_dir = tempdir.path().join("lancedb");
        std::fs::create_dir_all(&models_dir).unwrap();
        std::fs::create_dir_all(&training_dir).unwrap();
        std::fs::create_dir_all(&db_dir).unwrap();

        let store = EngineStore::new(db_dir.to_string_lossy().to_string());
        let embedding_engine = Arc::new(create_default_engine());

        EngineService {
            state: AppState {
                store: store.clone(),
                runtime: RuntimeEngine::new(
                    store.clone(),
                    models_dir,
                    "llama.cpp",
                    create_fake_backend(tempdir.path())
                        .to_string_lossy()
                        .to_string(),
                ),
                training: TrainingEngine::new(store.clone(), training_dir, "llama.cpp"),
                embedding_provider_name: embedding_engine.name().to_string(),
                embedding: embedding_engine,
                chunking: ChunkingConfig::default(),
            },
        }
    }

    fn create_fake_backend(root: &Path) -> PathBuf {
        if cfg!(windows) {
            let path = root.join("fake-llama.cmd");
            fs::write(&path, "@echo off\r\necho backend:%~4\r\n").unwrap();
            path
        } else {
            let path = root.join("fake-llama.sh");
            fs::write(&path, "#!/bin/sh\nprintf 'backend:%s\\n' \"$4\"\n").unwrap();
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;

                let mut permissions = fs::metadata(&path).unwrap().permissions();
                permissions.set_mode(0o755);
                fs::set_permissions(&path, permissions).unwrap();
            }
            path
        }
    }
}

fn bridge_inference_chunks(
    mut chunks: mpsc::Receiver<Result<String>>,
) -> ReceiverStream<Result<InferenceResponse, Status>> {
    let (sender, receiver) = mpsc::channel(8);
    tokio::spawn(async move {
        while let Some(chunk) = chunks.recv().await {
            match chunk {
                Ok(token) => {
                    if sender
                        .send(Ok(InferenceResponse {
                            token,
                            complete: false,
                            metrics: HashMap::new(),
                        }))
                        .await
                        .is_err()
                    {
                        return;
                    }
                }
                Err(error) => {
                    let _ = sender.send(Err(internal_status(error))).await;
                    return;
                }
            }
        }

        let _ = sender
            .send(Ok(InferenceResponse {
                token: String::new(),
                complete: true,
                metrics: HashMap::new(),
            }))
            .await;
    });

    ReceiverStream::new(receiver)
}

/// Opens and returns a `ContextEngine` configured from environment variables.
///
/// This reads configuration from the environment and initializes a `ContextEngine` with
/// the resulting settings. Relevant environment variables:
/// - `CONTEXT_DATA_DIR` (defaults to `./context-data`)
/// - `CONTEXT_ROOTS` (defaults to `workspace=.`)
/// - `CONTEXT_OPENVIKING_URL` and optional OpenViking subsettings:
///   `CONTEXT_OPENVIKING_API_KEY`, `CONTEXT_OPENVIKING_IMPORT_PATH`,
///   `CONTEXT_OPENVIKING_SYNC_PATH`, `CONTEXT_OPENVIKING_FIND_PATH`, `CONTEXT_OPENVIKING_READ_PATH`
/// - Dragonfly-related variables consulted by `dragonfly_config_from_env()`.
///
/// # Errors
///
/// Returns an error if parsing the configured roots or opening the engine fails.
///
/// # Examples
///
/// ```
/// # use std::env;
/// # use tokio_test::block_on;
/// // Set up minimal environment for example (in real use you would set these externally)
/// env::set_var("CONTEXT_DATA_DIR", "./tmp-context-data");
/// env::set_var("CONTEXT_ROOTS", "workspace=.");
/// // Attempt to open the engine
/// let engine = block_on(super::open_context_engine_from_env()).expect("open context engine");
/// // engine can now be used for context operations
/// drop(engine);
/// ```
async fn open_context_engine_from_env() -> Result<ContextEngine> {
    let data_dir =
        std::env::var("CONTEXT_DATA_DIR").unwrap_or_else(|_| "./context-data".to_string());
    let roots_env = std::env::var("CONTEXT_ROOTS").unwrap_or_else(|_| "workspace=.".to_string());
    let roots = parse_context_roots(&roots_env)?;
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

    ContextEngine::open(ContextConfig {
        data_dir: PathBuf::from(data_dir),
        roots,
        bridge,
        dragonfly: dragonfly_config_from_env(),
    })
    .await
    .map_err(Into::into)
}

/// Builds a DragonflyConfig from environment variables if any Dragonfly-related settings are present.
///
/// The configuration is enabled when `CONTEXT_DRAGONFLY_ENABLED` is set to a value other than `""`, `"0"`, or `"false"` (case-insensitive),
/// or when any of `CONTEXT_DRAGONFLY_ADDR`, `CONTEXT_DRAGONFLY_KEY_PREFIX`, or `CONTEXT_DRAGONFLY_RECENT_WINDOW` are provided.
/// Missing fields fall back to `DragonflyConfig::default()` values; `recent_window` is clamped to at least 1.
///
/// # Examples
///
/// ```
/// std::env::set_var("CONTEXT_DRAGONFLY_ADDR", "127.0.0.1:6379");
/// std::env::remove_var("CONTEXT_DRAGONFLY_ENABLED");
/// let cfg = crate::dragonfly_config_from_env();
/// assert!(cfg.is_some());
/// ```
fn dragonfly_config_from_env() -> Option<DragonflyConfig> {
    let enabled = std::env::var("CONTEXT_DRAGONFLY_ENABLED")
        .ok()
        .map(|value| {
            !matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "" | "0" | "false"
            )
        })
        .unwrap_or(false);
    let addr = std::env::var("CONTEXT_DRAGONFLY_ADDR").ok();
    let key_prefix = std::env::var("CONTEXT_DRAGONFLY_KEY_PREFIX").ok();
    let recent_window = std::env::var("CONTEXT_DRAGONFLY_RECENT_WINDOW")
        .ok()
        .and_then(|value| value.parse::<usize>().ok());

    if !(enabled || addr.is_some() || key_prefix.is_some() || recent_window.is_some()) {
        return None;
    }

    let defaults = DragonflyConfig::default();
    Some(DragonflyConfig {
        addr: addr.unwrap_or(defaults.addr),
        key_prefix: key_prefix.unwrap_or(defaults.key_prefix),
        recent_window: recent_window.unwrap_or(defaults.recent_window).max(1),
    })
}

/// Parses a semicolon-separated list of context roots into `ManagedRoot` instances.
///
/// Each non-empty entry is either `name=path` or just `path`. Entries are trimmed of
/// surrounding whitespace. For unnamed entries, the first is named `"workspace"` and
/// subsequent ones are named `"root-N"` (N starts at 2). Empty or whitespace-only
/// segments are ignored. Errors returned by `ManagedRoot::new` are propagated.
///
/// # Examples
///
/// ```
/// let input = "workspace=/data/ws;assets=/data/assets;/other/path";
/// let roots = parse_context_roots(input).unwrap();
/// assert_eq!(roots.len(), 3);
/// assert_eq!(roots[0].name(), "workspace");
/// assert_eq!(roots[1].name(), "assets");
/// assert_eq!(roots[2].name(), "root-3");
/// ```
fn parse_context_roots(value: &str) -> Result<Vec<ManagedRoot>> {
    let mut roots = Vec::new();
    for entry in value.split(';').filter(|entry| !entry.trim().is_empty()) {
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

#[tonic::async_trait]
impl ContextRpc for ContextGrpcService {
    /// Fetches the current status of the context engine, including counts, index size, embedding model, readiness, and managed roots.
    ///
    /// The returned `engine::ContextStatus` mirrors the internal engine status at the time of the call.
    ///
    /// # Returns
    ///
    /// A populated `engine::ContextStatus` with fields:
    /// - `document_count` — total number of documents indexed
    /// - `chunk_count` — total number of chunks indexed
    /// - `index_size_bytes` — approximate index size in bytes
    /// - `embedding_model` — name of the embedding provider/model
    /// - `ready` — whether the context engine is ready to serve requests
    /// - `managed_roots` — configured managed roots exposed by the engine
    ///
    /// # Examples
    ///
    /// ```no_run
    /// // Assume `svc` is an initialized ContextGrpcService and a Tokio runtime is running.
    /// # async fn example(svc: &crate::ContextGrpcService) {
    /// let resp = svc.get_context_status(tonic::Request::new(())).await.unwrap();
    /// let status = resp.get_ref();
    /// println!("documents: {}", status.document_count);
    /// println!("ready: {}", status.ready);
    /// # }
    /// ```
    async fn get_context_status(
        &self,
        _request: Request<()>,
    ) -> Result<Response<engine::ContextStatus>, Status> {
        let status = self.engine.status().await.map_err(internal_status)?;
        Ok(Response::new(engine::ContextStatus {
            document_count: status.document_count,
            chunk_count: status.chunk_count,
            index_size_bytes: status.index_size_bytes,
            embedding_model: status.embedding_model,
            ready: status.ready,
            managed_roots: status.managed_roots,
        }))
    }

    /// Lists available context resources.
    ///
    /// Retrieves resources from the underlying ContextEngine and returns them mapped into the protobuf `engine::ContextResourceList`.
    ///
    /// # Returns
    ///
    /// `engine::ContextResourceList` wrapped in a `Response` containing the mapped resources, or a gRPC `Status` on error.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use tonic::Request;
    /// # async fn example(svc: &crate::ContextGrpcService) -> Result<(), tonic::Status> {
    /// let resp = svc.list_resources(Request::new(())).await?;
    /// let list = resp.into_inner();
    /// println!("found {} resources", list.resources.len());
    /// # Ok(())
    /// # }
    /// ```
    async fn list_resources(
        &self,
        _request: Request<()>,
    ) -> Result<Response<engine::ContextResourceList>, Status> {
        let resources = self
            .engine
            .list_resources()
            .await
            .map_err(internal_status)?;
        Ok(Response::new(engine::ContextResourceList {
            resources: resources.into_iter().map(context_resource).collect(),
        }))
    }

    /// Upserts a context resource and returns the resulting resource plus how many chunks were indexed.
    ///
    /// Empty `title` or `previous_uri` fields in the request are treated as absent. The `layer` value
    /// is mapped to the engine's `ResourceLayer` before upsert. The response contains the stored
    /// resource representation and the number of chunks that were indexed as a result of the upsert.
    ///
    /// # Returns
    ///
    /// `ContextUpsertResourceResponse` containing the upserted `resource` and `chunks_indexed`.
    ///
    /// # Examples
    ///
    /// ```
    /// # async fn example(svc: crate::ContextGrpcService) {
    /// use tonic::Request;
    /// use crate::engine;
    ///
    /// let req = engine::ContextUpsertResourceRequest {
    ///     uri: "workspace://doc1".into(),
    ///     title: "".into(), // treated as absent
    ///     content: "Example content".into(),
    ///     layer: 1,
    ///     metadata: Default::default(),
    ///     previous_uri: "".into(),
    /// };
    ///
    /// let resp = svc.upsert_resource(Request::new(req)).await.unwrap().into_inner();
    /// assert!(resp.chunks_indexed >= 0);
    /// assert!(resp.resource.is_some());
    /// # }
    /// ```
    async fn upsert_resource(
        &self,
        request: Request<engine::ContextUpsertResourceRequest>,
    ) -> Result<Response<engine::ContextUpsertResourceResponse>, Status> {
        let request = request.into_inner();
        let outcome = self
            .engine
            .upsert_resource(ResourceUpsertRequest {
                uri: request.uri,
                title: if request.title.is_empty() {
                    None
                } else {
                    Some(request.title)
                },
                content: request.content,
                layer: resource_layer_from_proto(request.layer),
                metadata: request.metadata.into_iter().collect(),
                previous_uri: if request.previous_uri.is_empty() {
                    None
                } else {
                    Some(request.previous_uri)
                },
            })
            .await
            .map_err(internal_status)?;

        Ok(Response::new(engine::ContextUpsertResourceResponse {
            resource: Some(context_resource(outcome.resource)),
            chunks_indexed: outcome.chunks_indexed,
        }))
    }

    /// Deletes a context resource identified by the given URI from the context engine.
    ///
    /// Engine errors are translated into an internal gRPC status.
    ///
    /// # Returns
    ///
    /// `()` on success.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use tonic::Request;
    /// // Construct a request carrying the resource URI.
    /// let req = Request::new(engine::ContextDeleteResourceRequest { uri: "workspace://path/to/resource".into() });
    /// // Call the gRPC handler (example; `service` must be an instance of the gRPC service).
    /// // let resp = service.delete_resource(req).await?;
    /// ```
    async fn delete_resource(
        &self,
        request: Request<engine::ContextDeleteResourceRequest>,
    ) -> Result<Response<()>, Status> {
        self.engine
            .delete_resource(&request.into_inner().uri)
            .await
            .map_err(internal_status)?;
        Ok(Response::new(()))
    }

    /// Searches the context engine using the provided query and request options, returning matching context hits or graph facts along with the query duration.
    ///
    /// If `filters["kind"]` equals `"graph"` (case-insensitive) this will query graph facts and return graph-typed results; otherwise it performs a normal context search using optional scope, top-k, filters, layer, and rerank settings. The response contains the matched results and `query_time_ms` measured from request receipt to response construction.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use tonic::Request;
    /// # use crate::engine;
    /// # async fn example(svc: crate::ContextGrpcService) {
    /// let req = engine::ContextSearchRequest {
    ///     query: "find relevant facts".into(),
    ///     scope_uri: "".into(),
    ///     top_k: 0,
    ///     filters: Default::default(),
    ///     layer: 0,
    ///     rerank: false,
    /// };
    /// let resp = svc.search_context(Request::new(req)).await.unwrap();
    /// println!("query took {} ms and returned {} results", resp.get_ref().query_time_ms, resp.get_ref().results.len());
    /// # }
    /// ```
    async fn search_context(
        &self,
        request: Request<engine::ContextSearchRequest>,
    ) -> Result<Response<engine::ContextSearchResponse>, Status> {
        let started_at = Instant::now();
        let request = request.into_inner();
        let bounded_top_k = if request.top_k > 0 {
            std::cmp::min(request.top_k as usize, MAX_CONTEXT_TOP_K)
        } else {
            10
        };
        if request
            .filters
            .get("kind")
            .map(|value| value.eq_ignore_ascii_case("graph"))
            .unwrap_or(false)
        {
            let results = self
                .engine
                .graph_facts(&request.query, bounded_top_k)
                .await
                .map_err(internal_status)?;

            return Ok(Response::new(engine::ContextSearchResponse {
                results: results
                    .into_iter()
                    .enumerate()
                    .map(|(rank, fact)| context_graph_search_result(fact, rank))
                    .collect(),
                query_time_ms: started_at.elapsed().as_secs_f64() * 1000.0,
            }));
        }

        let results = self
            .engine
            .search_context(ContextSearchRequestModel {
                query: request.query,
                scope_uri: if request.scope_uri.is_empty() {
                    None
                } else {
                    Some(request.scope_uri)
                },
                top_k: (bounded_top_k > 0).then_some(bounded_top_k),
                filters: (!request.filters.is_empty())
                    .then_some(request.filters.into_iter().collect()),
                layer: context_search_layer(request.layer),
                rerank: request.rerank,
            })
            .await
            .map_err(internal_status)?;

        Ok(Response::new(engine::ContextSearchResponse {
            results: results.into_iter().map(context_search_result).collect(),
            query_time_ms: started_at.elapsed().as_secs_f64() * 1000.0,
        }))
    }

    /// Synchronizes the workspace rooted at `request.root`, optionally limited to a subpath, and returns counts of changed resources.
    ///
    /// The method delegates to the underlying `ContextEngine::sync_workspace` and maps its outcome into a `ContextWorkspaceSyncResponse`.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use tonic::Request;
    /// # use your_crate::engine;
    /// # async fn example(service: &ContextGrpcService) -> Result<(), tonic::Status> {
    /// let req = engine::ContextWorkspaceSyncRequest {
    ///     root: "workspace".to_string(),
    ///     path: "".to_string(), // empty to sync the entire root
    /// };
    /// let resp = service.sync_workspace(Request::new(req)).await?;
    /// let body = resp.into_inner();
    /// println!("indexed: {}", body.indexed_resources);
    /// # Ok(())
    /// # }
    /// ```
    async fn sync_workspace(
        &self,
        request: Request<engine::ContextWorkspaceSyncRequest>,
    ) -> Result<Response<engine::ContextWorkspaceSyncResponse>, Status> {
        let request = request.into_inner();
        let outcome = self
            .engine
            .sync_workspace(
                &request.root,
                (!request.path.is_empty()).then(|| PathBuf::from(request.path)),
            )
            .await
            .map_err(internal_status)?;

        Ok(Response::new(engine::ContextWorkspaceSyncResponse {
            root: outcome.root,
            prefix: outcome.prefix,
            indexed_resources: outcome.indexed_resources,
            reindexed_resources: outcome.reindexed_resources,
            deleted_resources: outcome.deleted_resources,
            skipped_files: outcome.skipped_files,
        }))
    }

    /// List files under a managed context root and path.
    ///
    /// Queries the context engine for the directory entries at the provided `root` and `path`,
    /// and returns them as a `ContextFileListResponse`. Engine errors are converted to gRPC
    /// internal statuses.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use tonic::Request;
    /// // assume `svc` is a ContextGrpcService instance and types are in scope
    /// let req = engine::ContextFileListRequest { root: "workspace".into(), path: "docs".into() };
    /// let resp = svc.list_files(Request::new(req)).await.unwrap();
    /// println!("found {} entries", resp.get_ref().entries.len());
    /// ```
    async fn list_files(
        &self,
        request: Request<engine::ContextFileListRequest>,
    ) -> Result<Response<engine::ContextFileListResponse>, Status> {
        let request = request.into_inner();
        let entries = self
            .engine
            .list_files(&request.root, PathBuf::from(request.path))
            .await
            .map_err(internal_status)?;
        Ok(Response::new(engine::ContextFileListResponse {
            entries: entries.into_iter().map(context_file_entry).collect(),
        }))
    }

    /// Reads a file from a managed workspace root and returns its contents along with the file version.
    ///
    /// Errors from the context engine are mapped to an internal gRPC status. If the engine does not
    /// provide a file version, the returned `version` is `0`.
    ///
    /// # Examples
    ///
    /// ```
    /// # tokio_test::block_on(async {
    /// use tonic::Request;
    /// let svc = /* ContextGrpcService instance */;
    /// let req = engine::ContextFileReadRequest {
    ///     root: "workspace".into(),
    ///     path: "docs/readme.md".into(),
    /// };
    /// let resp = svc.read_file(Request::new(req)).await.unwrap().into_inner();
    /// assert_eq!(resp.path, "docs/readme.md");
    /// // resp.content contains the file bytes and resp.version is the file version or 0
    /// # });
    /// ```
    async fn read_file(
        &self,
        request: Request<engine::ContextFileReadRequest>,
    ) -> Result<Response<engine::ContextFileReadResponse>, Status> {
        let request = request.into_inner();
        let path = PathBuf::from(&request.path);
        let content = self
            .engine
            .read_file(&request.root, path.clone())
            .await
            .map_err(internal_status)?;
        let version = self
            .engine
            .file_version(&request.root, path)
            .await
            .unwrap_or(0);
        Ok(Response::new(engine::ContextFileReadResponse {
            path: request.path,
            content,
            version,
        }))
    }

    /// Writes `content` to a file at `path` within the specified `root` and returns the resulting file `version`.
    ///
    /// The request's `version` is passed through to the underlying engine and the call returns the persisted
    /// `path` and the resolved `version` produced by the engine.
    ///
    /// # Returns
    ///
    /// `engine::ContextFileWriteResponse` containing the written `path` and the file `version`.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use tonic::Request;
    /// use crate::engine;
    ///
    /// // Assume `svc` is an initialized ContextGrpcService with a working `engine`.
    /// // let svc = ContextGrpcService::new(...);
    ///
    /// let req = Request::new(engine::ContextFileWriteRequest {
    ///     root: "workspace".into(),
    ///     path: "notes/todo.txt".into(),
    ///     content: b"Buy milk".to_vec(),
    ///     version: 0, // client-provided version or 0 for unconditional write
    /// });
    ///
    /// // let resp = tokio::runtime::Runtime::new().unwrap().block_on(svc.write_file(req)).unwrap();
    /// // assert_eq!(resp.get_ref().path, "notes/todo.txt");
    /// ```
    async fn write_file(
        &self,
        request: Request<engine::ContextFileWriteRequest>,
    ) -> Result<Response<engine::ContextFileWriteResponse>, Status> {
        let request = request.into_inner();
        let version = self
            .engine
            .write_file(
                &request.root,
                PathBuf::from(&request.path),
                &request.content,
                request.version,
            )
            .await
            .map_err(internal_status)?;
        Ok(Response::new(engine::ContextFileWriteResponse {
            path: request.path,
            version,
        }))
    }

    /// Deletes a file at the specified path within a managed root and returns whether it was removed.
    ///
    /// If a `version` is provided, the engine attempts a versioned delete; otherwise it performs an unconditional delete.
    ///
    /// # Returns
    ///
    /// A `ContextFileDeleteResponse` containing the original `path` and `deleted` set to `true` if the file was removed, `false` otherwise.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// // Illustrative usage
    /// let req = engine::ContextFileDeleteRequest {
    ///     root: "workspace".to_string(),
    ///     path: "notes/todo.txt".to_string(),
    ///     version: 0,
    /// };
    /// let resp = service.delete_file(Request::new(req)).await.unwrap().into_inner();
    /// assert_eq!(resp.path, "notes/todo.txt");
    /// // resp.deleted indicates whether the file was actually deleted
    /// ```
    async fn delete_file(
        &self,
        request: Request<engine::ContextFileDeleteRequest>,
    ) -> Result<Response<engine::ContextFileDeleteResponse>, Status> {
        let request = request.into_inner();
        let deleted = self
            .engine
            .delete_file(&request.root, PathBuf::from(&request.path), request.version)
            .await
            .map_err(internal_status)?;
        Ok(Response::new(engine::ContextFileDeleteResponse {
            path: request.path,
            deleted,
        }))
    }

    /// Moves a file within the specified managed root and returns the new version and paths.
    ///
    /// The request's `root`, `from_path`, `to_path`, and `version` are forwarded to the underlying
    /// context engine's `move_file`. Errors from the engine are mapped to an internal gRPC status.
    ///
    /// # Examples
    ///
    /// ```
    /// # use tonic::Request;
    /// # use crate::engine;
    /// # async fn example(svc: &crate::ContextGrpcService) {
    /// let req = engine::ContextFileMoveRequest {
    ///     root: "workspace".into(),
    ///     from_path: "old.txt".into(),
    ///     to_path: "new.txt".into(),
    ///     version: 0,
    /// };
    /// let resp = svc.move_file(Request::new(req)).await.unwrap().into_inner();
    /// assert_eq!(resp.from_path, "old.txt");
    /// assert_eq!(resp.to_path, "new.txt");
    /// // `version` contains the resulting file version returned by the engine.
    /// # }
    /// ```
    async fn move_file(
        &self,
        request: Request<engine::ContextFileMoveRequest>,
    ) -> Result<Response<engine::ContextFileMoveResponse>, Status> {
        let request = request.into_inner();
        let version = self
            .engine
            .move_file(
                &request.root,
                PathBuf::from(&request.from_path),
                PathBuf::from(&request.to_path),
                request.version,
            )
            .await
            .map_err(internal_status)?;
        Ok(Response::new(engine::ContextFileMoveResponse {
            from_path: request.from_path,
            to_path: request.to_path,
            version,
        }))
    }

    /// Appends an event to a session and returns the session's full history.
    ///
    /// Processes the provided session event (role, content, metadata), appends it to the session identified by
    /// `session_id`, and then fetches and returns the complete session history.
    ///
    /// # Returns
    ///
    /// An `engine::ContextSessionHistory` containing the `session_id` and the ordered `entries` for that session.
    ///
    /// # Examples
    ///
    /// ```
    /// use tonic::Request;
    /// # // The following types are from the crate's generated proto modules.
    /// # use crate::engine;
    ///
    /// // Build a request to append a session event.
    /// let req = engine::ContextSessionAppendRequest {
    ///     session_id: "session-123".into(),
    ///     role: "user".into(),
    ///     content: "Hello".into(),
    ///     metadata: vec![],
    /// };
    ///
    /// // Call the service method with `Request::new(req)` and await the response.
    /// // let response = service.append_session(Request::new(req)).await.unwrap();
    /// // assert_eq!(response.get_ref().session_id, "session-123");
    /// ```
    async fn append_session(
        &self,
        request: Request<engine::ContextSessionAppendRequest>,
    ) -> Result<Response<engine::ContextSessionHistory>, Status> {
        let request = request.into_inner();
        let session_id = request.session_id.clone();
        self.engine
            .append_session(SessionEventRequest {
                session_id: session_id.clone(),
                role: request.role,
                content: request.content,
                metadata: request.metadata.into_iter().collect(),
            })
            .await
            .map_err(internal_status)?;
        let entries = self
            .engine
            .list_sessions(&session_id)
            .await
            .map_err(internal_status)?;
        Ok(Response::new(engine::ContextSessionHistory {
            session_id,
            entries: entries.into_iter().map(context_session_entry).collect(),
        }))
    }

    /// Retrieves the full history for a context session identified by the request's session ID.
    ///
    /// Returns a `ContextSessionHistory` containing the requested `session_id` and the session's
    /// events converted into proto `entries`.
    ///
    /// # Examples
    ///
    /// ```
    /// # async fn example(svc: &ContextGrpcService) -> Result<(), tonic::Status> {
    /// let req = engine::ContextSessionGetRequest { session_id: "session-123".to_string() };
    /// let resp = svc.get_session(tonic::Request::new(req)).await?;
    /// assert_eq!(resp.get_ref().session_id, "session-123");
    /// # Ok(())
    /// # }
    /// ```
    async fn get_session(
        &self,
        request: Request<engine::ContextSessionGetRequest>,
    ) -> Result<Response<engine::ContextSessionHistory>, Status> {
        let request = request.into_inner();
        let entries = self
            .engine
            .list_sessions(&request.session_id)
            .await
            .map_err(internal_status)?;
        Ok(Response::new(engine::ContextSessionHistory {
            session_id: request.session_id,
            entries: entries.into_iter().map(context_session_entry).collect(),
        }))
    }
}

#[cfg(test)]
mod grpc_service_tests {
    use super::*;

    // -----------------------------------------------------------------------
    // parse_context_roots
    // -----------------------------------------------------------------------

    #[test]
    fn test_parse_context_roots_single_named() {
        let roots = parse_context_roots("workspace=.").unwrap();
        assert_eq!(roots.len(), 1);
    }

    #[test]
    fn test_parse_context_roots_multiple_named() {
        let roots = parse_context_roots("workspace=.;docs=./docs").unwrap();
        assert_eq!(roots.len(), 2);
    }

    #[test]
    fn test_parse_context_roots_without_equals_first_gets_workspace_name() {
        let roots = parse_context_roots(".").unwrap();
        assert_eq!(roots.len(), 1);
    }

    #[test]
    fn test_parse_context_roots_without_equals_subsequent_gets_root_n() {
        let roots = parse_context_roots(".;./docs").unwrap();
        assert_eq!(roots.len(), 2);
    }

    #[test]
    fn test_parse_context_roots_empty_string_returns_empty() {
        let roots = parse_context_roots("").unwrap();
        assert!(roots.is_empty());
    }

    #[test]
    fn test_parse_context_roots_whitespace_only_returns_empty() {
        let roots = parse_context_roots("   ;  ").unwrap();
        assert!(roots.is_empty());
    }

    #[test]
    fn test_parse_context_roots_trims_whitespace() {
        // Name and path with surrounding spaces should be trimmed
        let roots = parse_context_roots("  workspace  =  .  ").unwrap();
        assert_eq!(roots.len(), 1);
    }

    #[test]
    fn test_parse_context_roots_single_semicolon_ignored() {
        let roots = parse_context_roots(";").unwrap();
        assert!(roots.is_empty());
    }

    // -----------------------------------------------------------------------
    // dragonfly_config_from_env
    // -----------------------------------------------------------------------

    #[test]
    fn test_dragonfly_config_from_env_no_vars_returns_none() {
        // Ensure the relevant env vars are absent for this test
        std::env::remove_var("CONTEXT_DRAGONFLY_ENABLED");
        std::env::remove_var("CONTEXT_DRAGONFLY_ADDR");
        std::env::remove_var("CONTEXT_DRAGONFLY_KEY_PREFIX");
        std::env::remove_var("CONTEXT_DRAGONFLY_RECENT_WINDOW");

        let result = dragonfly_config_from_env();
        assert!(
            result.is_none(),
            "expected None when no dragonfly env vars are set"
        );
    }

    #[test]
    fn test_dragonfly_config_from_env_enabled_flag_true() {
        std::env::remove_var("CONTEXT_DRAGONFLY_ADDR");
        std::env::remove_var("CONTEXT_DRAGONFLY_KEY_PREFIX");
        std::env::remove_var("CONTEXT_DRAGONFLY_RECENT_WINDOW");
        std::env::set_var("CONTEXT_DRAGONFLY_ENABLED", "true");

        let result = dragonfly_config_from_env();
        std::env::remove_var("CONTEXT_DRAGONFLY_ENABLED");

        assert!(result.is_some(), "expected Some when enabled=true");
        let cfg = result.unwrap();
        // addr and key_prefix should use defaults when not set
        let defaults = DragonflyConfig::default();
        assert_eq!(cfg.addr, defaults.addr);
        assert_eq!(cfg.key_prefix, defaults.key_prefix);
    }

    #[test]
    fn test_dragonfly_config_from_env_enabled_false_no_other_vars_returns_none() {
        std::env::remove_var("CONTEXT_DRAGONFLY_ADDR");
        std::env::remove_var("CONTEXT_DRAGONFLY_KEY_PREFIX");
        std::env::remove_var("CONTEXT_DRAGONFLY_RECENT_WINDOW");
        std::env::set_var("CONTEXT_DRAGONFLY_ENABLED", "false");

        let result = dragonfly_config_from_env();
        std::env::remove_var("CONTEXT_DRAGONFLY_ENABLED");

        assert!(result.is_none());
    }

    #[test]
    fn test_dragonfly_config_from_env_enabled_zero_no_other_vars_returns_none() {
        std::env::remove_var("CONTEXT_DRAGONFLY_ADDR");
        std::env::remove_var("CONTEXT_DRAGONFLY_KEY_PREFIX");
        std::env::remove_var("CONTEXT_DRAGONFLY_RECENT_WINDOW");
        std::env::set_var("CONTEXT_DRAGONFLY_ENABLED", "0");

        let result = dragonfly_config_from_env();
        std::env::remove_var("CONTEXT_DRAGONFLY_ENABLED");

        assert!(result.is_none());
    }

    #[test]
    fn test_dragonfly_config_from_env_addr_triggers_some() {
        std::env::remove_var("CONTEXT_DRAGONFLY_ENABLED");
        std::env::remove_var("CONTEXT_DRAGONFLY_KEY_PREFIX");
        std::env::remove_var("CONTEXT_DRAGONFLY_RECENT_WINDOW");
        std::env::set_var("CONTEXT_DRAGONFLY_ADDR", "127.0.0.1:6380");

        let result = dragonfly_config_from_env();
        std::env::remove_var("CONTEXT_DRAGONFLY_ADDR");

        assert!(result.is_some());
        assert_eq!(result.unwrap().addr, "127.0.0.1:6380");
    }

    #[test]
    fn test_dragonfly_config_from_env_key_prefix_triggers_some() {
        std::env::remove_var("CONTEXT_DRAGONFLY_ENABLED");
        std::env::remove_var("CONTEXT_DRAGONFLY_ADDR");
        std::env::remove_var("CONTEXT_DRAGONFLY_RECENT_WINDOW");
        std::env::set_var("CONTEXT_DRAGONFLY_KEY_PREFIX", "myapp:");

        let result = dragonfly_config_from_env();
        std::env::remove_var("CONTEXT_DRAGONFLY_KEY_PREFIX");

        assert!(result.is_some());
        assert_eq!(result.unwrap().key_prefix, "myapp:");
    }

    #[test]
    fn test_dragonfly_config_from_env_recent_window_triggers_some() {
        std::env::remove_var("CONTEXT_DRAGONFLY_ENABLED");
        std::env::remove_var("CONTEXT_DRAGONFLY_ADDR");
        std::env::remove_var("CONTEXT_DRAGONFLY_KEY_PREFIX");
        std::env::set_var("CONTEXT_DRAGONFLY_RECENT_WINDOW", "20");

        let result = dragonfly_config_from_env();
        std::env::remove_var("CONTEXT_DRAGONFLY_RECENT_WINDOW");

        assert!(result.is_some());
        assert_eq!(result.unwrap().recent_window, 20);
    }

    #[test]
    fn test_dragonfly_config_from_env_recent_window_minimum_is_one() {
        std::env::remove_var("CONTEXT_DRAGONFLY_ENABLED");
        std::env::remove_var("CONTEXT_DRAGONFLY_ADDR");
        std::env::remove_var("CONTEXT_DRAGONFLY_KEY_PREFIX");
        // recent_window = 0 should be clamped to 1
        std::env::set_var("CONTEXT_DRAGONFLY_RECENT_WINDOW", "0");

        let result = dragonfly_config_from_env();
        std::env::remove_var("CONTEXT_DRAGONFLY_RECENT_WINDOW");

        // 0 cannot be parsed as usize 0 but max(1) applies
        // wait — "0".parse::<usize>() succeeds as 0, then .max(1) gives 1
        assert!(result.is_some());
        assert_eq!(result.unwrap().recent_window, 1);
    }

    #[test]
    fn test_dragonfly_config_from_env_invalid_window_ignored() {
        std::env::remove_var("CONTEXT_DRAGONFLY_ENABLED");
        std::env::remove_var("CONTEXT_DRAGONFLY_ADDR");
        std::env::remove_var("CONTEXT_DRAGONFLY_KEY_PREFIX");
        std::env::set_var("CONTEXT_DRAGONFLY_RECENT_WINDOW", "not-a-number");

        let result = dragonfly_config_from_env();
        std::env::remove_var("CONTEXT_DRAGONFLY_RECENT_WINDOW");

        // invalid parse → recent_window is None → no env var triggers → None
        assert!(result.is_none());
    }

    // -----------------------------------------------------------------------
    // resource_layer_from_proto
    // -----------------------------------------------------------------------

    #[test]
    fn test_resource_layer_from_proto_l0() {
        assert!(matches!(resource_layer_from_proto(1), ResourceLayer::L0));
    }

    /// Verifies that the integer 2 maps to `ResourceLayer::L1`.
    ///
    /// # Examples
    ///
    /// ```
    /// assert!(matches!(resource_layer_from_proto(2), ResourceLayer::L1));
    /// ```
    #[test]
    fn test_resource_layer_from_proto_l1() {
        assert!(matches!(resource_layer_from_proto(2), ResourceLayer::L1));
    }

    #[test]
    fn test_resource_layer_from_proto_l2_default() {
        assert!(matches!(resource_layer_from_proto(3), ResourceLayer::L2));
        assert!(matches!(resource_layer_from_proto(0), ResourceLayer::L2));
        assert!(matches!(resource_layer_from_proto(99), ResourceLayer::L2));
    }

    // -----------------------------------------------------------------------
    // context_search_layer
    // -----------------------------------------------------------------------

    #[test]
    fn test_context_search_layer_all_variants() {
        assert!(matches!(context_search_layer(1), Some(ResourceLayer::L0)));
        assert!(matches!(context_search_layer(2), Some(ResourceLayer::L1)));
        assert!(matches!(context_search_layer(3), Some(ResourceLayer::L2)));
        assert!(context_search_layer(0).is_none());
        assert!(context_search_layer(4).is_none());
        assert!(context_search_layer(-1).is_none());
    }

    // -----------------------------------------------------------------------
    // timestamp_millis
    // -----------------------------------------------------------------------

    #[test]
    fn test_timestamp_millis_zero() {
        let ts = timestamp_millis(0);
        assert_eq!(ts.seconds, 0);
        assert_eq!(ts.nanos, 0);
    }

    #[test]
    fn test_timestamp_millis_exact_second() {
        let ts = timestamp_millis(1000);
        assert_eq!(ts.seconds, 1);
        assert_eq!(ts.nanos, 0);
    }

    #[test]
    fn test_timestamp_millis_sub_second() {
        let ts = timestamp_millis(500);
        assert_eq!(ts.seconds, 0);
        assert_eq!(ts.nanos, 500_000_000);
    }

    #[test]
    fn test_timestamp_millis_mixed() {
        // 1500 ms = 1s + 500ms
        let ts = timestamp_millis(1500);
        assert_eq!(ts.seconds, 1);
        assert_eq!(ts.nanos, 500_000_000);
    }

    #[test]
    fn test_timestamp_millis_negative() {
        // Negative millis (before epoch) should use div_euclid
        let ts = timestamp_millis(-1);
        // div_euclid(-1, 1000) = -1, rem_euclid(-1, 1000) = 999
        assert_eq!(ts.seconds, -1);
        assert_eq!(ts.nanos, 999_000_000);
    }
}

fn stored_tool(tool: &Tool) -> StoredTool {
    StoredTool {
        name: tool.name.clone(),
        description: tool.description.clone(),
        parameters: tool
            .parameters
            .iter()
            .map(|parameter| StoredToolParameter {
                name: parameter.name.clone(),
                parameter_type: parameter.r#type.clone(),
                required: parameter.required,
                description: parameter.description.clone(),
            })
            .collect(),
    }
}

fn to_active_connection(record: MCPConnectionRecord) -> ActiveConnection {
    let tools = match serde_json::from_str::<Vec<StoredTool>>(&record.tools_json) {
        Ok(parsed_tools) => parsed_tools.into_iter().map(to_proto_tool).collect(),
        Err(_) => builtin_tools(),
    };
    ActiveConnection { record, tools }
}

fn server_name_from_url(server_url: &str) -> String {
    server_url
        .split("://")
        .nth(1)
        .unwrap_or(server_url)
        .split('/')
        .next()
        .unwrap_or("mcp-server")
        .to_string()
}
