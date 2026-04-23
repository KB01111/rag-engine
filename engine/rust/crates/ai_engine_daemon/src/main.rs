use std::collections::{BTreeMap, HashMap};
use std::pin::Pin;
use std::sync::Arc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use chunking::ChunkingConfig;
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

const MAX_TOP_K: i64 = 1000;

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

#[tokio::main]
async fn main() -> Result<()> {
    let addr =
        std::env::var("AI_ENGINE_DAEMON_ADDR").unwrap_or_else(|_| "127.0.0.1:50061".to_string());
    let lancedb_uri =
        std::env::var("AI_ENGINE_LANCEDB_URI").unwrap_or_else(|_| ".ai-engine/lancedb".to_string());
    let models_path =
        std::env::var("AI_ENGINE_MODELS_PATH").unwrap_or_else(|_| ".ai-engine/models".to_string());
    let training_dir = std::env::var("AI_ENGINE_TRAINING_DIR")
        .unwrap_or_else(|_| ".ai-engine/training".to_string());
    let backend = "llama.cpp".to_string();

    let store = EngineStore::new(lancedb_uri);
    let embedding_engine = Arc::new(create_default_engine());
    let embedding_provider_name = embedding_engine.name().to_string();
    let service = EngineService {
        state: AppState {
            store: store.clone(),
            runtime: RuntimeEngine::new(store.clone(), models_path, backend.clone()),
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
        .set_serving::<TrainingServer<EngineService>>()
        .await;
    health_reporter
        .set_serving::<McpServer<EngineService>>()
        .await;

    Server::builder()
        .add_service(health_service)
        .add_service(RuntimeServer::new(service.clone()))
        .add_service(RagServer::new(service.clone()))
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
        let loaded = models
            .into_iter()
            .filter(|model| model.status == "loaded")
            .map(model_info)
            .collect::<Vec<_>>();

        Ok(Response::new(RuntimeStatus {
            version: "1.0.0".to_string(),
            loaded_models: loaded,
            resources: Some(SystemResources {
                cpu_percent: 0.0,
                memory_used_bytes: 0,
                memory_total_bytes: 0,
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
            .map_err(not_found_status)?;
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
            .map_err(not_found_status)?;
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

        if first.model_id.is_empty() {
            return Err(Status::invalid_argument(
                "model_id is required for daemon-backed inference",
            ));
        }

        let chunks = self
            .state
            .runtime
            .stream_inference(&first.model_id, &first.prompt)
            .await
            .map_err(not_found_status)?;
        let (sender, receiver) = mpsc::channel(8);
        tokio::spawn(async move {
            for chunk in chunks {
                if sender
                    .send(Ok(InferenceResponse {
                        token: chunk.token,
                        complete: chunk.complete,
                        metrics: chunk.metrics,
                    }))
                    .await
                    .is_err()
                {
                    return;
                }
            }
        });

        Ok(Response::new(Box::pin(ReceiverStream::new(receiver))))
    }
}

#[tonic::async_trait]
impl Rag for EngineService {
    async fn upsert_document(
        &self,
        request: Request<UpsertRequest>,
    ) -> Result<Response<UpsertResponse>, Status> {
        let request = request.into_inner();
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

fn timestamp(seconds: i64) -> Timestamp {
    Timestamp { seconds, nanos: 0 }
}

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

fn not_found_status(error: impl std::fmt::Display) -> Status {
    Status::not_found(error.to_string())
}

fn now() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64
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
