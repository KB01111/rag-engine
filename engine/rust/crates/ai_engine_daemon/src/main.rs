use std::collections::{BTreeMap, HashMap};
use std::pin::Pin;
use std::sync::Arc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use chunking::ChunkingConfig;
use embedding::{create_default_engine, EmbeddingEngine};
use prost_types::{value, Struct, Timestamp, Value};
use runtime_engine::RuntimeEngine;
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
    LoadModelRequest, LogsRequest, McpConnection, McpConnectionRequest, ModelInfo,
    ModelList, RagStatus, RuntimeStatus, SearchRequest, SearchResponse, SearchResult,
    SystemResources, Tool, ToolList, ToolParameter, TrainingLog, TrainingRun, TrainingRunList,
    TrainingRunRequest, UpsertRequest, UpsertResponse,
};

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
    chunking: ChunkingConfig,
}

#[derive(Clone)]
struct EngineService {
    state: AppState,
}

#[tokio::main]
async fn main() -> Result<()> {
    let addr = std::env::var("AI_ENGINE_DAEMON_ADDR").unwrap_or_else(|_| "127.0.0.1:50061".to_string());
    let lancedb_uri =
        std::env::var("AI_ENGINE_LANCEDB_URI").unwrap_or_else(|_| ".ai-engine/lancedb".to_string());
    let models_path =
        std::env::var("AI_ENGINE_MODELS_PATH").unwrap_or_else(|_| ".ai-engine/models".to_string());
    let training_dir =
        std::env::var("AI_ENGINE_TRAINING_DIR").unwrap_or_else(|_| ".ai-engine/training".to_string());
    let backend = "llama.cpp".to_string();

    let store = EngineStore::new(lancedb_uri);
    let service = EngineService {
        state: AppState {
            store: store.clone(),
            runtime: RuntimeEngine::new(store.clone(), models_path, backend.clone()),
            training: TrainingEngine::new(store.clone(), training_dir, backend),
            embedding: Arc::new(create_default_engine()),
            chunking: ChunkingConfig::default(),
        },
    };

    let (health_reporter, health_service) = health_reporter();
    health_reporter
        .set_serving::<RuntimeServer<EngineService>>()
        .await;
    health_reporter.set_serving::<RagServer<EngineService>>().await;
    health_reporter
        .set_serving::<TrainingServer<EngineService>>()
        .await;
    health_reporter.set_serving::<McpServer<EngineService>>().await;

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

    async fn get_status(
        &self,
        _request: Request<()>,
    ) -> Result<Response<RuntimeStatus>, Status> {
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

    async fn list_models(
        &self,
        _request: Request<()>,
    ) -> Result<Response<ModelList>, Status> {
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

        let tokens = self.state.runtime.infer_tokens(&first.prompt);
        let (sender, receiver) = mpsc::channel(8);
        tokio::spawn(async move {
            let total = tokens.len();
            for (index, token) in tokens.into_iter().enumerate() {
                if sender
                    .send(Ok(InferenceResponse {
                        token,
                        complete: index + 1 == total,
                        metrics: HashMap::new(),
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
        let top_k = if request.top_k <= 0 { 10 } else { request.top_k as usize };

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

    async fn get_rag_status(
        &self,
        _request: Request<()>,
    ) -> Result<Response<RagStatus>, Status> {
        let documents = self.state.store.list_documents().await.map_err(internal_status)?;
        let chunks = self.state.store.list_chunks().await.map_err(internal_status)?;
        let index_size_bytes = chunks
            .iter()
            .map(|chunk| (chunk.vector.len() * std::mem::size_of::<f32>()) as i64)
            .sum();

        Ok(Response::new(RagStatus {
            document_count: documents.len() as i64,
            chunk_count: chunks.len() as i64,
            index_size_bytes,
            embedding_model: "mock".to_string(),
        }))
    }

    async fn list_documents(
        &self,
        _request: Request<()>,
    ) -> Result<Response<DocumentList>, Status> {
        let documents = self.state.store.list_documents().await.map_err(internal_status)?;
        let chunks = self.state.store.list_chunks().await.map_err(internal_status)?;

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
            .start_run(&request.name, &request.model_id, &request.dataset_path, &config_json)
            .await
            .map_err(internal_status)?;
        Ok(Response::new(training_run(run)))
    }

    async fn cancel_run(
        &self,
        request: Request<CancelRequest>,
    ) -> Result<Response<()>, Status> {
        self.state
            .training
            .cancel_run(&request.into_inner().run_id)
            .await
            .map_err(not_found_status)?;
        Ok(Response::new(()))
    }

    async fn list_runs(
        &self,
        _request: Request<()>,
    ) -> Result<Response<TrainingRunList>, Status> {
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

        let items = artifacts
            .into_iter()
            .filter_map(|path| {
                let metadata = std::fs::metadata(&path).ok()?;
                Some(Artifact {
                    name: path.file_name()?.to_string_lossy().to_string(),
                    path: path.to_string_lossy().to_string(),
                    size_bytes: metadata.len() as i64,
                    created_at: Some(timestamp(now())),
                })
            })
            .collect();
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

        self.state
            .store
            .upsert_mcp_connection(MCPConnectionRecord {
                connection_id: connection_id.clone(),
                server_url: request.server_url.clone(),
                server_name: "MCP Server".to_string(),
                connected: true,
                auth_json,
                tools_json: "default".to_string(),
                updated_at: now(),
            })
            .await
            .map_err(internal_status)?;

        Ok(Response::new(McpConnection {
            connection_id,
            connected: true,
            server_name: "MCP Server".to_string(),
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
        let connections = self
            .state
            .store
            .list_mcp_connections()
            .await
            .map_err(internal_status)?;
        let Some(connection) = connections
            .into_iter()
            .find(|connection| connection.server_url == request.server_url)
        else {
            return Err(Status::not_found("connection not found"));
        };

        let tools = if connection.tools_json == "default" || connection.tools_json.is_empty() {
            default_tools()
        } else {
            default_tools()
        };
        Ok(Response::new(ToolList { tools }))
    }

    async fn call_tool(
        &self,
        request: Request<CallToolRequest>,
    ) -> Result<Response<CallToolResponse>, Status> {
        let request = request.into_inner();
        Ok(Response::new(CallToolResponse {
            success: true,
            result: Some(Struct {
                fields: BTreeMap::from([
                    ("status".to_string(), string_value("success")),
                    ("tool".to_string(), string_value(&request.tool_name)),
                ]),
            }),
            error: String::new(),
        }))
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
    Timestamp {
        seconds,
        nanos: 0,
    }
}

fn default_tools() -> Vec<Tool> {
    vec![
        Tool {
            name: "get_weather".to_string(),
            description: "Get current weather for a location".to_string(),
            parameters: vec![ToolParameter {
                name: "location".to_string(),
                r#type: "string".to_string(),
                required: true,
                description: "City name".to_string(),
            }],
        },
        Tool {
            name: "search_files".to_string(),
            description: "Search files in a directory".to_string(),
            parameters: vec![
                ToolParameter {
                    name: "query".to_string(),
                    r#type: "string".to_string(),
                    required: true,
                    description: "Search query".to_string(),
                },
                ToolParameter {
                    name: "path".to_string(),
                    r#type: "string".to_string(),
                    required: false,
                    description: "Directory path".to_string(),
                },
            ],
        },
    ]
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
