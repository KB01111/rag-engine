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
use serde::Deserialize;
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

#[derive(Clone)]
struct ContextGrpcService {
    engine: ContextEngine,
}

#[derive(Debug, Deserialize)]
struct StoredTool {
    name: String,
    description: String,
    #[serde(default)]
    parameters: Vec<StoredToolParameter>,
}

#[derive(Debug, Deserialize)]
struct StoredToolParameter {
    name: String,
    #[serde(rename = "type")]
    parameter_type: String,
    required: bool,
    description: String,
}

/// Starts and serves the AI engine gRPC daemon, configuring services from environment variables.
///
/// The function reads these environment variables (with shown defaults) to configure the server:
/// - `AI_ENGINE_DAEMON_ADDR` (default `127.0.0.1:50061`)
/// - `AI_ENGINE_LANCEDB_URI` (default `.ai-engine/lancedb`)
/// - `AI_ENGINE_MODELS_PATH` (default `.ai-engine/models`)
/// - `AI_ENGINE_TRAINING_DIR` (default `.ai-engine/training`)
///
/// It initializes persistence, embedding, runtime, training, and context engines, registers gRPC
/// services and health reporting, and then binds and serves the tonic gRPC server on the chosen address.
///
/// # Examples
///
/// ```no_run
/// # use anyhow::Result;
/// # async fn _example() -> Result<()> {
/// crate::main().await?;
/// # Ok(())
/// # }
/// ```
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
    let context_service = ContextGrpcService {
        engine: open_context_engine_from_env().await?,
    };
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
            match serde_json::from_str::<Vec<StoredTool>>(&connection.tools_json) {
                Ok(parsed_tools) => parsed_tools.into_iter().map(to_proto_tool).collect(),
                Err(err) => {
                    eprintln!(
                        "Failed to parse tools_json: {}, falling back to default",
                        err
                    );
                    default_tools()
                }
            }
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

/// Convert a storage `TrainingRunRecord` into its RPC `TrainingRun` representation.
///
/// This maps timestamps and preserves fields; specifically, a `completed_at` value of `0`
/// is converted to `None` in the resulting `TrainingRun`.
///
/// # Examples
///
/// ```
/// let stored = storage::TrainingRunRecord {
///     id: "run1".to_string(),
///     name: "test".to_string(),
///     status: "running".to_string(),
///     started_at: 1_700_000_000,
///     completed_at: 0,
///     progress: 42,
///     error: None,
/// };
/// let proto = training_run(stored);
/// assert_eq!(proto.id, "run1");
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

/// Convert a `context_engine::ResourceSummary` into an `engine::ContextResource`.

///

/// The returned `engine::ContextResource` preserves the source `uri`, `title`,

/// `layer`, and converts the source `metadata` into the proto-compatible map.

///

/// # Examples

///

/// ```

/// use std::collections::HashMap;

/// let mut meta = HashMap::new();

/// meta.insert("k".to_string(), "v".to_string());

/// let src = context_engine::ResourceSummary {

///     uri: "urn:example".to_string(),

///     title: "Example".to_string(),

///     layer: "L1".to_string(),

///     metadata: meta,

/// };

/// let dst = crate::context_resource(src);

/// assert_eq!(dst.uri, "urn:example");

/// assert_eq!(dst.title, "Example");

/// assert_eq!(dst.layer, "L1");

/// assert_eq!(dst.metadata.get("k").and_then(|v| v.as_ref()), Some(&"v".to_string()));

/// ```
fn context_resource(resource: context_engine::ResourceSummary) -> engine::ContextResource {
    engine::ContextResource {
        uri: resource.uri,
        title: resource.title,
        layer: resource.layer,
        metadata: resource.metadata.into_iter().collect(),
    }
}

/// Convert a `context_engine::SearchHit` into an `engine::ContextSearchResult`.
///
/// The returned value mirrors the input hit: `uri`, `document_id`, `chunk_text`,
/// `score`, and `layer` are copied directly, and `metadata` is collected into the
/// target container type.
///
/// # Examples
///
/// ```
/// let hit = context_engine::SearchHit {
///     uri: "urn:doc:1".to_string(),
///     document_id: "doc1".to_string(),
///     chunk_text: "chunk".to_string(),
///     score: 0.75,
///     metadata: vec![("k".to_string(), "v".to_string())].into_iter().collect(),
///     layer: 1,
/// };
/// let res = context_search_result(hit);
/// assert_eq!(res.uri, "urn:doc:1");
/// assert_eq!(res.document_id, "doc1");
/// assert_eq!(res.chunk_text, "chunk");
/// assert_eq!(res.score, 0.75);
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

/// Build a ContextSearchResult from a graph fact, enriching metadata and deriving display fields.
///
/// The returned result synthesizes:
/// - `uri`: uses the fact's `resource_uri` if present, otherwise `viking://graph/{edge_id}`
/// - `document_id`: the edge id
/// - `chunk_text`: "`{subject_name} {relation} {object_name}`"
/// - `score`: `1.0 - rank * 0.01`, clamped to a minimum of `0.1`
/// - `metadata`: the fact's metadata plus graph-specific keys (`kind`, `subject_id`, `subject_type`, `subject_name`, `relation`, `object_id`, `object_type`, `object_name`, and optionally `session_id`)
/// - `layer`: set to `L1`
///
/// # Examples
///
/// ```
/// // Given a `fact: GraphFactRecord` and a zero-based rank:
/// let result = context_graph_search_result(fact, 2);
/// assert!(result.chunk_text.contains(" "));
/// assert!(result.score >= 0.1 && result.score <= 1.0);
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

/// Converts a context engine file entry into the protobuf-compatible context file entry.
///
/// Copies `name`, `path`, `is_dir`, and `version`, and converts `size_bytes` to `i64`.
///
/// # Examples
///
/// ```
/// let fe = context_engine::FileEntry {
///     name: "file.txt".into(),
///     path: "dir/file.txt".into(),
///     is_dir: false,
///     size_bytes: 256,
///     version: 2,
/// };
/// let out = context_file_entry(fe);
/// assert_eq!(out.name, "file.txt");
/// assert_eq!(out.path, "dir/file.txt");
/// assert_eq!(out.is_dir, false);
/// assert_eq!(out.size_bytes, 256_i64);
/// assert_eq!(out.version, 2);
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

/// Converts a Context Engine `SessionEntry` into a protobuf `ContextSessionEntry`.
///
/// The returned struct preserves session id, role, content, metadata map, and converts
/// the entry's millisecond timestamp into a `Timestamp`.
///
/// # Examples
///
/// ```
/// use std::collections::HashMap;
/// let entry = context_engine::SessionEntry {
///     session_id: "sess-1".to_string(),
///     role: "user".to_string(),
///     content: "hello".to_string(),
///     metadata: {
///         let mut m = HashMap::new();
///         m.insert("lang".to_string(), "en".to_string());
///         m
///     },
///     created_at: 1_650_000_000_000i64, // milliseconds
/// };
///
/// let proto = context_session_entry(entry);
/// assert_eq!(proto.session_id, "sess-1");
/// assert_eq!(proto.role, "user");
/// assert_eq!(proto.content, "hello");
/// assert!(proto.metadata.contains_key("lang"));
/// assert!(proto.created_at.is_some());
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

/// Maps a protobuf numeric layer value to the corresponding `ResourceLayer`.
///
/// The mapping is: `1` -> `ResourceLayer::L0`, `2` -> `ResourceLayer::L1`, any other value -> `ResourceLayer::L2`.
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

/// Convert a numeric layer identifier into a `ResourceLayer`.
///
/// Maps `1` → `ResourceLayer::L0`, `2` → `ResourceLayer::L1`, and `3` → `ResourceLayer::L2`.
///
/// # Parameters
///
/// - `layer`: numeric layer identifier where 1, 2, and 3 correspond to L0, L1, and L2 respectively.
///
/// # Returns
///
/// `Some(ResourceLayer)` for recognized layer values, `None` for any other value.
///
/// # Examples
///
/// ```
/// let r = context_search_layer(2);
/// assert_eq!(r, Some(ResourceLayer::L1));
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

/// Create a protobuf `Timestamp` representing the given Unix epoch seconds.
///
/// The resulting `Timestamp` has `seconds` set to `seconds` and `nanos` set to `0`.
///
/// # Examples
///
/// ```
/// let ts = timestamp(1_620_000_000);
/// assert_eq!(ts.seconds, 1_620_000_000);
/// assert_eq!(ts.nanos, 0);
/// ```
fn timestamp(seconds: i64) -> Timestamp {
    Timestamp { seconds, nanos: 0 }
}

/// Convert milliseconds since the UNIX epoch to a `Timestamp` with second and nanosecond fields.

///

/// The input is interpreted as milliseconds since UNIX epoch; the resulting `Timestamp`

/// has `seconds` set to the whole seconds and `nanos` set to the remaining milliseconds

/// converted to nanoseconds.

///

/// # Examples

///

/// ```

/// let ts = timestamp_millis(1_500); // 1.5 seconds since epoch

/// assert_eq!(ts.seconds, 1);

/// assert_eq!(ts.nanos, 500_000_000);

/// ```
fn timestamp_millis(millis: i64) -> Timestamp {
    let seconds = millis.div_euclid(1000);
    let nanos = (millis.rem_euclid(1000) * 1_000_000) as i32;
    Timestamp { seconds, nanos }
}

/// Provides default tool schemas used for MCP connections.
///
/// The returned vector contains two predefined tools:
/// - `get_weather`: accepts a required `location` string parameter.
/// - `search_files`: accepts a required `query` string and an optional `path` string parameter.
///
/// # Examples
///
/// ```
/// let tools = default_tools();
/// let names: Vec<_> = tools.iter().map(|t| t.name.as_str()).collect();
/// assert_eq!(names, vec!["get_weather", "search_files"]);
/// ```
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

/// Get the current UNIX timestamp in seconds.
///
/// The value is the number of whole seconds elapsed since the UNIX epoch (1970-01-01 UTC).
/// If the system clock is earlier than the UNIX epoch or the system time cannot be represented,
/// the function returns `0`.
///
/// # Examples
///
/// ```
/// let ts = now();
/// assert!(ts >= 0);
/// ```
fn now() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64
}

/// Opens a ContextEngine configured from environment variables.
///
/// Reads configuration from the following environment variables:
/// - `CONTEXT_DATA_DIR` (default: `./context-data`)
/// - `CONTEXT_ROOTS` (default: `workspace=.`)
/// - Optional OpenViking bridge config via `CONTEXT_OPENVIKING_URL` and related `CONTEXT_OPENVIKING_*` vars
/// - Optional Dragonfly config via `CONTEXT_DRAGONFLY_*` vars
///
/// The function parses `CONTEXT_ROOTS` into managed roots, builds an optional OpenViking bridge
/// configuration when `CONTEXT_OPENVIKING_URL` is present, and supplies a Dragonfly configuration
/// if enabled via environment. It then opens and returns a `ContextEngine` initialized with
/// the assembled configuration.
///
/// # Returns
///
/// `Ok(ContextEngine)` on success, `Err(anyhow::Error)` if the engine cannot be opened or parsing fails.
///
/// # Examples
///
/// ```no_run
/// use std::env;
///
/// // Optionally set environment variables for testing:
/// env::set_var("CONTEXT_DATA_DIR", "./tmp-context");
/// env::set_var("CONTEXT_ROOTS", "workspace=./;notes=./notes");
///
/// // Open the context engine (async context required).
/// // let engine = tokio::runtime::Runtime::new().unwrap().block_on(open_context_engine_from_env()).unwrap();
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

/// Build a Dragonfly configuration from environment variables if any relevant option is set.
///
/// Reads the following environment variables:
/// - `CONTEXT_DRAGONFLY_ENABLED` — enable flag; empty, `"0"`, or `"false"` (case-insensitive) are treated as disabled.
/// - `CONTEXT_DRAGONFLY_ADDR` — optional address override.
/// - `CONTEXT_DRAGONFLY_KEY_PREFIX` — optional key prefix override.
/// - `CONTEXT_DRAGONFLY_RECENT_WINDOW` — optional positive integer override for recent-window.
///
/// If none of these variables are present (or enabled), returns `None`.
///
/// # Examples
///
/// ```
/// // Ensure a clean environment for the example.
/// std::env::remove_var("CONTEXT_DRAGONFLY_ENABLED");
/// std::env::remove_var("CONTEXT_DRAGONFLY_ADDR");
/// std::env::remove_var("CONTEXT_DRAGONFLY_KEY_PREFIX");
/// std::env::remove_var("CONTEXT_DRAGONFLY_RECENT_WINDOW");
///
/// // Set an explicit override so a config is produced.
/// std::env::set_var("CONTEXT_DRAGONFLY_ADDR", "127.0.0.1:8080");
/// let cfg = crate::dragonfly_config_from_env();
/// assert!(cfg.is_some());
/// let cfg = cfg.unwrap();
/// assert_eq!(cfg.addr, "127.0.0.1:8080");
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

/// Parses a semicolon-separated list of context root specifications into `ManagedRoot` entries.
///
/// Each non-empty entry may be either `name=path` or just `path`. Entries are trimmed of
/// surrounding whitespace and empty entries are ignored. If a name is omitted the first
/// unnamed root is given the name `"workspace"` and subsequent unnamed roots are named
/// `"root-<n>"` (1-based).
///
/// Returns `Ok(Vec<ManagedRoot>)` on success or the underlying `ManagedRoot::new` error if any
/// entry fails validation.
///
/// # Examples
///
/// ```
/// let roots = parse_context_roots("workspace=.;data=./data;./other").unwrap();
/// assert_eq!(roots.len(), 3);
/// assert_eq!(roots[0].name(), "workspace");
/// assert_eq!(roots[1].name(), "data");
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
            format!("root-{}", roots.len() + 1)
        };
        roots.push(ManagedRoot::new(default_name, PathBuf::from(entry.trim()))?);
    }
    Ok(roots)
}

#[tonic::async_trait]
impl ContextRpc for ContextGrpcService {
    /// Return the current status of the context engine, including document and chunk counts, index size, embedding model name, readiness, and managed roots.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// // given a `ContextGrpcService` instance `svc`
    /// let resp = svc.get_context_status(tonic::Request::new(())).await.unwrap();
    /// let status = resp.into_inner();
    /// println!("documents: {}, chunks: {}", status.document_count, status.chunk_count);
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

    /// Returns the list of context-managed resources known to the engine.
    ///
    /// The response contains a `ContextResourceList` whose `resources` field is populated
    /// from the engine's resource summaries.
    ///
    /// # Errors
    ///
    /// Returns a gRPC `Status::internal` if the underlying context engine fails to list resources.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use tonic::Request;
    /// # async fn example(svc: &crate::ContextGrpcService) {
    /// let resp = svc.list_resources(Request::new(())).await.unwrap();
    /// let list = resp.into_inner();
    /// println!("found {} resources", list.resources.len());
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

    /// Upserts (creates or updates) a resource in the context engine and returns the stored resource plus how many chunks were indexed.
    ///
    /// The request's `title` and `previous_uri` are treated as optional: empty strings become `None`.
    ///
    /// # Returns
    ///
    /// A response containing the stored resource and the number of text chunks the engine indexed for it.
    ///
    /// # Examples
    ///
    /// ```
    /// use tonic::Request;
    /// use engine::ContextUpsertResourceRequest;
    ///
    /// let req = ContextUpsertResourceRequest {
    ///     uri: "viking://resource/1".to_string(),
    ///     title: "My Resource".to_string(),
    ///     content: "Some content to index".to_string(),
    ///     layer: 0,
    ///     metadata: std::collections::HashMap::new(),
    ///     previous_uri: "".to_string(),
    /// };
    ///
    /// // `service.upsert_resource(Request::new(req)).await` calls the method shown here.
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

    /// Deletes a resource identified by `uri` from the context engine.
    ///
    /// The request's `uri` field specifies which resource to remove.
    ///
    /// # Returns
    ///
    /// `Ok(Response(()))` on success, `Err(Status)` if the deletion fails.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let req = engine::ContextDeleteResourceRequest { uri: "viking://workspace/doc".into() };
    /// let resp = context_service.delete_resource(tonic::Request::new(req)).await?;
    /// assert_eq!(resp.into_inner(), ());
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

    /// Searches the context index or graph facts for the given query and returns ranked results.
    ///
    /// If the request contains a filter "kind" = "graph" (case-insensitive), performs a graph facts
    /// lookup and returns graph-formatted search results. Otherwise performs a regular context search
    /// honoring optional `scope_uri`, `top_k`, `filters`, `layer`, and `rerank`.
    ///
    /// The response includes the list of matched results and the query time in milliseconds.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use tonic::Request;
    /// use crate::engine;
    ///
    /// // build a request
    /// let req = engine::ContextSearchRequest {
    ///     query: "find relevant context".to_string(),
    ///     scope_uri: "".to_string(),
    ///     top_k: 0,
    ///     filters: std::collections::HashMap::new(),
    ///     layer: 0,
    ///     rerank: false,
    /// };
    ///
    /// // call the service (async context required)
    /// // let resp = my_service.search_context(Request::new(req)).await.unwrap();
    /// // println!("found {} results", resp.get_ref().results.len());
    /// ```
    async fn search_context(
        &self,
        request: Request<engine::ContextSearchRequest>,
    ) -> Result<Response<engine::ContextSearchResponse>, Status> {
        let started_at = Instant::now();
        let request = request.into_inner();
        if request
            .filters
            .get("kind")
            .map(|value| value.eq_ignore_ascii_case("graph"))
            .unwrap_or(false)
        {
            let top_k = if request.top_k > 0 {
                request.top_k as usize
            } else {
                10
            };
            let results = self
                .engine
                .graph_facts(&request.query, top_k)
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
                top_k: (request.top_k > 0).then_some(request.top_k as usize),
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

    /// Synchronizes a workspace root (and optional path) and returns counts of affected resources.
    ///
    /// Calls the context engine to sync the given `root` and optional `path`, then returns the
    /// resulting summary including the root, prefix, and counts of indexed, reindexed, deleted,
    /// and skipped items.
    ///
    /// # Examples
    ///
    /// ```
    /// # async fn example(service: &crate::ContextGrpcService) -> Result<(), tonic::Status> {
    /// use tonic::Request;
    /// let req = crate::engine::ContextWorkspaceSyncRequest {
    ///     root: "workspace".into(),
    ///     path: ".".into(),
    /// };
    /// let resp = service.sync_workspace(Request::new(req)).await?;
    /// let summary = resp.into_inner();
    /// println!("Indexed: {}", summary.indexed_resources);
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

    /// List files under a managed root and path and return their metadata as proto entries.
    ///
    /// The request's `root` selects the managed workspace and `path` is interpreted relative to that root.
    /// On success returns a `ContextFileListResponse` whose `entries` are the engine's file listings
    /// converted to protobuf `ContextFileEntry` values. Engine errors are translated into gRPC `Status` (internal).
    ///
    /// # Examples
    ///
    /// ```
    /// // Example usage (async context):
    /// // let resp = service.list_files(Request::new(engine::ContextFileListRequest { root: "workspace".into(), path: ".".into() })).await?;
    /// // let list = resp.into_inner();
    /// // assert!(list.entries.len() >= 0);
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

    /// Reads a file from the specified managed root and returns its content and version.
    ///
    /// Returns a `ContextFileReadResponse` containing the requested `path`, the file `content`,
    /// and the `version` (0 if no version information is available).
    ///
    /// # Examples
    ///
    /// ```
    /// # async fn example(svc: crate::ContextGrpcService) {
    /// let req = engine::ContextFileReadRequest {
    ///     root: "workspace".into(),
    ///     path: "README.md".into(),
    /// };
    /// let resp = svc.read_file(tonic::Request::new(req)).await.unwrap().into_inner();
    /// assert_eq!(resp.path, "README.md");
    /// # }
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

    /// Writes content to a file under the specified managed root and returns the saved path and resulting version.
    ///
    /// The request's `root` selects the managed workspace root, `path` is the file path within that root,
    /// `content` is stored as the file body, and `version` is an optional expected version used by the engine.
    /// On success, the response contains the (possibly normalized) `path` and the file's new `version`.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use tonic::Request;
    /// # use crate::engine;
    /// # async fn example(svc: &super::ContextGrpcService) -> Result<(), Box<dyn std::error::Error>> {
    /// let req = engine::ContextFileWriteRequest {
    ///     root: "workspace".into(),
    ///     path: "notes/todo.txt".into(),
    ///     content: "buy milk".into(),
    ///     version: 0,
    /// };
    /// let resp = svc.write_file(Request::new(req)).await?;
    /// let body = resp.into_inner();
    /// println!("wrote {} v{}", body.path, body.version);
    /// # Ok(()) }
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

    /// Deletes a file in the specified managed root at the given path and returns whether the file was removed.
    ///
    /// The response includes the original `path` and a `deleted` flag that is `true` when a file was actually deleted.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use tonic::Request;
    ///
    /// // `svc` is a ContextGrpcService instance available in your runtime.
    /// let req = Request::new(engine::ContextFileDeleteRequest {
    ///     root: "workspace".into(),
    ///     path: "docs/readme.md".into(),
    ///     version: 0,
    /// });
    ///
    /// let resp = futures::executor::block_on(svc.delete_file(req)).unwrap();
    /// let out = resp.into_inner();
    /// assert_eq!(out.path, "docs/readme.md");
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

    /// Moves a file within the specified managed root and returns the moved paths and the file's resulting version.
    ///
    /// The request's `root` selects the workspace root; `from_path` and `to_path` are the source and destination paths within that root. On success the response contains the original `from_path`, the destination `to_path`, and the updated `version`.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let req = engine::ContextFileMoveRequest {
    ///     root: "workspace".into(),
    ///     from_path: "docs/old.md".into(),
    ///     to_path: "docs/new.md".into(),
    ///     version: 0,
    /// };
    /// let resp = service.move_file(tonic::Request::new(req)).await?;
    /// let body = resp.into_inner();
    /// assert_eq!(body.from_path, "docs/old.md");
    /// assert_eq!(body.to_path, "docs/new.md");
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
    /// The request's `session_id`, `role`, `content`, and `metadata` are forwarded to the
    /// context engine; after the append succeeds the engine's session entries are loaded
    /// and returned as a `ContextSessionHistory`.
    ///
    /// # Returns
    ///
    /// `ContextSessionHistory` containing the `session_id` and all session entries after the append.
    ///
    /// # Examples
    ///
    /// ```
    /// # use tonic::Request;
    /// # use crate::engine;
    /// # async fn _example(svc: &crate::ContextGrpcService) {
    /// let req = engine::ContextSessionAppendRequest {
    ///     session_id: "s1".to_string(),
    ///     role: "user".to_string(),
    ///     content: "hello".to_string(),
    ///     metadata: std::collections::HashMap::new(),
    /// };
    /// let resp = svc.append_session(Request::new(req)).await.unwrap();
    /// let history = resp.into_inner();
    /// assert_eq!(history.session_id, "s1");
    /// # }
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

    /// Fetches the full history for a context session identified by `session_id`.
    ///
    /// The response contains the requested `session_id` and a chronological list of session entries.
    ///
    /// # Returns
    ///
    /// `engine::ContextSessionHistory` containing the session ID and a list of session entries.
    ///
    /// # Examples
    ///
    /// ```
    /// # use tonic::Request;
    /// # use crate::engine;
    /// # async fn example(svc: &crate::ContextGrpcService) -> Result<(), tonic::Status> {
    /// let req = engine::ContextSessionGetRequest { session_id: "session-123".to_string() };
    /// let resp = svc.get_session(Request::new(req)).await?;
    /// let history = resp.into_inner();
    /// assert_eq!(history.session_id, "session-123");
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
