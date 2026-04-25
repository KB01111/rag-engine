use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{anyhow, Result};
use async_trait::async_trait;
use futures::{stream, Stream, StreamExt};
use storage::{EngineStore, ModelRecord};
use tokio::sync::RwLock;

pub type InferenceStream = Pin<Box<dyn Stream<Item = Result<InferenceChunk>> + Send>>;

#[derive(Clone)]
pub struct RuntimeEngine {
    store: EngineStore,
    models_path: PathBuf,
    backend: Arc<dyn RuntimeBackend>,
}

#[derive(Debug, Clone)]
pub struct RuntimeInferenceRequest {
    pub model_id: String,
    pub prompt: String,
    pub parameters: HashMap<String, String>,
    pub context_refs: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct InferenceChunk {
    pub token: String,
    pub complete: bool,
    pub metrics: HashMap<String, String>,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct RuntimeParameters {
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<usize>,
    pub max_tokens: Option<usize>,
    pub seed: Option<u64>,
    pub stop: Vec<String>,
    pub truncate_sequence: Option<bool>,
    pub repetition_penalty: Option<f32>,
}

#[async_trait]
pub trait RuntimeBackend: Send + Sync {
    fn name(&self) -> &str;
    async fn list_models(&self, models: Vec<ModelRecord>) -> Result<Vec<ModelRecord>>;
    async fn load_model(&self, model: &ModelRecord) -> Result<()>;
    async fn unload_model(&self, model_id: &str) -> Result<()>;
    async fn stream_inference(
        &self,
        model: &ModelRecord,
        request: RuntimeInferenceRequest,
        parameters: RuntimeParameters,
    ) -> Result<InferenceStream>;
}

impl RuntimeEngine {
    pub fn new(
        store: EngineStore,
        models_path: impl Into<PathBuf>,
        backend: impl Into<String>,
    ) -> Self {
        Self::with_backend_name(store, models_path, backend.into())
    }

    pub fn with_backend_name(
        store: EngineStore,
        models_path: impl Into<PathBuf>,
        backend: impl Into<String>,
    ) -> Self {
        let backend = create_backend(backend.into());
        Self {
            store,
            models_path: models_path.into(),
            backend,
        }
    }

    pub fn with_backend(
        store: EngineStore,
        models_path: impl Into<PathBuf>,
        backend: Arc<dyn RuntimeBackend>,
    ) -> Self {
        Self {
            store,
            models_path: models_path.into(),
            backend,
        }
    }

    pub fn backend_name(&self) -> &str {
        self.backend.name()
    }

    pub async fn list_models(&self) -> Result<Vec<ModelRecord>> {
        self.discover_models().await?;
        self.backend
            .list_models(self.store.list_models().await?)
            .await
    }

    pub async fn load_model(&self, model_id: &str) -> Result<ModelRecord> {
        let mut models = self.list_models().await?;
        let Some(model) = models.iter_mut().find(|model| model.id == model_id) else {
            return Err(anyhow!("model not found: {model_id}"));
        };

        self.backend.load_model(model).await?;
        model.status = "loaded".to_string();
        model.backend = self.backend.name().to_string();
        model.updated_at = now();
        self.store.upsert_model(model.clone()).await?;
        Ok(model.clone())
    }

    pub async fn unload_model(&self, model_id: &str) -> Result<()> {
        let mut models = self.list_models().await?;
        let Some(model) = models.iter_mut().find(|model| model.id == model_id) else {
            return Err(anyhow!("model not found: {model_id}"));
        };

        self.backend.unload_model(model_id).await?;
        model.status = "discovered".to_string();
        model.backend = self.backend.name().to_string();
        model.updated_at = now();
        self.store.upsert_model(model.clone()).await?;
        Ok(())
    }

    pub async fn stream_inference(
        &self,
        request: RuntimeInferenceRequest,
    ) -> Result<Vec<InferenceChunk>> {
        let stream = self.stream_inference_stream(request).await?;
        stream.collect::<Vec<_>>().await.into_iter().collect()
    }

    pub async fn stream_inference_stream(
        &self,
        request: RuntimeInferenceRequest,
    ) -> Result<InferenceStream> {
        let parameters = RuntimeParameters::parse(&request.parameters)?;
        let model = self.ensure_loaded_model(&request.model_id).await?;
        self.backend
            .stream_inference(&model, request, parameters)
            .await
    }

    async fn ensure_loaded_model(&self, model_id: &str) -> Result<ModelRecord> {
        let models = self.list_models().await?;
        let Some(model) = models.into_iter().find(|model| model.id == model_id) else {
            return Err(anyhow!("model not found: {model_id}"));
        };

        if model.status == "loaded" {
            return Ok(model);
        }

        self.load_model(model_id).await
    }

    async fn discover_models(&self) -> Result<()> {
        tokio::fs::create_dir_all(&self.models_path).await?;

        let existing_models = self.store.list_models().await.unwrap_or_default();
        let mut existing_map: HashMap<String, ModelRecord> = existing_models
            .into_iter()
            .map(|model| (model.id.clone(), model))
            .collect();

        let models_path = self.models_path.clone();
        let discovered = tokio::task::spawn_blocking(move || {
            let mut models = Vec::new();
            if let Ok(entries) = fs::read_dir(&models_path) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if !is_model_file(&path) {
                        continue;
                    }
                    if let Ok(metadata) = entry.metadata() {
                        let name = entry.file_name().to_string_lossy().to_string();
                        models.push((name, path, metadata.len()));
                    }
                }
            }
            models
        })
        .await?;

        for (name, path, size_bytes) in discovered {
            let existing = existing_map.remove(&name);
            let status = existing
                .as_ref()
                .map(|m| m.status.clone())
                .unwrap_or_else(|| "discovered".to_string());
            let metadata_json = existing
                .as_ref()
                .map(|m| m.metadata_json.clone())
                .unwrap_or_else(|| "{}".to_string());

            self.store
                .upsert_model(ModelRecord {
                    id: name.clone(),
                    name: name.clone(),
                    path: path.to_string_lossy().to_string(),
                    backend: self.backend.name().to_string(),
                    status,
                    metadata_json,
                    size_bytes: size_bytes as i64,
                    updated_at: now(),
                })
                .await?;
        }
        Ok(())
    }
}

impl RuntimeParameters {
    pub fn parse(input: &HashMap<String, String>) -> Result<Self> {
        Ok(Self {
            temperature: parse_optional_f32(input, "temperature")?,
            top_p: parse_optional_f32(input, "top_p")?,
            top_k: parse_optional_usize(input, "top_k")?,
            max_tokens: parse_optional_usize(input, "max_tokens")?,
            seed: parse_optional_u64(input, "seed")?,
            stop: parse_stop(input.get("stop"))?,
            truncate_sequence: parse_optional_bool(input, "truncate_sequence")?,
            repetition_penalty: parse_optional_f32(input, "repetition_penalty")?,
        })
    }
}

#[derive(Debug)]
pub struct MockBackend {
    name: String,
}

impl MockBackend {
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into() }
    }
}

#[async_trait]
impl RuntimeBackend for MockBackend {
    fn name(&self) -> &str {
        &self.name
    }

    async fn list_models(&self, models: Vec<ModelRecord>) -> Result<Vec<ModelRecord>> {
        Ok(models)
    }

    async fn load_model(&self, _model: &ModelRecord) -> Result<()> {
        Ok(())
    }

    async fn unload_model(&self, _model_id: &str) -> Result<()> {
        Ok(())
    }

    async fn stream_inference(
        &self,
        model: &ModelRecord,
        request: RuntimeInferenceRequest,
        parameters: RuntimeParameters,
    ) -> Result<InferenceStream> {
        let tokens = mock_tokens(&request.prompt);
        let total = tokens.len();
        let chunks = tokens
            .into_iter()
            .enumerate()
            .map(|(index, token)| {
                Ok(InferenceChunk {
                    token,
                    complete: index + 1 == total,
                    metrics: runtime_metrics(self.name(), model, &request, &parameters),
                })
            })
            .collect::<Vec<_>>();
        Ok(Box::pin(stream::iter(chunks)))
    }
}

#[derive(Debug)]
struct UnavailableBackend {
    name: String,
    message: String,
}

#[async_trait]
impl RuntimeBackend for UnavailableBackend {
    fn name(&self) -> &str {
        &self.name
    }

    async fn list_models(&self, models: Vec<ModelRecord>) -> Result<Vec<ModelRecord>> {
        Ok(models)
    }

    async fn load_model(&self, _model: &ModelRecord) -> Result<()> {
        Err(anyhow!("{}", self.message))
    }

    async fn unload_model(&self, _model_id: &str) -> Result<()> {
        Ok(())
    }

    async fn stream_inference(
        &self,
        _model: &ModelRecord,
        _request: RuntimeInferenceRequest,
        _parameters: RuntimeParameters,
    ) -> Result<InferenceStream> {
        Err(anyhow!("{}", self.message))
    }
}

#[cfg(feature = "mistralrs-backend")]
mod mistralrs_backend {
    use super::*;
    use mistralrs::{
        ChatCompletionChunkResponse, Model, ModelBuilder, Response, TextMessageRole, TextMessages,
    };

    pub struct MistralRsBackend {
        models: RwLock<HashMap<String, Arc<Model>>>,
    }

    impl MistralRsBackend {
        pub fn new() -> Self {
            Self {
                models: RwLock::new(HashMap::new()),
            }
        }

        async fn get_or_load(&self, model: &ModelRecord) -> Result<Arc<Model>> {
            if let Some(existing) = self.models.read().await.get(&model.id).cloned() {
                return Ok(existing);
            }

            self.load_model(model).await?;
            self.models
                .read()
                .await
                .get(&model.id)
                .cloned()
                .ok_or_else(|| {
                    anyhow!("mistralrs model was not available after load: {}", model.id)
                })
        }
    }

    #[async_trait]
    impl RuntimeBackend for MistralRsBackend {
        fn name(&self) -> &str {
            "mistralrs"
        }

        async fn list_models(&self, models: Vec<ModelRecord>) -> Result<Vec<ModelRecord>> {
            Ok(models)
        }

        async fn load_model(&self, model: &ModelRecord) -> Result<()> {
            let loaded = ModelBuilder::new(model.path.clone()).build().await?;
            self.models
                .write()
                .await
                .insert(model.id.clone(), Arc::new(loaded));
            Ok(())
        }

        async fn unload_model(&self, model_id: &str) -> Result<()> {
            self.models.write().await.remove(model_id);
            Ok(())
        }

        async fn stream_inference(
            &self,
            model: &ModelRecord,
            request: RuntimeInferenceRequest,
            parameters: RuntimeParameters,
        ) -> Result<InferenceStream> {
            let loaded = self.get_or_load(model).await?;
            let messages =
                TextMessages::new().add_message(TextMessageRole::User, build_prompt(&request));
            let stream = loaded.stream_chat_request(messages).await?;
            let metrics = runtime_metrics(self.name(), model, &request, &parameters);
            Ok(Box::pin(stream.filter_map(move |response| {
                let metrics = metrics.clone();
                async move { mistralrs_response_chunk(response, metrics) }
            })))
        }
    }

    pub fn backend() -> Arc<dyn RuntimeBackend> {
        Arc::new(MistralRsBackend::new())
    }

    fn mistralrs_response_chunk(
        response: Response,
        metrics: HashMap<String, String>,
    ) -> Option<Result<InferenceChunk>> {
        match response {
            Response::Chunk(ChatCompletionChunkResponse { choices, .. }) => {
                let token = choices
                    .into_iter()
                    .filter_map(|choice| choice.delta.content)
                    .collect::<Vec<_>>()
                    .join("");
                if token.is_empty() {
                    None
                } else {
                    Some(Ok(InferenceChunk {
                        token,
                        complete: false,
                        metrics,
                    }))
                }
            }
            Response::Done(_) => Some(Ok(InferenceChunk {
                token: String::new(),
                complete: true,
                metrics,
            })),
            Response::ModelError(message, _) => Some(Err(anyhow!(message))),
            Response::CompletionModelError(message, _) => Some(Err(anyhow!(message))),
            Response::ValidationError(message) => Some(Err(anyhow!(message))),
            Response::InternalError(message) => Some(Err(anyhow!(message))),
            _ => None,
        }
    }
}

fn create_backend(name: String) -> Arc<dyn RuntimeBackend> {
    match normalize_backend_name(&name).as_str() {
        "mock" => Arc::new(MockBackend::new("mock")),
        "mistralrs" => create_mistralrs_backend(),
        other => Arc::new(UnavailableBackend {
            name: other.to_string(),
            message: format!("unsupported runtime backend: {other}"),
        }),
    }
}

#[cfg(feature = "mistralrs-backend")]
fn create_mistralrs_backend() -> Arc<dyn RuntimeBackend> {
    mistralrs_backend::backend()
}

#[cfg(not(feature = "mistralrs-backend"))]
fn create_mistralrs_backend() -> Arc<dyn RuntimeBackend> {
    Arc::new(UnavailableBackend {
        name: "mistralrs".to_string(),
        message: "runtime backend mistralrs requires building runtime_engine with the mistralrs-backend feature".to_string(),
    })
}

fn normalize_backend_name(name: &str) -> String {
    match name.trim().to_ascii_lowercase().as_str() {
        "" => "mistralrs".to_string(),
        "mistral.rs" | "mistral_rs" | "mistral-rs" => "mistralrs".to_string(),
        other => other.to_string(),
    }
}

fn runtime_metrics(
    backend: &str,
    model: &ModelRecord,
    request: &RuntimeInferenceRequest,
    parameters: &RuntimeParameters,
) -> HashMap<String, String> {
    let mut metrics = HashMap::from([
        ("backend".to_string(), backend.to_string()),
        ("model".to_string(), model.id.clone()),
        (
            "context_ref_count".to_string(),
            request.context_refs.len().to_string(),
        ),
    ]);
    if let Some(max_tokens) = parameters.max_tokens {
        metrics.insert("max_tokens".to_string(), max_tokens.to_string());
    }
    metrics
}

fn build_prompt(request: &RuntimeInferenceRequest) -> String {
    if request.context_refs.is_empty() {
        return request.prompt.clone();
    }
    format!(
        "Context references:\n- {}\n\n{}",
        request.context_refs.join("\n- "),
        request.prompt
    )
}

fn mock_tokens(prompt: &str) -> Vec<String> {
    let trimmed = prompt.trim();
    if trimmed.is_empty() {
        return vec![
            "No".to_string(),
            "prompt".to_string(),
            "provided.".to_string(),
        ];
    }

    let tokens = trimmed
        .split_whitespace()
        .map(String::from)
        .collect::<Vec<_>>();
    let mut result = Vec::with_capacity(tokens.len() * 2 - 1);
    for (index, token) in tokens.into_iter().enumerate() {
        if index > 0 {
            result.push(" ".to_string());
        }
        result.push(token);
    }
    result
}

fn parse_optional_f32(input: &HashMap<String, String>, key: &str) -> Result<Option<f32>> {
    input
        .get(key)
        .map(|value| {
            value.parse::<f32>().map_err(|_| {
                anyhow!("invalid runtime parameter {key}: expected float, got {value:?}")
            })
        })
        .transpose()
}

fn parse_optional_usize(input: &HashMap<String, String>, key: &str) -> Result<Option<usize>> {
    input
        .get(key)
        .map(|value| {
            value.parse::<usize>().map_err(|_| {
                anyhow!("invalid runtime parameter {key}: expected positive integer, got {value:?}")
            })
        })
        .transpose()
}

fn parse_optional_u64(input: &HashMap<String, String>, key: &str) -> Result<Option<u64>> {
    input
        .get(key)
        .map(|value| {
            value.parse::<u64>().map_err(|_| {
                anyhow!("invalid runtime parameter {key}: expected unsigned integer, got {value:?}")
            })
        })
        .transpose()
}

fn parse_optional_bool(input: &HashMap<String, String>, key: &str) -> Result<Option<bool>> {
    input
        .get(key)
        .map(|value| {
            value.parse::<bool>().map_err(|_| {
                anyhow!("invalid runtime parameter {key}: expected bool, got {value:?}")
            })
        })
        .transpose()
}

fn parse_stop(value: Option<&String>) -> Result<Vec<String>> {
    let Some(value) = value else {
        return Ok(Vec::new());
    };
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return Ok(Vec::new());
    }
    if trimmed.starts_with('[') && trimmed.ends_with(']') {
        return serde_json::from_str::<Vec<String>>(trimmed).map_err(|_| {
            anyhow!("invalid runtime parameter stop: expected string or JSON string array")
        });
    }
    Ok(vec![trimmed.to_string()])
}

fn is_model_file(path: &Path) -> bool {
    matches!(
        path.extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| ext.to_ascii_lowercase()),
        Some(ext) if matches!(ext.as_str(), "bin" | "gguf" | "ggml")
    )
}

fn now() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    use tempfile::tempdir;

    async fn test_engine(backend: &str) -> (RuntimeEngine, tempfile::TempDir, tempfile::TempDir) {
        let store_dir = tempdir().unwrap();
        let model_dir = tempdir().unwrap();
        tokio::fs::write(model_dir.path().join("local.gguf"), b"model")
            .await
            .unwrap();

        let store = EngineStore::new(store_dir.path().to_string_lossy().to_string());
        let engine = RuntimeEngine::new(store, model_dir.path(), backend);
        (engine, store_dir, model_dir)
    }

    #[tokio::test]
    async fn load_model_updates_store_status() {
        let (engine, _store_dir, _model_dir) = test_engine("mock").await;

        let loaded = engine.load_model("local.gguf").await.unwrap();

        assert_eq!(loaded.id, "local.gguf");
        assert_eq!(loaded.backend, "mock");
        assert_eq!(loaded.status, "loaded");
        assert!(engine
            .list_models()
            .await
            .unwrap()
            .into_iter()
            .any(|model| model.id == "local.gguf" && model.status == "loaded"));
    }

    #[tokio::test]
    async fn unload_model_clears_loaded_status() {
        let (engine, _store_dir, _model_dir) = test_engine("mock").await;
        engine.load_model("local.gguf").await.unwrap();

        engine.unload_model("local.gguf").await.unwrap();

        let model = engine
            .list_models()
            .await
            .unwrap()
            .into_iter()
            .find(|model| model.id == "local.gguf")
            .unwrap();
        assert_eq!(model.status, "discovered");
    }

    #[tokio::test]
    async fn stream_inference_emits_ordered_chunks_and_completion() {
        let (engine, _store_dir, _model_dir) = test_engine("mock").await;
        engine.load_model("local.gguf").await.unwrap();

        let chunks = engine
            .stream_inference(RuntimeInferenceRequest {
                model_id: "local.gguf".to_string(),
                prompt: "hello world".to_string(),
                parameters: HashMap::new(),
                context_refs: vec!["viking://resources/doc".to_string()],
            })
            .await
            .unwrap();

        assert_eq!(
            chunks
                .iter()
                .map(|chunk| chunk.token.as_str())
                .collect::<Vec<_>>(),
            vec!["hello", " ", "world"]
        );
        assert!(chunks.last().unwrap().complete);
        assert_eq!(chunks.last().unwrap().metrics["backend"], "mock");
    }

    #[tokio::test]
    async fn stream_inference_lazy_loads_discovered_model() {
        let (engine, _store_dir, _model_dir) = test_engine("mock").await;
        engine.list_models().await.unwrap();

        let chunks = engine
            .stream_inference(RuntimeInferenceRequest {
                model_id: "local.gguf".to_string(),
                prompt: "lazy".to_string(),
                parameters: HashMap::new(),
                context_refs: Vec::new(),
            })
            .await
            .unwrap();

        assert_eq!(chunks[0].token, "lazy");
        let model = engine
            .list_models()
            .await
            .unwrap()
            .into_iter()
            .find(|model| model.id == "local.gguf")
            .unwrap();
        assert_eq!(model.status, "loaded");
    }

    #[tokio::test]
    async fn stream_inference_returns_not_found_for_missing_model() {
        let (engine, _store_dir, _model_dir) = test_engine("mock").await;

        let error = engine
            .stream_inference(RuntimeInferenceRequest {
                model_id: "missing.gguf".to_string(),
                prompt: "hello".to_string(),
                parameters: HashMap::new(),
                context_refs: Vec::new(),
            })
            .await
            .unwrap_err();

        assert!(error.to_string().contains("model not found: missing.gguf"));
    }

    #[tokio::test]
    async fn stream_inference_rejects_malformed_known_parameters() {
        let (engine, _store_dir, _model_dir) = test_engine("mock").await;
        engine.load_model("local.gguf").await.unwrap();

        let error = engine
            .stream_inference(RuntimeInferenceRequest {
                model_id: "local.gguf".to_string(),
                prompt: "hello".to_string(),
                parameters: HashMap::from([("temperature".to_string(), "warm".to_string())]),
                context_refs: Vec::new(),
            })
            .await
            .unwrap_err();

        assert!(error
            .to_string()
            .contains("invalid runtime parameter temperature"));
    }

    #[test]
    fn parameter_parser_accepts_json_stop_and_ignores_unknown_keys() {
        let parameters = RuntimeParameters::parse(&HashMap::from([
            ("stop".to_string(), "[\"</s>\",\"END\"]".to_string()),
            ("future_option".to_string(), "kept-for-later".to_string()),
        ]))
        .unwrap();

        assert_eq!(parameters.stop, vec!["</s>", "END"]);
    }

    #[test]
    fn parameter_parser_rejects_malformed_stop_array() {
        let error = RuntimeParameters::parse(&HashMap::from([(
            "stop".to_string(),
            "[\"unterminated\"".to_string(),
        )]))
        .unwrap_err();

        assert!(error.to_string().contains("invalid runtime parameter stop"));
    }
}
