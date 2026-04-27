use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{anyhow, Result};
use async_trait::async_trait;
use futures::{stream, Stream, StreamExt};
use serde::{Deserialize, Serialize};
use storage::{EngineStore, ModelRecord};
use tokio::sync::RwLock;

pub type InferenceStream = Pin<Box<dyn Stream<Item = Result<InferenceChunk>> + Send>>;

#[derive(Debug, thiserror::Error)]
pub enum RuntimeError {
    #[error("invalid runtime parameter {parameter}: {details}")]
    InvalidParameter { parameter: String, details: String },
    #[error("model not found: {model_id}")]
    ModelNotFound { model_id: String },
    #[error("cloud backend not configured for model {model_id}")]
    CloudBackendUnavailable { model_id: String },
    #[error("{0}")]
    Other(#[from] anyhow::Error),
}

/// Persisted-as-JSON configuration for a cloud-hosted model.
///
/// Stashed inside `ModelRecord.metadata_json` under the `cloud` key so we
/// don't need to evolve the storage schema for v1. When `cloud` is present,
/// the engine routes inference through the cloud backend instead of the
/// local one.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CloudModelConfig {
    pub provider: String,
    pub base_url: String,
    pub api_key_env: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub deployment: Option<String>,
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub extra_headers: HashMap<String, String>,
    /// Model id of a fallback model to retry once when this model fails to
    /// start streaming (load failure, context overflow, OOM-like errors).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fallback_model_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct ModelMetadata {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    cloud: Option<CloudModelConfig>,
    #[serde(default, flatten)]
    extra: serde_json::Map<String, serde_json::Value>,
}

impl ModelMetadata {
    fn parse(raw: &str) -> Self {
        if raw.trim().is_empty() {
            return Self::default();
        }
        match serde_json::from_str(raw) {
            Ok(metadata) => metadata,
            Err(err) => {
                let length = raw.chars().count();
                eprintln!("warning: failed to parse model metadata JSON: {} (input length: {} chars)", err, length);
                Self::default()
            }
        }
    }

    fn to_json(&self) -> String {
        serde_json::to_string(self).unwrap_or_else(|_| "{}".to_string())
    }
}

pub(crate) fn cloud_config_from_metadata(metadata_json: &str) -> Option<CloudModelConfig> {
    ModelMetadata::parse(metadata_json).cloud
}

#[derive(Debug, Clone, Default)]
pub struct MistralRsConfig {
    pub force_cpu: bool,
    pub max_num_seqs: Option<usize>,
    pub auto_isq: Option<String>,
    pub paged_attn_block_size: Option<usize>,
    pub paged_attn_gpu_mem_ctx: Option<usize>,
    pub paged_attn_cache_dtype: Option<String>,
}

#[derive(Clone)]
pub struct RuntimeEngine {
    store: EngineStore,
    models_path: PathBuf,
    backend: Arc<dyn RuntimeBackend>,
    cloud_backend: Option<Arc<dyn RuntimeBackend>>,
    mistralrs_config: MistralRsConfig,
}

#[derive(Debug, Clone)]
pub struct RuntimeInferenceRequest {
    pub model_id: String,
    pub prompt: String,
    pub system_prompt: Option<String>,
    pub parameters: HashMap<String, String>,
    pub context_refs: Vec<String>,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct LoadModelOptions {
    pub max_num_seqs: Option<usize>,
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
    async fn load_model(&self, model: &ModelRecord, options: LoadModelOptions) -> Result<()>;
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
        _llama_cli: impl Into<String>,
        mistralrs_config: MistralRsConfig,
    ) -> Self {
        Self::with_backend_name(store, models_path, backend.into(), mistralrs_config)
    }

    pub fn with_backend_name(
        store: EngineStore,
        models_path: impl Into<PathBuf>,
        backend: impl Into<String>,
        mistralrs_config: MistralRsConfig,
    ) -> Self {
        let backend = create_backend(backend.into(), mistralrs_config.clone());
        let cloud_backend = create_openai_backend();
        Self {
            store,
            models_path: models_path.into(),
            backend,
            cloud_backend,
            mistralrs_config,
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
            cloud_backend: None,
            mistralrs_config: MistralRsConfig::default(),
        }
    }

    /// Set or replace the cloud backend used for models that have a
    /// `CloudModelConfig` in their metadata. Useful in tests for injecting a
    /// stub backend pointed at a wiremock server.
    pub fn with_cloud_backend(mut self, cloud_backend: Arc<dyn RuntimeBackend>) -> Self {
        self.cloud_backend = Some(cloud_backend);
        self
    }

    pub fn backend_name(&self) -> &str {
        self.backend.name()
    }

    pub fn cloud_backend_name(&self) -> Option<&str> {
        self.cloud_backend.as_ref().map(|b| b.name())
    }

    /// Pick the backend that should service requests for `model`. If the
    /// model carries a `CloudModelConfig` in its metadata, dispatch to the
    /// cloud backend; otherwise fall back to the configured local backend.
    fn select_backend(&self, model: &ModelRecord) -> Result<Arc<dyn RuntimeBackend>> {
        if cloud_config_from_metadata(&model.metadata_json).is_some() {
            return self
                .cloud_backend
                .clone()
                .ok_or_else(|| RuntimeError::CloudBackendUnavailable {
                    model_id: model.id.clone(),
                }
                .into());
        }
        Ok(self.backend.clone())
    }

    /// Register (or update) a cloud-hosted model in the store. The model
    /// becomes addressable by `model_id` for inference. Cloud-config is
    /// persisted in `metadata_json` so backend selection survives restarts.
    pub async fn register_cloud_model(
        &self,
        model_id: impl Into<String>,
        display_name: impl Into<String>,
        config: CloudModelConfig,
    ) -> Result<ModelRecord> {
        let model_id = model_id.into();
        let display_name = display_name.into();
        let mut metadata = ModelMetadata::default();
        metadata.cloud = Some(config);
        let backend_name = self
            .cloud_backend
            .as_ref()
            .ok_or_else(|| RuntimeError::CloudBackendUnavailable {
                model_id: model_id.clone(),
            })?
            .name()
            .to_string();
        let record = ModelRecord {
            id: model_id.clone(),
            name: display_name,
            path: String::new(),
            backend: backend_name,
            status: "registered".to_string(),
            metadata_json: metadata.to_json(),
            size_bytes: 0,
            updated_at: now(),
        };
        self.store.upsert_model(record.clone()).await?;
        Ok(record)
    }

    pub async fn list_models(&self) -> Result<Vec<ModelRecord>> {
        self.discover_models().await?;
        self.backend
            .list_models(self.store.list_models().await?)
            .await
    }

    pub async fn load_model(&self, model_id: &str) -> Result<ModelRecord> {
        self.load_model_with_options(model_id, HashMap::new()).await
    }

    pub async fn load_model_with_options(
        &self,
        model_id: &str,
        options: HashMap<String, String>,
    ) -> Result<ModelRecord> {
        let mut models = self.list_models().await?;
        let Some(model) = models.iter_mut().find(|model| model.id == model_id) else {
            return Err(RuntimeError::ModelNotFound {
                model_id: model_id.to_string(),
            }
            .into());
        };

        self.load_model_from_record(model.clone(), LoadModelOptions::parse(&options)?)
            .await
    }

    async fn load_model_from_record(
        &self,
        mut model: ModelRecord,
        options: LoadModelOptions,
    ) -> Result<ModelRecord> {
        let backend = self.select_backend(&model)?;
        backend.load_model(&model, options).await?;
        model.status = "loaded".to_string();
        model.backend = backend.name().to_string();
        model.updated_at = now();
        self.store.upsert_model(model.clone()).await?;
        Ok(model)
    }

    pub async fn unload_model(&self, model_id: &str) -> Result<()> {
        let mut models = self.list_models().await?;
        let Some(model) = models.iter_mut().find(|model| model.id == model_id) else {
            return Err(RuntimeError::ModelNotFound {
                model_id: model_id.to_string(),
            }
            .into());
        };

        let backend = self.select_backend(model)?;
        backend.unload_model(model_id).await?;
        model.status = "discovered".to_string();
        model.backend = backend.name().to_string();
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
        let backend = self.select_backend(&model)?;
        let outcome = backend
            .stream_inference(&model, request.clone(), parameters.clone())
            .await;
        match outcome {
            Ok(stream) => Ok(stream),
            Err(err) => {
                let fallback_id =
                    cloud_config_from_metadata(&model.metadata_json).and_then(|c| c.fallback_model_id);
                if let Some(fallback_id) = fallback_id {
                    self.try_fallback(request, fallback_id, err).await
                } else {
                    Err(err)
                }
            }
        }
    }

    async fn try_fallback(
        &self,
        mut request: RuntimeInferenceRequest,
        fallback_id: String,
        original_error: anyhow::Error,
    ) -> Result<InferenceStream> {
        request.model_id = fallback_id;
        let parameters = RuntimeParameters::parse(&request.parameters)?;
        let model = self
            .ensure_loaded_model(&request.model_id)
            .await
            .map_err(|e| anyhow!("primary failed: {original_error}; fallback resolution failed: {e}"))?;
        let backend = self
            .select_backend(&model)
            .map_err(|e| anyhow!("primary failed: {original_error}; fallback backend select failed: {e}"))?;
        backend
            .stream_inference(&model, request, parameters)
            .await
            .map_err(|e| anyhow!("primary failed: {original_error}; fallback also failed: {e}"))
    }

    async fn ensure_loaded_model(&self, model_id: &str) -> Result<ModelRecord> {
        let models = self.list_models().await?;
        let Some(model) = models.into_iter().find(|model| model.id == model_id) else {
            return Err(RuntimeError::ModelNotFound {
                model_id: model_id.to_string(),
            }
            .into());
        };

        if model.status == "loaded" {
            return Ok(model);
        }

        self.load_model_from_record(model, LoadModelOptions::default())
            .await
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

    async fn load_model(&self, _model: &ModelRecord, _options: LoadModelOptions) -> Result<()> {
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

    async fn load_model(&self, _model: &ModelRecord, _options: LoadModelOptions) -> Result<()> {
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
        ChatCompletionChunkResponse, IsqType, Model, ModelBuilder, RequestBuilder, Response,
        SamplingParams, StopTokens, TextMessageRole,
    };
    #[cfg(feature = "mistralrs-cuda")]
    use mistralrs::{MemoryGpuConfig, PagedAttentionMetaBuilder, PagedCacheType};

    pub struct MistralRsBackend {
        models: RwLock<HashMap<String, Arc<Model>>>,
        config: MistralRsConfig,
    }

    impl MistralRsBackend {
        pub fn new(config: MistralRsConfig) -> Self {
            Self {
                models: RwLock::new(HashMap::new()),
                config,
            }
        }

        async fn get_or_load(&self, model: &ModelRecord) -> Result<Arc<Model>> {
            // Fast path: check if already loaded
            if let Some(existing) = self.models.read().await.get(&model.id).cloned() {
                return Ok(existing);
            }

            // Acquire write lock to prevent concurrent loads
            let mut models = self.models.write().await;

            // Check again after acquiring write lock (double-checked locking)
            if let Some(existing) = models.get(&model.id).cloned() {
                return Ok(existing);
            }

            // Load the model using the shared builder
            let loaded = Arc::new(self.build_model(&model.path, &self.config).await?);
            models.insert(model.id.clone(), loaded.clone());
            Ok(loaded)
        }

        fn config_with_options(&self, options: &LoadModelOptions) -> MistralRsConfig {
            let mut config = self.config.clone();
            if let Some(max_num_seqs) = options.max_num_seqs {
                config.max_num_seqs = Some(max_num_seqs);
            }
            config
        }

        async fn build_model(&self, model_path: &str, config: &MistralRsConfig) -> Result<Model> {
            let mut builder = ModelBuilder::new(model_path.to_string());

            // Apply MistralRS configuration
            if config.force_cpu {
                builder = builder.with_force_cpu();
            }
            if let Some(max_num_seqs) = config.max_num_seqs {
                builder = builder.with_max_num_seqs(max_num_seqs);
            }
            if let Some(ref auto_isq) = config.auto_isq {
                let isq_type = parse_isq_type(auto_isq)?;
                builder = builder.with_isq(isq_type);
            }

            #[cfg(feature = "mistralrs-cuda")]
            {
                if paged_attention_configured(config) {
                    let context_size = config.paged_attn_gpu_mem_ctx.unwrap_or(8192);
                    let cache_type =
                        parse_paged_cache_type(config.paged_attn_cache_dtype.as_deref())?;
                    let mut paged = PagedAttentionMetaBuilder::default()
                        .with_gpu_memory(MemoryGpuConfig::ContextSize(context_size))
                        .with_paged_cache_type(cache_type);
                    if let Some(block_size) = config.paged_attn_block_size {
                        paged = paged.with_block_size(block_size);
                    }
                    builder = builder.with_paged_attn(paged.build()?);
                }
            }

            builder.build().await
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

        async fn load_model(&self, model: &ModelRecord, options: LoadModelOptions) -> Result<()> {
            let config = self.config_with_options(&options);
            let loaded = self.build_model(&model.path, &config).await?;
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
            validate_embedded_parameters(&parameters)?;
            let loaded = self.get_or_load(model).await?;
            let (system_message, user_message) = build_message_parts(&request);
            let mut builder =
                RequestBuilder::new().set_sampling(build_sampling_params(&parameters));
            if let Some(system_message) = system_message {
                builder = builder.add_message(TextMessageRole::System, system_message);
            }
            builder = builder.add_message(TextMessageRole::User, user_message);
            let stream = loaded.stream_chat_request(builder).await?;
            let metrics = runtime_metrics(self.name(), model, &request, &parameters);
            Ok(Box::pin(stream.filter_map(move |response| {
                let metrics = metrics.clone();
                async move { mistralrs_response_chunk(response, metrics) }
            })))
        }
    }

    pub(super) fn validate_embedded_parameters(parameters: &RuntimeParameters) -> Result<()> {
        if parameters.seed.is_some() {
            return Err(RuntimeError::InvalidParameter {
                parameter: "seed".to_string(),
                details: "seed is not supported by embedded mistralrs".to_string(),
            }
            .into());
        }
        Ok(())
    }

    pub fn backend(config: MistralRsConfig) -> Arc<dyn RuntimeBackend> {
        Arc::new(MistralRsBackend::new(config))
    }

    pub(super) fn parse_isq_type(value: &str) -> Result<IsqType> {
        match value.trim().to_lowercase().as_str() {
            "4" | "q4" | "q4k" => Ok(IsqType::Q4K),
            "q4_0" | "q4-0" => Ok(IsqType::Q4_0),
            "q5_0" | "q5-0" => Ok(IsqType::Q5_0),
            "q5k" => Ok(IsqType::Q5K),
            "q6k" => Ok(IsqType::Q6K),
            "8" | "q8" | "q8_0" | "q8-0" => Ok(IsqType::Q8_0),
            "hqq4" => Ok(IsqType::HQQ4),
            "hqq8" => Ok(IsqType::HQQ8),
            "f8e4m3" | "fp8" => Ok(IsqType::F8E4M3),
            other => Err(RuntimeError::InvalidParameter {
                parameter: "auto_isq".to_string(),
                details: format!(
                    "unsupported ISQ value {other:?}, expected q4_0, q4k, q5_0, q5k, q6k, q8_0, hqq4, hqq8, or f8e4m3"
                ),
            }
            .into()),
        }
    }

    pub(super) fn build_sampling_params(parameters: &RuntimeParameters) -> SamplingParams {
        let mut sampling = SamplingParams::neutral();
        sampling.temperature = parameters.temperature.map(f64::from);
        sampling.top_p = parameters.top_p.map(f64::from);
        sampling.top_k = parameters.top_k;
        sampling.max_len = parameters.max_tokens;
        sampling.repetition_penalty = parameters.repetition_penalty;
        if !parameters.stop.is_empty() {
            sampling.stop_toks = Some(StopTokens::Seqs(parameters.stop.clone()));
        }
        sampling
    }

    fn paged_attention_configured(config: &MistralRsConfig) -> bool {
        config.paged_attn_block_size.is_some()
            || config.paged_attn_gpu_mem_ctx.is_some()
            || config.paged_attn_cache_dtype.is_some()
    }

    #[cfg(feature = "mistralrs-cuda")]
    fn parse_paged_cache_type(value: Option<&str>) -> Result<PagedCacheType> {
        match value.unwrap_or("auto").trim().to_lowercase().as_str() {
            "" | "auto" | "f16" => Ok(PagedCacheType::Auto),
            "f8e4m3" | "fp8" => Ok(PagedCacheType::F8E4M3),
            other => Err(RuntimeError::InvalidParameter {
                parameter: "paged_attn_cache_dtype".to_string(),
                details: format!(
                    "unsupported PagedAttention cache dtype {other:?}, expected f16 or f8e4m3"
                ),
            }
            .into()),
        }
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
            Response::Done(done) => {
                let mut metrics = metrics;
                metrics.insert(
                    "prompt_tok_per_sec".to_string(),
                    done.usage.avg_prompt_tok_per_sec.to_string(),
                );
                metrics.insert(
                    "completion_tok_per_sec".to_string(),
                    done.usage.avg_compl_tok_per_sec.to_string(),
                );
                Some(Ok(InferenceChunk {
                    token: String::new(),
                    complete: true,
                    metrics,
                }))
            }
            Response::ModelError(message, _) => Some(Err(anyhow!(message))),
            Response::CompletionModelError(message, _) => Some(Err(anyhow!(message))),
            Response::ValidationError(err) => Some(Err(err.into())),
            Response::InternalError(err) => Some(Err(err.into())),
            _ => None,
        }
    }
}

#[cfg(feature = "openai-backend")]
mod openai_backend {
    use super::*;
    use async_openai::{
        config::OpenAIConfig,
        types::chat::{
            ChatCompletionRequestMessage, ChatCompletionRequestSystemMessageArgs,
            ChatCompletionRequestUserMessageArgs, CreateChatCompletionRequestArgs,
        },
        Client,
    };
    use std::env;

    /// Stateless cloud LLM backend backed by `async-openai`. Each request
    /// reads its routing details (base url, api key env var, deployment
    /// override, extra headers) from the model record's `metadata_json`.
    /// Nothing is loaded or persisted server-side; `load_model`/`unload_model`
    /// are therefore no-ops.
    pub struct OpenAiBackend;

    impl OpenAiBackend {
        pub fn new() -> Self {
            Self
        }

        fn build_client(cloud: &CloudModelConfig) -> Result<Client<OpenAIConfig>> {
            let api_key = env::var(&cloud.api_key_env).map_err(|_| {
                anyhow!(
                    "missing api key: env var {} is not set for cloud model",
                    cloud.api_key_env
                )
            })?;
            let mut config = OpenAIConfig::new()
                .with_api_base(cloud.base_url.clone())
                .with_api_key(api_key);
            if let Some(deployment) = cloud.deployment.as_deref() {
                // Azure-style "deployment" lives in the path; we surface it
                // via api_base override for v1. Native AzureConfig support is
                // tracked separately.
                config = config.with_api_base(format!(
                    "{}/deployments/{deployment}",
                    cloud.base_url.trim_end_matches('/')
                ));
            }
            for (key, value) in &cloud.extra_headers {
                config = config.with_header(key, value)?;
            }
            Ok(Client::with_config(config))
        }

        fn build_messages(request: &RuntimeInferenceRequest) -> Result<Vec<ChatCompletionRequestMessage>> {
            let mut messages: Vec<ChatCompletionRequestMessage> = Vec::new();
            let (system, user) = build_message_parts(request);
            if let Some(system) = system {
                let msg = ChatCompletionRequestSystemMessageArgs::default()
                    .content(system)
                    .build()?;
                messages.push(msg.into());
            }
            let user_msg = ChatCompletionRequestUserMessageArgs::default()
                .content(user)
                .build()?;
            messages.push(user_msg.into());
            Ok(messages)
        }
    }

    #[async_trait]
    impl RuntimeBackend for OpenAiBackend {
        fn name(&self) -> &str {
            "openai"
        }

        async fn list_models(&self, models: Vec<ModelRecord>) -> Result<Vec<ModelRecord>> {
            Ok(models)
        }

        async fn load_model(&self, _model: &ModelRecord, _options: LoadModelOptions) -> Result<()> {
            // Cloud models are stateless - nothing to load.
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
            let cloud = cloud_config_from_metadata(&model.metadata_json).ok_or_else(|| {
                anyhow!(
                    "openai backend invoked for model {} without CloudModelConfig",
                    model.id
                )
            })?;
            let client = Self::build_client(&cloud)?;

            let mut builder = CreateChatCompletionRequestArgs::default();
            // Use deployment name if provided (Azure pattern), otherwise the model id.
            let target_model = cloud.deployment.clone().unwrap_or_else(|| model.id.clone());
            builder.model(target_model);
            builder.messages(Self::build_messages(&request)?);
            builder.stream(true);
            if let Some(t) = parameters.temperature {
                builder.temperature(t);
            }
            if let Some(p) = parameters.top_p {
                builder.top_p(p);
            }
            if let Some(m) = parameters.max_tokens {
                let max_tokens = u32::try_from(m).map_err(|_| RuntimeError::InvalidParameter {
                    parameter: "max_tokens".to_string(),
                    details: format!("max_tokens value {} exceeds maximum allowed value {}", m, u32::MAX),
                })?;
                builder.max_tokens(max_tokens);
            }
            if let Some(s) = parameters.seed {
                if s > i64::MAX as u64 {
                    return Err(RuntimeError::InvalidParameter {
                        parameter: "seed".to_string(),
                        details: format!("seed value {} exceeds maximum allowed value {}", s, i64::MAX),
                    }
                    .into());
                }
                builder.seed(s as i64);
            }
            if !parameters.stop.is_empty() {
                builder.stop(parameters.stop.clone());
            }
            // top_k and repetition_penalty are not part of the OpenAI chat API;
            // they are silently dropped. truncate_sequence is also unused here.

            let req = builder.build()?;
            let metrics = runtime_metrics(self.name(), model, &request, &parameters);
            let stream = client.chat().create_stream(req).await?;

            use std::sync::Arc;
            use tokio::sync::Mutex;
            let error_flag = Arc::new(Mutex::new(false));
            let error_flag_clone = error_flag.clone();

            let transformed_stream = stream.filter_map(move |result| {
                let metrics = metrics.clone();
                let error_flag = error_flag.clone();
                async move {
                    match result {
                        Ok(resp) => {
                            let token = resp
                                .choices
                                .into_iter()
                                .filter_map(|c| c.delta.content)
                                .collect::<Vec<_>>()
                                .join("");
                            let mut chunk_metrics = metrics.clone();
                            if let Some(usage) = resp.usage {
                                chunk_metrics.insert(
                                    "prompt_tokens".to_string(),
                                    usage.prompt_tokens.to_string(),
                                );
                                chunk_metrics.insert(
                                    "completion_tokens".to_string(),
                                    usage.completion_tokens.to_string(),
                                );
                            }
                            if token.is_empty() {
                                None
                            } else {
                                Some(Ok(InferenceChunk {
                                    token,
                                    complete: false,
                                    metrics: chunk_metrics,
                                }))
                            }
                        }
                        Err(e) => {
                            *error_flag.lock().await = true;
                            Some(Err(anyhow!("openai stream error: {e}")))
                        }
                    }
                }
            }).chain(stream::once(async move {
                if !*error_flag_clone.lock().await {
                    Ok(InferenceChunk {
                        token: String::new(),
                        complete: true,
                        metrics: HashMap::new(),
                    })
                } else {
                    // Don't emit completion chunk if there was an error
                    Err(anyhow!("stream completed with errors"))
                }
            }));

            Ok(Box::pin(transformed_stream))
        }
    }

    pub fn backend() -> Arc<dyn RuntimeBackend> {
        Arc::new(OpenAiBackend::new())
    }
}

#[cfg(feature = "openai-backend")]
fn create_openai_backend() -> Option<Arc<dyn RuntimeBackend>> {
    Some(openai_backend::backend())
}

#[cfg(not(feature = "openai-backend"))]
fn create_openai_backend() -> Option<Arc<dyn RuntimeBackend>> {
    None
}

fn create_backend(name: String, config: MistralRsConfig) -> Arc<dyn RuntimeBackend> {
    match normalize_backend_name(&name).as_str() {
        "mock" => Arc::new(MockBackend::new("mock")),
        "mistralrs" => create_mistralrs_backend(config),
        other => Arc::new(UnavailableBackend {
            name: other.to_string(),
            message: format!("unsupported runtime backend: {other}"),
        }),
    }
}

#[cfg(feature = "mistralrs-backend")]
fn create_mistralrs_backend(config: MistralRsConfig) -> Arc<dyn RuntimeBackend> {
    mistralrs_backend::backend(config)
}

#[cfg(not(feature = "mistralrs-backend"))]
fn create_mistralrs_backend(_config: MistralRsConfig) -> Arc<dyn RuntimeBackend> {
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

fn build_message_parts(request: &RuntimeInferenceRequest) -> (Option<String>, String) {
    let mut system_parts = Vec::new();
    if let Some(system_prompt) = request
        .system_prompt
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        system_parts.push(system_prompt.to_string());
    }
    if !request.context_refs.is_empty() {
        system_parts.push(format!(
            "Context references:\n- {}",
            request.context_refs.join("\n- ")
        ));
    }
    let system = if system_parts.is_empty() {
        None
    } else {
        Some(system_parts.join("\n\n"))
    };
    (system, request.prompt.clone())
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
                RuntimeError::InvalidParameter {
                    parameter: key.to_string(),
                    details: format!("expected float, got {value:?}"),
                }
                .into()
            })
        })
        .transpose()
}

fn parse_optional_usize(input: &HashMap<String, String>, key: &str) -> Result<Option<usize>> {
    input
        .get(key)
        .map(|value| {
            value.parse::<usize>().map_err(|_| {
                RuntimeError::InvalidParameter {
                    parameter: key.to_string(),
                    details: format!("expected positive integer, got {value:?}"),
                }
                .into()
            })
        })
        .transpose()
}

fn parse_optional_u64(input: &HashMap<String, String>, key: &str) -> Result<Option<u64>> {
    input
        .get(key)
        .map(|value| {
            value.parse::<u64>().map_err(|_| {
                RuntimeError::InvalidParameter {
                    parameter: key.to_string(),
                    details: format!("expected unsigned integer, got {value:?}"),
                }
                .into()
            })
        })
        .transpose()
}

fn parse_optional_bool(input: &HashMap<String, String>, key: &str) -> Result<Option<bool>> {
    input
        .get(key)
        .map(|value| {
            value.parse::<bool>().map_err(|_| {
                RuntimeError::InvalidParameter {
                    parameter: key.to_string(),
                    details: format!("expected bool, got {value:?}"),
                }
                .into()
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
    if trimmed.starts_with('[') {
        return serde_json::from_str::<Vec<String>>(trimmed).map_err(|_| {
            RuntimeError::InvalidParameter {
                parameter: "stop".to_string(),
                details: "expected string or JSON string array".to_string(),
            }
            .into()
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
    use std::sync::Mutex;

    use tempfile::tempdir;

    async fn test_engine(backend: &str) -> (RuntimeEngine, tempfile::TempDir, tempfile::TempDir) {
        let store_dir = tempdir().unwrap();
        let model_dir = tempdir().unwrap();
        tokio::fs::write(model_dir.path().join("local.gguf"), b"model")
            .await
            .unwrap();

        let store = EngineStore::new(store_dir.path().to_string_lossy().to_string());
        let engine = RuntimeEngine::new(
            store,
            model_dir.path(),
            backend,
            "llama-cli",
            MistralRsConfig::default(),
        );
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
                system_prompt: None,
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
                system_prompt: None,
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
                system_prompt: None,
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
                system_prompt: None,
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

    #[tokio::test]
    async fn sampling_params_reach_backend() {
        let store_dir = tempdir().unwrap();
        let model_dir = tempdir().unwrap();
        tokio::fs::write(model_dir.path().join("local.gguf"), b"model")
            .await
            .unwrap();
        let backend = Arc::new(RecordingBackend::default());
        let engine = RuntimeEngine::with_backend(
            EngineStore::new(store_dir.path().to_string_lossy().to_string()),
            model_dir.path(),
            backend.clone(),
        );

        let chunks = engine
            .stream_inference(RuntimeInferenceRequest {
                model_id: "local.gguf".to_string(),
                prompt: "hello".to_string(),
                system_prompt: None,
                parameters: HashMap::from([
                    ("temperature".to_string(), "0.1".to_string()),
                    ("max_tokens".to_string(), "8".to_string()),
                    ("stop".to_string(), "[\"</s>\"]".to_string()),
                ]),
                context_refs: Vec::new(),
            })
            .await
            .unwrap();

        assert!(chunks.last().unwrap().complete);
        let parameters = backend.last_parameters.lock().unwrap().clone().unwrap();
        assert_eq!(parameters.temperature, Some(0.1));
        assert_eq!(parameters.max_tokens, Some(8));
        assert_eq!(parameters.stop, vec!["</s>"]);
    }

    #[test]
    fn system_prompt_and_context_refs_are_split_from_user_prompt() {
        let request = RuntimeInferenceRequest {
            model_id: "local.gguf".to_string(),
            prompt: "What changed?".to_string(),
            system_prompt: Some("Use terse answers.".to_string()),
            parameters: HashMap::new(),
            context_refs: vec!["viking://resources/doc-a".to_string()],
        };

        let (system, user) = build_message_parts(&request);

        assert_eq!(user, "What changed?");
        let system = system.unwrap();
        assert!(system.contains("Use terse answers."));
        assert!(system.contains("viking://resources/doc-a"));
    }

    #[cfg(feature = "mistralrs-backend")]
    #[test]
    fn isq_parser_accepts_q4k_q8_0_hqq8() {
        assert!(mistralrs_backend::parse_isq_type("q4k").is_ok());
        assert!(mistralrs_backend::parse_isq_type("q8_0").is_ok());
        assert!(mistralrs_backend::parse_isq_type("hqq8").is_ok());
        assert!(mistralrs_backend::parse_isq_type("8").is_ok());
    }

    #[cfg(feature = "mistralrs-backend")]
    #[test]
    fn embedded_mistralrs_rejects_seed() {
        let error = mistralrs_backend::validate_embedded_parameters(&RuntimeParameters {
            seed: Some(7),
            ..RuntimeParameters::default()
        })
        .unwrap_err();

        assert!(error.to_string().contains("seed is not supported"));
    }

    #[cfg(feature = "mistralrs-backend")]
    #[test]
    fn sampling_params_map_runtime_parameters() {
        let sampling = mistralrs_backend::build_sampling_params(&RuntimeParameters {
            temperature: Some(0.1),
            top_p: Some(0.9),
            top_k: Some(32),
            max_tokens: Some(8),
            stop: vec!["</s>".to_string()],
            repetition_penalty: Some(1.1),
            ..RuntimeParameters::default()
        });

        assert!((sampling.temperature.unwrap() - 0.1).abs() < 0.000001);
        assert!((sampling.top_p.unwrap() - 0.9).abs() < 0.000001);
        assert_eq!(sampling.top_k, Some(32));
        assert_eq!(sampling.max_len, Some(8));
        assert_eq!(sampling.repetition_penalty, Some(1.1));
        match sampling.stop_toks {
            Some(mistralrs::StopTokens::Seqs(values)) => assert_eq!(values, vec!["</s>"]),
            other => panic!("unexpected stop tokens: {other:?}"),
        }
    }

    #[derive(Default)]
    struct RecordingBackend {
        last_parameters: Mutex<Option<RuntimeParameters>>,
    }

    #[async_trait]
    impl RuntimeBackend for RecordingBackend {
        fn name(&self) -> &str {
            "recording"
        }

        async fn list_models(&self, models: Vec<ModelRecord>) -> Result<Vec<ModelRecord>> {
            Ok(models)
        }

        async fn load_model(&self, _model: &ModelRecord, _options: LoadModelOptions) -> Result<()> {
            Ok(())
        }

        async fn unload_model(&self, _model_id: &str) -> Result<()> {
            Ok(())
        }

        async fn stream_inference(
            &self,
            _model: &ModelRecord,
            _request: RuntimeInferenceRequest,
            parameters: RuntimeParameters,
        ) -> Result<InferenceStream> {
            *self.last_parameters.lock().unwrap() = Some(parameters);
            Ok(Box::pin(stream::iter([Ok(InferenceChunk {
                token: String::new(),
                complete: true,
                metrics: HashMap::new(),
            })])))
        }
    }

    #[derive(Default)]
    struct FailingBackend {
        attempts: Mutex<usize>,
    }

    #[async_trait]
    impl RuntimeBackend for FailingBackend {
        fn name(&self) -> &str {
            "failing"
        }
        async fn list_models(&self, models: Vec<ModelRecord>) -> Result<Vec<ModelRecord>> {
            Ok(models)
        }
        async fn load_model(&self, _: &ModelRecord, _: LoadModelOptions) -> Result<()> {
            Ok(())
        }
        async fn unload_model(&self, _: &str) -> Result<()> {
            Ok(())
        }
        async fn stream_inference(
            &self,
            _: &ModelRecord,
            _: RuntimeInferenceRequest,
            _: RuntimeParameters,
        ) -> Result<InferenceStream> {
            *self.attempts.lock().unwrap() += 1;
            Err(anyhow!("simulated backend failure"))
        }
    }

    #[test]
    fn cloud_config_round_trips_through_metadata_json() {
        let cfg = CloudModelConfig {
            provider: "openai".into(),
            base_url: "https://api.openai.com/v1".into(),
            api_key_env: "OPENAI_API_KEY".into(),
            deployment: None,
            extra_headers: HashMap::new(),
            fallback_model_id: Some("local.gguf".into()),
        };
        let mut metadata = ModelMetadata::default();
        metadata.cloud = Some(cfg.clone());
        let json = metadata.to_json();

        let parsed = cloud_config_from_metadata(&json).unwrap();
        assert_eq!(parsed, cfg);
    }

    #[test]
    fn cloud_config_absent_when_metadata_has_no_cloud_field() {
        assert!(cloud_config_from_metadata("{}").is_none());
        assert!(cloud_config_from_metadata("").is_none());
        assert!(cloud_config_from_metadata("{\"unrelated\":\"value\"}").is_none());
    }

    #[tokio::test]
    async fn register_cloud_model_persists_record_with_cloud_metadata() {
        let (engine, _s, _m) = test_engine("mock").await;
        let cloud_backend = Arc::new(RecordingBackend::default());
        let engine = engine.with_cloud_backend(cloud_backend.clone());

        let cfg = CloudModelConfig {
            provider: "openai".into(),
            base_url: "https://example/v1".into(),
            api_key_env: "TEST_OPENAI_KEY".into(),
            deployment: None,
            extra_headers: HashMap::new(),
            fallback_model_id: None,
        };
        let record = engine
            .register_cloud_model("gpt-4o", "GPT-4o", cfg.clone())
            .await
            .unwrap();

        assert_eq!(record.id, "gpt-4o");
        assert_eq!(record.backend, "recording");
        assert_eq!(record.status, "registered");
        assert_eq!(cloud_config_from_metadata(&record.metadata_json).unwrap(), cfg);
    }

    #[tokio::test]
    async fn select_backend_routes_cloud_model_to_cloud_backend() {
        let (engine, _s, _m) = test_engine("mock").await;
        let cloud_backend = Arc::new(RecordingBackend::default());
        let engine = engine.with_cloud_backend(cloud_backend.clone());

        let cfg = CloudModelConfig {
            provider: "openai".into(),
            base_url: "https://example/v1".into(),
            api_key_env: "TEST_OPENAI_KEY".into(),
            deployment: None,
            extra_headers: HashMap::new(),
            fallback_model_id: None,
        };
        engine
            .register_cloud_model("gpt-4o", "GPT-4o", cfg)
            .await
            .unwrap();

        let _chunks = engine
            .stream_inference(RuntimeInferenceRequest {
                model_id: "gpt-4o".to_string(),
                prompt: "hi".to_string(),
                system_prompt: None,
                parameters: HashMap::from([("temperature".to_string(), "0.3".to_string())]),
                context_refs: Vec::new(),
            })
            .await
            .unwrap();

        // The recording (cloud) backend should have observed the call.
        let observed = cloud_backend.last_parameters.lock().unwrap().clone().unwrap();
        assert_eq!(observed.temperature, Some(0.3));
    }

    #[tokio::test]
    async fn stream_inference_errors_when_cloud_backend_is_missing() {
        let (engine, _s, _m) = test_engine("mock").await;
        // No cloud backend installed.
        let cfg = CloudModelConfig {
            provider: "openai".into(),
            base_url: "https://example/v1".into(),
            api_key_env: "TEST_OPENAI_KEY".into(),
            deployment: None,
            extra_headers: HashMap::new(),
            fallback_model_id: None,
        };
        engine
            .register_cloud_model("gpt-4o", "GPT-4o", cfg)
            .await
            .unwrap();

        let err = engine
            .stream_inference(RuntimeInferenceRequest {
                model_id: "gpt-4o".to_string(),
                prompt: "hi".to_string(),
                system_prompt: None,
                parameters: HashMap::new(),
                context_refs: Vec::new(),
            })
            .await
            .unwrap_err();

        assert!(err.to_string().contains("cloud backend not configured"));
    }

    #[tokio::test]
    async fn fallback_kicks_in_when_primary_cloud_call_fails() {
        let (engine, _s, _m) = test_engine("mock").await;
        let failing = Arc::new(FailingBackend::default());
        let engine = engine.with_cloud_backend(failing.clone());

        // Register a cloud "primary" pointing to fallback = local.gguf (which mock backend handles).
        let cfg = CloudModelConfig {
            provider: "openai".into(),
            base_url: "https://example/v1".into(),
            api_key_env: "TEST_OPENAI_KEY".into(),
            deployment: None,
            extra_headers: HashMap::new(),
            fallback_model_id: Some("local.gguf".into()),
        };
        engine
            .register_cloud_model("flaky-cloud", "Flaky Cloud", cfg)
            .await
            .unwrap();

        let chunks = engine
            .stream_inference(RuntimeInferenceRequest {
                model_id: "flaky-cloud".to_string(),
                prompt: "hello world".to_string(),
                system_prompt: None,
                parameters: HashMap::new(),
                context_refs: Vec::new(),
            })
            .await
            .unwrap();

        assert_eq!(*failing.attempts.lock().unwrap(), 1);
        // Mock backend produced tokens for "hello world".
        let tokens: Vec<String> = chunks.iter().map(|c| c.token.clone()).collect();
        assert!(tokens.iter().any(|t| t == "hello"));
        assert!(chunks.last().unwrap().complete);
    }

    #[tokio::test]
    async fn fallback_not_attempted_when_unset() {
        let (engine, _s, _m) = test_engine("mock").await;
        let failing = Arc::new(FailingBackend::default());
        let engine = engine.with_cloud_backend(failing.clone());

        let cfg = CloudModelConfig {
            provider: "openai".into(),
            base_url: "https://example/v1".into(),
            api_key_env: "TEST_OPENAI_KEY".into(),
            deployment: None,
            extra_headers: HashMap::new(),
            fallback_model_id: None,
        };
        engine
            .register_cloud_model("only-cloud", "Only Cloud", cfg)
            .await
            .unwrap();

        let err = engine
            .stream_inference(RuntimeInferenceRequest {
                model_id: "only-cloud".to_string(),
                prompt: "hi".to_string(),
                system_prompt: None,
                parameters: HashMap::new(),
                context_refs: Vec::new(),
            })
            .await
            .unwrap_err();

        assert!(err.to_string().contains("simulated backend failure"));
        assert_eq!(*failing.attempts.lock().unwrap(), 1);
    }
}

impl LoadModelOptions {
    pub fn parse(input: &HashMap<String, String>) -> Result<Self> {
        let max_num_seqs = parse_optional_usize(input, "max_num_seqs")?;
        if let Some(0) = max_num_seqs {
            return Err(RuntimeError::InvalidParameter {
                parameter: "max_num_seqs".to_string(),
                details: "max_num_seqs must be greater than 0".to_string(),
            }
            .into());
        }
        Ok(Self { max_num_seqs })
    }
}
