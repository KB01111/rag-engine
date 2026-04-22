use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{anyhow, Result};
use storage::{EngineStore, ModelRecord};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InferenceChunk {
    pub token: String,
    pub complete: bool,
    pub metrics: HashMap<String, String>,
}

pub trait InferenceBackend: Send + Sync {
    fn backend_name(&self) -> &str;
    fn infer(&self, model: &ModelRecord, prompt: &str) -> Result<Vec<InferenceChunk>>;
}

#[derive(Debug, Clone)]
pub struct WhitespaceInferenceBackend {
    backend_name: String,
}

impl WhitespaceInferenceBackend {
    pub fn new(backend_name: impl Into<String>) -> Self {
        Self {
            backend_name: backend_name.into(),
        }
    }
}

impl InferenceBackend for WhitespaceInferenceBackend {
    fn backend_name(&self) -> &str {
        &self.backend_name
    }

    fn infer(&self, model: &ModelRecord, prompt: &str) -> Result<Vec<InferenceChunk>> {
        let trimmed = prompt.trim();
        let tokens = if trimmed.is_empty() {
            vec![
                "No".to_string(),
                "prompt".to_string(),
                "provided.".to_string(),
            ]
        } else {
            let split: Vec<String> = trimmed.split_whitespace().map(String::from).collect();
            let mut result = Vec::with_capacity(split.len() * 2 - 1);
            for (i, token) in split.into_iter().enumerate() {
                if i > 0 {
                    result.push(" ".to_string());
                }
                result.push(token);
            }
            result
        };

        let mut chunks = Vec::with_capacity(tokens.len().max(1));
        for (index, token) in tokens.into_iter().enumerate() {
            let complete = index + 1 == chunks.capacity();
            let mut metrics = HashMap::new();
            metrics.insert("backend".to_string(), self.backend_name.clone());
            metrics.insert("model".to_string(), model.id.clone());
            if complete {
                metrics.insert("status".to_string(), "complete".to_string());
            }
            chunks.push(InferenceChunk {
                token,
                complete,
                metrics,
            });
        }

        if chunks.is_empty() {
            chunks.push(InferenceChunk {
                token: String::new(),
                complete: true,
                metrics: HashMap::from([
                    ("backend".to_string(), self.backend_name.clone()),
                    ("model".to_string(), model.id.clone()),
                    ("status".to_string(), "complete".to_string()),
                ]),
            });
        } else if let Some(last) = chunks.last_mut() {
            last.complete = true;
            last.metrics
                .insert("status".to_string(), "complete".to_string());
        }

        Ok(chunks)
    }
}

#[derive(Clone)]
pub struct RuntimeEngine {
    store: EngineStore,
    models_path: PathBuf,
    backend: String,
    inference_backend: Arc<dyn InferenceBackend>,
}

impl RuntimeEngine {
    pub fn new(
        store: EngineStore,
        models_path: impl Into<PathBuf>,
        backend: impl Into<String>,
    ) -> Self {
        let backend = backend.into();
        Self {
            store,
            models_path: models_path.into(),
            inference_backend: Arc::new(WhitespaceInferenceBackend::new(backend.clone())),
            backend,
        }
    }

    pub fn with_inference_backend(
        store: EngineStore,
        models_path: impl Into<PathBuf>,
        backend: impl Into<String>,
        inference_backend: Arc<dyn InferenceBackend>,
    ) -> Self {
        Self {
            store,
            models_path: models_path.into(),
            backend: backend.into(),
            inference_backend,
        }
    }

    pub async fn list_models(&self) -> Result<Vec<ModelRecord>> {
        self.discover_models().await?;
        Ok(self.store.list_models().await?)
    }

    pub async fn load_model(&self, model_id: &str) -> Result<ModelRecord> {
        let mut models = self.list_models().await?;
        let Some(model) = models.iter_mut().find(|model| model.id == model_id) else {
            return Err(anyhow!("model not found: {model_id}"));
        };

        model.status = "loaded".to_string();
        model.updated_at = now();
        self.store.upsert_model(model.clone()).await?;
        Ok(model.clone())
    }

    pub async fn unload_model(&self, model_id: &str) -> Result<()> {
        let mut models = self.list_models().await?;
        let Some(model) = models.iter_mut().find(|model| model.id == model_id) else {
            return Err(anyhow!("model not found: {model_id}"));
        };

        model.status = "discovered".to_string();
        model.updated_at = now();
        self.store.upsert_model(model.clone()).await?;
        Ok(())
    }

    pub async fn stream_inference(
        &self,
        model_id: &str,
        prompt: &str,
    ) -> Result<Vec<InferenceChunk>> {
        let model = self.require_loaded_model(model_id).await?;
        self.inference_backend.infer(&model, prompt)
    }

    async fn discover_models(&self) -> Result<()> {
        tokio::fs::create_dir_all(&self.models_path).await?;

        // Load existing models to preserve their status
        let existing_models = self.store.list_models().await.unwrap_or_default();
        let mut existing_map: std::collections::HashMap<String, ModelRecord> = existing_models
            .into_iter()
            .map(|model| (model.id.clone(), model))
            .collect();

        // Discover models on filesystem using blocking task
        let models_path = self.models_path.clone();
        let backend = self.backend.clone();
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

        // Upsert discovered models, preserving existing status
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
                    backend: self.backend.clone(),
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

impl RuntimeEngine {
    async fn require_loaded_model(&self, model_id: &str) -> Result<ModelRecord> {
        let models = self.list_models().await?;
        let model = models
            .into_iter()
            .find(|candidate| candidate.id == model_id)
            .ok_or_else(|| anyhow!("model not found: {model_id}"))?;
        if model.status != "loaded" {
            return Err(anyhow!("model not loaded: {model_id}"));
        }
        Ok(model)
    }
}

fn is_model_file(path: &Path) -> bool {
    matches!(
        path.extension().and_then(|ext| ext.to_str()),
        Some("bin" | "gguf" | "ggml")
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
    use tempfile::tempdir;

    #[tokio::test]
    async fn load_model_then_stream_inference_emits_backend_metrics() -> Result<()> {
        let tempdir = tempdir()?;
        let db_path = tempdir.path().join("lancedb");
        let models_path = tempdir.path().join("models");
        tokio::fs::create_dir_all(&models_path).await?;
        tokio::fs::write(models_path.join("demo.gguf"), b"weights").await?;

        let store = EngineStore::new(db_path.to_string_lossy().as_ref());
        let engine = RuntimeEngine::new(store, &models_path, "llama.cpp");

        engine.load_model("demo.gguf").await?;
        let chunks = engine
            .stream_inference("demo.gguf", "hello runtime")
            .await?;

        assert!(!chunks.is_empty());
        assert_eq!(chunks.last().unwrap().complete, true);
        assert_eq!(
            chunks.last().unwrap().metrics.get("backend"),
            Some(&"llama.cpp".to_string())
        );
        assert_eq!(
            chunks.last().unwrap().metrics.get("status"),
            Some(&"complete".to_string())
        );
        Ok(())
    }

    #[tokio::test]
    async fn stream_inference_requires_loaded_model() -> Result<()> {
        let tempdir = tempdir()?;
        let db_path = tempdir.path().join("lancedb");
        let models_path = tempdir.path().join("models");
        tokio::fs::create_dir_all(&models_path).await?;
        tokio::fs::write(models_path.join("demo.gguf"), b"weights").await?;

        let store = EngineStore::new(db_path.to_string_lossy().as_ref());
        let engine = RuntimeEngine::new(store, &models_path, "llama.cpp");

        let err = engine
            .stream_inference("demo.gguf", "hello runtime")
            .await
            .expect_err("inference should require an explicitly loaded model");
        assert!(err.to_string().contains("model not loaded"));
        Ok(())
    }
}
