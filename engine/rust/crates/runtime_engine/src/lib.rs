use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{anyhow, Result};
use storage::{EngineStore, ModelRecord};

#[derive(Debug, Clone)]
pub struct RuntimeEngine {
    store: EngineStore,
    models_path: PathBuf,
    backend: String,
}

impl RuntimeEngine {
    pub fn new(
        store: EngineStore,
        models_path: impl Into<PathBuf>,
        backend: impl Into<String>,
    ) -> Self {
        Self {
            store,
            models_path: models_path.into(),
            backend: backend.into(),
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

    /// Mock implementation of token inference.
    /// Returns tokens without extra padding.
    /// This is a temporary implementation until a real backend is wired.
    pub fn infer_tokens(&self, prompt: &str) -> Vec<String> {
        let trimmed = prompt.trim();
        if trimmed.is_empty() {
            return vec![
                "No".to_string(),
                "prompt".to_string(),
                "provided.".to_string(),
            ];
        }

        let tokens: Vec<String> = trimmed.split_whitespace().map(String::from).collect();
        let mut result = Vec::with_capacity(tokens.len() * 2 - 1);
        for (i, token) in tokens.into_iter().enumerate() {
            if i > 0 {
                result.push(" ".to_string());
            }
            result.push(token);
        }
        result
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
