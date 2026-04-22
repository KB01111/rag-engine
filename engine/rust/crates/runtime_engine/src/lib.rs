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
    pub fn new(store: EngineStore, models_path: impl Into<PathBuf>, backend: impl Into<String>) -> Self {
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

    pub fn infer_tokens(&self, prompt: &str) -> Vec<String> {
        let trimmed = prompt.trim();
        if trimmed.is_empty() {
            return vec!["No".to_string(), " prompt".to_string(), " provided.".to_string()];
        }

        trimmed
            .split_whitespace()
            .map(|token| format!("{token} "))
            .collect()
    }

    async fn discover_models(&self) -> Result<()> {
        fs::create_dir_all(&self.models_path)?;
        for entry in fs::read_dir(&self.models_path)? {
            let entry = entry?;
            let path = entry.path();
            if !is_model_file(&path) {
                continue;
            }

            let metadata = entry.metadata()?;
            let name = entry.file_name().to_string_lossy().to_string();
            self.store
                .upsert_model(ModelRecord {
                    id: name.clone(),
                    name: name.clone(),
                    path: path.to_string_lossy().to_string(),
                    backend: self.backend.clone(),
                    status: "discovered".to_string(),
                    metadata_json: "{}".to_string(),
                    size_bytes: metadata.len() as i64,
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
