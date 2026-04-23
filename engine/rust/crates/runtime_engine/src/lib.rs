use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{anyhow, Context, Result};
use serde_json::json;
use storage::{EngineStore, ModelRecord};
use sysinfo::System;
use tokio::io::AsyncReadExt;
use tokio::process::Command;
use tokio::sync::mpsc;
use which::which;

#[derive(Debug, Clone)]
pub struct RuntimeResources {
    pub cpu_percent: f32,
    pub memory_used_bytes: u64,
    pub memory_total_bytes: u64,
}

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
    backend_name: String,
    backend_command: String,
}

impl RuntimeEngine {
    pub fn new(
        store: EngineStore,
        models_path: impl Into<PathBuf>,
        backend_name: impl Into<String>,
        backend_command: impl Into<String>,
    ) -> Self {
        Self {
            store,
            models_path: models_path.into(),
            backend_name: backend_name.into(),
            backend_command: backend_command.into(),
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
        let backend_command = self.resolve_backend_command()?;

        model.status = "loaded".to_string();
        model.metadata_json = json!({
            "backend_command": backend_command.to_string_lossy(),
            "loaded_at": now(),
        })
        .to_string();
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
        model.metadata_json = "{}".to_string();
        model.updated_at = now();
        self.store.upsert_model(model.clone()).await?;
        Ok(())
    }

    pub async fn stream_inference(
        &self,
        model_id: &str,
        prompt: &str,
        parameters: &HashMap<String, String>,
    ) -> Result<mpsc::Receiver<Result<String>>> {
        if prompt.trim().is_empty() {
            return Err(anyhow!("prompt is required"));
        }

        let model = self.loaded_model(model_id).await?;
        let backend_command = self.resolve_backend_command()?;
        let args = build_backend_args(&model.path, prompt, parameters);

        let mut child = Command::new(&backend_command)
            .args(args)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true)
            .spawn()
            .with_context(|| {
                format!(
                    "failed to start backend command {} for model {}",
                    backend_command.display(),
                    model_id
                )
            })?;

        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| anyhow!("backend command stdout was not piped"))?;
        let stderr = child
            .stderr
            .take()
            .ok_or_else(|| anyhow!("backend command stderr was not piped"))?;

        let (sender, receiver) = mpsc::channel(32);
        tokio::spawn(async move {
            let mut stdout = tokio::io::BufReader::new(stdout);
            let stderr_task = tokio::spawn(async move {
                let mut stderr = tokio::io::BufReader::new(stderr);
                let mut error_output = String::new();
                let _ = stderr.read_to_string(&mut error_output).await;
                error_output
            });

            let mut buffer = [0_u8; 1024];
            loop {
                match stdout.read(&mut buffer).await {
                    Ok(0) => break,
                    Ok(read) => {
                        let chunk = String::from_utf8_lossy(&buffer[..read]).to_string();
                        if sender.send(Ok(chunk)).await.is_err() {
                            let _ = child.kill().await;
                            return;
                        }
                    }
                    Err(error) => {
                        let _ = sender
                            .send(Err(anyhow!("failed to read backend output: {error}")))
                            .await;
                        let _ = child.kill().await;
                        return;
                    }
                }
            }

            let status = match child.wait().await {
                Ok(status) => status,
                Err(error) => {
                    let _ = sender
                        .send(Err(anyhow!("failed to wait for backend command: {error}")))
                        .await;
                    return;
                }
            };

            let stderr_output = stderr_task.await.unwrap_or_default();
            if !status.success() {
                let detail = stderr_output.trim();
                let message = if detail.is_empty() {
                    format!("backend command exited with status {status}")
                } else {
                    format!("backend command exited with status {status}: {detail}")
                };
                let _ = sender.send(Err(anyhow!(message))).await;
            }
        });

        Ok(receiver)
    }

    pub fn system_resources(&self) -> RuntimeResources {
        let mut system = System::new_all();
        system.refresh_cpu_usage();
        system.refresh_memory();

        RuntimeResources {
            cpu_percent: system.global_cpu_usage(),
            memory_used_bytes: system.used_memory(),
            memory_total_bytes: system.total_memory(),
        }
    }

    async fn loaded_model(&self, model_id: &str) -> Result<ModelRecord> {
        let mut models = self.list_models().await?;
        let Some(model) = models.iter_mut().find(|model| model.id == model_id) else {
            return Err(anyhow!("model not found: {model_id}"));
        };
        if model.status != "loaded" {
            return Err(anyhow!("model not loaded: {model_id}"));
        }
        Ok(model.clone())
    }

    fn resolve_backend_command(&self) -> Result<PathBuf> {
        let candidate = Path::new(&self.backend_command);
        if candidate.components().count() > 1 {
            if candidate.exists() {
                return Ok(candidate.to_path_buf());
            }
            return Err(anyhow!(
                "backend command not found at {}",
                candidate.to_string_lossy()
            ));
        }

        which(&self.backend_command)
            .map_err(|_| anyhow!("backend command not found: {}", self.backend_command))
    }

    async fn discover_models(&self) -> Result<()> {
        tokio::fs::create_dir_all(&self.models_path).await?;

        let existing_models = self.store.list_models().await.unwrap_or_default();
        let mut existing_map: HashMap<String, ModelRecord> = existing_models
            .into_iter()
            .map(|model| (model.id.clone(), model))
            .collect();

        let models_path = self.models_path.clone();
        let backend_name = self.backend_name.clone();
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
                        models.push((name, path, metadata.len(), backend_name.clone()));
                    }
                }
            }
            models
        })
        .await?;

        for (name, path, size_bytes, backend_name) in discovered {
            let existing = existing_map.remove(&name);
            let status = existing
                .as_ref()
                .map(|model| model.status.clone())
                .unwrap_or_else(|| "discovered".to_string());
            let metadata_json = existing
                .as_ref()
                .map(|model| model.metadata_json.clone())
                .unwrap_or_else(|| "{}".to_string());

            self.store
                .upsert_model(ModelRecord {
                    id: name.clone(),
                    name: name.clone(),
                    path: path.to_string_lossy().to_string(),
                    backend: backend_name,
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

fn build_backend_args(
    model_path: &str,
    prompt: &str,
    parameters: &HashMap<String, String>,
) -> Vec<String> {
    let mut args = vec![
        "-m".to_string(),
        model_path.to_string(),
        "-p".to_string(),
        prompt.to_string(),
        "-n".to_string(),
        parameters
            .get("n_predict")
            .cloned()
            .unwrap_or_else(|| "128".to_string()),
    ];

    if let Some(value) = parameters.get("temperature") {
        args.push("--temp".to_string());
        args.push(value.clone());
    }
    if let Some(value) = parameters.get("threads") {
        args.push("-t".to_string());
        args.push(value.clone());
    }
    if let Some(value) = parameters.get("ctx_size") {
        args.push("-c".to_string());
        args.push(value.clone());
    }
    if let Some(value) = parameters.get("top_k") {
        args.push("--top-k".to_string());
        args.push(value.clone());
    }
    if let Some(value) = parameters.get("top_p") {
        args.push("--top-p".to_string());
        args.push(value.clone());
    }

    args
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
    use std::collections::HashMap;
    use std::fs;
    use std::path::{Path, PathBuf};

    use tempfile::tempdir;

    use super::*;

    #[tokio::test]
    async fn load_model_requires_an_available_backend_command() {
        let tempdir = tempdir().unwrap();
        let models_dir = tempdir.path().join("models");
        tokio::fs::create_dir_all(&models_dir).await.unwrap();
        tokio::fs::write(models_dir.join("demo.gguf"), b"weights")
            .await
            .unwrap();

        let store = EngineStore::new(tempdir.path().join("db").to_string_lossy().to_string());
        let engine = RuntimeEngine::new(
            store,
            &models_dir,
            "llama.cpp",
            "__missing_backend_command__",
        );

        let err = engine.load_model("demo.gguf").await.unwrap_err();
        assert!(
            err.to_string().contains("backend command"),
            "expected backend command validation error, got {err:?}"
        );
    }

    #[tokio::test]
    async fn stream_inference_reads_output_from_the_backend_command() {
        let tempdir = tempdir().unwrap();
        let models_dir = tempdir.path().join("models");
        tokio::fs::create_dir_all(&models_dir).await.unwrap();
        tokio::fs::write(models_dir.join("demo.gguf"), b"weights")
            .await
            .unwrap();

        let backend = create_fake_backend(tempdir.path());
        let store = EngineStore::new(tempdir.path().join("db").to_string_lossy().to_string());
        let engine = RuntimeEngine::new(
            store,
            &models_dir,
            "llama.cpp",
            backend.to_string_lossy().to_string(),
        );

        engine.load_model("demo.gguf").await.unwrap();

        let mut stream = engine
            .stream_inference("demo.gguf", "hello from runtime", &HashMap::new())
            .await
            .unwrap();

        let mut output = String::new();
        while let Some(chunk) = stream.recv().await {
            output.push_str(&chunk.unwrap());
        }

        assert!(
            output.contains("backend:hello from runtime"),
            "expected backend output, got {output:?}"
        );
    }

    fn create_fake_backend(root: &Path) -> PathBuf {
        if cfg!(windows) {
            let path = root.join("fake-llama.cmd");
            fs::write(
                &path,
                "@echo off\r\necho backend:%~4\r\n",
            )
            .unwrap();
            path
        } else {
            let path = root.join("fake-llama.sh");
            fs::write(
                &path,
                "#!/bin/sh\nPROMPT=\"\"\nwhile [ \"$#\" -gt 0 ]; do\n  if [ \"$1\" = \"-p\" ]; then\n    shift\n    PROMPT=\"$1\"\n  fi\n  shift\ndone\nprintf 'backend:%s\\n' \"$PROMPT\"\n",
            )
            .unwrap();
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