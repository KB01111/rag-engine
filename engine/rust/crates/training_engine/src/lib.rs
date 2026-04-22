// TODO: This training_engine crate is not included in the documented Rust architecture.
// Either:
// 1. Move functionality into existing canonical crates (rag_engine, embedding, chunking, storage), OR
// 2. Update project architecture docs to formally add training_engine (and runtime_engine, ai_engine_daemon)
//    with precise responsibilities, boundaries, and API definitions.
// Currently TrainingEngine, EngineStore, and cancellations live here but their role is unclear.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{anyhow, Result};
use storage::{EngineStore, TrainingLogRecord, TrainingRunRecord};
use tokio::sync::Mutex;
use tokio::time::{sleep, Duration};
use uuid::Uuid;

#[derive(Debug, Clone)]
pub struct TrainingEngine {
    store: EngineStore,
    working_dir: PathBuf,
    backend: String,
    cancellations: Arc<Mutex<HashMap<String, Arc<AtomicBool>>>>,
}

impl TrainingEngine {
    pub fn new(store: EngineStore, working_dir: impl Into<PathBuf>, backend: impl Into<String>) -> Self {
        Self {
            store,
            working_dir: working_dir.into(),
            backend: backend.into(),
            cancellations: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    pub async fn start_run(
        &self,
        name: &str,
        model_id: &str,
        dataset_path: &str,
        config_json: &str,
    ) -> Result<TrainingRunRecord> {
        tokio::fs::create_dir_all(&self.working_dir).await?;

        let run_id = format!("run-{}", Uuid::new_v4());
        let artifact_dir = self.working_dir.join(&run_id);
        tokio::fs::create_dir_all(&artifact_dir).await?;

        let run = TrainingRunRecord {
            id: run_id.clone(),
            name: name.to_string(),
            model_id: model_id.to_string(),
            dataset_path: dataset_path.to_string(),
            status: "running".to_string(),
            progress: 0.0,
            error: String::new(),
            backend: self.backend.clone(),
            artifact_dir: artifact_dir.to_string_lossy().to_string(),
            config_json: config_json.to_string(),
            started_at: now(),
            completed_at: 0,
        };
        self.store.upsert_training_run(run.clone()).await?;
        self.store
            .append_training_logs(vec![TrainingLogRecord {
                run_id: run_id.clone(),
                level: "info".to_string(),
                message: "training started".to_string(),
                fields_json: "{}".to_string(),
                timestamp: now(),
            }])
            .await?;

        let cancelled = Arc::new(AtomicBool::new(false));
        self.cancellations
            .lock()
            .await
            .insert(run_id.clone(), cancelled.clone());

        let store = self.store.clone();
        let backend = self.backend.clone();
        let task_run = run.clone();
        tokio::spawn(async move {
            for step in 1..=5 {
                sleep(Duration::from_millis(250)).await;
                if cancelled.load(Ordering::SeqCst) {
                    return;
                }

                let progress = step as f32 / 5.0;
                let _ = store
                    .upsert_training_run(TrainingRunRecord {
                        progress,
                        ..task_run.clone()
                    })
                    .await;
                let _ = store
                    .append_training_logs(vec![TrainingLogRecord {
                        run_id: task_run.id.clone(),
                        level: "info".to_string(),
                        message: format!("training progress {:.0}%", progress * 100.0),
                        fields_json: format!("{{\"backend\":\"{backend}\"}}"),
                        timestamp: now(),
                    }])
                    .await;
            }

            let artifact_path = PathBuf::from(&task_run.artifact_dir).join("summary.txt");
            let _ = tokio::fs::write(&artifact_path, "training complete\n").await;
            let _ = store
                .upsert_training_run(TrainingRunRecord {
                    status: "completed".to_string(),
                    progress: 1.0,
                    completed_at: now(),
                    ..task_run.clone()
                })
                .await;
        });

        Ok(run)
    }

    pub async fn cancel_run(&self, run_id: &str) -> Result<()> {
        if let Some(flag) = self.cancellations.lock().await.get(run_id).cloned() {
            flag.store(true, Ordering::SeqCst);
        }

        let mut runs = self.store.list_training_runs().await?;
        let Some(run) = runs.iter_mut().find(|run| run.id == run_id) else {
            return Err(anyhow!("run not found: {run_id}"));
        };

        // Do not overwrite terminal states
        if matches!(run.status.as_str(), "completed" | "failed" | "cancelled") {
            return Ok(());
        }

        run.status = "cancelled".to_string();
        run.completed_at = now();
        self.store.upsert_training_run(run.clone()).await?;
        self.store
            .append_training_logs(vec![TrainingLogRecord {
                run_id: run_id.to_string(),
                level: "warn".to_string(),
                message: "training cancelled".to_string(),
                fields_json: "{}".to_string(),
                timestamp: now(),
            }])
            .await?;
        Ok(())
    }

    pub async fn list_runs(&self) -> Result<Vec<TrainingRunRecord>> {
        Ok(self.store.list_training_runs().await?)
    }

    pub async fn list_logs(&self, run_id: &str) -> Result<Vec<TrainingLogRecord>> {
        Ok(self.store.list_training_logs(run_id).await?)
    }

    pub async fn list_artifacts(&self, run_id: &str) -> Result<Vec<PathBuf>> {
        let runs = self.store.list_training_runs().await?;
        let Some(run) = runs.into_iter().find(|run| run.id == run_id) else {
            return Err(anyhow!("run not found: {run_id}"));
        };

        let mut artifacts = Vec::new();
        let mut entries = tokio::fs::read_dir(&run.artifact_dir).await?;
        while let Some(entry) = entries.next_entry().await? {
            if entry.file_type().await?.is_file() {
                artifacts.push(entry.path());
            }
        }
        Ok(artifacts)
    }
}

fn now() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use anyhow::Result;
    use storage::EngineStore;
    use tempfile::tempdir;
    use tokio::time::{sleep, timeout};

    use super::TrainingEngine;

    #[tokio::test]
    async fn start_run_completes_and_writes_artifact() -> Result<()> {
        let tempdir = tempdir()?;
        let db_path = tempdir.path().join("lancedb");
        let work_path = tempdir.path().join("training");
        let store = EngineStore::new(db_path.to_string_lossy().as_ref());
        let engine = TrainingEngine::new(store, &work_path, "llama.cpp");

        let run = engine
            .start_run("demo", "model.gguf", "dataset.jsonl", "{\"epochs\":1}")
            .await?;

        let completed = timeout(Duration::from_secs(5), async {
            loop {
                let runs = engine.list_runs().await.expect("list_runs should succeed");
                let current = runs
                    .into_iter()
                    .find(|candidate| candidate.id == run.id)
                    .expect("run should exist");
                if current.status == "completed" {
                    break current;
                }
                sleep(Duration::from_millis(100)).await;
            }
        })
        .await
        .expect("run should complete within timeout");

        assert_eq!(completed.progress, 1.0);
        let artifacts = engine.list_artifacts(&run.id).await?;
        assert!(artifacts.iter().any(|path| path.ends_with("summary.txt")));
        Ok(())
    }

    #[tokio::test]
    async fn cancel_run_stays_cancelled() -> Result<()> {
        let tempdir = tempdir()?;
        let db_path = tempdir.path().join("lancedb");
        let work_path = tempdir.path().join("training");
        let store = EngineStore::new(db_path.to_string_lossy().as_ref());
        let engine = TrainingEngine::new(store, &work_path, "llama.cpp");

        let run = engine
            .start_run("demo", "model.gguf", "dataset.jsonl", "{\"epochs\":1}")
            .await?;

        engine.cancel_run(&run.id).await?;
        sleep(Duration::from_secs(2)).await;

        let runs = engine.list_runs().await?;
        let cancelled = runs
            .into_iter()
            .find(|candidate| candidate.id == run.id)
            .expect("run should exist");
        assert_eq!(cancelled.status, "cancelled");
        assert!(cancelled.completed_at > 0);
        Ok(())
    }
}