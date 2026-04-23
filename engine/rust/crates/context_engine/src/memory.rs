use std::future::Future;
use std::pin::Pin;

use serde::{Deserialize, Serialize};

use crate::engine::ContextError;
use crate::model::SessionEntry;

pub type ProviderFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DragonflyConfig {
    pub addr: String,
    pub key_prefix: String,
    pub recent_window: usize,
}

impl Default for DragonflyConfig {
    fn default() -> Self {
        Self {
            addr: "127.0.0.1:6379".to_string(),
            key_prefix: "ai-engine:sessions".to_string(),
            recent_window: 10,
        }
    }
}

pub trait WorkingMemoryProvider: Send + Sync {
    fn append_session_entry<'a>(
        &'a self,
        entry: SessionEntry,
    ) -> ProviderFuture<'a, Result<(), ContextError>>;

    fn recent_entries<'a>(
        &'a self,
        session_id: &'a str,
        limit: usize,
    ) -> ProviderFuture<'a, Result<Vec<SessionEntry>, ContextError>>;
}
