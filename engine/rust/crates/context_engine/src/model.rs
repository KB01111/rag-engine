use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ResourceLayer {
    L0,
    L1,
    L2,
}

impl ResourceLayer {
    pub fn as_str(self) -> &'static str {
        match self {
            ResourceLayer::L0 => "l0",
            ResourceLayer::L1 => "l1",
            ResourceLayer::L2 => "l2",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayeredContent {
    pub l0: String,
    pub l1: String,
    pub l2: String,
}

impl LayeredContent {
    pub fn from_full_text(text: &str) -> Self {
        let cleaned = text.trim().replace("\r\n", "\n");
        let paragraphs = split_paragraphs(&cleaned);

        let l0 = build_abstract(&paragraphs, 240);
        let l1 = build_overview(&paragraphs, 960);
        let l2 = cleaned;

        Self { l0, l1, l2 }
    }

    pub fn layer(&self, layer: ResourceLayer) -> &str {
        match layer {
            ResourceLayer::L0 => &self.l0,
            ResourceLayer::L1 => &self.l1,
            ResourceLayer::L2 => &self.l2,
        }
    }
}

fn split_paragraphs(text: &str) -> Vec<String> {
    text.split("\n\n")
        .map(|part| part.trim())
        .filter(|part| !part.is_empty())
        .map(|part| part.to_string())
        .collect()
}

fn build_abstract(paragraphs: &[String], max_chars: usize) -> String {
    if paragraphs.is_empty() {
        return String::new();
    }

    clip_text(&paragraphs[0], max_chars)
}

fn build_overview(paragraphs: &[String], max_chars: usize) -> String {
    if paragraphs.is_empty() {
        return String::new();
    }

    let mut combined = String::new();
    for paragraph in paragraphs.iter().take(3) {
        if !combined.is_empty() {
            combined.push_str("\n\n");
        }
        combined.push_str(paragraph);
        if combined.len() >= max_chars {
            break;
        }
    }

    clip_text(&combined, max_chars)
}

fn clip_text(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        return text.to_string();
    }

    text.chars().take(max_chars).collect()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceSummary {
    pub uri: String,
    pub title: String,
    pub layer: String,
    pub metadata: BTreeMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchHit {
    pub uri: String,
    pub document_id: String,
    pub chunk_text: String,
    pub score: f32,
    pub metadata: BTreeMap<String, String>,
    pub layer: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileEntry {
    pub name: String,
    pub path: String,
    pub is_dir: bool,
    pub size_bytes: u64,
    pub version: Option<i64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionEntry {
    pub session_id: String,
    pub role: String,
    pub content: String,
    pub metadata: BTreeMap<String, String>,
    pub created_at: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatusResponse {
    pub document_count: i64,
    pub chunk_count: i64,
    pub index_size_bytes: i64,
    pub embedding_model: String,
    pub ready: bool,
    pub managed_roots: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceRecord {
    pub resource_id: String,
    pub uri: String,
    pub title: String,
    pub root: String,
    pub path: String,
    pub layer: String,
    pub metadata: BTreeMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUpsertRequest {
    pub uri: String,
    pub title: Option<String>,
    pub content: String,
    pub layer: ResourceLayer,
    pub metadata: BTreeMap<String, String>,
    pub previous_uri: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpsertOutcome {
    pub resource: ResourceSummary,
    pub chunks_indexed: i32,
    pub reused_chunks: i32,
    pub reindexed_chunks: i32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeleteOutcome {
    pub deleted: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchRequest {
    pub query: String,
    pub scope_uri: Option<String>,
    pub top_k: Option<usize>,
    pub filters: Option<BTreeMap<String, String>>,
    pub layer: Option<ResourceLayer>,
    pub rerank: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionEventRequest {
    pub session_id: String,
    pub role: String,
    pub content: String,
    pub metadata: BTreeMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkspaceSyncOutcome {
    pub root: String,
    pub prefix: Option<String>,
    pub indexed_resources: i32,
    pub reindexed_resources: i32,
    pub deleted_resources: i32,
    pub skipped_files: i32,
}
