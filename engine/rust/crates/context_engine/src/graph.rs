use std::collections::BTreeMap;
use std::future::Future;
use std::pin::Pin;

use serde::{Deserialize, Serialize};

use crate::engine::ContextError;

pub type ProviderFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GraphNodeKind {
    User,
    Project,
    Concept,
    Document,
    Generic,
}

impl GraphNodeKind {
    pub fn from_metadata(value: &str) -> Self {
        match value.trim().to_ascii_lowercase().as_str() {
            "user" => Self::User,
            "project" => Self::Project,
            "concept" => Self::Concept,
            "document" => Self::Document,
            _ => Self::Generic,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::User => "user",
            Self::Project => "project",
            Self::Concept => "concept",
            Self::Document => "document",
            Self::Generic => "generic",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum GraphRelationKind {
    InterestedIn,
    Recommended,
    PartOfProject,
    Prefers,
    Dislikes,
    Uses,
    RelatedTo,
}

impl GraphRelationKind {
    pub fn from_metadata(value: &str) -> Self {
        match value.trim().to_ascii_uppercase().as_str() {
            "INTERESTED_IN" => Self::InterestedIn,
            "RECOMMENDED" => Self::Recommended,
            "PART_OF_PROJECT" => Self::PartOfProject,
            "PREFERS" => Self::Prefers,
            "DISLIKES" => Self::Dislikes,
            "USES" => Self::Uses,
            _ => Self::RelatedTo,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::InterestedIn => "INTERESTED_IN",
            Self::Recommended => "RECOMMENDED",
            Self::PartOfProject => "PART_OF_PROJECT",
            Self::Prefers => "PREFERS",
            Self::Dislikes => "DISLIKES",
            Self::Uses => "USES",
            Self::RelatedTo => "RELATED_TO",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GraphNode {
    pub id: String,
    pub name: String,
    pub kind: GraphNodeKind,
    pub metadata: BTreeMap<String, String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GraphFactRecord {
    pub edge_id: String,
    pub subject: GraphNode,
    pub relation: GraphRelationKind,
    pub object: GraphNode,
    pub metadata: BTreeMap<String, String>,
    pub resource_uri: Option<String>,
    pub session_id: Option<String>,
    pub updated_at: i64,
}

pub trait GraphStore: Send + Sync {
    fn upsert_fact<'a>(
        &'a self,
        fact: GraphFactRecord,
    ) -> ProviderFuture<'a, Result<(), ContextError>>;

    fn related_facts<'a>(
        &'a self,
        query: &'a str,
        limit: usize,
    ) -> ProviderFuture<'a, Result<Vec<GraphFactRecord>, ContextError>>;
}
