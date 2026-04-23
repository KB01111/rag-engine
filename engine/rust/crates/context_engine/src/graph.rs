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
    /// Maps a metadata string to the corresponding `GraphNodeKind`.
    ///
    /// Comparison is case-insensitive and ignores surrounding whitespace. Recognized inputs:
    /// - `"user"` -> `GraphNodeKind::User`
    /// - `"project"` -> `GraphNodeKind::Project`
    /// - `"concept"` -> `GraphNodeKind::Concept`
    /// - `"document"` -> `GraphNodeKind::Document`
    /// Any other value yields `GraphNodeKind::Generic`.
    ///
    /// # Examples
    ///
    /// ```
    /// assert!(matches!(GraphNodeKind::from_metadata(" User "), GraphNodeKind::User));
    /// assert!(matches!(GraphNodeKind::from_metadata("PROJECT"), GraphNodeKind::Project));
    /// assert!(matches!(GraphNodeKind::from_metadata("unknown"), GraphNodeKind::Generic));
    /// ```
    pub fn from_metadata(value: &str) -> Self {
        match value.trim().to_ascii_lowercase().as_str() {
            "user" => Self::User,
            "project" => Self::Project,
            "concept" => Self::Concept,
            "document" => Self::Document,
            _ => Self::Generic,
        }
    }

    /// Canonical lowercase identifier for this node kind.
    ///
    /// # Returns
    ///
    /// The lowercase name corresponding to the variant: `"user"`, `"project"`, `"concept"`,
    /// `"document"`, or `"generic"`.
    ///
    /// # Examples
    ///
    /// ```
    /// let k = GraphNodeKind::User;
    /// assert_eq!(k.as_str(), "user");
    /// ```
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
    /// Parses a metadata string into a `GraphRelationKind`.
    ///
    /// The input is interpreted case-insensitively and may include surrounding whitespace.
    /// Unrecognized values are mapped to `GraphRelationKind::RelatedTo`.
    ///
    /// # Examples
    ///
    /// ```
    /// use crate::graph::GraphRelationKind;
    ///
    /// assert_eq!(GraphRelationKind::from_metadata("INTERESTED_IN"), GraphRelationKind::InterestedIn);
    /// assert_eq!(GraphRelationKind::from_metadata(" recommended "), GraphRelationKind::Recommended);
    /// assert_eq!(GraphRelationKind::from_metadata("unknown_value"), GraphRelationKind::RelatedTo);
    /// ```
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

    /// Uppercase identifier for this relation kind used in metadata and serialization.
    ///
    /// # Returns
    ///
    /// The uppercase identifier as a `&'static str` (e.g., `"INTERESTED_IN"`, `"RELATED_TO"`).
    ///
    /// # Examples
    ///
    /// ```
    /// let s = GraphRelationKind::Uses.as_str();
    /// assert_eq!(s, "USES");
    /// ```
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
