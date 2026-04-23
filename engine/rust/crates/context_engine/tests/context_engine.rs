use std::collections::BTreeMap;

use context_engine::{
    ContextConfig, ContextEngine, DragonflyConfig, ManagedRoot, ResourceLayer,
    ResourceUpsertRequest, SearchRequest, SessionEventRequest, VikingUri,
};
use tempfile::tempdir;

#[tokio::test]
async fn parses_and_roundtrips_viking_resource_uri() {
    let uri = VikingUri::parse("viking://resources/workspace/docs/readme.md").unwrap();

    assert_eq!(uri.namespace().as_str(), "resources");
    assert_eq!(uri.resource_root().unwrap(), "workspace");
    assert_eq!(uri.resource_path().unwrap(), "docs/readme.md");
    assert_eq!(
        uri.to_string(),
        "viking://resources/workspace/docs/readme.md"
    );
}

#[tokio::test]
async fn layered_content_keeps_all_three_levels() {
    let layered = context_engine::LayeredContent::from_full_text(
        "Intro paragraph.\n\nSecond paragraph with more detail.\n\nThird paragraph ends here.",
    );

    assert!(layered.l0.len() < layered.l1.len());
    assert_eq!(
        layered.l2,
        "Intro paragraph.\n\nSecond paragraph with more detail.\n\nThird paragraph ends here."
    );
    assert_eq!(layered.layer(ResourceLayer::L2), layered.l2);
}

#[tokio::test]
async fn managed_root_rejects_path_escape() {
    let dir = tempdir().unwrap();
    let root = ManagedRoot {
        name: "workspace".to_string(),
        path: dir.path().to_path_buf(),
    };
    let engine = ContextEngine::open(ContextConfig {
        data_dir: dir.path().join("data"),
        roots: vec![root],
        bridge: None,
        dragonfly: None,
    })
    .await
    .unwrap();

    let err = engine
        .read_file("workspace", "../outside.txt")
        .await
        .unwrap_err();
    let message = err.to_string();
    assert!(message.contains("outside managed root") || message.contains("escape"));
}

/// Verifies that incremental upserts reuse unchanged content chunks and only reindex modified chunks.
///
/// This test upserts a multi-paragraph resource, then upserts it again with a single paragraph changed
/// and asserts that the engine reports two reused chunks and one reindexed chunk.
///
/// # Examples
///
/// ```no_run
/// // Conceptual example: initial upsert followed by a small change should reuse unchanged chunks.
/// // The test harness sets up a temporary workspace and asserts reused/reindexed counts.
/// ```
#[tokio::test]
async fn incremental_upsert_reuses_unchanged_chunks() {
    let dir = tempdir().unwrap();
    let root_dir = dir.path().join("workspace");
    std::fs::create_dir_all(&root_dir).unwrap();

    let engine = ContextEngine::open(ContextConfig {
        data_dir: dir.path().join("data"),
        roots: vec![ManagedRoot {
            name: "workspace".to_string(),
            path: root_dir.clone(),
        }],
        bridge: None,
        dragonfly: None,
    })
    .await
    .unwrap();

    let initial = engine
        .upsert_resource(ResourceUpsertRequest {
            uri: "viking://resources/workspace/notes.md".to_string(),
            title: Some("Notes".to_string()),
            content: "Alpha paragraph.\n\nBeta paragraph.\n\nGamma paragraph.".to_string(),
            layer: ResourceLayer::L2,
            metadata: BTreeMap::new(),
            previous_uri: None,
        })
        .await
        .unwrap();

    let updated = engine
        .upsert_resource(ResourceUpsertRequest {
            uri: "viking://resources/workspace/notes.md".to_string(),
            title: Some("Notes".to_string()),
            content: "Alpha paragraph.\n\nBeta paragraph revised.\n\nGamma paragraph.".to_string(),
            layer: ResourceLayer::L2,
            metadata: BTreeMap::new(),
            previous_uri: Some(initial.resource.uri.clone()),
        })
        .await
        .unwrap();

    assert_eq!(updated.reused_chunks, 2);
    assert_eq!(updated.reindexed_chunks, 1);
}

/// Verifies that a text resource indexed into the engine is discoverable via lexical search.
///
/// # Examples
///
/// ```rust
/// # async fn example(engine: &ContextEngine) -> anyhow::Result<()> {
/// engine.upsert_resource(ResourceUpsertRequest {
///     uri: "viking://resources/workspace/fox.txt".to_string(),
///     title: Some("Fox".to_string()),
///     content: "The quick brown fox jumps over the lazy dog.".to_string(),
///     layer: ResourceLayer::L2,
///     metadata: std::collections::BTreeMap::new(),
///     previous_uri: None,
/// }).await?;
///
/// let hits = engine.search_context(SearchRequest {
///     query: "quick fox".to_string(),
///     scope_uri: None,
///     top_k: Some(5),
///     filters: None,
///     layer: Some(ResourceLayer::L2),
///     rerank: Some(true),
/// }).await?;
///
/// assert!(hits.iter().any(|hit| hit.uri.contains("fox.txt")));
/// # Ok(()) }
/// ```
#[tokio::test]
async fn lexical_search_finds_indexed_resource() {
    let dir = tempdir().unwrap();
    let root_dir = dir.path().join("workspace");
    std::fs::create_dir_all(&root_dir).unwrap();
    std::fs::write(
        root_dir.join("fox.txt"),
        "The quick brown fox jumps over the lazy dog.",
    )
    .unwrap();

    let engine = ContextEngine::open(ContextConfig {
        data_dir: dir.path().join("data"),
        roots: vec![ManagedRoot {
            name: "workspace".to_string(),
            path: root_dir,
        }],
        bridge: None,
        dragonfly: None,
    })
    .await
    .unwrap();

    engine
        .upsert_resource(ResourceUpsertRequest {
            uri: "viking://resources/workspace/fox.txt".to_string(),
            title: Some("Fox".to_string()),
            content: "The quick brown fox jumps over the lazy dog.".to_string(),
            layer: ResourceLayer::L2,
            metadata: BTreeMap::new(),
            previous_uri: None,
        })
        .await
        .unwrap();

    let hits = engine
        .search_context(SearchRequest {
            query: "quick fox".to_string(),
            scope_uri: None,
            top_k: Some(5),
            filters: None,
            layer: Some(ResourceLayer::L2),
            rerank: Some(true),
        })
        .await
        .unwrap();

    assert!(!hits.is_empty());
    assert!(hits.iter().any(|hit| hit.uri.contains("fox.txt")));
}

/// Verifies that writing, moving, and deleting files keep the search index synchronized with the filesystem.
///
/// The test writes a file into a managed workspace, asserts the content is discoverable via `search_context`,
/// moves the file and asserts search results reflect the new path, then deletes the file and asserts the index
/// no longer returns results for the deleted path.
///
/// # Examples
///
/// ```
/// // Run the test with `cargo test` (the test itself performs the operations against a temporary directory).
/// file_operations_keep_index_in_sync();
/// ```
#[tokio::test]
async fn file_operations_keep_index_in_sync() {
    let dir = tempdir().unwrap();
    let root_dir = dir.path().join("workspace");
    std::fs::create_dir_all(&root_dir).unwrap();

    let engine = ContextEngine::open(ContextConfig {
        data_dir: dir.path().join("data"),
        roots: vec![ManagedRoot {
            name: "workspace".to_string(),
            path: root_dir,
        }],
        bridge: None,
        dragonfly: None,
    })
    .await
    .unwrap();

    let version = engine
        .write_file(
            "workspace",
            "notes.md",
            "Launch checklist and rollout plan.",
            None,
        )
        .await
        .unwrap();
    assert!(version > 0);

    let initial_hits = engine
        .search_context(SearchRequest {
            query: "rollout plan".to_string(),
            scope_uri: Some("viking://resources/workspace/".to_string()),
            top_k: Some(5),
            filters: None,
            layer: Some(ResourceLayer::L2),
            rerank: Some(true),
        })
        .await
        .unwrap();
    assert!(initial_hits
        .iter()
        .any(|hit| hit.uri.ends_with("/notes.md")));

    let moved_version = engine
        .move_file("workspace", "notes.md", "archive/notes.md", Some(version))
        .await
        .unwrap();
    assert!(moved_version > 0);

    let moved_hits = engine
        .search_context(SearchRequest {
            query: "rollout plan".to_string(),
            scope_uri: Some("viking://resources/workspace/archive".to_string()),
            top_k: Some(5),
            filters: None,
            layer: Some(ResourceLayer::L2),
            rerank: Some(true),
        })
        .await
        .unwrap();
    assert!(moved_hits
        .iter()
        .any(|hit| hit.uri.ends_with("/archive/notes.md")));

    let deleted = engine
        .delete_file("workspace", "archive/notes.md", Some(moved_version))
        .await
        .unwrap();
    assert!(deleted);

    let final_hits = engine
        .search_context(SearchRequest {
            query: "rollout plan".to_string(),
            scope_uri: Some("viking://resources/workspace/".to_string()),
            top_k: Some(5),
            filters: None,
            layer: Some(ResourceLayer::L2),
            rerank: Some(true),
        })
        .await
        .unwrap();
    assert!(!final_hits
        .iter()
        .any(|hit| hit.uri.ends_with("/archive/notes.md")));
}

/// Verifies that syncing a managed workspace indexes new files and prunes resources for files removed from disk.
///
/// This test:
/// - Creates a temporary managed root with two files under `docs/`.
/// - Runs `sync_workspace` and asserts both files are indexed.
/// - Removes one file from disk, re-runs `sync_workspace`, and asserts the deleted resource is pruned.
/// - Confirms `list_resources` reflects the removal.
///
/// # Examples
///
/// ```rust
/// // Creates workspace with docs/alpha.md and docs/beta.md, syncs, then deletes beta.md and re-syncs.
/// ```
#[tokio::test]
async fn workspace_sync_indexes_files_and_prunes_missing_resources() {
    let dir = tempdir().unwrap();
    let root_dir = dir.path().join("workspace");
    std::fs::create_dir_all(root_dir.join("docs")).unwrap();
    std::fs::write(
        root_dir.join("docs").join("alpha.md"),
        "Alpha architecture notes.",
    )
    .unwrap();
    std::fs::write(
        root_dir.join("docs").join("beta.md"),
        "Beta context summary.",
    )
    .unwrap();

    let engine = ContextEngine::open(ContextConfig {
        data_dir: dir.path().join("data"),
        roots: vec![ManagedRoot {
            name: "workspace".to_string(),
            path: root_dir.clone(),
        }],
        bridge: None,
        dragonfly: None,
    })
    .await
    .unwrap();

    let first_sync = engine
        .sync_workspace("workspace", Some(std::path::PathBuf::from("docs")))
        .await
        .unwrap();
    assert_eq!(first_sync.indexed_resources, 2);
    assert_eq!(first_sync.deleted_resources, 0);

    std::fs::remove_file(root_dir.join("docs").join("beta.md")).unwrap();
    let second_sync = engine
        .sync_workspace("workspace", Some(std::path::PathBuf::from("docs")))
        .await
        .unwrap();
    assert_eq!(second_sync.deleted_resources, 1);

    let resources = engine.list_resources().await.unwrap();
    assert!(resources
        .iter()
        .any(|resource| resource.uri.ends_with("/docs/alpha.md")));
    assert!(!resources
        .iter()
        .any(|resource| resource.uri.ends_with("/docs/beta.md")));
}
/// Verifies that graph-typed resources are materialized into facts with provenance and that Dragonfly session recent-windowing works.
///
/// This test upserts a graph resource carrying provenance metadata (subject, object, relation, session id), asserts that
/// `graph_facts` returns the expected materialized fact with the provided identifiers and provenance, and confirms the
/// graph resource is discoverable via `search_context` when filtered by graph kind. It then appends multiple session
/// events for the same session id and verifies `recent_session_entries` returns only the most recent entries according
/// to the configured `recent_window`, while `list_sessions` reports the full session history length.
///
/// # Examples
///
/// ```no_run
/// // Demonstrates the expected high-level interactions used by the test:
/// # async fn example(engine: &ContextEngine) {
/// let facts = engine.graph_facts("Project X", 5).await.unwrap();
/// assert!(!facts.is_empty());
///
/// engine.append_session(SessionEventRequest {
///     session_id: "sess-graph".to_string(),
///     role: "user".to_string(),
///     content: "Example".to_string(),
///     metadata: std::collections::BTreeMap::new(),
/// }).await.unwrap();
///
/// let recent = engine.recent_session_entries("sess-graph", 2).await.unwrap();
/// assert!(recent.len() <= 2);
/// # }
/// ```
#[tokio::test]
async fn graph_resources_materialize_facts_with_provenance_and_recent_window() {
    let dir = tempdir().unwrap();
    let root_dir = dir.path().join("workspace");
    std::fs::create_dir_all(&root_dir).unwrap();

    let engine = ContextEngine::open(ContextConfig {
        data_dir: dir.path().join("data"),
        roots: vec![ManagedRoot {
            name: "workspace".to_string(),
            path: root_dir,
        }],
        bridge: None,
        dragonfly: Some(DragonflyConfig {
            addr: "memory://dragonfly".to_string(),
            key_prefix: "test:sessions".to_string(),
            recent_window: 2,
        }),
    })
    .await
    .unwrap();

    let graph_metadata = BTreeMap::from([
        ("kind".to_string(), "graph".to_string()),
        ("subject_id".to_string(), "project-x".to_string()),
        ("subject_type".to_string(), "project".to_string()),
        ("subject_name".to_string(), "Project X".to_string()),
        ("relation".to_string(), "USES".to_string()),
        ("object_id".to_string(), "dragonfly".to_string()),
        ("object_type".to_string(), "concept".to_string()),
        ("object_name".to_string(), "Dragonfly".to_string()),
        ("session_id".to_string(), "sess-graph".to_string()),
        ("source".to_string(), "test".to_string()),
    ]);

    engine
        .upsert_resource(ResourceUpsertRequest {
            uri: "viking://resources/workspace/graph/project-x.md".to_string(),
            title: Some("Project X Graph".to_string()),
            content: "Project X uses Dragonfly for hot memory.".to_string(),
            layer: ResourceLayer::L1,
            metadata: graph_metadata,
            previous_uri: None,
        })
        .await
        .unwrap();

    let facts = engine.graph_facts("Project X", 5).await.unwrap();
    assert_eq!(facts.len(), 1);
    assert_eq!(facts[0].subject.id, "project-x");
    assert_eq!(facts[0].subject.kind.as_str(), "project");
    assert_eq!(facts[0].relation.as_str(), "USES");
    assert_eq!(facts[0].object.id, "dragonfly");
    assert_eq!(facts[0].object.kind.as_str(), "concept");
    assert_eq!(
        facts[0].resource_uri.as_deref(),
        Some("viking://resources/workspace/graph/project-x.md")
    );
    assert_eq!(
        facts[0].metadata.get("session_id").map(String::as_str),
        Some("sess-graph")
    );

    let search = engine
        .search_context(SearchRequest {
            query: "Project X Dragonfly".to_string(),
            scope_uri: None,
            top_k: Some(5),
            filters: Some(BTreeMap::from([("kind".to_string(), "graph".to_string())])),
            layer: Some(ResourceLayer::L1),
            rerank: Some(true),
        })
        .await
        .unwrap();
    assert!(search
        .iter()
        .any(|hit| hit.uri.ends_with("/graph/project-x.md")));

    engine
        .append_session(SessionEventRequest {
            session_id: "sess-graph".to_string(),
            role: "user".to_string(),
            content: "We moved off Redis.".to_string(),
            metadata: BTreeMap::new(),
        })
        .await
        .unwrap();
    engine
        .append_session(SessionEventRequest {
            session_id: "sess-graph".to_string(),
            role: "assistant".to_string(),
            content: "Dragonfly works well.".to_string(),
            metadata: BTreeMap::new(),
        })
        .await
        .unwrap();
    engine
        .append_session(SessionEventRequest {
            session_id: "sess-graph".to_string(),
            role: "user".to_string(),
            content: "Keep that preference.".to_string(),
            metadata: BTreeMap::new(),
        })
        .await
        .unwrap();

    let recent = engine
        .recent_session_entries("sess-graph", 2)
        .await
        .unwrap();
    assert_eq!(recent.len(), 2);
    assert_eq!(recent[0].content, "Dragonfly works well.");
    assert_eq!(recent[1].content, "Keep that preference.");

    let full_history = engine.list_sessions("sess-graph").await.unwrap();
    assert_eq!(full_history.len(), 3);
}
