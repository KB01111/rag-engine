use std::collections::BTreeMap;

use context_engine::{
    ContextConfig, ContextEngine, ManagedRoot, ResourceLayer, ResourceUpsertRequest, SearchRequest,
    VikingUri,
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

    assert!(updated.reused_chunks > 0);
    assert!(updated.reindexed_chunks > 0);
}

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
