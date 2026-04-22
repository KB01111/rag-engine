pub mod bridge;
pub mod engine;
pub mod model;
pub mod uri;

pub use bridge::{
    OpenVikingBridgeClient, OpenVikingBridgeConfig, RemoteResourcePayload, RemoteResourceSummary,
    RemoteSearchRequest, RemoteSearchResponse,
};
pub use engine::{ContextConfig, ContextEngine, ContextError, ManagedRoot};
pub use model::{
    DeleteOutcome, FileEntry, LayeredContent, ResourceLayer, ResourceSummary,
    ResourceUpsertRequest, SearchHit, SearchRequest, SessionEntry, SessionEventRequest,
    StatusResponse, UpsertOutcome, WorkspaceSyncOutcome,
};
pub use uri::{VikingNamespace, VikingUri};
