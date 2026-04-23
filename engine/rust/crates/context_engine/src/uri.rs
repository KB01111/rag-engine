use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum UriError {
    #[error("invalid viking uri")]
    Invalid,
    #[error("unsupported namespace")]
    UnsupportedNamespace,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum VikingNamespace {
    Resources,
    UserMemories,
    AgentMemories,
    AgentSkills,
    Session,
}

impl VikingNamespace {
    pub fn as_str(&self) -> &'static str {
        match self {
            VikingNamespace::Resources => "resources",
            VikingNamespace::UserMemories => "user/memories",
            VikingNamespace::AgentMemories => "agent/memories",
            VikingNamespace::AgentSkills => "agent/skills",
            VikingNamespace::Session => "session",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VikingUri {
    namespace: VikingNamespace,
    segments: Vec<String>,
}

impl VikingUri {
    pub fn parse(input: &str) -> Result<Self, UriError> {
        let rest = input.strip_prefix("viking://").ok_or(UriError::Invalid)?;
        let segments: Vec<String> = rest
            .split('/')
            .filter(|s| !s.is_empty())
            .map(|s| s.to_string())
            .collect();

        if segments.is_empty() {
            return Err(UriError::Invalid);
        }

        match segments[0].as_str() {
            "resources" => {
                if segments.len() < 3 {
                    return Err(UriError::Invalid);
                }
                Ok(Self {
                    namespace: VikingNamespace::Resources,
                    segments: segments[1..].to_vec(),
                })
            }
            "user" if segments.get(1).map(|s| s.as_str()) == Some("memories") => {
                if segments.len() < 3 {
                    return Err(UriError::Invalid);
                }
                Ok(Self {
                    namespace: VikingNamespace::UserMemories,
                    segments: segments[2..].to_vec(),
                })
            }
            "agent" if segments.get(1).map(|s| s.as_str()) == Some("memories") => {
                if segments.len() < 3 {
                    return Err(UriError::Invalid);
                }
                Ok(Self {
                    namespace: VikingNamespace::AgentMemories,
                    segments: segments[2..].to_vec(),
                })
            }
            "agent" if segments.get(1).map(|s| s.as_str()) == Some("skills") => {
                if segments.len() < 3 {
                    return Err(UriError::Invalid);
                }
                Ok(Self {
                    namespace: VikingNamespace::AgentSkills,
                    segments: segments[2..].to_vec(),
                })
            }
            "session" => {
                if segments.len() < 2 {
                    return Err(UriError::Invalid);
                }
                Ok(Self {
                    namespace: VikingNamespace::Session,
                    segments: segments[1..].to_vec(),
                })
            }
            _ => Err(UriError::UnsupportedNamespace),
        }
    }

    pub fn resource(root: impl Into<String>, path: impl Into<String>) -> Self {
        let root_str = root.into();
        let path = path.into();
        let mut segments = vec![root_str];
        segments.extend(
            path.split('/')
                .filter(|segment| !segment.is_empty())
                .map(|segment| segment.to_string()),
        );
        Self {
            namespace: VikingNamespace::Resources,
            segments,
        }
    }

    pub fn user_memory(id: impl Into<String>) -> Result<Self, UriError> {
        let id_str = id.into();
        if id_str.is_empty() {
            return Err(UriError::Invalid);
        }
        Ok(Self {
            namespace: VikingNamespace::UserMemories,
            segments: vec![id_str],
        })
    }

    pub fn agent_memory(id: impl Into<String>) -> Result<Self, UriError> {
        let id_str = id.into();
        if id_str.is_empty() {
            return Err(UriError::Invalid);
        }
        Ok(Self {
            namespace: VikingNamespace::AgentMemories,
            segments: vec![id_str],
        })
    }

    pub fn agent_skill(id: impl Into<String>) -> Result<Self, UriError> {
        let id_str = id.into();
        if id_str.is_empty() {
            return Err(UriError::Invalid);
        }
        Ok(Self {
            namespace: VikingNamespace::AgentSkills,
            segments: vec![id_str],
        })
    }

    pub fn session(id: impl Into<String>) -> Result<Self, UriError> {
        let id_str = id.into();
        if id_str.is_empty() {
            return Err(UriError::Invalid);
        }
        Ok(Self {
            namespace: VikingNamespace::Session,
            segments: vec![id_str],
        })
    }

    pub fn namespace(&self) -> &VikingNamespace {
        &self.namespace
    }

    pub fn resource_root(&self) -> Option<&str> {
        (self.namespace == VikingNamespace::Resources).then_some(self.segments.first()?.as_str())
    }

    pub fn resource_path(&self) -> Option<String> {
        (self.namespace == VikingNamespace::Resources && self.segments.len() > 1)
            .then(|| self.segments[1..].join("/"))
    }

    pub fn leaf_id(&self) -> Option<&str> {
        self.segments.last().map(|segment| segment.as_str())
    }
}

impl std::fmt::Display for VikingUri {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "viking://{}", self.namespace.as_str())?;
        for segment in &self.segments {
            write!(f, "/{}", segment)?;
        }
        Ok(())
    }
}
