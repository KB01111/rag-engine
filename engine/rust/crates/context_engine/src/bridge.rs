use std::collections::BTreeMap;

use reqwest::Client;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use url::Url;

use crate::model::ResourceLayer;

#[derive(Debug, Error)]
pub enum BridgeError {
    #[error("bridge is disabled")]
    Disabled,
    #[error("request failed: {0}")]
    Request(#[from] reqwest::Error),
    #[error("invalid bridge response: {0}")]
    Response(String),
    #[error("url error: {0}")]
    Url(#[from] url::ParseError),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenVikingBridgeConfig {
    pub base_url: String,
    pub token: Option<String>,
    pub import_path: String,
    pub sync_path: String,
    pub find_path: String,
    pub read_path: String,
}

impl OpenVikingBridgeConfig {
    pub fn new(base_url: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into(),
            token: None,
            import_path: "/v1/resources".to_string(),
            sync_path: "/v1/resources/sync".to_string(),
            find_path: "/v1/search".to_string(),
            read_path: "/v1/resources/read".to_string(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RemoteResourcePayload {
    pub uri: String,
    pub title: String,
    pub content: String,
    pub layer: ResourceLayer,
    pub metadata: BTreeMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RemoteResourceSummary {
    pub uri: String,
    pub title: String,
    pub layer: String,
    pub metadata: BTreeMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RemoteSearchRequest {
    pub query: String,
    pub top_k: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RemoteSearchResponse {
    pub resources: Vec<RemoteResourceSummary>,
}

#[derive(Clone)]
pub struct OpenVikingBridgeClient {
    client: Client,
    config: OpenVikingBridgeConfig,
}

impl OpenVikingBridgeClient {
    pub fn new(config: OpenVikingBridgeConfig) -> Self {
        Self {
            client: Client::new(),
            config,
        }
    }

    fn base_url(&self) -> Result<Url, BridgeError> {
        Ok(Url::parse(&self.config.base_url)?)
    }

    fn request(&self, path: &str) -> Result<reqwest::RequestBuilder, BridgeError> {
        let url = self.base_url()?.join(path.trim_start_matches('/'))?;
        let mut request = self
            .client
            .post(url)
            .header("content-type", "application/json");
        if let Some(token) = &self.config.token {
            request = request.bearer_auth(token);
        }
        Ok(request)
    }

    fn get_request(&self, path: &str) -> Result<reqwest::RequestBuilder, BridgeError> {
        let url = self.base_url()?.join(path.trim_start_matches('/'))?;
        let mut request = self.client.get(url);
        if let Some(token) = &self.config.token {
            request = request.bearer_auth(token);
        }
        Ok(request)
    }

    pub async fn import_resource(
        &self,
        payload: RemoteResourcePayload,
    ) -> Result<RemoteResourceSummary, BridgeError> {
        let response = self
            .request(&self.config.import_path)?
            .json(&payload)
            .send()
            .await?
            .error_for_status()?;
        let body = response.json::<RemoteResourceSummary>().await?;
        Ok(body)
    }

    pub async fn sync_resource(
        &self,
        payload: RemoteResourcePayload,
    ) -> Result<RemoteResourceSummary, BridgeError> {
        let response = self
            .request(&self.config.sync_path)?
            .json(&payload)
            .send()
            .await?
            .error_for_status()?;
        let body = response.json::<RemoteResourceSummary>().await?;
        Ok(body)
    }

    pub async fn find_resources(
        &self,
        query: impl Into<String>,
        top_k: usize,
    ) -> Result<Vec<RemoteResourceSummary>, BridgeError> {
        let response = self
            .request(&self.config.find_path)?
            .json(&RemoteSearchRequest {
                query: query.into(),
                top_k,
            })
            .send()
            .await?
            .error_for_status()?;
        let body = response.json::<RemoteSearchResponse>().await?;
        Ok(body.resources)
    }

    pub async fn read_resource(
        &self,
        uri: impl Into<String>,
    ) -> Result<RemoteResourcePayload, BridgeError> {
        let response = self
            .get_request(&format!(
                "{}/{}",
                self.config.read_path.trim_end_matches('/'),
                uri.into()
            ))?
            .send()
            .await?
            .error_for_status()?;
        let body = response.json::<RemoteResourcePayload>().await?;
        Ok(body)
    }
}
