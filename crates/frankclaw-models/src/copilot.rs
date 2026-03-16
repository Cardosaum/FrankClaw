#![forbid(unsafe_code)]

//! GitHub Copilot provider.
//!
//! Exchanges a GitHub OAuth token for a Copilot API token, then delegates
//! to `OpenAiProvider` using the Copilot-derived base URL.

use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
use reqwest::Client;
use secrecy::{ExposeSecret, SecretString};
use tracing::{debug, warn};

use frankclaw_core::error::{ConfigValidation, Internal, Provider, Result};
use frankclaw_core::model::*;

use crate::OpenAiProvider;

/// Well-known GitHub OAuth client ID for Copilot.
const COPILOT_CLIENT_ID: &str = "Iv1.b507a08c87ecfe98";

/// Default Copilot models available.
const DEFAULT_MODELS: &[&str] = &[
    "claude-sonnet-4.6",
    "gpt-4o",
    "gpt-4.1",
    "gpt-4.1-mini",
    "o3-mini",
];

/// Cached Copilot API token with expiry.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct CopilotTokenCache {
    token: String,
    expires_at: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    endpoints: Option<CopilotEndpoints>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct CopilotEndpoints {
    api: String,
}

/// GitHub device flow response.
#[derive(Debug, serde::Deserialize)]
pub struct DeviceCodeResponse {
    pub device_code: String,
    pub user_code: String,
    pub verification_uri: String,
    pub interval: u64,
}

/// GitHub OAuth token exchange response.
#[derive(Debug, serde::Deserialize)]
struct TokenExchangeResponse {
    #[serde(default)]
    access_token: Option<String>,
    #[serde(default)]
    error: Option<String>,
}

/// Copilot API token exchange response.
#[derive(Debug, serde::Deserialize)]
struct CopilotTokenResponse {
    token: String,
    expires_at: u64,
    #[serde(default)]
    endpoints: Option<CopilotEndpoints>,
}

/// Resolve a Copilot API token from a GitHub OAuth token.
///
/// Caches the result in the given file path (0o600 permissions).
pub async fn resolve_copilot_api_token(
    github_token: &SecretString,
    cache_path: &Path,
) -> Result<(SecretString, String)> {
    // Try cache first
    if let Ok(cached) = read_token_cache(cache_path) {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        // Refresh 60s before expiry
        if cached.expires_at > now + 60 {
            let base_url = cached
                .endpoints
                .as_ref()
                .map(|ep| ep.api.clone())
                .unwrap_or_else(default_copilot_api_url);
            return Ok((SecretString::from(cached.token), base_url));
        }
    }

    // Exchange GitHub token for Copilot API token
    let client = Client::builder()
        .timeout(std::time::Duration::from_secs(15))
        .build()
        .map_err(|e| {
            Provider {
                msg: format!("failed to build HTTP client: {e}"),
            }
            .build()
        })?;

    let response = client
        .get("https://api.github.com/copilot_internal/v2/token")
        .header(
            "Authorization",
            format!("token {}", github_token.expose_secret()),
        )
        .header("User-Agent", "FrankClaw/0.1")
        .header("Accept", "application/json")
        .send()
        .await
        .map_err(|e| {
            Provider {
                msg: format!("Copilot token exchange failed: {e}"),
            }
            .build()
        })?;

    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        return Err(Provider {
            msg: format!("Copilot token exchange returned HTTP {status}: {body}"),
        }
        .build());
    }

    let token_response: CopilotTokenResponse = response.json().await.map_err(|e| {
        Provider {
            msg: format!("invalid Copilot token response: {e}"),
        }
        .build()
    })?;

    let base_url = token_response
        .endpoints
        .as_ref()
        .map(|ep| ep.api.clone())
        .unwrap_or_else(default_copilot_api_url);

    // Cache the token
    let cache = CopilotTokenCache {
        token: token_response.token.clone(),
        expires_at: token_response.expires_at,
        endpoints: token_response.endpoints,
    };
    if let Err(e) = write_token_cache(cache_path, &cache) {
        warn!("failed to cache Copilot token: {e}");
    }

    Ok((SecretString::from(token_response.token), base_url))
}

fn default_copilot_api_url() -> String {
    "https://api.githubcopilot.com/v1".to_string()
}

fn read_token_cache(
    path: &Path,
) -> std::result::Result<CopilotTokenCache, Box<dyn std::error::Error>> {
    let data = std::fs::read_to_string(path)?;
    let cache: CopilotTokenCache = serde_json::from_str(&data)?;
    Ok(cache)
}

fn write_token_cache(
    path: &Path,
    cache: &CopilotTokenCache,
) -> std::result::Result<(), Box<dyn std::error::Error>> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let data = serde_json::to_string_pretty(cache)?;
    std::fs::write(path, &data)?;
    restrict_file_permissions(path);
    Ok(())
}

fn restrict_file_permissions(path: &Path) {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600));
    }
}

/// Derive the Copilot API base URL by parsing the semicolon-delimited token
/// for `proxy-ep=` parameter.
pub fn derive_copilot_base_url(token: &str) -> Option<String> {
    for part in token.split(';') {
        let part = part.trim();
        if let Some(proxy_ep) = part.strip_prefix("proxy-ep=") {
            let proxy_ep = proxy_ep.trim();
            if !proxy_ep.is_empty() {
                // Convert proxy endpoint to API prefix
                // e.g., "proxy.github.com" -> "https://api.proxy.github.com/v1"
                if proxy_ep.starts_with("http://") || proxy_ep.starts_with("https://") {
                    return Some(format!("{}/v1", proxy_ep.trim_end_matches('/')));
                }
                return Some(format!("https://{}/v1", proxy_ep.trim_end_matches('/')));
            }
        }
    }
    None
}

/// GitHub OAuth device flow: request a device code.
pub async fn request_device_code() -> Result<DeviceCodeResponse> {
    let client = Client::builder()
        .timeout(std::time::Duration::from_secs(15))
        .build()
        .map_err(|e| {
            Internal {
                msg: format!("failed to build HTTP client: {e}"),
            }
            .build()
        })?;

    let response = client
        .post("https://github.com/login/device/code")
        .header("Accept", "application/json")
        .form(&[("client_id", COPILOT_CLIENT_ID), ("scope", "copilot")])
        .send()
        .await
        .map_err(|e| {
            Internal {
                msg: format!("device code request failed: {e}"),
            }
            .build()
        })?;

    if !response.status().is_success() {
        let body = response.text().await.unwrap_or_default();
        return Err(Internal {
            msg: format!("device code request failed: {body}"),
        }
        .build());
    }

    response.json().await.map_err(|e| {
        Internal {
            msg: format!("invalid device code response: {e}"),
        }
        .build()
    })
}

/// Poll for an access token after the user has entered the device code.
pub async fn poll_for_access_token(device_code: &str, interval: u64) -> Result<SecretString> {
    let client = Client::builder()
        .timeout(std::time::Duration::from_secs(15))
        .build()
        .map_err(|e| {
            Internal {
                msg: format!("failed to build HTTP client: {e}"),
            }
            .build()
        })?;

    let mut poll_interval = interval.max(5);

    loop {
        tokio::time::sleep(std::time::Duration::from_secs(poll_interval)).await;

        let response = client
            .post("https://github.com/login/oauth/access_token")
            .header("Accept", "application/json")
            .form(&[
                ("client_id", COPILOT_CLIENT_ID),
                ("device_code", device_code),
                ("grant_type", "urn:ietf:params:oauth:grant-type:device_code"),
            ])
            .send()
            .await
            .map_err(|e| {
                Internal {
                    msg: format!("token poll failed: {e}"),
                }
                .build()
            })?;

        let body: TokenExchangeResponse = response.json().await.map_err(|e| {
            Internal {
                msg: format!("invalid token response: {e}"),
            }
            .build()
        })?;

        if let Some(token) = body.access_token {
            if !token.is_empty() {
                return Ok(SecretString::from(token));
            }
        }

        match body.error.as_deref() {
            Some("authorization_pending") => {
                debug!("still waiting for user authorization");
                continue;
            }
            Some("slow_down") => {
                poll_interval += 5;
                debug!(new_interval = poll_interval, "slowing down polling");
                continue;
            }
            Some("expired_token") => {
                return Err(Internal {
                    msg: "device code expired — please restart the login flow".to_string(),
                }
                .build());
            }
            Some(other) => {
                return Err(Internal {
                    msg: format!("GitHub OAuth error: {other}"),
                }
                .build());
            }
            None => {
                return Err(Internal {
                    msg: "unexpected empty token response".to_string(),
                }
                .build());
            }
        }
    }
}

/// Store a GitHub access token securely.
pub fn store_github_token(state_dir: &Path, token: &SecretString) -> Result<PathBuf> {
    let creds_dir = state_dir.join("credentials");
    std::fs::create_dir_all(&creds_dir).map_err(|e| {
        Internal {
            msg: format!("failed to create credentials directory: {e}"),
        }
        .build()
    })?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(&creds_dir, std::fs::Permissions::from_mode(0o700));
    }

    let token_path = creds_dir.join("github-copilot.json");
    let data = serde_json::json!({
        "github_token": token.expose_secret(),
    });
    std::fs::write(
        &token_path,
        serde_json::to_string_pretty(&data).unwrap_or_default(),
    )
    .map_err(|e| {
        Internal {
            msg: format!("failed to write token file: {e}"),
        }
        .build()
    })?;
    restrict_file_permissions(&token_path);
    Ok(token_path)
}

/// Load a stored GitHub access token.
pub fn load_github_token(state_dir: &Path) -> Result<SecretString> {
    let token_path = state_dir.join("credentials/github-copilot.json");
    let data = std::fs::read_to_string(&token_path).map_err(|e| {
        ConfigValidation {
            msg: format!(
                "GitHub Copilot token not found at {}: {e}. Run: frankclaw login github-copilot",
                token_path.display()
            ),
        }
        .build()
    })?;
    let parsed: serde_json::Value = serde_json::from_str(&data).map_err(|e| {
        ConfigValidation {
            msg: format!("invalid Copilot token file: {e}"),
        }
        .build()
    })?;
    let token = parsed["github_token"].as_str().ok_or_else(|| {
        ConfigValidation {
            msg: "Copilot token file missing github_token field".to_string(),
        }
        .build()
    })?;
    Ok(SecretString::from(token.to_string()))
}

/// GitHub Copilot model provider.
///
/// Wraps `OpenAiProvider`, auto-refreshing the Copilot API token before requests.
pub struct CopilotProvider {
    id: String,
    github_token: SecretString,
    cache_path: PathBuf,
    models: Vec<String>,
    inner: tokio::sync::RwLock<Option<CopilotInner>>,
}

struct CopilotInner {
    provider: OpenAiProvider,
    expires_at: u64,
}

impl CopilotProvider {
    pub fn new(
        id: impl Into<String>,
        github_token: SecretString,
        state_dir: &Path,
        models: Vec<String>,
    ) -> Self {
        let models = if models.is_empty() {
            DEFAULT_MODELS.iter().map(|s| (*s).to_string()).collect()
        } else {
            models
        };

        let cache_path = state_dir.join("credentials/github-copilot-token.json");

        Self {
            id: id.into(),
            github_token,
            cache_path,
            models,
            inner: tokio::sync::RwLock::new(None),
        }
    }

    async fn ensure_provider(&self) -> Result<()> {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        // Check if current provider is still valid
        {
            let inner = self.inner.read().await;
            if let Some(ref inner) = *inner {
                if inner.expires_at > now + 60 {
                    return Ok(());
                }
            }
        }

        // Refresh token
        let (api_token, base_url) =
            resolve_copilot_api_token(&self.github_token, &self.cache_path).await?;

        // Read expires_at from cache
        let expires_at = read_token_cache(&self.cache_path)
            .map(|c| c.expires_at)
            .unwrap_or(now + 3600);

        let provider =
            OpenAiProvider::new(self.id.clone(), base_url, api_token, self.models.clone())?;

        let mut inner = self.inner.write().await;
        *inner = Some(CopilotInner {
            provider,
            expires_at,
        });

        Ok(())
    }
}

#[async_trait]
impl ModelProvider for CopilotProvider {
    fn id(&self) -> &str {
        &self.id
    }

    async fn complete(
        &self,
        request: CompletionRequest,
        stream_tx: Option<tokio::sync::mpsc::Sender<StreamDelta>>,
    ) -> Result<CompletionResponse> {
        self.ensure_provider().await?;
        let inner = self.inner.read().await;
        let inner = inner.as_ref().ok_or_else(|| {
            Provider {
                msg: "Copilot provider not initialized".to_string(),
            }
            .build()
        })?;
        inner.provider.complete(request, stream_tx).await
    }

    async fn list_models(&self) -> Result<Vec<ModelDef>> {
        Ok(self
            .models
            .iter()
            .map(|id| ModelDef {
                id: id.clone(),
                name: id.clone(),
                api: ModelApi::OpenaiCompletions,
                reasoning: false,
                input: vec![InputModality::Text],
                cost: ModelCost::default(),
                context_window: 128_000,
                max_output_tokens: 4096,
                compat: ModelCompat {
                    supports_tools: true,
                    supports_streaming: true,
                    supports_system_message: true,
                    ..Default::default()
                },
            })
            .collect())
    }

    async fn health(&self) -> bool {
        self.ensure_provider().await.is_ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn derive_copilot_base_url_extracts_proxy_ep() {
        let token =
            "tid=abc123;exp=1234567890;sku=free;proxy-ep=copilot-proxy.example.com;st=dotcom";
        assert_eq!(
            derive_copilot_base_url(token),
            Some("https://copilot-proxy.example.com/v1".to_string())
        );
    }

    #[test]
    fn derive_copilot_base_url_handles_https_prefix() {
        let token = "tid=abc;proxy-ep=https://proxy.example.com";
        assert_eq!(
            derive_copilot_base_url(token),
            Some("https://proxy.example.com/v1".to_string())
        );
    }

    #[test]
    fn derive_copilot_base_url_returns_none_without_proxy() {
        let token = "tid=abc;exp=1234567890;sku=free";
        assert_eq!(derive_copilot_base_url(token), None);
    }

    #[test]
    fn derive_copilot_base_url_handles_empty_proxy() {
        let token = "proxy-ep=";
        assert_eq!(derive_copilot_base_url(token), None);
    }

    #[test]
    fn default_models_are_populated() {
        assert!(!DEFAULT_MODELS.is_empty());
        assert!(DEFAULT_MODELS.contains(&"gpt-4o"));
    }

    #[test]
    fn token_cache_roundtrip() {
        let dir =
            std::env::temp_dir().join(format!("frankclaw-copilot-test-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("test-token.json");

        let cache = CopilotTokenCache {
            token: "test-token-value".into(),
            expires_at: 9999999999,
            endpoints: Some(CopilotEndpoints {
                api: "https://api.test.com".into(),
            }),
        };

        write_token_cache(&path, &cache).expect("write should succeed");
        let loaded = read_token_cache(&path).expect("read should succeed");
        assert_eq!(loaded.token, "test-token-value");
        assert_eq!(loaded.expires_at, 9999999999);
        assert_eq!(loaded.endpoints.unwrap().api, "https://api.test.com");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn store_and_load_github_token_roundtrip() {
        let dir =
            std::env::temp_dir().join(format!("frankclaw-copilot-gh-test-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);

        let token = SecretString::from("ghp_test_token_12345".to_string());
        let path = store_github_token(&dir, &token).expect("store should succeed");
        assert!(path.exists());

        let loaded = load_github_token(&dir).expect("load should succeed");
        assert_eq!(loaded.expose_secret(), "ghp_test_token_12345");

        let _ = std::fs::remove_dir_all(&dir);
    }
}
