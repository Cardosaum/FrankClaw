use std::net::SocketAddr;
use std::sync::Arc;

use axum::{
    extract::{
        ConnectInfo, Json, State, WebSocketUpgrade,
    },
    http::{HeaderMap, StatusCode},
    response::IntoResponse,
    routing::{get, post},
    Router,
};
use tower_http::{
    compression::CompressionLayer,
    cors::{Any, CorsLayer},
    trace::TraceLayer,
};
use tracing::info;

use frankclaw_core::channel::InboundMessage;
use frankclaw_core::config::{BindMode, FrankClawConfig};
use frankclaw_runtime::Runtime;
use frankclaw_sessions::SqliteSessionStore;

use crate::auth::{authenticate, validate_bind_auth, AuthCredential};
use crate::rate_limit::AuthRateLimiter;
use crate::state::GatewayState;

/// Build and start the gateway server.
pub async fn run(
    config: FrankClawConfig,
    sessions: Arc<SqliteSessionStore>,
    runtime: Arc<Runtime>,
) -> anyhow::Result<()> {
    // Validate that bind + auth combination is safe.
    validate_bind_auth(&config.gateway.bind, &config.gateway.auth)?;

    let rate_limiter = Arc::new(AuthRateLimiter::new(config.gateway.rate_limit.clone()));
    let bind_addr = resolve_bind_addr(&config.gateway.bind, config.gateway.port);
    let channels = Arc::new(frankclaw_channels::load_from_config(&config)?);
    let state = GatewayState::new(config, sessions, runtime, channels);
    start_channel_runtime(state.clone());

    let app = build_router(state.clone(), rate_limiter);

    let listener = tokio::net::TcpListener::bind(&bind_addr).await?;
    info!(%bind_addr, "gateway listening");

    axum::serve(
        listener,
        app.into_make_service_with_connect_info::<SocketAddr>(),
    )
    .with_graceful_shutdown(shutdown_signal(state.shutdown.clone()))
    .await?;

    info!("gateway stopped");
    Ok(())
}

fn build_router(
    state: Arc<GatewayState>,
    rate_limiter: Arc<AuthRateLimiter>,
) -> Router {
    Router::new()
        // WebSocket endpoint.
        .route("/ws", get(ws_handler))
        // Health probes (no auth required).
        .route("/health", get(health_handler))
        .route("/ready", get(readiness_handler))
        // Local web channel ingress / polling.
        .route("/api/web/inbound", post(web_inbound_handler))
        .route("/api/web/outbound", get(web_outbound_handler))
        // State.
        .with_state(AppState {
            gateway: state,
            rate_limiter,
        })
        // Middleware layers.
        .layer(TraceLayer::new_for_http())
        .layer(CompressionLayer::new())
        .layer(
            CorsLayer::new()
                .allow_origin(Any)
                .allow_methods(Any)
                .allow_headers(Any),
        )
}

#[derive(Clone)]
struct AppState {
    gateway: Arc<GatewayState>,
    rate_limiter: Arc<AuthRateLimiter>,
}

/// WebSocket upgrade handler with auth.
async fn ws_handler(
    ws: WebSocketUpgrade,
    State(state): State<AppState>,
    ConnectInfo(addr): ConnectInfo<SocketAddr>,
    headers: HeaderMap,
) -> impl IntoResponse {
    let config = state.gateway.current_config();
    // Extract credential from the configured auth mode.
    let credential = extract_credential(&headers, &config.gateway.auth);

    // Authenticate.
    match authenticate(
        &config.gateway.auth,
        &credential,
        Some(&addr),
        &state.rate_limiter,
    ) {
        Ok(role) => {
            let conn_id = state.gateway.alloc_conn_id();
            let gw = state.gateway.clone();

            ws.on_upgrade(move |socket| {
                crate::ws::handle_ws_connection(socket, gw, conn_id, role, Some(addr))
            })
            .into_response()
        }
        Err(e) => {
            let status = StatusCode::from_u16(e.status_code()).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
            (status, e.to_string()).into_response()
        }
    }
}

/// Extract auth credential from HTTP headers.
fn extract_credential(
    headers: &HeaderMap,
    mode: &frankclaw_core::auth::AuthMode,
) -> AuthCredential {
    match mode {
        frankclaw_core::auth::AuthMode::Token { .. } => {
            if let Some(auth) = headers.get("authorization") {
                if let Ok(value) = auth.to_str() {
                    if let Some(token) = value.strip_prefix("Bearer ") {
                        return AuthCredential::BearerToken(secrecy::SecretString::from(
                            token.to_string(),
                        ));
                    }
                }
            }
        }
        frankclaw_core::auth::AuthMode::Password { .. } => {
            if let Some(password) = headers.get("x-frankclaw-password") {
                if let Ok(value) = password.to_str() {
                    return AuthCredential::Password(secrecy::SecretString::from(
                        value.to_string(),
                    ));
                }
            }
            if let Some(auth) = headers.get("authorization") {
                if let Ok(value) = auth.to_str() {
                    if let Some(password) = value.strip_prefix("Password ") {
                        return AuthCredential::Password(secrecy::SecretString::from(
                            password.to_string(),
                        ));
                    }
                }
            }
        }
        frankclaw_core::auth::AuthMode::TrustedProxy { identity_header } => {
            if let Some(identity) = headers.get(identity_header.as_str()) {
                if let Ok(value) = identity.to_str() {
                    return AuthCredential::ProxyIdentity(value.to_string());
                }
            }
        }
        frankclaw_core::auth::AuthMode::Tailscale => {
            for header_name in [
                "tailscale-user-login",
                "tailscale-user-name",
                "x-tailscale-user-login",
            ] {
                if let Some(identity) = headers.get(header_name) {
                    if let Ok(value) = identity.to_str() {
                        return AuthCredential::TailscaleIdentity(value.to_string());
                    }
                }
            }
        }
        frankclaw_core::auth::AuthMode::None => {}
    }

    AuthCredential::None
}

/// Health check (always 200 — proves the process is running).
async fn health_handler() -> StatusCode {
    StatusCode::OK
}

/// Readiness check (200 when gateway is ready to serve).
async fn readiness_handler(State(state): State<AppState>) -> StatusCode {
    if state.gateway.shutdown.is_cancelled() {
        StatusCode::SERVICE_UNAVAILABLE
    } else {
        StatusCode::OK
    }
}

#[derive(Debug, serde::Deserialize)]
struct WebInboundRequest {
    sender_id: String,
    message: String,
    #[serde(default = "default_web_account_id")]
    account_id: String,
    sender_name: Option<String>,
    thread_id: Option<String>,
    #[serde(default)]
    is_group: bool,
    #[serde(default)]
    is_mention: bool,
}

fn default_web_account_id() -> String {
    "default".to_string()
}

async fn web_inbound_handler(
    State(state): State<AppState>,
    ConnectInfo(addr): ConnectInfo<SocketAddr>,
    headers: HeaderMap,
    Json(body): Json<WebInboundRequest>,
) -> impl IntoResponse {
    if let Err(response) = require_http_auth(&state, addr, &headers) {
        return response;
    }

    let inbound = InboundMessage {
        channel: frankclaw_core::types::ChannelId::new("web"),
        account_id: body.account_id,
        sender_id: body.sender_id,
        sender_name: body.sender_name,
        thread_id: body.thread_id,
        is_group: body.is_group,
        is_mention: body.is_mention,
        text: Some(body.message),
        attachments: Vec::new(),
        platform_message_id: None,
        timestamp: chrono::Utc::now(),
    };

    match process_inbound_message(state.gateway.clone(), inbound).await {
        Ok(()) => (
            StatusCode::ACCEPTED,
            Json(serde_json::json!({ "status": "accepted" })),
        )
            .into_response(),
        Err(err) => (
            StatusCode::from_u16(err.status_code()).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
            Json(serde_json::json!({ "error": err.to_string() })),
        )
            .into_response(),
    }
}

async fn web_outbound_handler(
    State(state): State<AppState>,
    ConnectInfo(addr): ConnectInfo<SocketAddr>,
    headers: HeaderMap,
) -> impl IntoResponse {
    if let Err(response) = require_http_auth(&state, addr, &headers) {
        return response;
    }

    let Some(web) = state.gateway.web_channel() else {
        return (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({ "error": "web channel not configured" })),
        )
            .into_response();
    };

    let messages = web.drain_outbound().await;
    (
        StatusCode::OK,
        Json(serde_json::json!({ "messages": messages })),
    )
        .into_response()
}

fn resolve_bind_addr(mode: &BindMode, port: u16) -> String {
    match mode {
        BindMode::Loopback => format!("127.0.0.1:{port}"),
        BindMode::Lan => format!("0.0.0.0:{port}"),
        BindMode::Address(addr) => format!("{addr}:{port}"),
    }
}

async fn shutdown_signal(token: tokio_util::sync::CancellationToken) {
    tokio::select! {
        _ = token.cancelled() => {}
        _ = tokio::signal::ctrl_c() => {
            info!("received ctrl-c, initiating graceful shutdown");
            token.cancel();
        }
    }
}

fn require_http_auth(
    state: &AppState,
    addr: SocketAddr,
    headers: &HeaderMap,
) -> std::result::Result<(), axum::response::Response> {
    let config = state.gateway.current_config();
    let credential = extract_credential(headers, &config.gateway.auth);
    match authenticate(
        &config.gateway.auth,
        &credential,
        Some(&addr),
        &state.rate_limiter,
    ) {
        Ok(_) => Ok(()),
        Err(err) => Err((
            StatusCode::from_u16(err.status_code())
                .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
            err.to_string(),
        )
            .into_response()),
    }
}

fn start_channel_runtime(state: Arc<GatewayState>) {
    let (inbound_tx, mut inbound_rx) = tokio::sync::mpsc::channel::<InboundMessage>(256);

    for plugin in state.channels.channels().values() {
        let plugin = plugin.clone();
        let tx = inbound_tx.clone();
        tokio::spawn(async move {
            if let Err(err) = plugin.start(tx).await {
                tracing::error!(channel = %plugin.id(), error = %err, "channel stopped with error");
            }
        });
    }

    tokio::spawn(async move {
        while let Some(inbound) = inbound_rx.recv().await {
            if let Err(err) = process_inbound_message(state.clone(), inbound).await {
                tracing::warn!(error = %err, "inbound message processing failed");
            }
        }
    });
}

async fn process_inbound_message(
    state: Arc<GatewayState>,
    inbound: InboundMessage,
) -> frankclaw_core::error::Result<()> {
    let config = state.current_config();
    let text = inbound
        .text
        .as_deref()
        .map(str::trim)
        .filter(|text| !text.is_empty())
        .ok_or_else(|| frankclaw_core::error::FrankClawError::InvalidRequest {
            msg: "inbound message text is required".into(),
        })?;

    if text.len() > config.security.max_webhook_body_bytes {
        return Err(frankclaw_core::error::FrankClawError::RequestTooLarge {
            max_bytes: config.security.max_webhook_body_bytes,
        });
    }

    if inbound.is_group && !inbound.is_mention {
        return Ok(());
    }

    let session_key = state.runtime.session_key_for_inbound(&inbound);

    let response = state
        .runtime
        .chat(frankclaw_runtime::ChatRequest {
            agent_id: None,
            session_key: Some(session_key.clone()),
            message: text.to_string(),
            model_id: None,
            max_tokens: None,
            temperature: None,
        })
        .await?;

    if let Some(channel) = state.channel(&inbound.channel) {
        let _ = channel
            .send(frankclaw_core::channel::OutboundMessage {
                channel: inbound.channel.clone(),
                account_id: inbound.account_id.clone(),
                to: inbound.sender_id.clone(),
                thread_id: inbound.thread_id.clone(),
                text: response.content.clone(),
                attachments: Vec::new(),
                reply_to: inbound.platform_message_id.clone(),
            })
            .await?;
    }

    let event = frankclaw_core::protocol::Frame::Event(
        frankclaw_core::protocol::EventFrame {
            event: frankclaw_core::protocol::EventType::ChatComplete,
            payload: serde_json::json!({
                "channel": inbound.channel.as_str(),
                "account_id": inbound.account_id,
                "session_key": session_key.as_str(),
                "content": response.content,
            }),
        },
    );
    if let Ok(json) = serde_json::to_string(&event) {
        let _ = state.broadcast.send(json);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::{HeaderMap, HeaderValue};
    use secrecy::ExposeSecret;

    #[test]
    fn extracts_password_header_for_password_mode() {
        let mut headers = HeaderMap::new();
        headers.insert("x-frankclaw-password", HeaderValue::from_static("secret"));

        match extract_credential(
            &headers,
            &frankclaw_core::auth::AuthMode::Password {
                hash: "hash".into(),
            },
        ) {
            AuthCredential::Password(password) => {
                assert_eq!(password.expose_secret(), "secret");
            }
            _ => panic!("expected password credential"),
        }
    }

    #[test]
    fn extracts_trusted_proxy_identity() {
        let mut headers = HeaderMap::new();
        headers.insert("x-auth-user", HeaderValue::from_static("alice@example.com"));

        match extract_credential(
            &headers,
            &frankclaw_core::auth::AuthMode::TrustedProxy {
                identity_header: "x-auth-user".into(),
            },
        ) {
            AuthCredential::ProxyIdentity(identity) => {
                assert_eq!(identity, "alice@example.com");
            }
            _ => panic!("expected proxy identity"),
        }
    }
}
