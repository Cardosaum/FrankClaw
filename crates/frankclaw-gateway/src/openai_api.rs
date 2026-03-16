//! OpenAI-compatible HTTP API endpoints.
//!
//! Implements `/v1/chat/completions` and `/v1/models` so FrankClaw can serve as
//! a drop-in replacement for OpenAI-compatible clients (Continue, Cursor, etc.).

use std::sync::Arc;

use axum::{
    Json,
    extract::State,
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
};
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use frankclaw_core::model::StreamDelta;

use crate::state::GatewayState;

/// Authenticate using Bearer token from the Authorization header.
/// Falls back to the gateway's configured auth mode.
fn authenticate_bearer(
    state: &Arc<GatewayState>,
    headers: &HeaderMap,
) -> Result<(), (StatusCode, Json<ErrorResponse>)> {
    let config = state.current_config();

    // Extract Bearer token.
    let token = headers
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "))
        .map(str::to_string);

    let credential = match (&config.gateway.auth, token) {
        (frankclaw_core::auth::AuthMode::None, _) => return Ok(()),
        (_, Some(t)) => crate::auth::AuthCredential::BearerToken(secrecy::SecretString::from(t)),
        (_, None) => {
            return Err((
                StatusCode::UNAUTHORIZED,
                Json(ErrorResponse::new(
                    "invalid_api_key",
                    "Missing Authorization: Bearer <token> header",
                )),
            ));
        }
    };

    // Use a no-op rate limiter for the API path — the main rate limiter
    // requires SocketAddr which we don't always have here.
    let limiter = crate::rate_limit::AuthRateLimiter::new(Default::default());
    match crate::auth::authenticate(&config.gateway.auth, &credential, None, &limiter) {
        Ok(_) => Ok(()),
        Err(_) => Err((
            StatusCode::UNAUTHORIZED,
            Json(ErrorResponse::new("invalid_api_key", "Invalid API key")),
        )),
    }
}

// --------------------------------------------------------------------------
// Request / Response types (OpenAI-compatible)
// --------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ApiMessage>,
    #[serde(default)]
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub stream: Option<bool>,
}

#[derive(Debug, Deserialize)]
pub struct ApiMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: &'static str,
    pub created: i64,
    pub model: String,
    pub choices: Vec<Choice>,
    pub usage: ApiUsage,
}

#[derive(Debug, Serialize)]
pub struct Choice {
    pub index: u32,
    pub message: ApiResponseMessage,
    pub finish_reason: &'static str,
}

#[derive(Debug, Serialize)]
pub struct ApiResponseMessage {
    pub role: &'static str,
    pub content: String,
}

#[derive(Debug, Serialize)]
pub struct ApiUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

#[derive(Debug, Serialize)]
pub struct StreamChunk {
    pub id: String,
    pub object: &'static str,
    pub created: i64,
    pub model: String,
    pub choices: Vec<StreamChoice>,
}

#[derive(Debug, Serialize)]
pub struct StreamChoice {
    pub index: u32,
    pub delta: StreamDeltaContent,
    pub finish_reason: Option<&'static str>,
}

#[derive(Debug, Serialize)]
pub struct StreamDeltaContent {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct ErrorResponse {
    pub error: ApiError,
}

#[derive(Debug, Serialize)]
pub struct ApiError {
    pub message: String,
    #[serde(rename = "type")]
    pub error_type: String,
    pub code: Option<String>,
}

impl ErrorResponse {
    fn new(error_type: &str, message: &str) -> Self {
        Self {
            error: ApiError {
                message: message.into(),
                error_type: error_type.into(),
                code: None,
            },
        }
    }
}

#[derive(Debug, Serialize)]
pub struct ModelsResponse {
    pub object: &'static str,
    pub data: Vec<ModelObject>,
}

#[derive(Debug, Serialize)]
pub struct ModelObject {
    pub id: String,
    pub object: &'static str,
    pub created: i64,
    pub owned_by: String,
}

// --------------------------------------------------------------------------
// Handler: POST /v1/chat/completions
// --------------------------------------------------------------------------

pub async fn chat_completions_handler(
    State(state): State<Arc<GatewayState>>,
    headers: HeaderMap,
    Json(body): Json<ChatCompletionRequest>,
) -> Response {
    if let Err(resp) = authenticate_bearer(&state, &headers) {
        return resp.into_response();
    }

    let stream = body.stream.unwrap_or(false);

    // Extract the last user message as the chat input.
    let message = body
        .messages
        .iter()
        .rev()
        .find(|m| m.role == "user")
        .map(|m| m.content.clone())
        .unwrap_or_default();

    if message.trim().is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse::new(
                "invalid_request_error",
                "No user message provided",
            )),
        )
            .into_response();
    }

    let cancel_token = CancellationToken::new();
    let request_id = format!("chatcmpl-{}", uuid::Uuid::new_v4());

    if stream {
        handle_streaming(state, body, message, request_id, cancel_token).await
    } else {
        handle_non_streaming(state, body, message, request_id, cancel_token).await
    }
}

async fn handle_non_streaming(
    state: Arc<GatewayState>,
    body: ChatCompletionRequest,
    message: String,
    request_id: String,
    cancel_token: CancellationToken,
) -> Response {
    let result = state
        .runtime
        .chat(frankclaw_runtime::ChatRequest {
            agent_id: None,
            session_key: None,
            message,
            attachments: Vec::new(),
            model_id: Some(body.model.clone()),
            max_tokens: body.max_tokens,
            temperature: body.temperature,
            stream_tx: None,
            thinking_budget: None,
            channel_id: None,
            channel_capabilities: None,
            canvas: Some(state.canvas.clone()),
            cancel_token: Some(cancel_token),
            approval_tx: None,
        })
        .await;

    match result {
        Ok(response) => {
            let created = chrono::Utc::now().timestamp();
            Json(ChatCompletionResponse {
                id: request_id,
                object: "chat.completion",
                created,
                model: response.model_id,
                choices: vec![Choice {
                    index: 0,
                    message: ApiResponseMessage {
                        role: "assistant",
                        content: response.content,
                    },
                    finish_reason: "stop",
                }],
                usage: ApiUsage {
                    prompt_tokens: response.usage.input_tokens,
                    completion_tokens: response.usage.output_tokens,
                    total_tokens: response.usage.input_tokens + response.usage.output_tokens,
                },
            })
            .into_response()
        }
        Err(err) => (
            StatusCode::from_u16(err.status_code()).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
            Json(ErrorResponse::new("server_error", &err.to_string())),
        )
            .into_response(),
    }
}

async fn handle_streaming(
    state: Arc<GatewayState>,
    body: ChatCompletionRequest,
    message: String,
    request_id: String,
    cancel_token: CancellationToken,
) -> Response {
    let (delta_tx, mut delta_rx) = mpsc::channel::<StreamDelta>(64);
    let model_id = body.model.clone();
    let created = chrono::Utc::now().timestamp();

    // Spawn the chat call.
    tokio::spawn(async move {
        let _ = state
            .runtime
            .chat(frankclaw_runtime::ChatRequest {
                agent_id: None,
                session_key: None,
                message,
                attachments: Vec::new(),
                model_id: Some(body.model),
                max_tokens: body.max_tokens,
                temperature: body.temperature,
                stream_tx: Some(delta_tx),
                thinking_budget: None,
                channel_id: None,
                channel_capabilities: None,
                canvas: Some(state.canvas.clone()),
                cancel_token: Some(cancel_token),
                approval_tx: None,
            })
            .await;
    });

    // Build SSE stream using a channel-based approach (no async_stream dep).
    let (sse_tx, sse_rx) = mpsc::channel::<Result<String, std::convert::Infallible>>(64);

    tokio::spawn(async move {
        // Initial chunk with role.
        let initial = StreamChunk {
            id: request_id.clone(),
            object: "chat.completion.chunk",
            created,
            model: model_id.clone(),
            choices: vec![StreamChoice {
                index: 0,
                delta: StreamDeltaContent {
                    role: Some("assistant"),
                    content: None,
                },
                finish_reason: None,
            }],
        };
        let _ = sse_tx
            .send(Ok(format!(
                "data: {}\n\n",
                serde_json::to_string(&initial).unwrap_or_default()
            )))
            .await;

        while let Some(delta) = delta_rx.recv().await {
            let msg = match delta {
                StreamDelta::Text(text) => {
                    let chunk = StreamChunk {
                        id: request_id.clone(),
                        object: "chat.completion.chunk",
                        created,
                        model: model_id.clone(),
                        choices: vec![StreamChoice {
                            index: 0,
                            delta: StreamDeltaContent {
                                role: None,
                                content: Some(text),
                            },
                            finish_reason: None,
                        }],
                    };
                    format!(
                        "data: {}\n\n",
                        serde_json::to_string(&chunk).unwrap_or_default()
                    )
                }
                StreamDelta::Done { .. } => {
                    let chunk = StreamChunk {
                        id: request_id.clone(),
                        object: "chat.completion.chunk",
                        created,
                        model: model_id.clone(),
                        choices: vec![StreamChoice {
                            index: 0,
                            delta: StreamDeltaContent {
                                role: None,
                                content: None,
                            },
                            finish_reason: Some("stop"),
                        }],
                    };
                    let done = format!(
                        "data: {}\n\ndata: [DONE]\n\n",
                        serde_json::to_string(&chunk).unwrap_or_default()
                    );
                    let _ = sse_tx.send(Ok(done)).await;
                    break;
                }
                StreamDelta::Error(msg) => {
                    let err = serde_json::json!({
                        "error": { "message": msg, "type": "server_error" }
                    });
                    let _ = sse_tx.send(Ok(format!("data: {}\n\n", err))).await;
                    break;
                }
                // Skip tool-related deltas — the OpenAI compat API is text-only.
                _ => continue,
            };
            if sse_tx.send(Ok(msg)).await.is_err() {
                break;
            }
        }
    });

    let body = axum::body::Body::from_stream(tokio_stream::wrappers::ReceiverStream::new(sse_rx));

    Response::builder()
        .status(StatusCode::OK)
        .header("content-type", "text/event-stream")
        .header("cache-control", "no-cache")
        .header("connection", "keep-alive")
        .body(body)
        .unwrap_or_else(|_| StatusCode::INTERNAL_SERVER_ERROR.into_response())
}

// --------------------------------------------------------------------------
// Handler: GET /v1/models
// --------------------------------------------------------------------------

pub async fn models_handler(
    State(state): State<Arc<GatewayState>>,
    headers: HeaderMap,
) -> Response {
    if let Err(resp) = authenticate_bearer(&state, &headers) {
        return resp.into_response();
    }

    let models: Vec<ModelObject> = state
        .runtime
        .list_models()
        .iter()
        .map(|m| ModelObject {
            id: m.id.clone(),
            object: "model",
            created: 0,
            owned_by: format!("{:?}", m.api).to_lowercase(),
        })
        .collect();

    Json(ModelsResponse {
        object: "list",
        data: models,
    })
    .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_response_serializes_correctly() {
        let err = ErrorResponse::new("invalid_request_error", "test message");
        let json = serde_json::to_value(&err).unwrap();
        assert_eq!(json["error"]["type"], "invalid_request_error");
        assert_eq!(json["error"]["message"], "test message");
    }

    #[test]
    fn chat_completion_response_matches_openai_format() {
        let resp = ChatCompletionResponse {
            id: "chatcmpl-123".into(),
            object: "chat.completion",
            created: 1000,
            model: "gpt-4".into(),
            choices: vec![Choice {
                index: 0,
                message: ApiResponseMessage {
                    role: "assistant",
                    content: "Hello!".into(),
                },
                finish_reason: "stop",
            }],
            usage: ApiUsage {
                prompt_tokens: 10,
                completion_tokens: 5,
                total_tokens: 15,
            },
        };
        let json = serde_json::to_value(&resp).unwrap();
        assert_eq!(json["object"], "chat.completion");
        assert_eq!(json["choices"][0]["message"]["role"], "assistant");
        assert_eq!(json["choices"][0]["finish_reason"], "stop");
        assert_eq!(json["usage"]["total_tokens"], 15);
    }

    #[test]
    fn stream_chunk_matches_openai_format() {
        let chunk = StreamChunk {
            id: "chatcmpl-123".into(),
            object: "chat.completion.chunk",
            created: 1000,
            model: "gpt-4".into(),
            choices: vec![StreamChoice {
                index: 0,
                delta: StreamDeltaContent {
                    role: None,
                    content: Some("Hello".into()),
                },
                finish_reason: None,
            }],
        };
        let json = serde_json::to_value(&chunk).unwrap();
        assert_eq!(json["object"], "chat.completion.chunk");
        assert_eq!(json["choices"][0]["delta"]["content"], "Hello");
        assert!(json["choices"][0]["delta"].get("role").is_none());
        assert!(json["choices"][0]["finish_reason"].is_null());
    }

    #[test]
    fn models_response_matches_openai_format() {
        let resp = ModelsResponse {
            object: "list",
            data: vec![ModelObject {
                id: "gpt-4".into(),
                object: "model",
                created: 0,
                owned_by: "openai".into(),
            }],
        };
        let json = serde_json::to_value(&resp).unwrap();
        assert_eq!(json["object"], "list");
        assert_eq!(json["data"][0]["id"], "gpt-4");
        assert_eq!(json["data"][0]["object"], "model");
    }
}
