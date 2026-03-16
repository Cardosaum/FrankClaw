//! ACP protocol server: JSON-RPC 2.0 over NDJSON stdin/stdout.
//!
//! Implements the Agent Communication Protocol for interop with other
//! AI agent frameworks. Reads requests from stdin, writes responses to stdout.

use std::sync::Arc;
use std::time::{Duration, Instant};

use dashmap::DashMap;
use tokio::sync::mpsc;
use tracing::debug;

use frankclaw_core::model::StreamDelta;
use frankclaw_core::types::SessionKey;
use frankclaw_runtime::{ChatRequest, Runtime};

use crate::acp_transport::{
    self, INTERNAL_ERROR, INVALID_PARAMS, JsonRpcRequest, JsonRpcResponse, METHOD_NOT_FOUND,
    RATE_LIMITED, SESSION_NOT_FOUND,
};

/// ACP server configuration.
#[derive(Debug, Clone)]
pub struct AcpServerOptions {
    /// Maximum concurrent sessions.
    pub max_sessions: usize,
    /// Session TTL before eviction.
    pub session_ttl: Duration,
    /// Session creation rate limit: max requests per 10-second window.
    pub rate_limit_per_window: u32,
}

impl Default for AcpServerOptions {
    fn default() -> Self {
        Self {
            max_sessions: 5000,
            session_ttl: Duration::from_secs(24 * 60 * 60), // 24h
            rate_limit_per_window: 120,
        }
    }
}

/// An active ACP session.
#[derive(Debug, Clone)]
pub struct AcpSession {
    pub session_id: String,
    pub session_key: SessionKey,
    pub created_at: Instant,
    pub last_touched: Instant,
}

/// ACP server state.
pub struct AcpServer {
    runtime: Arc<Runtime>,
    sessions: DashMap<String, AcpSession>,
    options: AcpServerOptions,
    canvas: Arc<crate::canvas::CanvasStore>,
    // Simple rate limiter: (window_start, count).
    rate_window: std::sync::Mutex<(Instant, u32)>,
}

impl AcpServer {
    pub fn new(runtime: Arc<Runtime>, options: AcpServerOptions) -> Self {
        Self {
            runtime,
            sessions: DashMap::new(),
            options,
            canvas: crate::canvas::CanvasStore::new(),
            rate_window: std::sync::Mutex::new((Instant::now(), 0)),
        }
    }

    /// Handle a single JSON-RPC request.
    pub async fn handle_request(&self, req: JsonRpcRequest) -> JsonRpcResponse {
        match req.method.as_str() {
            "initialize" => self.handle_initialize(req),
            "newSession" => self.handle_new_session(req),
            "loadSession" => self.handle_load_session(req),
            "prompt" => self.handle_prompt(req).await,
            "listTools" => self.handle_list_tools(req),
            "callTool" => self.handle_call_tool(req).await,
            _ => JsonRpcResponse::error(
                req.id,
                METHOD_NOT_FOUND,
                format!("unknown method '{}'", req.method),
            ),
        }
    }

    fn handle_initialize(&self, req: JsonRpcRequest) -> JsonRpcResponse {
        JsonRpcResponse::success(
            req.id,
            serde_json::json!({
                "name": "frankclaw",
                "version": env!("CARGO_PKG_VERSION"),
                "capabilities": {
                    "streaming": true,
                    "tools": true,
                    "sessions": true,
                }
            }),
        )
    }

    fn handle_new_session(&self, req: JsonRpcRequest) -> JsonRpcResponse {
        // Rate limit session creation.
        if !self.check_rate_limit() {
            return JsonRpcResponse::error(
                req.id,
                RATE_LIMITED,
                "session creation rate limit exceeded",
            );
        }

        // Evict expired sessions.
        self.evict_expired();

        if self.sessions.len() >= self.options.max_sessions {
            // Evict LRU session.
            self.evict_lru();
        }

        let session_id = uuid::Uuid::new_v4().to_string();
        let session_key = SessionKey::from_raw(format!("acp:{session_id}"));

        let now = Instant::now();
        self.sessions.insert(
            session_id.clone(),
            AcpSession {
                session_id: session_id.clone(),
                session_key,
                created_at: now,
                last_touched: now,
            },
        );

        debug!(session_id = %session_id, "ACP session created");
        JsonRpcResponse::success(req.id, serde_json::json!({ "sessionId": session_id }))
    }

    fn handle_load_session(&self, req: JsonRpcRequest) -> JsonRpcResponse {
        let session_id = req
            .params
            .get("sessionId")
            .and_then(|v| v.as_str())
            .unwrap_or("");

        if session_id.is_empty() {
            return JsonRpcResponse::error(req.id, INVALID_PARAMS, "sessionId is required");
        }

        if let Some(mut entry) = self.sessions.get_mut(session_id) {
            entry.last_touched = Instant::now();
            JsonRpcResponse::success(
                req.id,
                serde_json::json!({
                    "sessionId": session_id,
                    "loaded": true,
                }),
            )
        } else {
            JsonRpcResponse::error(
                req.id,
                SESSION_NOT_FOUND,
                format!("session '{session_id}' not found"),
            )
        }
    }

    async fn handle_prompt(&self, req: JsonRpcRequest) -> JsonRpcResponse {
        let session_id = req
            .params
            .get("sessionId")
            .and_then(|v| v.as_str())
            .unwrap_or("");

        let message = req
            .params
            .get("message")
            .and_then(|v| v.as_str())
            .unwrap_or("");

        if message.is_empty() {
            return JsonRpcResponse::error(req.id, INVALID_PARAMS, "message is required");
        }

        // Enforce prompt size limit.
        if message.len() > 2 * 1024 * 1024 {
            return JsonRpcResponse::error(req.id, INVALID_PARAMS, "message exceeds 2MB limit");
        }

        let session_key = if session_id.is_empty() {
            None
        } else if let Some(mut entry) = self.sessions.get_mut(session_id) {
            entry.last_touched = Instant::now();
            Some(entry.session_key.clone())
        } else {
            return JsonRpcResponse::error(
                req.id,
                SESSION_NOT_FOUND,
                format!("session '{session_id}' not found"),
            );
        };

        let (stream_tx, mut stream_rx) = mpsc::channel::<StreamDelta>(64);

        let chat_req = ChatRequest {
            agent_id: None,
            session_key,
            message: message.to_string(),
            attachments: Vec::new(),
            model_id: req
                .params
                .get("model")
                .and_then(|v| v.as_str())
                .map(String::from),
            max_tokens: req
                .params
                .get("maxTokens")
                .and_then(|v| v.as_u64())
                .map(|n| n as u32),
            temperature: req
                .params
                .get("temperature")
                .and_then(|v| v.as_f64())
                .map(|f| f as f32),
            stream_tx: Some(stream_tx),
            thinking_budget: None,
            channel_id: None,
            channel_capabilities: None,
            canvas: Some(self.canvas.clone()),
            cancel_token: None,
            approval_tx: None,
        };

        // Stream deltas as NDJSON events to stdout.
        let rt = self.runtime.clone();
        let (_result_tx, _result_rx) = mpsc::channel::<serde_json::Value>(1);

        tokio::spawn(async move {
            // Forward stream events.
            while let Some(delta) = stream_rx.recv().await {
                let event = match delta {
                    StreamDelta::Text(text) => serde_json::json!({"type": "text", "text": text}),
                    StreamDelta::ToolCallStart { id, name } => {
                        serde_json::json!({"type": "tool_start", "id": id, "name": name})
                    }
                    StreamDelta::ToolCallEnd { id } => {
                        serde_json::json!({"type": "tool_end", "id": id})
                    }
                    StreamDelta::Done { usage } => {
                        serde_json::json!({"type": "done", "usage": usage})
                    }
                    StreamDelta::Error(msg) => {
                        serde_json::json!({"type": "error", "message": msg})
                    }
                    _ => continue,
                };
                // Write event as NDJSON (best-effort).
                let _ = acp_transport::write_response(&JsonRpcResponse {
                    jsonrpc: "2.0".into(),
                    id: None,
                    result: Some(event),
                    error: None,
                });
            }
        });

        let chat_result = rt.chat(chat_req).await;

        match chat_result {
            Ok(resp) => {
                // Update session key if created.
                if !session_id.is_empty() {
                    if let Some(mut entry) = self.sessions.get_mut(session_id) {
                        entry.session_key = resp.session_key.clone();
                        entry.last_touched = Instant::now();
                    }
                }

                JsonRpcResponse::success(
                    req.id,
                    serde_json::json!({
                        "content": resp.content,
                        "model": resp.model_id,
                        "sessionKey": resp.session_key.to_string(),
                        "usage": resp.usage,
                    }),
                )
            }
            Err(e) => JsonRpcResponse::error(req.id, INTERNAL_ERROR, e.to_string()),
        }
    }

    fn handle_list_tools(&self, req: JsonRpcRequest) -> JsonRpcResponse {
        match self.runtime.list_tools(None) {
            Ok(tools) => JsonRpcResponse::success(req.id, serde_json::json!({ "tools": tools })),
            Err(e) => JsonRpcResponse::error(req.id, INTERNAL_ERROR, e.to_string()),
        }
    }

    async fn handle_call_tool(&self, req: JsonRpcRequest) -> JsonRpcResponse {
        let tool_name = match req.params.get("name").and_then(|v| v.as_str()) {
            Some(name) if !name.is_empty() => name.to_string(),
            _ => {
                return JsonRpcResponse::error(req.id, INVALID_PARAMS, "tool name is required");
            }
        };

        let args = req
            .params
            .get("arguments")
            .cloned()
            .unwrap_or(serde_json::json!({}));

        let session_id = req
            .params
            .get("sessionId")
            .and_then(|v| v.as_str())
            .unwrap_or("");

        let session_key = if !session_id.is_empty() {
            self.sessions.get(session_id).map(|e| e.session_key.clone())
        } else {
            None
        };

        let tool_request = frankclaw_runtime::ToolRequest {
            agent_id: None,
            session_key,
            tool_name: tool_name.clone(),
            arguments: args,
            canvas: Some(self.canvas.clone()),
        };

        match self.runtime.invoke_tool(tool_request).await {
            Ok(output) => JsonRpcResponse::success(
                req.id,
                serde_json::json!({
                    "name": output.name,
                    "output": output.output,
                }),
            ),
            Err(e) => JsonRpcResponse::error(req.id, INTERNAL_ERROR, e.to_string()),
        }
    }

    fn check_rate_limit(&self) -> bool {
        let mut guard = self.rate_window.lock().expect("rate limiter lock");
        let now = Instant::now();
        let window = Duration::from_secs(10);

        if now.duration_since(guard.0) > window {
            *guard = (now, 1);
            true
        } else if guard.1 < self.options.rate_limit_per_window {
            guard.1 += 1;
            true
        } else {
            false
        }
    }

    fn evict_expired(&self) {
        let ttl = self.options.session_ttl;
        let now = Instant::now();
        self.sessions
            .retain(|_, session| now.duration_since(session.last_touched) < ttl);
    }

    fn evict_lru(&self) {
        if let Some(lru_key) = self
            .sessions
            .iter()
            .min_by_key(|e| e.last_touched)
            .map(|e| e.key().clone())
        {
            debug!(session_id = %lru_key, "evicting LRU ACP session");
            self.sessions.remove(&lru_key);
        }
    }

    /// Session count (for testing).
    pub fn session_count(&self) -> usize {
        self.sessions.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn options_defaults() {
        let opts = AcpServerOptions::default();
        assert_eq!(opts.max_sessions, 5000);
        assert_eq!(opts.session_ttl, Duration::from_secs(86400));
        assert_eq!(opts.rate_limit_per_window, 120);
    }

    #[test]
    fn session_ttl_eviction() {
        let sessions: DashMap<String, AcpSession> = DashMap::new();
        let ttl = Duration::from_millis(1);

        sessions.insert(
            "old".to_string(),
            AcpSession {
                session_id: "old".to_string(),
                session_key: SessionKey::from_raw("acp:old".to_string()),
                created_at: Instant::now() - Duration::from_secs(100),
                last_touched: Instant::now() - Duration::from_secs(100),
            },
        );

        std::thread::sleep(Duration::from_millis(5));

        let now = Instant::now();
        sessions.retain(|_, session| now.duration_since(session.last_touched) < ttl);
        assert_eq!(sessions.len(), 0);
    }

    #[test]
    fn session_lru_eviction() {
        let sessions: DashMap<String, AcpSession> = DashMap::new();
        let old = Instant::now() - Duration::from_secs(200);
        let recent = Instant::now() - Duration::from_secs(10);

        sessions.insert(
            "old".to_string(),
            AcpSession {
                session_id: "old".to_string(),
                session_key: SessionKey::from_raw("acp:old".to_string()),
                created_at: old,
                last_touched: old,
            },
        );
        sessions.insert(
            "recent".to_string(),
            AcpSession {
                session_id: "recent".to_string(),
                session_key: SessionKey::from_raw("acp:recent".to_string()),
                created_at: recent,
                last_touched: recent,
            },
        );

        // Evict LRU.
        if let Some(lru_key) = sessions
            .iter()
            .min_by_key(|e| e.last_touched)
            .map(|e| e.key().clone())
        {
            sessions.remove(&lru_key);
        }

        assert_eq!(sessions.len(), 1);
        assert!(sessions.contains_key("recent"));
    }

    #[test]
    fn rate_limiter_logic() {
        let window = std::sync::Mutex::new((Instant::now(), 0u32));
        let limit = 3u32;
        let window_duration = Duration::from_secs(10);

        for _ in 0..3 {
            let mut guard = window.lock().unwrap();
            let now = Instant::now();
            if now.duration_since(guard.0) > window_duration {
                *guard = (now, 1);
            } else {
                assert!(guard.1 < limit);
                guard.1 += 1;
            }
        }

        // 4th should exceed.
        let guard = window.lock().unwrap();
        assert!(guard.1 >= limit);
    }

    #[test]
    fn acp_session_fields() {
        let now = Instant::now();
        let session = AcpSession {
            session_id: "test-123".to_string(),
            session_key: SessionKey::from_raw("acp:test-123".to_string()),
            created_at: now,
            last_touched: now,
        };
        assert_eq!(session.session_id, "test-123");
        assert_eq!(session.created_at, session.last_touched);
    }
}
