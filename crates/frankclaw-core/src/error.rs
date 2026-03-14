use snafu::Snafu;

use crate::types::{AgentId, ChannelId, SessionKey};

/// Unified error hierarchy. Every variant is explicit — no catch-all.
/// Error messages never contain secret material.
///
/// Each variant carries an implicit `snafu::Location` that records the
/// file and line where the error was constructed, available via the
/// [`FrankClawError::location`] method.
#[derive(Debug, Snafu)]
#[snafu(visibility(pub))]
pub enum FrankClawError {
    // ── Auth ──────────────────────────────────────────────
    #[snafu(display("authentication required"))]
    AuthRequired {
        #[snafu(implicit)]
        location: snafu::Location,
    },

    #[snafu(display("authentication failed"))]
    AuthFailed {
        #[snafu(implicit)]
        location: snafu::Location,
    },

    #[snafu(display("rate limited (retry after {retry_after_secs}s)"))]
    RateLimited {
        retry_after_secs: u64,
        #[snafu(implicit)]
        location: snafu::Location,
    },

    #[snafu(display("insufficient permissions for method {method}"))]
    Forbidden {
        method: String,
        #[snafu(implicit)]
        location: snafu::Location,
    },

    // ── Session ──────────────────────────────────────────
    #[snafu(display("session not found: {key}"))]
    SessionNotFound {
        key: SessionKey,
        #[snafu(implicit)]
        location: snafu::Location,
    },

    #[snafu(display("session storage error: {msg}"))]
    SessionStorage {
        msg: String,
        #[snafu(implicit)]
        location: snafu::Location,
    },

    // ── Channel ──────────────────────────────────────────
    #[snafu(display("channel {channel} error: {msg}"))]
    Channel {
        channel: ChannelId,
        msg: String,
        #[snafu(implicit)]
        location: snafu::Location,
    },

    #[snafu(display("channel {channel} not configured"))]
    ChannelNotConfigured {
        channel: ChannelId,
        #[snafu(implicit)]
        location: snafu::Location,
    },

    #[snafu(display("channel {channel} is disabled"))]
    ChannelDisabled {
        channel: ChannelId,
        #[snafu(implicit)]
        location: snafu::Location,
    },

    #[snafu(display("sender blocked by policy on channel {channel}"))]
    SenderBlocked {
        channel: ChannelId,
        #[snafu(implicit)]
        location: snafu::Location,
    },

    // ── Agent ────────────────────────────────────────────
    #[snafu(display("agent {agent_id} not found"))]
    AgentNotFound {
        agent_id: AgentId,
        #[snafu(implicit)]
        location: snafu::Location,
    },

    #[snafu(display("agent runtime error: {msg}"))]
    AgentRuntime {
        msg: String,
        #[snafu(implicit)]
        location: snafu::Location,
    },

    #[snafu(display("agent turn cancelled"))]
    TurnCancelled {
        #[snafu(implicit)]
        location: snafu::Location,
    },

    #[snafu(display("sandbox error: {msg}"))]
    Sandbox {
        msg: String,
        #[snafu(implicit)]
        location: snafu::Location,
    },

    // ── Model ────────────────────────────────────────────
    #[snafu(display("model provider error: {msg}"))]
    ModelProvider {
        msg: String,
        #[snafu(implicit)]
        location: snafu::Location,
    },

    #[snafu(display("all model providers failed"))]
    AllProvidersFailed {
        #[snafu(implicit)]
        location: snafu::Location,
    },

    #[snafu(display("model not found: {model_id}"))]
    ModelNotFound {
        model_id: String,
        #[snafu(implicit)]
        location: snafu::Location,
    },

    // ── Config ───────────────────────────────────────────
    #[snafu(display("config validation error: {msg}"))]
    ConfigValidation {
        msg: String,
        #[snafu(implicit)]
        location: snafu::Location,
    },

    #[snafu(display("config I/O error: {msg}"))]
    ConfigIo {
        msg: String,
        #[snafu(implicit)]
        location: snafu::Location,
    },

    // ── Protocol ─────────────────────────────────────────
    #[snafu(display("invalid request: {msg}"))]
    InvalidRequest {
        msg: String,
        #[snafu(implicit)]
        location: snafu::Location,
    },

    #[snafu(display("unknown method: {method}"))]
    UnknownMethod {
        method: String,
        #[snafu(implicit)]
        location: snafu::Location,
    },

    #[snafu(display("request too large (max {max_bytes} bytes)"))]
    RequestTooLarge {
        max_bytes: usize,
        #[snafu(implicit)]
        location: snafu::Location,
    },

    // ── Media ────────────────────────────────────────────
    #[snafu(display("media file too large (max {max_bytes} bytes)"))]
    MediaTooLarge {
        max_bytes: u64,
        #[snafu(implicit)]
        location: snafu::Location,
    },

    #[snafu(display("media fetch blocked: {reason}"))]
    MediaFetchBlocked {
        reason: String,
        #[snafu(implicit)]
        location: snafu::Location,
    },

    #[snafu(display("unsupported media type: {mime}"))]
    UnsupportedMediaType {
        mime: String,
        #[snafu(implicit)]
        location: snafu::Location,
    },

    #[snafu(display("malware detected in file '{filename}': {detail}"))]
    MalwareDetected {
        filename: String,
        detail: String,
        #[snafu(implicit)]
        location: snafu::Location,
    },

    // ── Crypto ───────────────────────────────────────────
    #[snafu(display("cryptographic operation failed"), context(false))]
    Crypto {
        source: frankclaw_crypto::CryptoError,
        #[snafu(implicit)]
        location: snafu::Location,
    },

    // ── Internal ─────────────────────────────────────────
    #[snafu(display("internal error: {msg}"))]
    Internal {
        msg: String,
        #[snafu(implicit)]
        location: snafu::Location,
    },

    #[snafu(display("shutdown in progress"))]
    ShuttingDown {
        #[snafu(implicit)]
        location: snafu::Location,
    },
}

impl FrankClawError {
    /// Where the error was created (file, line, column).
    pub fn location(&self) -> &snafu::Location {
        match self {
            Self::AuthRequired { location, .. }
            | Self::AuthFailed { location, .. }
            | Self::RateLimited { location, .. }
            | Self::Forbidden { location, .. }
            | Self::SessionNotFound { location, .. }
            | Self::SessionStorage { location, .. }
            | Self::Channel { location, .. }
            | Self::ChannelNotConfigured { location, .. }
            | Self::ChannelDisabled { location, .. }
            | Self::SenderBlocked { location, .. }
            | Self::AgentNotFound { location, .. }
            | Self::AgentRuntime { location, .. }
            | Self::TurnCancelled { location, .. }
            | Self::Sandbox { location, .. }
            | Self::ModelProvider { location, .. }
            | Self::AllProvidersFailed { location, .. }
            | Self::ModelNotFound { location, .. }
            | Self::ConfigValidation { location, .. }
            | Self::ConfigIo { location, .. }
            | Self::InvalidRequest { location, .. }
            | Self::UnknownMethod { location, .. }
            | Self::RequestTooLarge { location, .. }
            | Self::MediaTooLarge { location, .. }
            | Self::MediaFetchBlocked { location, .. }
            | Self::UnsupportedMediaType { location, .. }
            | Self::MalwareDetected { location, .. }
            | Self::Crypto { location, .. }
            | Self::Internal { location, .. }
            | Self::ShuttingDown { location, .. } => location,
        }
    }

    /// Whether the client should retry this request.
    pub fn is_retryable(&self) -> bool {
        matches!(
            self,
            Self::RateLimited { .. }
                | Self::ModelProvider { .. }
                | Self::AllProvidersFailed { .. }
                | Self::Internal { .. }
        )
    }

    /// HTTP-like status code for protocol responses.
    pub fn status_code(&self) -> u16 {
        match self {
            Self::AuthRequired { .. } | Self::AuthFailed { .. } => 401,
            Self::RateLimited { .. } => 429,
            Self::Forbidden { .. } => 403,
            Self::SessionNotFound { .. }
            | Self::AgentNotFound { .. }
            | Self::ModelNotFound { .. }
            | Self::ChannelNotConfigured { .. } => 404,
            Self::InvalidRequest { .. } | Self::UnknownMethod { .. } => 400,
            Self::RequestTooLarge { .. } | Self::MediaTooLarge { .. } => 413,
            Self::MediaFetchBlocked { .. }
            | Self::MalwareDetected { .. }
            | Self::SenderBlocked { .. } => 403,
            Self::ConfigValidation { .. } => 422,
            Self::SessionStorage { .. }
            | Self::Channel { .. }
            | Self::ChannelDisabled { .. }
            | Self::AgentRuntime { .. }
            | Self::TurnCancelled { .. }
            | Self::Sandbox { .. }
            | Self::ModelProvider { .. }
            | Self::AllProvidersFailed { .. }
            | Self::ConfigIo { .. }
            | Self::UnsupportedMediaType { .. }
            | Self::Crypto { .. }
            | Self::Internal { .. }
            | Self::ShuttingDown { .. } => 500,
        }
    }
}

pub type Result<T> = std::result::Result<T, FrankClawError>;
