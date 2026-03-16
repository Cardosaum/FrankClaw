//! Service trait abstractions for tools.
//!
//! These traits decouple `frankclaw-tools` from concrete implementations in
//! `frankclaw-media`, `frankclaw-channels`, and `frankclaw-cron`, avoiding
//! circular dependencies.

use async_trait::async_trait;

use crate::error::Result;

/// Content returned by a URL fetch.
#[derive(Debug, Clone)]
pub struct FetchedContent {
    pub bytes: Vec<u8>,
    pub content_type: String,
    pub final_url: String,
}

/// SSRF-safe URL fetcher.
#[async_trait]
pub trait Fetcher: Send + Sync + 'static {
    async fn fetch(&self, url: &str) -> Result<FetchedContent>;
}

/// Send outbound messages through a channel adapter.
#[async_trait]
pub trait MessageSender: Send + Sync + 'static {
    async fn send_text(
        &self,
        channel: &str,
        account_id: &str,
        to: &str,
        text: &str,
        thread_id: Option<&str>,
        reply_to: Option<&str>,
    ) -> Result<String>;

    /// Send an emoji reaction to a message.
    async fn send_reaction(
        &self,
        channel: &str,
        account_id: &str,
        to: &str,
        thread_id: Option<&str>,
        platform_message_id: &str,
        emoji: &str,
    ) -> Result<()>;
}

/// Semantic memory search service.
#[async_trait]
pub trait MemorySearch: Send + Sync + 'static {
    /// Search memory using a text query.
    /// Returns matching chunks with relevance scores.
    async fn search(&self, query: &str, limit: usize) -> Result<Vec<MemorySearchResult>>;

    /// List all indexed memory sources.
    async fn list_sources(&self) -> Result<Vec<serde_json::Value>>;
}

/// A memory search result.
#[derive(Debug, Clone, serde::Serialize)]
pub struct MemorySearchResult {
    pub source: String,
    pub text: String,
    pub score: f32,
    pub line_start: usize,
    pub line_end: usize,
}

/// Audio transcription service.
#[async_trait]
pub trait AudioTranscriber: Send + Sync + 'static {
    /// Transcribe audio data to text.
    async fn transcribe(&self, data: &[u8], mime: &str, filename: &str) -> Result<String>;
}

/// Manage scheduled cron jobs.
#[async_trait]
pub trait CronManager: Send + Sync + 'static {
    async fn list_jobs(&self) -> Vec<serde_json::Value>;
    async fn add_job(
        &self,
        id: &str,
        schedule: &str,
        agent_id: &str,
        session_key: &str,
        prompt: &str,
        enabled: bool,
    ) -> Result<()>;
    async fn remove_job(&self, id: &str) -> Result<bool>;
}
