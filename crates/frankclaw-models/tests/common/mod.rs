//! Shared test helpers for `frankclaw-models` integration and smoke tests.

#![allow(
    dead_code,
    reason = "shared helpers are consumed selectively by smoke and integration test binaries"
)]

use frankclaw_core::model::{CompletionMessage, CompletionRequest};
use frankclaw_core::types::Role;
use secrecy::SecretString;

/// Returns the OpenAI API key from `OPENAI_API_KEY`, if set and non-empty.
pub fn openai_key() -> Option<SecretString> {
    std::env::var("OPENAI_API_KEY")
        .ok()
        .filter(|k| !k.trim().is_empty())
        .map(SecretString::from)
}

/// Returns the OpenAI-compatible base URL. If `OPENAI_API_KEY` looks like an
/// OpenRouter key (`sk-or-*`), use the OpenRouter endpoint automatically.
pub fn openai_base_url() -> String {
    let key = std::env::var("OPENAI_API_KEY").unwrap_or_default();
    if key.starts_with("sk-or-") {
        "https://openrouter.ai/api/v1".into()
    } else {
        std::env::var("OPENAI_BASE_URL").unwrap_or_else(|_| "https://api.openai.com/v1".into())
    }
}

/// Pick a model that works with the detected endpoint.
pub fn openai_model() -> String {
    let key = std::env::var("OPENAI_API_KEY").unwrap_or_default();
    if key.starts_with("sk-or-") {
        // OpenRouter model IDs use provider prefix
        "openai/gpt-4o-mini".into()
    } else {
        "gpt-4o-mini".into()
    }
}

/// Returns the Anthropic API key from `ANTHROPIC_API_KEY`, if set and non-empty.
pub fn anthropic_key() -> Option<SecretString> {
    std::env::var("ANTHROPIC_API_KEY")
        .ok()
        .filter(|k| !k.trim().is_empty())
        .map(SecretString::from)
}

/// Checks whether a local Ollama instance is reachable on port 11434.
pub fn ollama_available() -> bool {
    std::net::TcpStream::connect_timeout(
        &"127.0.0.1:11434".parse().expect("valid addr"),
        std::time::Duration::from_secs(2),
    )
    .is_ok()
}

/// Builds a minimal [`CompletionRequest`] with a single user message.
pub fn simple_request(model: &str, prompt: &str) -> CompletionRequest {
    CompletionRequest {
        model_id: model.into(),
        messages: vec![CompletionMessage::text(Role::User, prompt)],
        max_tokens: Some(50),
        temperature: Some(0.0),
        system: None,
        tools: Vec::new(),
        thinking_budget: None,
        parallel_tool_calls: None,
        seed: None,
        response_format: None,
        reasoning_effort: None,
    }
}
