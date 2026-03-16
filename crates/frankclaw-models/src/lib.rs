#![forbid(unsafe_code)]

mod anthropic;
pub mod cache;
pub mod catalog;
pub mod circuit_breaker;
pub mod copilot;
pub mod cost_guard;
pub mod costs;
mod failover;
mod ollama;
mod openai;
mod openai_compat;
pub mod retry;
pub mod routing;
mod sse;

pub use anthropic::AnthropicProvider;
pub use cache::{ResponseCache, ResponseCacheConfig};
pub use circuit_breaker::{CircuitBreaker, CircuitBreakerConfig, CircuitState};
pub use copilot::CopilotProvider;
pub use cost_guard::{CostGuard, CostGuardConfig, CostLimitExceeded, ModelTokens};
pub use costs::{default_cost, model_cost};
pub use failover::{FailoverChain, ProviderHealth};
pub use ollama::OllamaProvider;
pub use openai::OpenAiProvider;
pub use retry::{RetryConfig, is_retryable_error, retry_backoff_delay};
pub use routing::{
    ScoreBreakdown, ScorerConfig, ScorerWeights, TaskComplexity, Tier, classify_message,
    response_is_uncertain, score_complexity, score_complexity_with_config,
};
