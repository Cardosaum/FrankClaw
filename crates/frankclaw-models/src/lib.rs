#![forbid(unsafe_code)]

pub mod cache;
pub mod catalog;
pub mod circuit_breaker;
pub mod cost_guard;
pub mod costs;
mod failover;
mod openai;
mod openai_compat;
mod anthropic;
mod ollama;
pub mod retry;
pub mod routing;
mod sse;

pub use cache::{ResponseCache, ResponseCacheConfig};
pub use circuit_breaker::{CircuitBreaker, CircuitBreakerConfig, CircuitState};
pub use cost_guard::{CostGuard, CostGuardConfig, CostLimitExceeded, ModelTokens};
pub use costs::{model_cost, default_cost};
pub use failover::{FailoverChain, ProviderHealth};
pub use openai::OpenAiProvider;
pub use anthropic::AnthropicProvider;
pub use ollama::OllamaProvider;
pub use retry::{RetryConfig, is_retryable_error, retry_backoff_delay};
pub use routing::{
    classify_message, response_is_uncertain, score_complexity, score_complexity_with_config,
    ScoreBreakdown, ScorerConfig, ScorerWeights, TaskComplexity, Tier,
};
