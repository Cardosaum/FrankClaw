use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use tracing::warn;

use frankclaw_core::error::{AllProvidersFailed, FrankClawError, Result};
use frankclaw_core::model::{
    CompletionRequest, CompletionResponse, ModelDef, ModelProvider, StreamDelta,
};

use crate::circuit_breaker::{CircuitBreaker, CircuitBreakerConfig};
use crate::retry::{RetryConfig, is_retryable_error, retry_backoff_delay};

#[derive(Debug, Clone)]
pub struct ProviderHealth {
    pub provider_id: String,
    pub healthy: bool,
}

struct ProviderEntry {
    provider: Arc<dyn ModelProvider>,
    breaker: CircuitBreaker,
    /// Cached model IDs this provider serves (populated at construction).
    model_ids: Vec<String>,
}

/// Failover chain with per-provider circuit breakers and retry with backoff.
///
/// On each call:
/// 1. Skip providers whose circuit breaker is open.
/// 2. Retry transient failures with exponential backoff.
/// 3. Record success/failure in the circuit breaker.
/// 4. On permanent failure, try the next provider.
pub struct FailoverChain {
    entries: Vec<ProviderEntry>,
    retry_config: RetryConfig,
}

impl FailoverChain {
    /// Create a failover chain without model routing (tries all providers).
    pub fn new(providers: Vec<Arc<dyn ModelProvider>>, cooldown_secs: u64) -> Self {
        let with_models = providers.into_iter().map(|p| (p, Vec::new())).collect();
        Self::with_model_routing(with_models, cooldown_secs)
    }

    /// Create a failover chain with model routing — each provider is paired
    /// with the model IDs it serves. Providers whose list doesn't contain
    /// the requested model are skipped, avoiding unnecessary 404s.
    pub fn with_model_routing(
        providers: Vec<(Arc<dyn ModelProvider>, Vec<String>)>,
        cooldown_secs: u64,
    ) -> Self {
        let breaker_config = CircuitBreakerConfig {
            recovery_timeout: Duration::from_secs(cooldown_secs),
            ..CircuitBreakerConfig::default()
        };
        let entries = providers
            .into_iter()
            .map(|(provider, model_ids)| ProviderEntry {
                provider,
                breaker: CircuitBreaker::new(breaker_config.clone()),
                model_ids,
            })
            .collect();
        Self {
            entries,
            retry_config: RetryConfig::default(),
        }
    }

    /// Try each provider in order with circuit breaker + retry.
    ///
    /// Providers that don't list the requested model are skipped to avoid
    /// unnecessary 404s (e.g. sending a Claude model ID to OpenAI).
    pub async fn complete(
        &self,
        request: CompletionRequest,
        stream_tx: Option<tokio::sync::mpsc::Sender<StreamDelta>>,
    ) -> Result<CompletionResponse> {
        let mut last_error = None;

        for entry in &self.entries {
            let id = entry.provider.id().to_string();

            // Skip providers that don't serve the requested model.
            if !entry.model_ids.is_empty()
                && !entry.model_ids.iter().any(|id| id == &request.model_id)
            {
                continue;
            }

            // Skip if circuit breaker is open.
            if !entry.breaker.check_allowed() {
                warn!(provider = %id, "circuit breaker open, skipping provider");
                continue;
            }

            // Retry loop with backoff for transient failures.
            let mut attempt_result: Option<
                std::result::Result<CompletionResponse, FrankClawError>,
            > = None;
            for attempt in 0..=self.retry_config.max_retries {
                let mut forward_task = None;
                let streamed_any = Arc::new(AtomicBool::new(false));
                let provider_stream_tx = stream_tx.as_ref().map(|stream_tx| {
                    let (proxy_tx, mut proxy_rx) = tokio::sync::mpsc::channel(64);
                    let target_tx = stream_tx.clone();
                    let streamed_any = streamed_any.clone();
                    forward_task = Some(tokio::spawn(async move {
                        while let Some(delta) = proxy_rx.recv().await {
                            streamed_any.store(true, Ordering::Relaxed);
                            let _ = target_tx.send(delta).await;
                        }
                    }));
                    proxy_tx
                });

                let result = entry
                    .provider
                    .complete(request.clone(), provider_stream_tx)
                    .await;
                if let Some(task) = forward_task {
                    let _ = task.await;
                }

                match result {
                    Ok(response) => {
                        entry.breaker.record_success();
                        return Ok(response);
                    }
                    Err(e) => {
                        // If we already streamed data, we can't retry or failover.
                        if stream_tx.is_some() && streamed_any.load(Ordering::Relaxed) {
                            entry.breaker.record_failure();
                            return Err(e);
                        }

                        let err_str = e.to_string();
                        let retryable = is_retryable_error(&err_str);

                        if !retryable || attempt == self.retry_config.max_retries {
                            // Not retryable or out of retries — record failure and try next provider.
                            attempt_result = Some(Err(e));
                            break;
                        }

                        let delay = retry_backoff_delay(attempt);
                        warn!(
                            provider = %id,
                            attempt = attempt + 1,
                            max_retries = self.retry_config.max_retries,
                            delay_ms = delay.as_millis() as u64,
                            error = %e,
                            "retrying after transient error"
                        );
                        tokio::time::sleep(delay).await;
                    }
                }
            }

            // If we got here, the provider failed after all retries.
            if let Some(Err(e)) = attempt_result {
                let err_str = e.to_string();
                // Only trip the circuit breaker for transient failures.
                if is_retryable_error(&err_str) {
                    entry.breaker.record_failure();
                }
                warn!(provider = %id, error = %e, "provider failed, trying next");
                last_error = Some(e);
            }
        }

        Err(last_error.unwrap_or_else(|| AllProvidersFailed.build()))
    }

    /// List models from all providers (skip those with open circuit breakers).
    pub async fn list_models(&self) -> Result<Vec<ModelDef>> {
        let mut all = Vec::new();
        for entry in &self.entries {
            match entry.provider.list_models().await {
                Ok(models) => all.extend(models),
                Err(e) => {
                    warn!(provider = %entry.provider.id(), error = %e, "failed to list models");
                }
            }
        }
        Ok(all)
    }

    pub async fn health(&self) -> Vec<ProviderHealth> {
        let mut health = Vec::with_capacity(self.entries.len());
        for entry in &self.entries {
            health.push(ProviderHealth {
                provider_id: entry.provider.id().to_string(),
                healthy: entry.provider.health().await,
            });
        }
        health
    }
}
