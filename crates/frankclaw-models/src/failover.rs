use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use dashmap::DashMap;
use tracing::warn;

use frankclaw_core::error::{FrankClawError, Result};
use frankclaw_core::model::{
    CompletionRequest, CompletionResponse, ModelDef, ModelProvider, StreamDelta,
};

#[derive(Debug, Clone)]
pub struct ProviderHealth {
    pub provider_id: String,
    pub healthy: bool,
}

/// Failover chain that tries providers in order, with cooldowns on failure.
pub struct FailoverChain {
    providers: Vec<Arc<dyn ModelProvider>>,
    cooldowns: DashMap<String, Instant>,
    cooldown_duration: Duration,
}

impl FailoverChain {
    pub fn new(providers: Vec<Arc<dyn ModelProvider>>, cooldown_secs: u64) -> Self {
        Self {
            providers,
            cooldowns: DashMap::new(),
            cooldown_duration: Duration::from_secs(cooldown_secs),
        }
    }

    /// Try each provider in order. Skip cooled-down providers.
    pub async fn complete(
        &self,
        request: CompletionRequest,
        stream_tx: Option<tokio::sync::mpsc::Sender<StreamDelta>>,
    ) -> Result<CompletionResponse> {
        let mut last_error = None;

        for provider in &self.providers {
            let id = provider.id().to_string();

            // Skip if still cooling down.
            if let Some(until) = self.cooldowns.get(&id) {
                if Instant::now() < *until {
                    continue;
                }
                // Cooldown expired, remove it.
                drop(until);
                self.cooldowns.remove(&id);
            }

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

            let result = provider.complete(request.clone(), provider_stream_tx).await;
            if let Some(task) = forward_task {
                let _ = task.await;
            }

            match result {
                Ok(response) => return Ok(response),
                Err(e) => {
                    if stream_tx.is_some() && streamed_any.load(Ordering::Relaxed) {
                        return Err(e);
                    }
                    warn!(provider = %id, error = %e, "provider failed, trying next");
                    self.cooldowns
                        .insert(id, Instant::now() + self.cooldown_duration);
                    last_error = Some(e);
                }
            }
        }

        Err(last_error.unwrap_or(FrankClawError::AllProvidersFailed))
    }

    /// List models from all non-cooled-down providers.
    pub async fn list_models(&self) -> Result<Vec<ModelDef>> {
        let mut all = Vec::new();
        for provider in &self.providers {
            match provider.list_models().await {
                Ok(models) => all.extend(models),
                Err(e) => {
                    warn!(provider = %provider.id(), error = %e, "failed to list models");
                }
            }
        }
        Ok(all)
    }

    pub async fn health(&self) -> Vec<ProviderHealth> {
        let mut health = Vec::with_capacity(self.providers.len());
        for provider in &self.providers {
            health.push(ProviderHealth {
                provider_id: provider.id().to_string(),
                healthy: provider.health().await,
            });
        }
        health
    }
}
