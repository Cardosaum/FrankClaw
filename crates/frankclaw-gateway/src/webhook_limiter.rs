use std::sync::Arc;

use dashmap::DashMap;
use tokio::sync::Semaphore;

/// Per-mapping concurrency limiter and fixed-window rate counter.
pub struct WebhookLimiter {
    semaphores: DashMap<String, Arc<Semaphore>>,
    rate_counters: DashMap<String, RateCounter>,
}

struct RateCounter {
    count: u32,
    limit: u32,
    window_start: std::time::Instant,
}

impl Default for WebhookLimiter {
    fn default() -> Self {
        Self::new()
    }
}

impl WebhookLimiter {
    pub fn new() -> Self {
        Self {
            semaphores: DashMap::new(),
            rate_counters: DashMap::new(),
        }
    }

    /// Acquire a concurrency permit for the given mapping.
    /// Returns None if max_concurrent is 0 (unlimited).
    pub async fn acquire_concurrency(
        &self,
        mapping_id: &str,
        max_concurrent: usize,
    ) -> Option<tokio::sync::OwnedSemaphorePermit> {
        if max_concurrent == 0 {
            return None;
        }
        let sem = self
            .semaphores
            .entry(mapping_id.to_string())
            .or_insert_with(|| Arc::new(Semaphore::new(max_concurrent)))
            .clone();
        sem.acquire_owned().await.ok()
    }

    /// Check and increment the rate counter. Returns true if allowed.
    pub fn check_rate(&self, mapping_id: &str, limit_per_minute: u32) -> bool {
        if limit_per_minute == 0 {
            return true;
        }
        let now = std::time::Instant::now();
        let mut entry = self
            .rate_counters
            .entry(mapping_id.to_string())
            .or_insert_with(|| RateCounter {
                count: 0,
                limit: limit_per_minute,
                window_start: now,
            });
        let counter = entry.value_mut();

        // Reset window if expired (1 minute).
        if now.duration_since(counter.window_start).as_secs() >= 60 {
            counter.count = 0;
            counter.window_start = now;
        }

        if counter.count >= counter.limit {
            return false;
        }
        counter.count += 1;
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn concurrency_limits_permits() {
        let limiter = WebhookLimiter::new();
        let p1 = limiter.acquire_concurrency("test", 2).await;
        let p2 = limiter.acquire_concurrency("test", 2).await;
        assert!(p1.is_some());
        assert!(p2.is_some());
        // Third should block, but we can test that the semaphore is full
        // by using try_acquire on the semaphore directly.
        let sem = limiter
            .semaphores
            .get("test")
            .expect("semaphore entry should exist after acquiring permits")
            .clone();
        let _err = sem
            .try_acquire()
            .expect_err("third permit should fail while the semaphore is full");
    }

    #[test]
    fn rate_limit_allows_within_window() {
        let limiter = WebhookLimiter::new();
        assert!(limiter.check_rate("test", 3));
        assert!(limiter.check_rate("test", 3));
        assert!(limiter.check_rate("test", 3));
        assert!(!limiter.check_rate("test", 3));
    }

    #[test]
    fn rate_limit_zero_means_unlimited() {
        let limiter = WebhookLimiter::new();
        for _ in 0..100 {
            assert!(limiter.check_rate("test", 0));
        }
    }
}
