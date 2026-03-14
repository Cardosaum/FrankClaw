//! Circuit breaker for LLM providers.
//!
//! Prevents hammering a failing provider by tracking consecutive transient
//! failures and temporarily disabling the provider. Allows probe calls
//! after a recovery timeout to test if the provider has recovered.
//!
//! Derived from IronClaw (MIT OR Apache-2.0, Copyright (c) 2024-2025 NEAR AI Inc.)

use std::sync::Mutex;
use std::time::{Duration, Instant};

/// Circuit breaker configuration.
#[derive(Debug, Clone)]
pub struct CircuitBreakerConfig {
    /// Consecutive transient failures before opening the circuit.
    pub failure_threshold: u32,
    /// How long the circuit stays open before allowing probe calls.
    pub recovery_timeout: Duration,
    /// Number of successful probes needed in half-open state to close.
    pub half_open_successes_needed: u32,
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            failure_threshold: 5,
            recovery_timeout: Duration::from_secs(30),
            half_open_successes_needed: 2,
        }
    }
}

/// Circuit breaker state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CircuitState {
    /// Normal operation — tracking consecutive failures.
    Closed,
    /// Rejecting all calls — waiting for recovery timeout.
    Open,
    /// Allowing probe calls to test if the provider has recovered.
    HalfOpen,
}

struct BreakerState {
    state: CircuitState,
    consecutive_failures: u32,
    opened_at: Option<Instant>,
    half_open_successes: u32,
}

/// Circuit breaker that wraps a provider and tracks failure state.
pub struct CircuitBreaker {
    state: Mutex<BreakerState>,
    config: CircuitBreakerConfig,
}

impl CircuitBreaker {
    pub fn new(config: CircuitBreakerConfig) -> Self {
        Self {
            state: Mutex::new(BreakerState {
                state: CircuitState::Closed,
                consecutive_failures: 0,
                opened_at: None,
                half_open_successes: 0,
            }),
            config,
        }
    }

    /// Current circuit state (may transition Open → HalfOpen if timeout elapsed).
    pub fn circuit_state(&self) -> CircuitState {
        let mut s = self.state.lock().expect("invariant: breaker mutex not poisoned");
        maybe_transition_to_half_open(&mut s, &self.config);
        s.state
    }

    pub fn consecutive_failures(&self) -> u32 {
        self.state
            .lock()
            .expect("invariant: breaker mutex not poisoned")
            .consecutive_failures
    }

    /// Check if a call is allowed. Returns `false` if the circuit is open
    /// and the recovery timeout hasn't elapsed yet.
    pub fn check_allowed(&self) -> bool {
        let mut s = self.state.lock().expect("invariant: breaker mutex not poisoned");
        maybe_transition_to_half_open(&mut s, &self.config);
        // Open state blocks; Closed and HalfOpen allow.
        s.state != CircuitState::Open
    }

    /// Record a successful call.
    pub fn record_success(&self) {
        let mut s = self.state.lock().expect("invariant: breaker mutex not poisoned");
        match s.state {
            CircuitState::Closed => {
                s.consecutive_failures = 0;
            }
            CircuitState::HalfOpen => {
                s.half_open_successes += 1;
                if s.half_open_successes >= self.config.half_open_successes_needed {
                    s.state = CircuitState::Closed;
                    s.consecutive_failures = 0;
                    s.half_open_successes = 0;
                    s.opened_at = None;
                }
            }
            CircuitState::Open => {}
        }
    }

    /// Record a transient failure. Non-transient errors should NOT be recorded
    /// (e.g., auth failures, context length exceeded, model not found).
    pub fn record_failure(&self) {
        let mut s = self.state.lock().expect("invariant: breaker mutex not poisoned");
        match s.state {
            CircuitState::Closed => {
                s.consecutive_failures += 1;
                if s.consecutive_failures >= self.config.failure_threshold {
                    s.state = CircuitState::Open;
                    s.opened_at = Some(Instant::now());
                    s.half_open_successes = 0;
                }
            }
            CircuitState::HalfOpen => {
                // Any failure in half-open reopens the circuit.
                s.state = CircuitState::Open;
                s.opened_at = Some(Instant::now());
                s.half_open_successes = 0;
            }
            CircuitState::Open => {}
        }
    }
}

fn maybe_transition_to_half_open(s: &mut BreakerState, config: &CircuitBreakerConfig) {
    if s.state == CircuitState::Open
        && s.opened_at
            .is_some_and(|opened_at| opened_at.elapsed() >= config.recovery_timeout)
    {
        s.state = CircuitState::HalfOpen;
        s.half_open_successes = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::{fixture, rstest};

    #[fixture]
    fn breaker() -> CircuitBreaker {
        CircuitBreaker::new(CircuitBreakerConfig {
            failure_threshold: 3,
            recovery_timeout: Duration::from_millis(50),
            half_open_successes_needed: 2,
        })
    }

    #[rstest]
    fn closed_allows_calls_and_resets_on_success(breaker: CircuitBreaker) {
        assert!(breaker.check_allowed());
        assert_eq!(breaker.circuit_state(), CircuitState::Closed);

        breaker.record_failure();
        breaker.record_failure();
        assert_eq!(breaker.consecutive_failures(), 2);

        breaker.record_success();
        assert_eq!(breaker.consecutive_failures(), 0);
        assert_eq!(breaker.circuit_state(), CircuitState::Closed);
    }

    #[rstest]
    fn failures_trip_circuit_to_open(breaker: CircuitBreaker) {
        for _ in 0..3 {
            breaker.record_failure();
        }
        assert_eq!(breaker.circuit_state(), CircuitState::Open);
        assert!(!breaker.check_allowed());
    }

    #[rstest]
    fn open_rejects_immediately(breaker: CircuitBreaker) {
        for _ in 0..3 {
            breaker.record_failure();
        }
        assert!(!breaker.check_allowed());
    }

    #[rstest]
    fn recovery_timeout_transitions_to_half_open(breaker: CircuitBreaker) {
        for _ in 0..3 {
            breaker.record_failure();
        }
        assert_eq!(breaker.circuit_state(), CircuitState::Open);

        std::thread::sleep(Duration::from_millis(60));
        assert_eq!(breaker.circuit_state(), CircuitState::HalfOpen);
        assert!(breaker.check_allowed());
    }

    #[rstest]
    fn half_open_success_closes_circuit(breaker: CircuitBreaker) {
        for _ in 0..3 {
            breaker.record_failure();
        }
        std::thread::sleep(Duration::from_millis(60));

        assert_eq!(breaker.circuit_state(), CircuitState::HalfOpen);
        breaker.record_success();
        assert_eq!(breaker.circuit_state(), CircuitState::HalfOpen); // Needs 2
        breaker.record_success();
        assert_eq!(breaker.circuit_state(), CircuitState::Closed);
    }

    #[rstest]
    fn half_open_failure_reopens_circuit(breaker: CircuitBreaker) {
        for _ in 0..3 {
            breaker.record_failure();
        }
        std::thread::sleep(Duration::from_millis(60));

        assert_eq!(breaker.circuit_state(), CircuitState::HalfOpen);
        breaker.record_failure();
        assert_eq!(breaker.circuit_state(), CircuitState::Open);
        assert!(!breaker.check_allowed());
    }

    #[rstest]
    fn success_in_closed_resets_failure_count(breaker: CircuitBreaker) {
        breaker.record_failure();
        breaker.record_failure();
        assert_eq!(breaker.consecutive_failures(), 2);
        breaker.record_success();
        assert_eq!(breaker.consecutive_failures(), 0);
        // Two more failures should NOT trip (we're back at 0).
        breaker.record_failure();
        breaker.record_failure();
        assert_eq!(breaker.circuit_state(), CircuitState::Closed);
    }

    #[test]
    fn default_config_values() {
        let config = CircuitBreakerConfig::default();
        assert_eq!(config.failure_threshold, 5);
        assert_eq!(config.recovery_timeout, Duration::from_secs(30));
        assert_eq!(config.half_open_successes_needed, 2);
    }
}
