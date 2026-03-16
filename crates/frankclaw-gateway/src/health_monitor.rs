#![allow(
    dead_code,
    reason = "health monitor wiring is present but not fully integrated yet"
)]

//! Periodic channel health monitor.
//!
//! Polls all registered channels on a configurable interval, broadcasts
//! health status changes to connected clients, and optionally restarts
//! unhealthy channels with rate-limiting guardrails.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use tracing::{debug, info, warn};

use frankclaw_core::channel::HealthStatus;
use frankclaw_core::protocol::{EventFrame, EventType, Frame};
use frankclaw_core::types::ChannelId;

use crate::broadcast::BroadcastHandle;
use crate::state::GatewayState;

/// Default health check interval (5 minutes).
const DEFAULT_CHECK_INTERVAL_SECS: u64 = 5 * 60;

/// Grace period after startup before first health check (60 seconds).
const STARTUP_GRACE_SECS: u64 = 60;

/// Minimum time between restarts of the same channel (10 minutes).
const RESTART_COOLDOWN_SECS: u64 = 10 * 60;

/// Maximum restarts per channel per hour.
const MAX_RESTARTS_PER_HOUR: usize = 6;

/// Per-channel tracking state for the health monitor.
#[derive(Debug)]
struct ChannelHealth {
    /// Last observed health status.
    last_status: Option<HealthStatus>,
    /// When the last restart was attempted.
    last_restart_at: Option<Instant>,
    /// Restart timestamps within the last hour (for rate limiting).
    restart_history: Vec<Instant>,
}

impl ChannelHealth {
    fn new() -> Self {
        Self {
            last_status: None,
            last_restart_at: None,
            restart_history: Vec::new(),
        }
    }

    /// Record a restart attempt and return whether it's allowed.
    fn try_restart(&mut self) -> bool {
        let now = Instant::now();

        // Check cooldown.
        if let Some(last) = self.last_restart_at
            && now.duration_since(last) < Duration::from_secs(RESTART_COOLDOWN_SECS)
        {
            return false;
        }

        // Prune old history (older than 1 hour).
        let one_hour_ago = now
            .checked_sub(Duration::from_secs(3600))
            .expect("one-hour lookback should remain within `Instant` bounds");
        self.restart_history.retain(|&t| t > one_hour_ago);

        // Check rate limit.
        if self.restart_history.len() >= MAX_RESTARTS_PER_HOUR {
            return false;
        }

        self.last_restart_at = Some(now);
        self.restart_history.push(now);
        true
    }

    /// Check if the status changed from the last observation.
    fn status_changed(&self, new_status: &HealthStatus) -> bool {
        match (&self.last_status, new_status) {
            (None, _) => true,
            (Some(HealthStatus::Connected), HealthStatus::Connected) => false,
            (Some(HealthStatus::NotConfigured), HealthStatus::NotConfigured) => false,
            (
                Some(HealthStatus::Degraded { reason: old }),
                HealthStatus::Degraded { reason: new },
            ) => old != new,
            (
                Some(HealthStatus::Disconnected { reason: old }),
                HealthStatus::Disconnected { reason: new },
            ) => old != new,
            _ => true,
        }
    }
}

/// Returns whether the health status indicates the channel needs attention.
fn is_unhealthy(status: &HealthStatus) -> bool {
    matches!(status, HealthStatus::Disconnected { .. })
}

/// Start the periodic channel health monitor as a background task.
pub fn start_health_monitor(state: Arc<GatewayState>) {
    let shutdown = state.shutdown.clone();
    let interval_secs = state
        .current_config()
        .gateway
        .health_check_interval_secs
        .unwrap_or(DEFAULT_CHECK_INTERVAL_SECS);

    if interval_secs == 0 {
        info!("channel health monitor disabled (interval = 0)");
        return;
    }

    tokio::spawn(async move {
        // Startup grace period.
        tokio::select! {
            () = tokio::time::sleep(Duration::from_secs(STARTUP_GRACE_SECS)) => {}
            () = shutdown.cancelled() => return,
        }

        let mut tracker: HashMap<ChannelId, ChannelHealth> = HashMap::new();
        let mut interval = tokio::time::interval(Duration::from_secs(interval_secs));
        interval.tick().await; // consume the immediate first tick

        loop {
            tokio::select! {
                _tick = interval.tick() => {
                    check_all_channels(&state, &mut tracker).await;
                }
                () = shutdown.cancelled() => {
                    debug!("health monitor shutting down");
                    return;
                }
            }
        }
    });
}

/// Poll all channels and broadcast any status changes.
async fn check_all_channels(
    state: &Arc<GatewayState>,
    tracker: &mut HashMap<ChannelId, ChannelHealth>,
) {
    for (channel_id, channel) in state.channels.channels() {
        let status = channel.health().await;
        let entry = tracker
            .entry(channel_id.clone())
            .or_insert_with(ChannelHealth::new);

        if entry.status_changed(&status) {
            info!(
                channel = %channel_id,
                status = ?status,
                "channel health status changed"
            );
            broadcast_health_event(&state.broadcast, channel_id, &status);
        }

        // Auto-restart disconnected channels.
        if is_unhealthy(&status) && entry.try_restart() {
            warn!(channel = %channel_id, "attempting channel restart");
            let (restart_tx, _) = tokio::sync::mpsc::channel(256);
            match channel.stop().await {
                Ok(()) => {
                    if let Err(e) = channel.start(restart_tx).await {
                        warn!(channel = %channel_id, error = %e, "channel restart failed");
                    } else {
                        info!(channel = %channel_id, "channel restarted successfully");
                    }
                }
                Err(e) => {
                    warn!(channel = %channel_id, error = %e, "channel stop failed during restart");
                }
            }
        }

        entry.last_status = Some(status);
    }
}

/// Broadcast a ChannelHealth event to all connected clients.
fn broadcast_health_event(
    broadcast: &BroadcastHandle,
    channel_id: &ChannelId,
    status: &HealthStatus,
) {
    let payload = serde_json::json!({
        "channel": channel_id.as_str(),
        "status": status,
    });
    let event = Frame::Event(EventFrame {
        event: EventType::ChannelHealth,
        payload,
    });
    if let Ok(json) = serde_json::to_string(&event) {
        let _ = broadcast.send(json);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn channel_health_detects_status_change() {
        let mut ch = ChannelHealth::new();

        // First observation is always a change.
        assert!(ch.status_changed(&HealthStatus::Connected));
        ch.last_status = Some(HealthStatus::Connected);

        // Same status = no change.
        assert!(!ch.status_changed(&HealthStatus::Connected));

        // Different status = change.
        assert!(ch.status_changed(&HealthStatus::Disconnected {
            reason: "timeout".into()
        }));

        ch.last_status = Some(HealthStatus::Disconnected {
            reason: "timeout".into(),
        });

        // Same disconnect reason = no change.
        assert!(!ch.status_changed(&HealthStatus::Disconnected {
            reason: "timeout".into()
        }));

        // Different reason = change.
        assert!(ch.status_changed(&HealthStatus::Disconnected {
            reason: "auth failure".into()
        }));
    }

    #[test]
    fn channel_health_restart_respects_cooldown() {
        let mut ch = ChannelHealth::new();

        // First restart should be allowed.
        assert!(ch.try_restart());

        // Immediate second restart should be denied (cooldown).
        assert!(!ch.try_restart());
    }

    #[test]
    fn channel_health_restart_rate_limiting() {
        let mut ch = ChannelHealth::new();

        // Manually fill restart history to max.
        let now = Instant::now();
        for _ in 0..MAX_RESTARTS_PER_HOUR {
            ch.restart_history.push(now);
        }
        // Even with no cooldown issue, rate limit should kick in.
        ch.last_restart_at = Some(
            now.checked_sub(Duration::from_secs(RESTART_COOLDOWN_SECS + 1))
                .expect("recent instant subtraction should succeed"),
        );
        assert!(!ch.try_restart());
    }

    #[test]
    fn is_unhealthy_identifies_disconnected() {
        assert!(is_unhealthy(&HealthStatus::Disconnected {
            reason: "test".into()
        }));
        assert!(!is_unhealthy(&HealthStatus::Connected));
        assert!(!is_unhealthy(&HealthStatus::NotConfigured));
        assert!(!is_unhealthy(&HealthStatus::Degraded {
            reason: "slow".into()
        }));
    }
}
