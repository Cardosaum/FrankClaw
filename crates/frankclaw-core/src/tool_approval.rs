//! Interactive tool approval types.
//!
//! When a tool requires human approval before execution, the runtime sends an
//! `ApprovalRequest` and waits for an `ApprovalDecision`.

use serde::{Deserialize, Serialize};

/// A request for human approval of a tool invocation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApprovalRequest {
    /// Unique ID for this approval request.
    pub approval_id: String,
    /// Tool name being invoked.
    pub tool_name: String,
    /// Tool arguments (may be large — consider truncating for display).
    pub tool_args: serde_json::Value,
    /// Risk level of the tool.
    pub risk_level: String,
    /// Session key context.
    pub session_key: String,
    /// Agent ID context.
    pub agent_id: String,
}

/// The human's decision on a tool approval request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ApprovalDecision {
    /// Allow this single invocation.
    AllowOnce,
    /// Allow all future invocations of this tool in this session.
    AllowAlways,
    /// Deny this invocation.
    Deny,
}

/// Channel for sending approval requests from the runtime to the gateway.
pub type ApprovalRequestTx = tokio::sync::mpsc::Sender<(
    ApprovalRequest,
    tokio::sync::oneshot::Sender<ApprovalDecision>,
)>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn approval_decision_serializes() {
        assert_eq!(
            serde_json::to_string(&ApprovalDecision::AllowOnce).unwrap(),
            "\"allow_once\""
        );
        assert_eq!(
            serde_json::to_string(&ApprovalDecision::Deny).unwrap(),
            "\"deny\""
        );
    }

    #[test]
    fn approval_request_serializes() {
        let req = ApprovalRequest {
            approval_id: "test-123".into(),
            tool_name: "bash".into(),
            tool_args: serde_json::json!({"command": "rm -rf /"}),
            risk_level: "destructive".into(),
            session_key: "session-1".into(),
            agent_id: "agent-1".into(),
        };
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["tool_name"], "bash");
        assert_eq!(json["risk_level"], "destructive");
    }
}
