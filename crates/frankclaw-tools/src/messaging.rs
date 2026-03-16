//! Channel messaging tool: send messages via configured channels.

use async_trait::async_trait;

use frankclaw_core::error::{AgentRuntime, FrankClawError, InvalidRequest, Result};
use frankclaw_core::model::{ToolDef, ToolRiskLevel};

use crate::{Tool, ToolContext};

/// Maximum text length for outbound messages.
const MAX_TEXT_LEN: usize = 4000;

fn agent_runtime_err(msg: impl Into<String>) -> FrankClawError {
    AgentRuntime { msg: msg.into() }.build()
}

fn invalid_request_err(msg: impl Into<String>) -> FrankClawError {
    InvalidRequest { msg: msg.into() }.build()
}

pub struct MessageSendTool;

#[async_trait]
impl Tool for MessageSendTool {
    fn definition(&self) -> ToolDef {
        ToolDef {
            name: "message_send".into(),
            description: "Send a text message via a configured channel \
                (Telegram, Discord, Slack, etc.)."
                .into(),
            parameters: serde_json::json!({
                "type": "object",
                "required": ["channel", "to", "text"],
                "properties": {
                    "channel": {
                        "type": "string",
                        "description": "Channel name (e.g., 'telegram', 'discord', 'slack')."
                    },
                    "to": {
                        "type": "string",
                        "description": "Recipient identifier (chat ID, channel ID, etc.)."
                    },
                    "text": {
                        "type": "string",
                        "description": "Message text (max 4000 chars)."
                    },
                    "account_id": {
                        "type": "string",
                        "description": "Account ID to send from. Default: 'default'."
                    },
                    "thread_id": {
                        "type": "string",
                        "description": "Optional thread/topic ID for threaded channels."
                    },
                    "reply_to": {
                        "type": "string",
                        "description": "Optional message ID to reply to."
                    }
                }
            }),
            risk_level: ToolRiskLevel::Mutating,
        }
    }

    async fn invoke(&self, args: serde_json::Value, ctx: ToolContext) -> Result<serde_json::Value> {
        let channels = ctx.channels.as_ref().ok_or_else(|| {
            agent_runtime_err("message.send is not available: no channel service configured")
        })?;

        let channel = args
            .get("channel")
            .and_then(|v| v.as_str())
            .map(str::trim)
            .filter(|v| !v.is_empty())
            .ok_or_else(|| invalid_request_err("message.send requires a non-empty channel"))?;

        let to = args
            .get("to")
            .and_then(|v| v.as_str())
            .map(str::trim)
            .filter(|v| !v.is_empty())
            .ok_or_else(|| {
                invalid_request_err("message.send requires a non-empty 'to' recipient")
            })?;

        let text = args
            .get("text")
            .and_then(|v| v.as_str())
            .ok_or_else(|| invalid_request_err("message.send requires text"))?;

        if text.len() > MAX_TEXT_LEN {
            return Err(invalid_request_err(format!(
                "message.send text exceeds {MAX_TEXT_LEN} char limit"
            )));
        }

        let account_id = args
            .get("account_id")
            .and_then(|v| v.as_str())
            .unwrap_or("default");

        let thread_id = args
            .get("thread_id")
            .and_then(|v| v.as_str())
            .filter(|v| !v.is_empty());

        let reply_to = args
            .get("reply_to")
            .and_then(|v| v.as_str())
            .filter(|v| !v.is_empty());

        let message_id = channels
            .send_text(channel, account_id, to, text, thread_id, reply_to)
            .await?;

        Ok(serde_json::json!({
            "status": "sent",
            "channel": channel,
            "to": to,
            "message_id": message_id,
        }))
    }
}

pub struct MessageReactTool;

#[async_trait]
impl Tool for MessageReactTool {
    fn definition(&self) -> ToolDef {
        ToolDef {
            name: "message_react".into(),
            description: "Send an emoji reaction to a message on a channel.".into(),
            parameters: serde_json::json!({
                "type": "object",
                "required": ["channel", "to", "message_id", "emoji"],
                "properties": {
                    "channel": {
                        "type": "string",
                        "description": "Channel name (e.g., 'telegram', 'discord', 'slack')."
                    },
                    "to": {
                        "type": "string",
                        "description": "Chat/channel ID where the message lives."
                    },
                    "message_id": {
                        "type": "string",
                        "description": "Platform message ID to react to."
                    },
                    "emoji": {
                        "type": "string",
                        "description": "Emoji to react with (e.g., '👍', '❤️', '✅')."
                    },
                    "account_id": {
                        "type": "string",
                        "description": "Account ID to send from. Default: 'default'."
                    },
                    "thread_id": {
                        "type": "string",
                        "description": "Optional thread/topic ID."
                    }
                }
            }),
            risk_level: ToolRiskLevel::Mutating,
        }
    }

    async fn invoke(&self, args: serde_json::Value, ctx: ToolContext) -> Result<serde_json::Value> {
        let channels = ctx.channels.as_ref().ok_or_else(|| {
            agent_runtime_err("message.react is not available: no channel service configured")
        })?;

        let channel = args
            .get("channel")
            .and_then(|v| v.as_str())
            .map(str::trim)
            .filter(|v| !v.is_empty())
            .ok_or_else(|| invalid_request_err("message.react requires a non-empty channel"))?;

        let to = args
            .get("to")
            .and_then(|v| v.as_str())
            .map(str::trim)
            .filter(|v| !v.is_empty())
            .ok_or_else(|| invalid_request_err("message.react requires a non-empty 'to'"))?;

        let message_id = args
            .get("message_id")
            .and_then(|v| v.as_str())
            .map(str::trim)
            .filter(|v| !v.is_empty())
            .ok_or_else(|| invalid_request_err("message.react requires a message_id"))?;

        let emoji = args
            .get("emoji")
            .and_then(|v| v.as_str())
            .map(str::trim)
            .filter(|v| !v.is_empty())
            .ok_or_else(|| invalid_request_err("message.react requires an emoji"))?;

        // Sanity check: emoji should be short.
        if emoji.len() > 32 {
            return Err(invalid_request_err("emoji is too long"));
        }

        let account_id = args
            .get("account_id")
            .and_then(|v| v.as_str())
            .unwrap_or("default");

        let thread_id = args
            .get("thread_id")
            .and_then(|v| v.as_str())
            .filter(|v| !v.is_empty());

        channels
            .send_reaction(channel, account_id, to, thread_id, message_id, emoji)
            .await?;

        Ok(serde_json::json!({
            "status": "reacted",
            "channel": channel,
            "message_id": message_id,
            "emoji": emoji,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn message_send_definition_is_valid() {
        let tool = MessageSendTool;
        let def = tool.definition();
        assert_eq!(def.name, "message_send");
        assert_eq!(def.risk_level, ToolRiskLevel::Mutating);
    }

    #[test]
    fn message_react_definition_is_valid() {
        let tool = MessageReactTool;
        let def = tool.definition();
        assert_eq!(def.name, "message_react");
        assert_eq!(def.risk_level, ToolRiskLevel::Mutating);
    }
}
