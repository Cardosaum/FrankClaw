#![forbid(unsafe_code)]

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;

use frankclaw_core::error::{FrankClawError, Result};
use frankclaw_core::model::ToolDef;
use frankclaw_core::session::SessionStore;
use frankclaw_core::types::{AgentId, SessionKey};

#[derive(Clone)]
pub struct ToolContext {
    pub agent_id: AgentId,
    pub session_key: Option<SessionKey>,
    pub sessions: Arc<dyn SessionStore>,
}

#[derive(Debug, Clone)]
pub struct ToolOutput {
    pub name: String,
    pub output: serde_json::Value,
}

#[async_trait]
pub trait Tool: Send + Sync + 'static {
    fn definition(&self) -> ToolDef;

    async fn invoke(&self, args: serde_json::Value, ctx: ToolContext) -> Result<serde_json::Value>;
}

pub struct ToolRegistry {
    tools: HashMap<String, Arc<dyn Tool>>,
}

impl ToolRegistry {
    pub fn with_builtins() -> Self {
        let mut registry = Self {
            tools: HashMap::new(),
        };
        registry.register(Arc::new(SessionInspectTool));
        registry
    }

    pub fn register(&mut self, tool: Arc<dyn Tool>) {
        self.tools.insert(tool.definition().name.clone(), tool);
    }

    pub fn validate_names(&self, names: &[String]) -> Result<()> {
        for name in names {
            if !self.tools.contains_key(name) {
                return Err(FrankClawError::ConfigValidation {
                    msg: format!("unknown tool '{}'", name),
                });
            }
        }
        Ok(())
    }

    pub fn definitions(&self, names: &[String]) -> Result<Vec<ToolDef>> {
        self.validate_names(names)?;
        Ok(names
            .iter()
            .filter_map(|name| self.tools.get(name))
            .map(|tool| tool.definition())
            .collect())
    }

    pub async fn invoke_allowed(
        &self,
        allowed_tools: &[String],
        name: &str,
        args: serde_json::Value,
        ctx: ToolContext,
    ) -> Result<ToolOutput> {
        if !allowed_tools.iter().any(|allowed| allowed == name) {
            return Err(FrankClawError::AgentRuntime {
                msg: format!("tool '{}' is not allowed for agent '{}'", name, ctx.agent_id),
            });
        }

        let tool = self
            .tools
            .get(name)
            .ok_or_else(|| FrankClawError::InvalidRequest {
                msg: format!("unknown tool '{}'", name),
            })?;
        let output = tool.invoke(args, ctx).await?;
        Ok(ToolOutput {
            name: name.to_string(),
            output,
        })
    }
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::with_builtins()
    }
}

struct SessionInspectTool;

#[async_trait]
impl Tool for SessionInspectTool {
    fn definition(&self) -> ToolDef {
        ToolDef {
            name: "session.inspect".into(),
            description: "Inspect one session entry and recent transcript messages.".into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "session_key": {
                        "type": "string",
                        "description": "Optional explicit session key. Defaults to the current tool context session."
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 100,
                        "description": "Maximum transcript entries to return."
                    }
                }
            }),
        }
    }

    async fn invoke(&self, args: serde_json::Value, ctx: ToolContext) -> Result<serde_json::Value> {
        let session_key = args
            .get("session_key")
            .and_then(|value| value.as_str())
            .map(SessionKey::from_raw)
            .or(ctx.session_key)
            .ok_or_else(|| FrankClawError::InvalidRequest {
                msg: "session.inspect requires a session_key".into(),
            })?;
        let limit = args
            .get("limit")
            .and_then(|value| value.as_u64())
            .unwrap_or(20)
            .clamp(1, 100) as usize;

        let session = ctx.sessions.get(&session_key).await?;
        let entries = ctx.sessions.get_transcript(&session_key, limit, None).await?;

        Ok(serde_json::json!({
            "session": session,
            "entries": entries,
        }))
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::sync::Arc;

    use async_trait::async_trait;
    use chrono::Utc;
    use tokio::sync::Mutex;

    use frankclaw_core::session::{
        PruningConfig, SessionEntry, SessionScoping, SessionStore, TranscriptEntry,
    };
    use frankclaw_core::types::{ChannelId, Role};

    use super::*;

    #[derive(Default)]
    struct MockSessionStore {
        sessions: Mutex<BTreeMap<String, SessionEntry>>,
        transcripts: Mutex<BTreeMap<String, Vec<TranscriptEntry>>>,
    }

    #[async_trait]
    impl SessionStore for MockSessionStore {
        async fn get(&self, key: &SessionKey) -> Result<Option<SessionEntry>> {
            Ok(self.sessions.lock().await.get(key.as_str()).cloned())
        }

        async fn upsert(&self, entry: &SessionEntry) -> Result<()> {
            self.sessions
                .lock()
                .await
                .insert(entry.key.as_str().to_string(), entry.clone());
            Ok(())
        }

        async fn delete(&self, key: &SessionKey) -> Result<()> {
            self.sessions.lock().await.remove(key.as_str());
            self.transcripts.lock().await.remove(key.as_str());
            Ok(())
        }

        async fn list(
            &self,
            _agent_id: &AgentId,
            _limit: usize,
            _offset: usize,
        ) -> Result<Vec<SessionEntry>> {
            Ok(self.sessions.lock().await.values().cloned().collect())
        }

        async fn append_transcript(&self, key: &SessionKey, entry: &TranscriptEntry) -> Result<()> {
            self.transcripts
                .lock()
                .await
                .entry(key.as_str().to_string())
                .or_default()
                .push(entry.clone());
            Ok(())
        }

        async fn get_transcript(
            &self,
            key: &SessionKey,
            limit: usize,
            _before_seq: Option<u64>,
        ) -> Result<Vec<TranscriptEntry>> {
            Ok(self
                .transcripts
                .lock()
                .await
                .get(key.as_str())
                .cloned()
                .unwrap_or_default()
                .into_iter()
                .take(limit)
                .collect())
        }

        async fn clear_transcript(&self, key: &SessionKey) -> Result<()> {
            self.transcripts.lock().await.remove(key.as_str());
            Ok(())
        }

        async fn maintenance(&self, _config: &PruningConfig) -> Result<u64> {
            Ok(0)
        }
    }

    #[tokio::test]
    async fn session_inspect_returns_session_and_entries() {
        let store = Arc::new(MockSessionStore::default());
        let key = SessionKey::from_raw("main:web:default");
        store
            .upsert(&SessionEntry {
                key: key.clone(),
                agent_id: AgentId::default_agent(),
                channel: ChannelId::new("web"),
                account_id: "default".into(),
                scoping: SessionScoping::PerChannelPeer,
                created_at: Utc::now(),
                last_message_at: Some(Utc::now()),
                thread_id: None,
                metadata: serde_json::json!({}),
            })
            .await
            .expect("session should upsert");
        store
            .append_transcript(
                &key,
                &TranscriptEntry {
                    seq: 1,
                    role: Role::User,
                    content: "hello".into(),
                    timestamp: Utc::now(),
                    metadata: None,
                },
            )
            .await
            .expect("transcript should append");

        let registry = ToolRegistry::with_builtins();
        let result = registry
            .invoke_allowed(
                &["session.inspect".into()],
                "session.inspect",
                serde_json::json!({ "limit": 5 }),
                ToolContext {
                    agent_id: AgentId::default_agent(),
                    session_key: Some(key.clone()),
                    sessions: store as Arc<dyn SessionStore>,
                },
            )
            .await
            .expect("tool should succeed");

        assert_eq!(result.name, "session.inspect");
        assert_eq!(result.output["session"]["key"], serde_json::json!(key.as_str()));
        assert_eq!(result.output["entries"][0]["content"], serde_json::json!("hello"));
    }

    #[tokio::test]
    async fn invoke_allowed_rejects_unlisted_tools() {
        let registry = ToolRegistry::with_builtins();
        let err = registry
            .invoke_allowed(
                &[],
                "session.inspect",
                serde_json::json!({}),
                ToolContext {
                    agent_id: AgentId::default_agent(),
                    session_key: None,
                    sessions: Arc::new(MockSessionStore::default()),
                },
            )
            .await
            .expect_err("tool should be rejected");

        assert!(err.to_string().contains("not allowed"));
    }
}
