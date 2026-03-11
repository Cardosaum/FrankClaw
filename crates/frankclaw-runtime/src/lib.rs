#![forbid(unsafe_code)]

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use chrono::Utc;
use secrecy::SecretString;

use frankclaw_core::config::{AgentDef, FrankClawConfig, ProviderConfig};
use frankclaw_core::error::{FrankClawError, Result};
use frankclaw_core::channel::InboundMessage;
use frankclaw_core::model::{
    CompletionMessage, CompletionRequest, ModelDef, ModelProvider, Usage,
};
use frankclaw_core::session::{SessionEntry, SessionStore, TranscriptEntry};
use frankclaw_core::types::{AgentId, ChannelId, Role, SessionKey};
use frankclaw_models::{
    AnthropicProvider, FailoverChain, OllamaProvider, OpenAiProvider,
};
use frankclaw_plugin_sdk::{SkillManifest, load_workspace_skills};
use frankclaw_tools::{ToolContext, ToolOutput, ToolRegistry};

pub struct Runtime {
    config: FrankClawConfig,
    sessions: Arc<dyn SessionStore>,
    models: FailoverChain,
    model_defs: Vec<ModelDef>,
    channel_ids: Vec<ChannelId>,
    tools: ToolRegistry,
    skill_manifests: HashMap<AgentId, Vec<SkillManifest>>,
}

#[derive(Debug, Clone)]
pub struct ChatRequest {
    pub agent_id: Option<AgentId>,
    pub session_key: Option<SessionKey>,
    pub message: String,
    pub model_id: Option<String>,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
}

#[derive(Debug, Clone)]
pub struct ChatResponse {
    pub session_key: SessionKey,
    pub model_id: String,
    pub content: String,
    pub usage: Usage,
}

#[derive(Debug, Clone)]
pub struct ToolRequest {
    pub agent_id: Option<AgentId>,
    pub session_key: Option<SessionKey>,
    pub tool_name: String,
    pub arguments: serde_json::Value,
}

impl Runtime {
    pub async fn from_config(
        config: &FrankClawConfig,
        sessions: Arc<dyn SessionStore>,
    ) -> Result<Self> {
        let providers = build_providers(config)?;
        Self::from_providers(config, sessions, providers).await
    }

    pub async fn from_providers(
        config: &FrankClawConfig,
        sessions: Arc<dyn SessionStore>,
        providers: Vec<Arc<dyn ModelProvider>>,
    ) -> Result<Self> {
        let cooldown_secs = config
            .models
            .providers
            .iter()
            .map(|provider| provider.cooldown_secs)
            .max()
            .unwrap_or(30)
            .max(1);
        let models = FailoverChain::new(providers, cooldown_secs);
        let model_defs = models.list_models().await?;
        let channel_ids = config
            .channels
            .iter()
            .filter_map(|(channel_id, channel)| {
                if channel.enabled {
                    Some(channel_id.clone())
                } else {
                    None
                }
            })
            .collect();
        let tools = ToolRegistry::with_builtins();
        for (agent_id, agent) in &config.agents.agents {
            tools.validate_names(&agent.tools).map_err(|err| {
                FrankClawError::ConfigValidation {
                    msg: format!("agent '{}' has invalid tool config: {}", agent_id, err),
                }
            })?;
        }
        let skill_manifests = load_agent_skills(config)?;

        Ok(Self {
            config: config.clone(),
            sessions,
            models,
            model_defs,
            channel_ids,
            tools,
            skill_manifests,
        })
    }

    pub fn list_models(&self) -> &[ModelDef] {
        &self.model_defs
    }

    pub fn list_channels(&self) -> &[ChannelId] {
        &self.channel_ids
    }

    pub fn list_tools(&self, agent_id: Option<&AgentId>) -> Result<Vec<frankclaw_core::model::ToolDef>> {
        let (_, agent) = self.resolve_agent(agent_id)?;
        self.tools.definitions(&agent.tools)
    }

    pub fn list_skills(&self, agent_id: Option<&AgentId>) -> Result<&[SkillManifest]> {
        let (agent_id, _) = self.resolve_agent(agent_id)?;
        Ok(self
            .skill_manifests
            .get(&agent_id)
            .map(|skills| skills.as_slice())
            .unwrap_or(&[]))
    }

    pub fn session_key_for_inbound(
        &self,
        inbound: &InboundMessage,
    ) -> SessionKey {
        let account_scope = self.config.session.scoping.resolve_inbound_account_scope(
            &inbound.account_id,
            &inbound.sender_id,
            inbound.thread_id.as_deref(),
            inbound.is_group,
        );

        SessionKey::new(
            &self.config.agents.default_agent,
            &inbound.channel,
            &account_scope,
        )
    }

    pub async fn chat(&self, request: ChatRequest) -> Result<ChatResponse> {
        if request.message.trim().is_empty() {
            return Err(FrankClawError::InvalidRequest {
                msg: "message is required".into(),
            });
        }

        let (agent_id, agent) = self.resolve_agent(request.agent_id.as_ref())?;
        let model_id = self.resolve_model_id(&agent, request.model_id.as_deref())?;
        let session_key = self.resolve_session_key(&agent_id, request.session_key)?;
        let history = self.sessions.get_transcript(&session_key, 200, None).await?;
        let next_seq = history.last().map(|entry| entry.seq + 1).unwrap_or(1);

        self.ensure_session(&session_key, &agent_id).await?;

        let request_messages = history
            .iter()
            .map(|entry| CompletionMessage {
                role: entry.role,
                content: entry.content.clone(),
            })
            .chain(std::iter::once(CompletionMessage {
                role: Role::User,
                content: request.message.clone(),
            }))
            .collect();

        self.sessions
            .append_transcript(
                &session_key,
                &TranscriptEntry {
                    seq: next_seq,
                    role: Role::User,
                    content: request.message,
                    timestamp: Utc::now(),
                    metadata: None,
                },
            )
            .await?;

        let response = self
            .models
            .complete(
                CompletionRequest {
                    model_id: model_id.clone(),
                    messages: request_messages,
                    max_tokens: request.max_tokens,
                    temperature: request.temperature,
                    system: self.build_system_prompt(&agent_id, &agent),
                    tools: Vec::new(),
                },
                None,
            )
            .await?;

        self.sessions
            .append_transcript(
                &session_key,
                &TranscriptEntry {
                    seq: next_seq + 1,
                    role: Role::Assistant,
                    content: response.content.clone(),
                    timestamp: Utc::now(),
                    metadata: None,
                },
            )
            .await?;

        Ok(ChatResponse {
            session_key,
            model_id,
            content: response.content,
            usage: response.usage,
        })
    }

    pub async fn invoke_tool(&self, request: ToolRequest) -> Result<ToolOutput> {
        let (agent_id, agent) = self.resolve_agent(request.agent_id.as_ref())?;
        self.tools
            .invoke_allowed(
                &agent.tools,
                &request.tool_name,
                request.arguments,
                ToolContext {
                    agent_id,
                    session_key: request.session_key,
                    sessions: self.sessions.clone(),
                },
            )
            .await
    }

    fn resolve_agent(&self, requested: Option<&AgentId>) -> Result<(AgentId, AgentDef)> {
        let agent_id = requested
            .cloned()
            .unwrap_or_else(|| self.config.agents.default_agent.clone());
        let agent = self
            .config
            .agents
            .agents
            .get(&agent_id)
            .cloned()
            .ok_or_else(|| FrankClawError::AgentNotFound {
                agent_id: agent_id.clone(),
            })?;
        Ok((agent_id, agent))
    }

    fn build_system_prompt(&self, agent_id: &AgentId, agent: &AgentDef) -> Option<String> {
        let skill_prompts = self
            .skill_manifests
            .get(agent_id)
            .map(|skills| {
                skills
                    .iter()
                    .map(|skill| format!("[Skill: {}]\n{}", skill.name, skill.prompt.trim()))
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();

        match (agent.system_prompt.as_deref(), skill_prompts.is_empty()) {
            (None, true) => None,
            (Some(system), true) => Some(system.to_string()),
            (None, false) => Some(skill_prompts.join("\n\n")),
            (Some(system), false) => Some(format!(
                "{}\n\n{}",
                system.trim(),
                skill_prompts.join("\n\n")
            )),
        }
    }

    fn resolve_model_id(
        &self,
        agent: &AgentDef,
        requested: Option<&str>,
    ) -> Result<String> {
        if let Some(model_id) = requested {
            return Ok(model_id.to_string());
        }
        if let Some(model_id) = &agent.model {
            return Ok(model_id.clone());
        }
        if let Some(model_id) = &self.config.models.default_model {
            return Ok(model_id.clone());
        }
        self.model_defs
            .first()
            .map(|model| model.id.clone())
            .ok_or_else(|| FrankClawError::ConfigValidation {
                msg: "no model providers are configured".into(),
            })
    }

    fn resolve_session_key(
        &self,
        agent_id: &AgentId,
        explicit: Option<SessionKey>,
    ) -> Result<SessionKey> {
        if let Some(session_key) = explicit {
            if let Some((session_agent_id, _, _)) = session_key.parse() {
                if &session_agent_id != agent_id {
                    return Err(FrankClawError::InvalidRequest {
                        msg: format!(
                            "session '{}' does not belong to agent '{}'",
                            session_key, agent_id
                        ),
                    });
                }
            }
            return Ok(session_key);
        }

        Ok(SessionKey::new(
            agent_id,
            &ChannelId::new("web"),
            "control",
        ))
    }

    async fn ensure_session(
        &self,
        session_key: &SessionKey,
        agent_id: &AgentId,
    ) -> Result<()> {
        if self.sessions.get(session_key).await?.is_some() {
            return Ok(());
        }

        let (channel, account_id) = session_key
            .parse()
            .map(|(_, channel, account_id)| (channel, account_id))
            .unwrap_or_else(|| (ChannelId::new("web"), "control".to_string()));

        self.sessions
            .upsert(&SessionEntry {
                key: session_key.clone(),
                agent_id: agent_id.clone(),
                channel,
                account_id,
                scoping: self.config.session.scoping,
                created_at: Utc::now(),
                last_message_at: None,
                thread_id: None,
                metadata: serde_json::json!({}),
            })
            .await
    }
}

fn build_providers(
    config: &FrankClawConfig,
) -> Result<Vec<Arc<dyn frankclaw_core::model::ModelProvider>>> {
    let mut providers: Vec<Arc<dyn frankclaw_core::model::ModelProvider>> = Vec::new();
    let mut seen_ids = HashSet::new();

    for provider in &config.models.providers {
        if !seen_ids.insert(provider.id.clone()) {
            return Err(FrankClawError::ConfigValidation {
                msg: format!("duplicate model provider id '{}'", provider.id),
            });
        }

        let provider_impl: Arc<dyn frankclaw_core::model::ModelProvider> =
            match provider.api.as_str() {
                "openai" => Arc::new(OpenAiProvider::new(
                    provider.id.clone(),
                    provider
                        .base_url
                        .clone()
                        .unwrap_or_else(|| "https://api.openai.com/v1".to_string()),
                    resolve_secret(provider, "OPENAI_API_KEY")?,
                    provider.models.clone(),
                )),
                "anthropic" => Arc::new(AnthropicProvider::new(
                    provider.id.clone(),
                    resolve_secret(provider, "ANTHROPIC_API_KEY")?,
                    provider.models.clone(),
                )),
                "ollama" => Arc::new(OllamaProvider::new(
                    provider.id.clone(),
                    provider.base_url.clone(),
                )),
                other => {
                    return Err(FrankClawError::ConfigValidation {
                        msg: format!(
                            "unsupported model provider api '{}'; expected openai, anthropic, or ollama",
                            other
                        ),
                    });
                }
            };
        providers.push(provider_impl);
    }

    Ok(providers)
}

fn load_agent_skills(config: &FrankClawConfig) -> Result<HashMap<AgentId, Vec<SkillManifest>>> {
    let mut manifests = HashMap::new();

    for (agent_id, agent) in &config.agents.agents {
        if agent.skills.is_empty() {
            manifests.insert(agent_id.clone(), Vec::new());
            continue;
        }

        let workspace = agent.workspace.as_ref().ok_or_else(|| FrankClawError::ConfigValidation {
            msg: format!("agent '{}' declares skills but has no workspace", agent_id),
        })?;
        let skills = load_workspace_skills(workspace, &agent.skills)?;
        for skill in &skills {
            for tool in &skill.tools {
                if !agent.tools.iter().any(|allowed| allowed == tool) {
                    return Err(FrankClawError::ConfigValidation {
                        msg: format!(
                            "agent '{}' skill '{}' requires tool '{}' but the agent does not allow it",
                            agent_id, skill.id, tool
                        ),
                    });
                }
            }
        }
        manifests.insert(agent_id.clone(), skills);
    }

    Ok(manifests)
}

fn resolve_secret(provider: &ProviderConfig, default_env: &str) -> Result<SecretString> {
    let env_key = provider
        .api_key_ref
        .as_deref()
        .unwrap_or(default_env)
        .trim();
    if env_key.is_empty() {
        return Err(FrankClawError::ConfigValidation {
            msg: format!("provider '{}' requires an api_key_ref", provider.id),
        });
    }

    let value = std::env::var(env_key).map_err(|_| FrankClawError::ConfigValidation {
        msg: format!(
            "provider '{}' references missing environment variable '{}'",
            provider.id, env_key
        ),
    })?;

    if value.trim().is_empty() {
        return Err(FrankClawError::ConfigValidation {
            msg: format!(
                "provider '{}' environment variable '{}' is empty",
                provider.id, env_key
            ),
        });
    }

    Ok(SecretString::from(value))
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use frankclaw_core::model::{
        CompletionResponse, FinishReason, InputModality, ModelApi, ModelCompat, ModelCost,
    };
    use frankclaw_core::session::SessionStore;
    use frankclaw_sessions::SqliteSessionStore;
    use std::sync::Mutex;

    struct MockProvider {
        id: String,
        model_id: String,
        response: Option<String>,
        seen_system: Mutex<Vec<Option<String>>>,
    }

    #[async_trait]
    impl ModelProvider for MockProvider {
        fn id(&self) -> &str {
            &self.id
        }

        async fn complete(
            &self,
            request: CompletionRequest,
            _stream_tx: Option<tokio::sync::mpsc::Sender<frankclaw_core::model::StreamDelta>>,
        ) -> Result<CompletionResponse> {
            self.seen_system
                .lock()
                .expect("system capture should lock")
                .push(request.system.clone());
            match &self.response {
                Some(content) => Ok(CompletionResponse {
                    content: content.clone(),
                    tool_calls: Vec::new(),
                    usage: Usage {
                        input_tokens: 4,
                        output_tokens: 2,
                        cache_read_tokens: None,
                        cache_write_tokens: None,
                    },
                    finish_reason: FinishReason::Stop,
                }),
                None => Err(FrankClawError::AllProvidersFailed),
            }
        }

        async fn list_models(&self) -> Result<Vec<ModelDef>> {
            Ok(vec![ModelDef {
                id: self.model_id.clone(),
                name: self.model_id.clone(),
                api: ModelApi::Ollama,
                reasoning: false,
                input: vec![InputModality::Text],
                cost: ModelCost::default(),
                context_window: 8192,
                max_output_tokens: 1024,
                compat: ModelCompat::default(),
            }])
        }

        async fn health(&self) -> bool {
            true
        }
    }

    #[tokio::test]
    async fn runtime_fails_over_to_next_provider_and_persists_history() {
        let temp = std::env::temp_dir().join(format!(
            "frankclaw-runtime-{}.db",
            uuid::Uuid::new_v4()
        ));
        let sessions = Arc::new(SqliteSessionStore::open(&temp, None).expect("sessions should open"));
        let mut config = FrankClawConfig::default();
        config.models.providers = vec![
            ProviderConfig {
                id: "primary".into(),
                api: "ollama".into(),
                base_url: None,
                api_key_ref: None,
                models: vec!["mock-primary".into()],
                cooldown_secs: 1,
            },
            ProviderConfig {
                id: "secondary".into(),
                api: "ollama".into(),
                base_url: None,
                api_key_ref: None,
                models: vec!["mock-secondary".into()],
                cooldown_secs: 1,
            },
        ];

        let runtime = Runtime::from_providers(
            &config,
            sessions.clone() as Arc<dyn SessionStore>,
            vec![
                Arc::new(MockProvider {
                    id: "primary".into(),
                    model_id: "mock-primary".into(),
                    response: None,
                    seen_system: Mutex::new(Vec::new()),
                }),
                Arc::new(MockProvider {
                    id: "secondary".into(),
                    model_id: "mock-secondary".into(),
                    response: Some("fallback reply".into()),
                    seen_system: Mutex::new(Vec::new()),
                }),
            ],
        )
        .await
        .expect("runtime should build");

        let response = runtime
            .chat(ChatRequest {
                agent_id: None,
                session_key: None,
                message: "hello".into(),
                model_id: Some("mock-secondary".into()),
                max_tokens: None,
                temperature: None,
            })
            .await
            .expect("chat should succeed");

        assert_eq!(response.content, "fallback reply");
        let transcript = sessions
            .get_transcript(&response.session_key, 10, None)
            .await
            .expect("transcript should load");
        assert_eq!(transcript.len(), 2);
        assert_eq!(transcript[0].role, Role::User);
        assert_eq!(transcript[1].role, Role::Assistant);

        let _ = std::fs::remove_file(temp);
    }

    #[tokio::test]
    async fn runtime_invokes_allowed_tool() {
        let temp = std::env::temp_dir().join(format!(
            "frankclaw-runtime-tools-{}.db",
            uuid::Uuid::new_v4()
        ));
        let sessions = Arc::new(SqliteSessionStore::open(&temp, None).expect("sessions should open"));
        let mut config = FrankClawConfig::default();
        config.agents.agents.get_mut(&AgentId::default_agent()).unwrap().tools =
            vec!["session.inspect".into()];
        let runtime = Runtime::from_providers(
            &config,
            sessions.clone() as Arc<dyn SessionStore>,
            vec![Arc::new(MockProvider {
                id: "primary".into(),
                model_id: "mock-primary".into(),
                response: Some("reply".into()),
                seen_system: Mutex::new(Vec::new()),
            })],
        )
        .await
        .expect("runtime should build");

        let chat = runtime
            .chat(ChatRequest {
                agent_id: None,
                session_key: None,
                message: "hello".into(),
                model_id: Some("mock-primary".into()),
                max_tokens: None,
                temperature: None,
            })
            .await
            .expect("chat should succeed");

        let tool = runtime
            .invoke_tool(ToolRequest {
                agent_id: None,
                session_key: Some(chat.session_key.clone()),
                tool_name: "session.inspect".into(),
                arguments: serde_json::json!({ "limit": 5 }),
            })
            .await
            .expect("tool should run");

        assert_eq!(tool.name, "session.inspect");
        assert_eq!(
            tool.output["session"]["key"],
            serde_json::json!(chat.session_key.as_str())
        );

        let _ = std::fs::remove_file(temp);
    }

    #[tokio::test]
    async fn runtime_includes_skill_prompt_in_system_message() {
        let temp = std::env::temp_dir().join(format!(
            "frankclaw-runtime-skill-{}",
            uuid::Uuid::new_v4()
        ));
        let workspace = temp.join("workspace");
        let skill_dir = workspace.join(".frankclaw/skills/briefing");
        std::fs::create_dir_all(&skill_dir).expect("skill dir should exist");
        std::fs::write(
            skill_dir.join("skill.json"),
            serde_json::json!({
                "id": "briefing",
                "name": "Briefing",
                "prompt": "Summarize in a terse operational style.",
                "tools": []
            })
            .to_string(),
        )
        .expect("skill manifest should write");

        let sessions = Arc::new(SqliteSessionStore::open(&temp.join("sessions.db"), None).expect("sessions should open"));
        let mut config = FrankClawConfig::default();
        let agent = config.agents.agents.get_mut(&AgentId::default_agent()).unwrap();
        agent.workspace = Some(workspace.clone());
        agent.system_prompt = Some("Base system".into());
        agent.skills = vec!["briefing".into()];

        let provider = Arc::new(MockProvider {
            id: "primary".into(),
            model_id: "mock-primary".into(),
            response: Some("reply".into()),
            seen_system: Mutex::new(Vec::new()),
        });
        let runtime = Runtime::from_providers(
            &config,
            sessions as Arc<dyn SessionStore>,
            vec![provider.clone()],
        )
        .await
        .expect("runtime should build");

        runtime
            .chat(ChatRequest {
                agent_id: None,
                session_key: None,
                message: "hello".into(),
                model_id: Some("mock-primary".into()),
                max_tokens: None,
                temperature: None,
            })
            .await
            .expect("chat should succeed");

        let seen = provider.seen_system.lock().expect("system capture should lock");
        let system = seen
            .last()
            .and_then(|value| value.clone())
            .expect("system prompt should be captured");
        assert!(system.contains("Base system"));
        assert!(system.contains("[Skill: Briefing]"));
        assert!(system.contains("Summarize in a terse operational style."));

        let _ = std::fs::remove_dir_all(temp);
    }
}
