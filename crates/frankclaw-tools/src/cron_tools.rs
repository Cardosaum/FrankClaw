//! Cron management tools: list, add, and remove scheduled jobs.

use async_trait::async_trait;

use frankclaw_core::error::{AgentRuntime, InvalidRequest, Result};
use frankclaw_core::model::{ToolDef, ToolRiskLevel};

use crate::{Tool, ToolContext};

/// Maximum prompt length for cron jobs.
const MAX_PROMPT_LEN: usize = 2000;

fn get_cron(ctx: &ToolContext) -> Result<&dyn frankclaw_core::tool_services::CronManager> {
    ctx.cron.as_deref().ok_or_else(|| {
        AgentRuntime {
            msg: "cron tools are not available: no cron service configured".to_string(),
        }
        .build()
    })
}

// --------------------------------------------------------------------------
// cron.list
// --------------------------------------------------------------------------

pub struct CronListTool;

#[async_trait]
impl Tool for CronListTool {
    fn definition(&self) -> ToolDef {
        ToolDef {
            name: "cron_list".into(),
            description:
                "List all scheduled cron jobs with their schedule, status, and last run info."
                    .into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {}
            }),
            risk_level: ToolRiskLevel::ReadOnly,
        }
    }

    async fn invoke(
        &self,
        _args: serde_json::Value,
        ctx: ToolContext,
    ) -> Result<serde_json::Value> {
        let cron = get_cron(&ctx)?;
        let jobs = cron.list_jobs().await;
        Ok(serde_json::json!({
            "jobs": jobs,
            "count": jobs.len(),
        }))
    }
}

// --------------------------------------------------------------------------
// cron.add
// --------------------------------------------------------------------------

pub struct CronAddTool;

#[async_trait]
impl Tool for CronAddTool {
    fn definition(&self) -> ToolDef {
        ToolDef {
            name: "cron_add".into(),
            description: "Add a new scheduled cron job. The job will run the given prompt \
                on the specified schedule (cron expression)."
                .into(),
            parameters: serde_json::json!({
                "type": "object",
                "required": ["schedule", "prompt"],
                "properties": {
                    "schedule": {
                        "type": "string",
                        "description": "Cron expression (e.g., '0 */6 * * * *' for every 6 hours)."
                    },
                    "prompt": {
                        "type": "string",
                        "description": "Prompt text to execute on each run (max 2000 chars)."
                    },
                    "agent_id": {
                        "type": "string",
                        "description": "Agent to run the job as. Defaults to current agent."
                    },
                    "enabled": {
                        "type": "boolean",
                        "description": "Whether the job is enabled. Default: true."
                    }
                }
            }),
            risk_level: ToolRiskLevel::Mutating,
        }
    }

    async fn invoke(&self, args: serde_json::Value, ctx: ToolContext) -> Result<serde_json::Value> {
        let cron = get_cron(&ctx)?;

        let schedule = args
            .get("schedule")
            .and_then(|v| v.as_str())
            .map(str::trim)
            .filter(|v| !v.is_empty())
            .ok_or_else(|| {
                InvalidRequest {
                    msg: "cron.add requires a non-empty schedule".to_string(),
                }
                .build()
            })?;

        let prompt = args
            .get("prompt")
            .and_then(|v| v.as_str())
            .map(str::trim)
            .filter(|v| !v.is_empty())
            .ok_or_else(|| {
                InvalidRequest {
                    msg: "cron.add requires a non-empty prompt".to_string(),
                }
                .build()
            })?;

        if prompt.len() > MAX_PROMPT_LEN {
            return Err(InvalidRequest {
                msg: format!("cron.add prompt exceeds {MAX_PROMPT_LEN} char limit"),
            }
            .build());
        }

        let agent_id = args
            .get("agent_id")
            .and_then(|v| v.as_str())
            .unwrap_or(ctx.agent_id.as_str());

        let enabled = args
            .get("enabled")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(true);

        let job_id = uuid::Uuid::new_v4().to_string();
        let session_key = ctx.session_key.as_ref().map_or_else(
            || format!("{agent_id}:cron:{}", &job_id[..8]),
            |sk| sk.as_str().to_string(),
        );

        cron.add_job(&job_id, schedule, agent_id, &session_key, prompt, enabled)
            .await?;

        Ok(serde_json::json!({
            "status": "created",
            "job_id": job_id,
            "schedule": schedule,
            "agent_id": agent_id,
            "enabled": enabled,
        }))
    }
}

// --------------------------------------------------------------------------
// cron.remove
// --------------------------------------------------------------------------

pub struct CronRemoveTool;

#[async_trait]
impl Tool for CronRemoveTool {
    fn definition(&self) -> ToolDef {
        ToolDef {
            name: "cron_remove".into(),
            description: "Remove a scheduled cron job by ID.".into(),
            parameters: serde_json::json!({
                "type": "object",
                "required": ["job_id"],
                "properties": {
                    "job_id": {
                        "type": "string",
                        "description": "The job ID to remove."
                    }
                }
            }),
            risk_level: ToolRiskLevel::Destructive,
        }
    }

    async fn invoke(&self, args: serde_json::Value, ctx: ToolContext) -> Result<serde_json::Value> {
        let cron = get_cron(&ctx)?;

        let job_id = args
            .get("job_id")
            .and_then(|v| v.as_str())
            .map(str::trim)
            .filter(|v| !v.is_empty())
            .ok_or_else(|| {
                InvalidRequest {
                    msg: "cron.remove requires a non-empty job_id".to_string(),
                }
                .build()
            })?;

        let existed = cron.remove_job(job_id).await?;

        Ok(serde_json::json!({
            "job_id": job_id,
            "removed": existed,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cron_list_definition_is_valid() {
        let tool = CronListTool;
        let def = tool.definition();
        assert_eq!(def.name, "cron_list");
        assert_eq!(def.risk_level, ToolRiskLevel::ReadOnly);
    }

    #[test]
    fn cron_add_definition_is_valid() {
        let tool = CronAddTool;
        let def = tool.definition();
        assert_eq!(def.name, "cron_add");
        assert_eq!(def.risk_level, ToolRiskLevel::Mutating);
    }

    #[test]
    fn cron_remove_definition_is_valid() {
        let tool = CronRemoveTool;
        let def = tool.definition();
        assert_eq!(def.name, "cron_remove");
        assert_eq!(def.risk_level, ToolRiskLevel::Destructive);
    }
}
