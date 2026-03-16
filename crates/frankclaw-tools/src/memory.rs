//! Memory tools: read files from the agent's memory directory.

use std::path::Path;

use async_trait::async_trait;

use frankclaw_core::error::{AgentRuntime, FrankClawError, InvalidRequest, Result};
use frankclaw_core::model::{ToolDef, ToolRiskLevel};

use crate::{Tool, ToolContext};

/// Maximum output size (100 KB).
const MAX_OUTPUT_BYTES: usize = 100 * 1024;

/// Default memory subdirectory name within workspace.
const MEMORY_DIR: &str = "memory";

fn agent_runtime_err(msg: impl Into<String>) -> FrankClawError {
    AgentRuntime { msg: msg.into() }.build()
}

fn invalid_request_err(msg: impl Into<String>) -> FrankClawError {
    InvalidRequest { msg: msg.into() }.build()
}

// --------------------------------------------------------------------------
// memory.get
// --------------------------------------------------------------------------

pub struct MemoryGetTool;

#[async_trait]
impl Tool for MemoryGetTool {
    fn definition(&self) -> ToolDef {
        ToolDef {
            name: "memory_get".into(),
            description: "Read a file from the agent's memory directory. \
                Use this to retrieve stored notes, context, or reference data."
                .into(),
            parameters: serde_json::json!({
                "type": "object",
                "required": ["path"],
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path within the memory directory."
                    },
                    "from": {
                        "type": "integer",
                        "minimum": 0,
                        "description": "Starting line number (0-based). Default: 0."
                    },
                    "lines": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 5000,
                        "description": "Maximum lines to return. Default: 500."
                    }
                }
            }),
            risk_level: ToolRiskLevel::ReadOnly,
        }
    }

    async fn invoke(&self, args: serde_json::Value, ctx: ToolContext) -> Result<serde_json::Value> {
        let workspace = ctx.workspace.as_deref().ok_or_else(|| {
            agent_runtime_err("memory.get is not available: no workspace directory configured")
        })?;

        let memory_dir = workspace.join(MEMORY_DIR);
        if !memory_dir.exists() {
            return Err(agent_runtime_err("memory directory does not exist"));
        }

        let path_str = args
            .get("path")
            .and_then(|v| v.as_str())
            .map(str::trim)
            .filter(|v| !v.is_empty())
            .ok_or_else(|| invalid_request_err("memory.get requires a non-empty path"))?;

        // Security: validate the path doesn't escape memory directory.
        validate_memory_path(&memory_dir, path_str)?;

        let resolved = memory_dir.join(path_str);
        let from = args
            .get("from")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0) as usize;
        let max_lines = args
            .get("lines")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(500)
            .clamp(1, 5000) as usize;

        let content = tokio::fs::read_to_string(&resolved).await.map_err(|e| {
            agent_runtime_err(format!("failed to read memory file '{path_str}': {e}"))
        })?;

        let lines: Vec<&str> = content.lines().collect();
        let total_lines = lines.len();
        let selected: String = lines
            .into_iter()
            .skip(from)
            .take(max_lines)
            .collect::<Vec<_>>()
            .join("\n");

        let truncated = selected.len() > MAX_OUTPUT_BYTES;
        let output = if truncated {
            selected[..MAX_OUTPUT_BYTES].to_string()
        } else {
            selected
        };

        Ok(serde_json::json!({
            "path": path_str,
            "content": output,
            "total_lines": total_lines,
            "from": from,
            "truncated": truncated,
        }))
    }
}

fn validate_memory_path(memory_dir: &Path, requested: &str) -> Result<()> {
    if requested.is_empty() {
        return Err(invalid_request_err("memory path must not be empty"));
    }

    if requested.starts_with('/') || requested.starts_with('\\') {
        return Err(invalid_request_err("memory path must be relative"));
    }

    for component in Path::new(requested).components() {
        if component == std::path::Component::ParentDir {
            return Err(invalid_request_err(
                "memory path must not contain '..' components",
            ));
        }
    }

    // If the resolved path exists, verify it's inside memory_dir.
    let resolved = memory_dir.join(requested);
    if resolved.exists() {
        let canonical = resolved
            .canonicalize()
            .map_err(|e| agent_runtime_err(format!("failed to resolve memory path: {e}")))?;
        let dir_canonical = memory_dir
            .canonicalize()
            .map_err(|e| agent_runtime_err(format!("failed to resolve memory directory: {e}")))?;
        if !canonical.starts_with(&dir_canonical) {
            return Err(invalid_request_err(
                "memory path escapes the memory directory",
            ));
        }
    }

    Ok(())
}

// --------------------------------------------------------------------------
// memory.search
// --------------------------------------------------------------------------

pub struct MemorySearchTool;

#[async_trait]
impl Tool for MemorySearchTool {
    fn definition(&self) -> ToolDef {
        ToolDef {
            name: "memory_search".into(),
            description: "Search the agent's memory using semantic/keyword hybrid search. \
                Returns the most relevant memory chunks ranked by relevance score."
                .into(),
            parameters: serde_json::json!({
                "type": "object",
                "required": ["query"],
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query text."
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 20,
                        "description": "Maximum number of results. Default: 5."
                    }
                }
            }),
            risk_level: ToolRiskLevel::ReadOnly,
        }
    }

    async fn invoke(&self, args: serde_json::Value, ctx: ToolContext) -> Result<serde_json::Value> {
        let memory = ctx.memory_search.as_ref().ok_or_else(|| {
            agent_runtime_err("memory.search is not available: no memory service configured")
        })?;

        let query = args
            .get("query")
            .and_then(|v| v.as_str())
            .map(str::trim)
            .filter(|v| !v.is_empty())
            .ok_or_else(|| {
                invalid_request_err("memory.search requires a non-empty 'query' string")
            })?;

        let limit = args
            .get("limit")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(5)
            .clamp(1, 20) as usize;

        let results = memory.search(query, limit).await?;

        Ok(serde_json::json!({
            "query": query,
            "count": results.len(),
            "results": results,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn validates_memory_paths() {
        let memory_dir = PathBuf::from("/tmp/frankclaw-memory-test");
        let _ = std::fs::create_dir_all(&memory_dir);

        // Absolute path rejected.
        validate_memory_path(&memory_dir, "/etc/passwd")
            .expect_err("absolute memory paths should be rejected");

        // Parent traversal rejected.
        validate_memory_path(&memory_dir, "../secrets.txt")
            .expect_err("parent traversal should be rejected");

        // Empty path rejected.
        validate_memory_path(&memory_dir, "").expect_err("empty path should be rejected");

        // Normal relative path accepted.
        validate_memory_path(&memory_dir, "notes.md")
            .expect("normal relative memory paths should be accepted");
    }

    #[test]
    fn memory_get_definition_is_valid() {
        let tool = MemoryGetTool;
        let def = tool.definition();
        assert_eq!(def.name, "memory_get");
        assert_eq!(def.risk_level, ToolRiskLevel::ReadOnly);
    }

    #[test]
    fn memory_search_definition_is_valid() {
        let tool = MemorySearchTool;
        let def = tool.definition();
        assert_eq!(def.name, "memory_search");
        assert_eq!(def.risk_level, ToolRiskLevel::ReadOnly);
    }
}
