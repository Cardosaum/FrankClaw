//! File system tools: read, write, and edit files within a workspace directory.

use std::path::{Path, PathBuf};

use async_trait::async_trait;

use frankclaw_core::error::{AgentRuntime, FrankClawError, InvalidRequest, Result};
use frankclaw_core::model::{ToolDef, ToolRiskLevel};

use crate::{Tool, ToolContext};

/// Maximum file read output (200 KB).
const MAX_READ_BYTES: usize = 200 * 1024;

/// Maximum write content (1 MB).
const MAX_WRITE_BYTES: usize = 1024 * 1024;

fn agent_runtime_err(msg: impl Into<String>) -> FrankClawError {
    AgentRuntime { msg: msg.into() }.build()
}

fn invalid_request_err(msg: impl Into<String>) -> FrankClawError {
    InvalidRequest { msg: msg.into() }.build()
}

/// Validate and resolve a requested path within the workspace.
///
/// - Rejects absolute paths
/// - Rejects `..` components
/// - Resolves to an absolute path inside the workspace
/// - Rejects symlinks that escape the workspace
pub(crate) fn validate_workspace_path(workspace: &Path, requested: &str) -> Result<PathBuf> {
    let requested = requested.trim();
    if requested.is_empty() {
        return Err(invalid_request_err("file path must not be empty"));
    }

    // Reject absolute paths.
    if requested.starts_with('/') || requested.starts_with('\\') {
        return Err(invalid_request_err(
            "file path must be relative to the workspace directory",
        ));
    }

    // Reject .. traversal.
    for component in Path::new(requested).components() {
        if component == std::path::Component::ParentDir {
            return Err(invalid_request_err(
                "file path must not contain '..' components",
            ));
        }
    }

    let resolved = workspace.join(requested);

    // If the path exists, canonicalize and verify it's inside workspace.
    if resolved.exists() {
        let canonical = resolved
            .canonicalize()
            .map_err(|e| agent_runtime_err(format!("failed to resolve path: {e}")))?;
        let workspace_canonical = workspace
            .canonicalize()
            .map_err(|e| agent_runtime_err(format!("failed to resolve workspace: {e}")))?;
        if !canonical.starts_with(&workspace_canonical) {
            return Err(invalid_request_err(
                "file path escapes the workspace directory (symlink?)",
            ));
        }
        Ok(canonical)
    } else {
        // For new files, verify the parent exists and is inside workspace.
        if let Some(parent) = resolved.parent().filter(|p| p.exists()) {
            let parent_canonical = parent.canonicalize().map_err(|e| {
                agent_runtime_err(format!("failed to resolve parent directory: {e}"))
            })?;
            let workspace_canonical = workspace
                .canonicalize()
                .map_err(|e| agent_runtime_err(format!("failed to resolve workspace: {e}")))?;
            if !parent_canonical.starts_with(&workspace_canonical) {
                return Err(invalid_request_err(
                    "file path escapes the workspace directory",
                ));
            }
        }
        Ok(resolved)
    }
}

// --------------------------------------------------------------------------
// file.read
// --------------------------------------------------------------------------

pub struct FileReadTool;

#[async_trait]
impl Tool for FileReadTool {
    fn definition(&self) -> ToolDef {
        ToolDef {
            name: "file_read".into(),
            description: "Read a file from the workspace directory. \
                Returns the file content with line numbers."
                .into(),
            parameters: serde_json::json!({
                "type": "object",
                "required": ["path"],
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path within the workspace."
                    },
                    "offset": {
                        "type": "integer",
                        "minimum": 0,
                        "description": "Starting line number (0-based). Default: 0."
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 10000,
                        "description": "Maximum lines to return. Default: 2000."
                    }
                }
            }),
            risk_level: ToolRiskLevel::ReadOnly,
        }
    }

    async fn invoke(&self, args: serde_json::Value, ctx: ToolContext) -> Result<serde_json::Value> {
        let workspace = ctx.require_workspace()?;
        let path_str = args
            .get("path")
            .and_then(|v| v.as_str())
            .ok_or_else(|| invalid_request_err("file.read requires a path"))?;
        let resolved = validate_workspace_path(workspace, path_str)?;
        let offset = args
            .get("offset")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0) as usize;
        let limit = args
            .get("limit")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(2000)
            .clamp(1, 10000) as usize;

        let content = tokio::fs::read_to_string(&resolved)
            .await
            .map_err(|e| agent_runtime_err(format!("failed to read file '{path_str}': {e}")))?;

        let lines: Vec<&str> = content.lines().collect();
        let total_lines = lines.len();
        let selected: Vec<String> = lines
            .into_iter()
            .skip(offset)
            .take(limit)
            .enumerate()
            .map(|(i, line)| format!("{:>6}\t{}", offset + i + 1, line))
            .collect();

        let output = selected.join("\n");
        let truncated = output.len() > MAX_READ_BYTES;
        let final_output = if truncated {
            output[..MAX_READ_BYTES].to_string()
        } else {
            output
        };

        Ok(serde_json::json!({
            "path": path_str,
            "content": final_output,
            "total_lines": total_lines,
            "offset": offset,
            "lines_returned": selected.len().min(if truncated { limit } else { selected.len() }),
            "truncated": truncated,
        }))
    }
}

// --------------------------------------------------------------------------
// file.write
// --------------------------------------------------------------------------

pub struct FileWriteTool;

#[async_trait]
impl Tool for FileWriteTool {
    fn definition(&self) -> ToolDef {
        ToolDef {
            name: "file_write".into(),
            description: "Create or overwrite a file in the workspace directory.".into(),
            parameters: serde_json::json!({
                "type": "object",
                "required": ["path", "content"],
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path within the workspace."
                    },
                    "content": {
                        "type": "string",
                        "description": "File content to write (max 1MB)."
                    }
                }
            }),
            risk_level: ToolRiskLevel::Mutating,
        }
    }

    async fn invoke(&self, args: serde_json::Value, ctx: ToolContext) -> Result<serde_json::Value> {
        let workspace = ctx.require_workspace()?;
        let path_str = args
            .get("path")
            .and_then(|v| v.as_str())
            .ok_or_else(|| invalid_request_err("file.write requires a path"))?;
        let content = args
            .get("content")
            .and_then(|v| v.as_str())
            .ok_or_else(|| invalid_request_err("file.write requires content"))?;

        if content.len() > MAX_WRITE_BYTES {
            return Err(invalid_request_err(format!(
                "file.write content exceeds {MAX_WRITE_BYTES} byte limit"
            )));
        }

        let resolved = validate_workspace_path(workspace, path_str)?;

        // Create parent directories.
        if let Some(parent) = resolved.parent() {
            tokio::fs::create_dir_all(parent).await.map_err(|e| {
                agent_runtime_err(format!(
                    "failed to create directories for '{path_str}': {e}"
                ))
            })?;
        }

        tokio::fs::write(&resolved, content)
            .await
            .map_err(|e| agent_runtime_err(format!("failed to write file '{path_str}': {e}")))?;

        // Set file permissions to owner-only.
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let _ =
                tokio::fs::set_permissions(&resolved, std::fs::Permissions::from_mode(0o600)).await;
        }

        Ok(serde_json::json!({
            "path": path_str,
            "bytes_written": content.len(),
            "status": "ok",
        }))
    }
}

// --------------------------------------------------------------------------
// file.edit
// --------------------------------------------------------------------------

pub struct FileEditTool;

#[async_trait]
impl Tool for FileEditTool {
    fn definition(&self) -> ToolDef {
        ToolDef {
            name: "file_edit".into(),
            description: "Search and replace text in a file. \
                The old_text must match exactly once in the file."
                .into(),
            parameters: serde_json::json!({
                "type": "object",
                "required": ["path", "old_text", "new_text"],
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path within the workspace."
                    },
                    "old_text": {
                        "type": "string",
                        "description": "Exact text to find and replace. Must match exactly once."
                    },
                    "new_text": {
                        "type": "string",
                        "description": "Replacement text."
                    }
                }
            }),
            risk_level: ToolRiskLevel::Mutating,
        }
    }

    async fn invoke(&self, args: serde_json::Value, ctx: ToolContext) -> Result<serde_json::Value> {
        let workspace = ctx.require_workspace()?;
        let path_str = args
            .get("path")
            .and_then(|v| v.as_str())
            .ok_or_else(|| invalid_request_err("file.edit requires a path"))?;
        let old_text = args
            .get("old_text")
            .and_then(|v| v.as_str())
            .ok_or_else(|| invalid_request_err("file.edit requires old_text"))?;
        let new_text = args
            .get("new_text")
            .and_then(|v| v.as_str())
            .ok_or_else(|| invalid_request_err("file.edit requires new_text"))?;

        if old_text.is_empty() {
            return Err(invalid_request_err("file.edit old_text must not be empty"));
        }

        let resolved = validate_workspace_path(workspace, path_str)?;
        let content = tokio::fs::read_to_string(&resolved)
            .await
            .map_err(|e| agent_runtime_err(format!("failed to read file '{path_str}': {e}")))?;

        let match_count = content.matches(old_text).count();
        if match_count == 0 {
            return Err(agent_runtime_err(format!(
                "file.edit: old_text not found in '{path_str}'"
            )));
        }
        if match_count > 1 {
            return Err(agent_runtime_err(format!(
                "file.edit: old_text matches {match_count} times in '{path_str}' (must match exactly once)"
            )));
        }

        let new_content = content.replacen(old_text, new_text, 1);

        if new_content.len() > MAX_WRITE_BYTES {
            return Err(invalid_request_err(format!(
                "file.edit result would exceed {MAX_WRITE_BYTES} byte limit"
            )));
        }

        tokio::fs::write(&resolved, &new_content)
            .await
            .map_err(|e| agent_runtime_err(format!("failed to write file '{path_str}': {e}")))?;

        Ok(serde_json::json!({
            "path": path_str,
            "status": "ok",
            "bytes_written": new_content.len(),
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn validates_relative_paths() {
        let workspace = PathBuf::from("/tmp/test-workspace");
        let _ = std::fs::create_dir_all(&workspace);

        // Absolute paths rejected.
        validate_workspace_path(&workspace, "/etc/passwd")
            .expect_err("absolute paths should be rejected");

        // Parent traversal rejected.
        validate_workspace_path(&workspace, "../etc/passwd")
            .expect_err("parent traversal should be rejected");
        validate_workspace_path(&workspace, "foo/../../etc/passwd")
            .expect_err("nested parent traversal should be rejected");

        // Empty path rejected.
        validate_workspace_path(&workspace, "").expect_err("empty path should be rejected");
    }

    #[test]
    fn validates_normal_relative_path() {
        let workspace = std::env::temp_dir().join("frankclaw-file-test");
        let _ = std::fs::create_dir_all(&workspace);

        // Normal relative path should work.
        let result = validate_workspace_path(&workspace, "hello.txt");
        result.expect("normal relative path should be valid");
    }

    #[tokio::test]
    async fn file_edit_rejects_ambiguous_match() {
        let workspace = std::env::temp_dir().join("frankclaw-edit-test");
        let _ = std::fs::create_dir_all(&workspace);
        let test_file = workspace.join("test.txt");
        let _ = std::fs::write(&test_file, "aaa bbb aaa");

        let tool = FileEditTool;
        let ctx = crate::test_tool_context(Some(workspace.clone()));

        let result = tool
            .invoke(
                serde_json::json!({
                    "path": "test.txt",
                    "old_text": "aaa",
                    "new_text": "ccc"
                }),
                ctx,
            )
            .await;

        let err = result
            .expect_err("ambiguous replacement should be rejected")
            .to_string();
        assert!(err.contains("matches 2 times"));

        // Cleanup.
        let _ = std::fs::remove_dir_all(&workspace);
    }

    #[tokio::test]
    async fn file_read_write_roundtrip() {
        let workspace = std::env::temp_dir().join("frankclaw-rw-test");
        let _ = std::fs::create_dir_all(&workspace);

        let ctx = crate::test_tool_context(Some(workspace.clone()));

        // Write.
        let write_tool = FileWriteTool;
        let result = write_tool
            .invoke(
                serde_json::json!({
                    "path": "hello.txt",
                    "content": "line1\nline2\nline3"
                }),
                ctx.clone(),
            )
            .await
            .expect("write should succeed");
        assert_eq!(result["status"], "ok");

        // Read.
        let read_tool = FileReadTool;
        let result = read_tool
            .invoke(serde_json::json!({ "path": "hello.txt" }), ctx)
            .await
            .expect("read should succeed");
        assert_eq!(result["total_lines"], 3);
        let content = result["content"]
            .as_str()
            .expect("file.read should return string content");
        assert!(content.contains("line1"));
        assert!(content.contains("line3"));

        // Cleanup.
        let _ = std::fs::remove_dir_all(&workspace);
    }
}
