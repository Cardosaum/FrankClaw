//! Audio transcription tool — transcribes audio files from the workspace
//! using the configured media understanding pipeline.

use std::path::Path;

use async_trait::async_trait;
use base64::Engine;

use frankclaw_core::error::{FrankClawError, Result};
use frankclaw_core::model::{ToolDef, ToolRiskLevel};

use crate::file::validate_workspace_path;
use crate::{Tool, ToolContext};

/// Maximum audio file size (25 MB — Whisper limit).
const MAX_AUDIO_BYTES: u64 = 25 * 1024 * 1024;

/// Supported audio extensions with their MIME types.
const SUPPORTED_AUDIO: &[(&str, &str)] = &[
    ("mp3", "audio/mpeg"),
    ("wav", "audio/wav"),
    ("ogg", "audio/ogg"),
    ("m4a", "audio/mp4"),
    ("flac", "audio/flac"),
    ("webm", "audio/webm"),
    ("aac", "audio/aac"),
    ("wma", "audio/x-ms-wma"),
];

fn audio_mime_from_extension(path: &Path) -> Result<&'static str> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase())
        .unwrap_or_default();
    SUPPORTED_AUDIO
        .iter()
        .find(|(e, _)| *e == ext)
        .map(|(_, mime)| *mime)
        .ok_or_else(|| FrankClawError::InvalidRequest {
            msg: format!(
                "unsupported audio format '.{}'. Supported: {}",
                ext,
                SUPPORTED_AUDIO
                    .iter()
                    .map(|(e, _)| format!(".{e}"))
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        })
}

pub struct AudioTranscribeTool;

#[async_trait]
impl Tool for AudioTranscribeTool {
    fn definition(&self) -> ToolDef {
        ToolDef {
            name: "audio.transcribe".into(),
            description: "Transcribe an audio file from the workspace to text. \
                Supports MP3, WAV, OGG, M4A, FLAC, WebM, AAC, and WMA formats. \
                Returns the transcription text."
                .into(),
            parameters: serde_json::json!({
                "type": "object",
                "required": ["path"],
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path to the audio file within the workspace."
                    }
                }
            }),
            risk_level: ToolRiskLevel::ReadOnly,
        }
    }

    async fn invoke(&self, args: serde_json::Value, ctx: ToolContext) -> Result<serde_json::Value> {
        let workspace = ctx.workspace.as_deref().ok_or_else(|| FrankClawError::AgentRuntime {
            msg: "audio.transcribe is not available: no workspace directory configured".into(),
        })?;

        let path_str = args
            .get("path")
            .and_then(|v| v.as_str())
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .ok_or_else(|| FrankClawError::InvalidRequest {
                msg: "audio.transcribe requires a 'path' string".into(),
            })?;

        let resolved = validate_workspace_path(workspace, path_str)?;
        let _mime = audio_mime_from_extension(&resolved)?;

        let metadata = tokio::fs::metadata(&resolved).await.map_err(|e| {
            FrankClawError::AgentRuntime {
                msg: format!("failed to read '{}': {e}", path_str),
            }
        })?;

        if metadata.len() > MAX_AUDIO_BYTES {
            return Err(FrankClawError::InvalidRequest {
                msg: format!(
                    "audio file '{}' exceeds {} MB limit",
                    path_str,
                    MAX_AUDIO_BYTES / (1024 * 1024)
                ),
            });
        }

        // Read the file — actual transcription requires the understanding pipeline
        // to be wired in. For now, return the file info so it can be processed
        // by the runtime's understanding chain.
        let bytes = tokio::fs::read(&resolved).await.map_err(|e| {
            FrankClawError::AgentRuntime {
                msg: format!("failed to read audio '{}': {e}", path_str),
            }
        })?;

        // Signal that this tool output needs audio transcription processing.
        // The runtime will pick this up and route through the understanding chain.
        Ok(serde_json::json!({
            "path": path_str,
            "size_bytes": bytes.len(),
            "status": "audio_file_loaded",
            "_audio_transcription_request": {
                "data_base64": base64::engine::general_purpose::STANDARD.encode(&bytes),
                "mime": _mime,
                "filename": resolved.file_name().and_then(|f| f.to_str()).unwrap_or("audio"),
            }
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn audio_mime_mp3() {
        let path = Path::new("voice.mp3");
        assert_eq!(audio_mime_from_extension(path).unwrap(), "audio/mpeg");
    }

    #[test]
    fn audio_mime_wav() {
        let path = Path::new("recording.WAV");
        assert_eq!(audio_mime_from_extension(path).unwrap(), "audio/wav");
    }

    #[test]
    fn audio_mime_unsupported() {
        let path = Path::new("video.mp4");
        assert!(audio_mime_from_extension(path).is_err());
    }

    #[test]
    fn audio_transcribe_definition_is_valid() {
        let tool = AudioTranscribeTool;
        let def = tool.definition();
        assert_eq!(def.name, "audio.transcribe");
        assert_eq!(def.risk_level, ToolRiskLevel::ReadOnly);
    }
}
