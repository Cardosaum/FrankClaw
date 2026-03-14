//! Media understanding pipeline: vision description and audio transcription.
//!
//! Processes media attachments to extract textual understanding that can be
//! injected into model context. Supports image description via vision-capable
//! models and audio transcription via the OpenAI Whisper API.

use frankclaw_core::error::{FrankClawError, Result};
use frankclaw_core::media::{classify_extension, classify_mime, MediaKind};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tracing::{debug, warn};

/// Maximum image size for vision analysis (5 MB).
const MAX_IMAGE_BYTES: usize = 5 * 1024 * 1024;

/// Maximum audio size for transcription (25 MB — OpenAI Whisper limit).
const MAX_AUDIO_BYTES: usize = 25 * 1024 * 1024;

/// A media attachment to be processed for understanding.
#[derive(Debug, Clone)]
pub struct MediaAttachment {
    /// Raw file content.
    pub data: Vec<u8>,
    /// MIME type (e.g. "image/jpeg", "audio/mp3").
    pub mime: String,
    /// Original filename if available.
    pub filename: Option<String>,
    /// Attachment index in the original message (for ordering output).
    pub index: usize,
}

impl MediaAttachment {
    /// Determine the media kind from MIME type, falling back to filename extension.
    pub fn kind(&self) -> MediaKind {
        let from_mime = classify_mime(&self.mime);
        if from_mime != MediaKind::Unknown {
            return from_mime;
        }
        if let Some(ref name) = self.filename {
            classify_extension(name)
        } else {
            MediaKind::Unknown
        }
    }
}

/// Output from media understanding.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnderstandingOutput {
    /// What kind of understanding was produced.
    pub kind: UnderstandingKind,
    /// The attachment index this output corresponds to.
    pub attachment_index: usize,
    /// Extracted text (description or transcription).
    pub text: String,
}

/// The type of understanding output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UnderstandingKind {
    ImageDescription,
    AudioTranscription,
}

/// Trait for media understanding providers.
///
/// Each provider handles one capability (vision or audio transcription).
#[async_trait]
pub trait UnderstandingProvider: Send + Sync {
    /// Provider identifier.
    fn id(&self) -> &str;

    /// What media kind this provider handles.
    fn handles(&self) -> MediaKind;

    /// Process a media attachment and return textual understanding.
    async fn process(&self, attachment: &MediaAttachment) -> Result<UnderstandingOutput>;
}

/// Process a batch of media attachments through available providers.
///
/// Returns understanding outputs for each attachment that has a matching provider.
/// Attachments without a matching provider are silently skipped.
/// Errors on individual attachments are logged but don't fail the batch.
pub async fn process_attachments(
    attachments: &[MediaAttachment],
    providers: &[Box<dyn UnderstandingProvider>],
) -> Vec<UnderstandingOutput> {
    let mut outputs = Vec::new();

    for attachment in attachments {
        let kind = attachment.kind();

        // Find a provider for this media kind.
        let Some(provider) = providers.iter().find(|p| p.handles() == kind) else {
            debug!(
                kind = ?kind,
                index = attachment.index,
                "no understanding provider for attachment kind, skipping"
            );
            continue;
        };

        // Validate size limits.
        let max_bytes = match kind {
            MediaKind::Image => MAX_IMAGE_BYTES,
            MediaKind::Audio => MAX_AUDIO_BYTES,
            _ => continue,
        };

        if attachment.data.len() > max_bytes {
            warn!(
                index = attachment.index,
                size = attachment.data.len(),
                max = max_bytes,
                "attachment exceeds size limit, skipping"
            );
            continue;
        }

        match provider.process(attachment).await {
            Ok(output) => outputs.push(output),
            Err(e) => {
                warn!(
                    index = attachment.index,
                    provider = provider.id(),
                    error = %e,
                    "media understanding failed for attachment"
                );
            }
        }
    }

    // Sort by attachment index to preserve message ordering.
    outputs.sort_by_key(|o| o.attachment_index);
    outputs
}

/// Format understanding outputs as context text for injection into model messages.
pub fn format_as_context(outputs: &[UnderstandingOutput]) -> String {
    if outputs.is_empty() {
        return String::new();
    }

    let mut parts = Vec::with_capacity(outputs.len());
    for output in outputs {
        let label = match output.kind {
            UnderstandingKind::ImageDescription => "Image description",
            UnderstandingKind::AudioTranscription => "Audio transcription",
        };
        parts.push(format!("[{label}]: {}", output.text));
    }
    parts.join("\n\n")
}

// ── Vision provider ─────────────────────────────────────────────────────

/// Vision provider that describes images using a vision-capable model API.
///
/// Sends the image as base64 to the configured API endpoint and gets back
/// a textual description. Compatible with OpenAI vision and Anthropic vision APIs.
pub struct VisionProvider {
    client: reqwest::Client,
    api_base: String,
    api_key: secrecy::SecretString,
    model: String,
}

impl VisionProvider {
    /// Create a new vision provider.
    ///
    /// - `api_base`: Base URL (e.g. "https://api.openai.com/v1")
    /// - `api_key`: API key for authentication
    /// - `model`: Model ID with vision support (e.g. "gpt-4o")
    pub fn new(
        api_base: impl Into<String>,
        api_key: secrecy::SecretString,
        model: impl Into<String>,
    ) -> Self {
        Self {
            client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(60))
                .build()
                .expect("invariant: HTTP client build should not fail"),
            api_base: api_base.into(),
            api_key,
            model: model.into(),
        }
    }

    /// Build the OpenAI-compatible vision request body.
    fn build_request(&self, attachment: &MediaAttachment) -> serde_json::Value {
        use base64::Engine;
        let b64 = base64::engine::general_purpose::STANDARD.encode(&attachment.data);
        let data_url = format!("data:{};base64,{}", attachment.mime, b64);

        serde_json::json!({
            "model": self.model,
            "max_tokens": 512,
            "messages": [{
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Describe this image concisely. Focus on the key content, text visible, and any important details."
                    },
                    {
                        "type": "image_url",
                        "image_url": { "url": data_url }
                    }
                ]
            }]
        })
    }
}

#[async_trait]
impl UnderstandingProvider for VisionProvider {
    fn id(&self) -> &str {
        "vision"
    }

    fn handles(&self) -> MediaKind {
        MediaKind::Image
    }

    async fn process(&self, attachment: &MediaAttachment) -> Result<UnderstandingOutput> {
        use secrecy::ExposeSecret;

        let body = self.build_request(attachment);
        let url = format!("{}/chat/completions", self.api_base);

        let response = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key.expose_secret()))
            .json(&body)
            .send()
            .await
            .map_err(|e| FrankClawError::ModelProvider {
                msg: format!("request failed: {e}"),
            })?;

        if !response.status().is_success() {
            let status = response.status();
            let body_text = response.text().await.unwrap_or_default();
            return Err(FrankClawError::ModelProvider {
                msg: format!("HTTP {status}: {body_text}"),
            });
        }

        let json: serde_json::Value =
            response
                .json()
                .await
                .map_err(|e| FrankClawError::ModelProvider {
                    msg: format!("vision: invalid JSON response: {e}"),
                })?;

        let text = json["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or("(no description)")
            .to_string();

        Ok(UnderstandingOutput {
            kind: UnderstandingKind::ImageDescription,
            attachment_index: attachment.index,
            text,
        })
    }
}

// ── Whisper provider ────────────────────────────────────────────────────

/// Audio transcription provider using the OpenAI Whisper API.
///
/// Sends audio files to the `/v1/audio/transcriptions` endpoint.
/// Compatible with OpenAI and any API that follows the same interface.
pub struct WhisperProvider {
    client: reqwest::Client,
    api_base: String,
    api_key: secrecy::SecretString,
    model: String,
}

impl WhisperProvider {
    /// Create a new Whisper provider.
    ///
    /// - `api_base`: Base URL (e.g. "https://api.openai.com/v1")
    /// - `api_key`: API key for authentication
    /// - `model`: Model ID (e.g. "whisper-1")
    pub fn new(
        api_base: impl Into<String>,
        api_key: secrecy::SecretString,
        model: impl Into<String>,
    ) -> Self {
        Self {
            client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(120))
                .build()
                .expect("invariant: HTTP client build should not fail"),
            api_base: api_base.into(),
            api_key,
            model: model.into(),
        }
    }
}

#[async_trait]
impl UnderstandingProvider for WhisperProvider {
    fn id(&self) -> &str {
        "whisper"
    }

    fn handles(&self) -> MediaKind {
        MediaKind::Audio
    }

    async fn process(&self, attachment: &MediaAttachment) -> Result<UnderstandingOutput> {
        use secrecy::ExposeSecret;

        let filename = attachment
            .filename
            .clone()
            .unwrap_or_else(|| format!("audio.{}", frankclaw_core::media::safe_extension_for_mime(&attachment.mime)));

        let file_part = reqwest::multipart::Part::bytes(attachment.data.clone())
            .file_name(filename)
            .mime_str(&attachment.mime)
            .map_err(|e| FrankClawError::ModelProvider {
                msg: format!("invalid MIME type: {e}"),
            })?;

        let form = reqwest::multipart::Form::new()
            .text("model", self.model.clone())
            .text("response_format", "json")
            .part("file", file_part);

        let url = format!("{}/audio/transcriptions", self.api_base);

        let response = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key.expose_secret()))
            .multipart(form)
            .send()
            .await
            .map_err(|e| FrankClawError::ModelProvider {
                msg: format!("request failed: {e}"),
            })?;

        if !response.status().is_success() {
            let status = response.status();
            let body_text = response.text().await.unwrap_or_default();
            return Err(FrankClawError::ModelProvider {
                msg: format!("HTTP {status}: {body_text}"),
            });
        }

        let json: serde_json::Value =
            response
                .json()
                .await
                .map_err(|e| FrankClawError::ModelProvider {
                    msg: format!("whisper: invalid JSON response: {e}"),
                })?;

        let text = json["text"]
            .as_str()
            .unwrap_or("")
            .to_string();

        if text.is_empty() {
            return Err(FrankClawError::ModelProvider {
                msg: "transcription returned empty text".into(),
            });
        }

        Ok(UnderstandingOutput {
            kind: UnderstandingKind::AudioTranscription,
            attachment_index: attachment.index,
            text,
        })
    }
}

// ── Anthropic vision provider ───────────────────────────────────────────

/// Vision provider that describes images using the Anthropic Messages API.
///
/// Uses the `messages` endpoint with base64 image content blocks,
/// which differs from the OpenAI-compatible vision format.
pub struct AnthropicVisionProvider {
    client: reqwest::Client,
    api_base: String,
    api_key: secrecy::SecretString,
    model: String,
}

impl AnthropicVisionProvider {
    /// Create a new Anthropic vision provider.
    ///
    /// - `api_base`: Base URL (e.g. "https://api.anthropic.com")
    /// - `api_key`: API key for authentication
    /// - `model`: Model ID with vision support (e.g. "claude-sonnet-4-20250514")
    pub fn new(
        api_base: impl Into<String>,
        api_key: secrecy::SecretString,
        model: impl Into<String>,
    ) -> Self {
        Self {
            client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(60))
                .build()
                .expect("invariant: HTTP client build should not fail"),
            api_base: api_base.into(),
            api_key,
            model: model.into(),
        }
    }
}

#[async_trait]
impl UnderstandingProvider for AnthropicVisionProvider {
    fn id(&self) -> &str {
        "anthropic-vision"
    }

    fn handles(&self) -> MediaKind {
        MediaKind::Image
    }

    async fn process(&self, attachment: &MediaAttachment) -> Result<UnderstandingOutput> {
        use base64::Engine;
        use secrecy::ExposeSecret;

        let b64 = base64::engine::general_purpose::STANDARD.encode(&attachment.data);

        // Anthropic Messages API format with image content blocks.
        let body = serde_json::json!({
            "model": self.model,
            "max_tokens": 512,
            "messages": [{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": attachment.mime,
                            "data": b64
                        }
                    },
                    {
                        "type": "text",
                        "text": "Describe this image concisely. Focus on the key content, text visible, and any important details."
                    }
                ]
            }]
        });

        let url = format!("{}/v1/messages", self.api_base);

        let response = self
            .client
            .post(&url)
            .header("x-api-key", self.api_key.expose_secret())
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| FrankClawError::ModelProvider {
                msg: format!("anthropic vision request failed: {e}"),
            })?;

        if !response.status().is_success() {
            let status = response.status();
            let body_text = response.text().await.unwrap_or_default();
            return Err(FrankClawError::ModelProvider {
                msg: format!("anthropic vision HTTP {status}: {body_text}"),
            });
        }

        let json: serde_json::Value =
            response
                .json()
                .await
                .map_err(|e| FrankClawError::ModelProvider {
                    msg: format!("anthropic vision: invalid JSON response: {e}"),
                })?;

        // Anthropic response: content[0].text
        let text = json["content"][0]["text"]
            .as_str()
            .unwrap_or("(no description)")
            .to_string();

        Ok(UnderstandingOutput {
            kind: UnderstandingKind::ImageDescription,
            attachment_index: attachment.index,
            text,
        })
    }
}

// ── Ollama vision provider ──────────────────────────────────────────────

/// Vision provider for local Ollama multimodal models (e.g., llava, bakllava).
///
/// Uses the Ollama `/api/generate` endpoint with inline base64 images.
pub struct OllamaVisionProvider {
    client: reqwest::Client,
    api_base: String,
    model: String,
}

impl OllamaVisionProvider {
    /// Create a new Ollama vision provider.
    ///
    /// - `api_base`: Base URL (e.g. "http://localhost:11434")
    /// - `model`: Model name (e.g. "llava")
    pub fn new(api_base: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(120))
                .build()
                .expect("invariant: HTTP client build should not fail"),
            api_base: api_base.into(),
            model: model.into(),
        }
    }
}

#[async_trait]
impl UnderstandingProvider for OllamaVisionProvider {
    fn id(&self) -> &str {
        "ollama-vision"
    }

    fn handles(&self) -> MediaKind {
        MediaKind::Image
    }

    async fn process(&self, attachment: &MediaAttachment) -> Result<UnderstandingOutput> {
        use base64::Engine;

        let b64 = base64::engine::general_purpose::STANDARD.encode(&attachment.data);

        let body = serde_json::json!({
            "model": self.model,
            "prompt": "Describe this image concisely. Focus on the key content, text visible, and any important details.",
            "images": [b64],
            "stream": false
        });

        let url = format!("{}/api/generate", self.api_base);

        let response = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(|e| FrankClawError::ModelProvider {
                msg: format!("ollama vision request failed: {e}"),
            })?;

        if !response.status().is_success() {
            let status = response.status();
            let body_text = response.text().await.unwrap_or_default();
            return Err(FrankClawError::ModelProvider {
                msg: format!("ollama vision HTTP {status}: {body_text}"),
            });
        }

        let json: serde_json::Value =
            response
                .json()
                .await
                .map_err(|e| FrankClawError::ModelProvider {
                    msg: format!("ollama vision: invalid JSON response: {e}"),
                })?;

        let text = json["response"]
            .as_str()
            .unwrap_or("(no description)")
            .trim()
            .to_string();

        if text.is_empty() {
            return Err(FrankClawError::ModelProvider {
                msg: "ollama vision returned empty response".into(),
            });
        }

        Ok(UnderstandingOutput {
            kind: UnderstandingKind::ImageDescription,
            attachment_index: attachment.index,
            text,
        })
    }
}

// ── Understanding chain ─────────────────────────────────────────────────

/// Ordered fallback chain of understanding providers.
///
/// Tries each provider for the matching media kind in order. If a provider
/// fails, the next one is tried. Only returns an error if all providers fail.
pub struct UnderstandingChain {
    providers: Vec<Box<dyn UnderstandingProvider>>,
}

impl UnderstandingChain {
    pub fn new(providers: Vec<Box<dyn UnderstandingProvider>>) -> Self {
        Self { providers }
    }

    /// Process a single attachment through the chain.
    /// Returns the first successful result, or the last error if all fail.
    pub async fn process(&self, attachment: &MediaAttachment) -> Result<UnderstandingOutput> {
        let kind = attachment.kind();
        let matching: Vec<_> = self.providers.iter().filter(|p| p.handles() == kind).collect();

        if matching.is_empty() {
            return Err(FrankClawError::ModelProvider {
                msg: format!("no understanding provider for {:?}", kind),
            });
        }

        let mut last_error = None;
        for provider in &matching {
            match provider.process(attachment).await {
                Ok(output) => return Ok(output),
                Err(e) => {
                    warn!(
                        provider = provider.id(),
                        error = %e,
                        "understanding provider failed, trying next"
                    );
                    last_error = Some(e);
                }
            }
        }

        Err(last_error.unwrap_or_else(|| FrankClawError::ModelProvider {
            msg: "all understanding providers failed".into(),
        }))
    }

    /// Build providers from config, returning an empty vec if disabled.
    pub fn from_config(config: &frankclaw_core::config::MediaUnderstandingConfig) -> Vec<Box<dyn UnderstandingProvider>> {
        if !config.enabled {
            return Vec::new();
        }

        let mut providers: Vec<Box<dyn UnderstandingProvider>> = Vec::new();

        // Vision provider.
        match config.vision_provider.as_str() {
            "openai" => {
                if let Some(api_key) = resolve_api_key(config.vision_api_key_ref.as_deref()) {
                    let base_url = config
                        .vision_base_url
                        .as_deref()
                        .unwrap_or("https://api.openai.com/v1");
                    let model = config
                        .vision_model
                        .as_deref()
                        .unwrap_or("gpt-4o");
                    providers.push(Box::new(VisionProvider::new(base_url, api_key, model)));
                }
            }
            "anthropic" => {
                if let Some(api_key) = resolve_api_key(config.vision_api_key_ref.as_deref()) {
                    let base_url = config
                        .vision_base_url
                        .as_deref()
                        .unwrap_or("https://api.anthropic.com");
                    let model = config
                        .vision_model
                        .as_deref()
                        .unwrap_or("claude-sonnet-4-20250514");
                    providers.push(Box::new(AnthropicVisionProvider::new(base_url, api_key, model)));
                }
            }
            "ollama" => {
                let base_url = config
                    .vision_base_url
                    .as_deref()
                    .unwrap_or("http://localhost:11434");
                let model = config
                    .vision_model
                    .as_deref()
                    .unwrap_or("llava");
                providers.push(Box::new(OllamaVisionProvider::new(base_url, model)));
            }
            _ => {}
        }

        // Transcription provider.
        match config.transcription_provider.as_str() {
            "openai" => {
                if let Some(api_key) = resolve_api_key(config.transcription_api_key_ref.as_deref()) {
                    let base_url = config
                        .transcription_base_url
                        .as_deref()
                        .unwrap_or("https://api.openai.com/v1");
                    let model = config
                        .transcription_model
                        .as_deref()
                        .unwrap_or("whisper-1");
                    providers.push(Box::new(WhisperProvider::new(base_url, api_key, model)));
                }
            }
            _ => {}
        }

        providers
    }
}

/// Resolve an API key from an environment variable reference.
fn resolve_api_key(env_ref: Option<&str>) -> Option<secrecy::SecretString> {
    let var_name = env_ref?;
    std::env::var(var_name)
        .ok()
        .filter(|v| !v.trim().is_empty())
        .map(secrecy::SecretString::from)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn image_attachment(index: usize) -> MediaAttachment {
        MediaAttachment {
            data: vec![0xFF, 0xD8, 0xFF], // JPEG magic bytes
            mime: "image/jpeg".into(),
            filename: Some("photo.jpg".into()),
            index,
        }
    }

    fn audio_attachment(index: usize) -> MediaAttachment {
        MediaAttachment {
            data: vec![0x49, 0x44, 0x33], // ID3 tag
            mime: "audio/mpeg".into(),
            filename: Some("voice.mp3".into()),
            index,
        }
    }

    #[test]
    fn attachment_kind_from_mime() {
        let img = image_attachment(0);
        assert_eq!(img.kind(), MediaKind::Image);

        let aud = audio_attachment(0);
        assert_eq!(aud.kind(), MediaKind::Audio);
    }

    #[test]
    fn attachment_kind_falls_back_to_extension() {
        let att = MediaAttachment {
            data: vec![],
            mime: "application/octet-stream".into(),
            filename: Some("recording.wav".into()),
            index: 0,
        };
        assert_eq!(att.kind(), MediaKind::Audio);
    }

    #[test]
    fn attachment_kind_unknown_without_hints() {
        let att = MediaAttachment {
            data: vec![],
            mime: "application/octet-stream".into(),
            filename: None,
            index: 0,
        };
        assert_eq!(att.kind(), MediaKind::Unknown);
    }

    #[test]
    fn format_context_empty() {
        assert_eq!(format_as_context(&[]), "");
    }

    #[test]
    fn format_context_single_image() {
        let outputs = vec![UnderstandingOutput {
            kind: UnderstandingKind::ImageDescription,
            attachment_index: 0,
            text: "A cat sitting on a keyboard".into(),
        }];
        let ctx = format_as_context(&outputs);
        assert_eq!(ctx, "[Image description]: A cat sitting on a keyboard");
    }

    #[test]
    fn format_context_mixed() {
        let outputs = vec![
            UnderstandingOutput {
                kind: UnderstandingKind::AudioTranscription,
                attachment_index: 0,
                text: "Hello world".into(),
            },
            UnderstandingOutput {
                kind: UnderstandingKind::ImageDescription,
                attachment_index: 1,
                text: "A diagram of architecture".into(),
            },
        ];
        let ctx = format_as_context(&outputs);
        assert!(ctx.contains("[Audio transcription]: Hello world"));
        assert!(ctx.contains("[Image description]: A diagram of architecture"));
    }

    #[tokio::test]
    async fn process_attachments_skips_unknown_kinds() {
        let attachments = vec![MediaAttachment {
            data: vec![0x00],
            mime: "application/octet-stream".into(),
            filename: Some("mystery.bin".into()),
            index: 0,
        }];
        let providers: Vec<Box<dyn UnderstandingProvider>> = vec![];
        let results = process_attachments(&attachments, &providers).await;
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn process_attachments_skips_oversized() {
        // Create a "provider" that would succeed but attachment is too large.
        struct FakeVision;
        #[async_trait]
        impl UnderstandingProvider for FakeVision {
            fn id(&self) -> &str { "fake-vision" }
            fn handles(&self) -> MediaKind { MediaKind::Image }
            async fn process(&self, att: &MediaAttachment) -> Result<UnderstandingOutput> {
                Ok(UnderstandingOutput {
                    kind: UnderstandingKind::ImageDescription,
                    attachment_index: att.index,
                    text: "should not reach here".into(),
                })
            }
        }

        let oversized = MediaAttachment {
            data: vec![0xFF; MAX_IMAGE_BYTES + 1],
            mime: "image/png".into(),
            filename: Some("huge.png".into()),
            index: 0,
        };

        let providers: Vec<Box<dyn UnderstandingProvider>> = vec![Box::new(FakeVision)];
        let results = process_attachments(&[oversized], &providers).await;
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn process_attachments_collects_successes() {
        struct FakeVision;
        #[async_trait]
        impl UnderstandingProvider for FakeVision {
            fn id(&self) -> &str { "fake-vision" }
            fn handles(&self) -> MediaKind { MediaKind::Image }
            async fn process(&self, att: &MediaAttachment) -> Result<UnderstandingOutput> {
                Ok(UnderstandingOutput {
                    kind: UnderstandingKind::ImageDescription,
                    attachment_index: att.index,
                    text: format!("described image {}", att.index),
                })
            }
        }

        let attachments = vec![image_attachment(0), image_attachment(1)];
        let providers: Vec<Box<dyn UnderstandingProvider>> = vec![Box::new(FakeVision)];
        let results = process_attachments(&attachments, &providers).await;
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].text, "described image 0");
        assert_eq!(results[1].text, "described image 1");
    }

    #[tokio::test]
    async fn process_attachments_handles_provider_errors_gracefully() {
        struct FailingProvider;
        #[async_trait]
        impl UnderstandingProvider for FailingProvider {
            fn id(&self) -> &str { "failing" }
            fn handles(&self) -> MediaKind { MediaKind::Image }
            async fn process(&self, _att: &MediaAttachment) -> Result<UnderstandingOutput> {
                Err(FrankClawError::ModelProvider {
                    msg: "intentional test failure".into(),
                })
            }
        }

        let attachments = vec![image_attachment(0)];
        let providers: Vec<Box<dyn UnderstandingProvider>> = vec![Box::new(FailingProvider)];
        let results = process_attachments(&attachments, &providers).await;
        assert!(results.is_empty()); // Error logged, not propagated
    }

    #[tokio::test]
    async fn chain_returns_first_success() {
        struct FailingVision;
        #[async_trait]
        impl UnderstandingProvider for FailingVision {
            fn id(&self) -> &str { "failing-vision" }
            fn handles(&self) -> MediaKind { MediaKind::Image }
            async fn process(&self, _att: &MediaAttachment) -> Result<UnderstandingOutput> {
                Err(FrankClawError::ModelProvider {
                    msg: "provider 1 failed".into(),
                })
            }
        }

        struct SucceedingVision;
        #[async_trait]
        impl UnderstandingProvider for SucceedingVision {
            fn id(&self) -> &str { "succeeding-vision" }
            fn handles(&self) -> MediaKind { MediaKind::Image }
            async fn process(&self, att: &MediaAttachment) -> Result<UnderstandingOutput> {
                Ok(UnderstandingOutput {
                    kind: UnderstandingKind::ImageDescription,
                    attachment_index: att.index,
                    text: "fallback description".into(),
                })
            }
        }

        let chain = UnderstandingChain::new(vec![
            Box::new(FailingVision),
            Box::new(SucceedingVision),
        ]);

        let result = chain.process(&image_attachment(0)).await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap().text, "fallback description");
    }

    #[tokio::test]
    async fn chain_errors_when_all_fail() {
        struct FailingVision;
        #[async_trait]
        impl UnderstandingProvider for FailingVision {
            fn id(&self) -> &str { "failing" }
            fn handles(&self) -> MediaKind { MediaKind::Image }
            async fn process(&self, _att: &MediaAttachment) -> Result<UnderstandingOutput> {
                Err(FrankClawError::ModelProvider {
                    msg: "all fail".into(),
                })
            }
        }

        let chain = UnderstandingChain::new(vec![Box::new(FailingVision)]);
        let result = chain.process(&image_attachment(0)).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn chain_errors_when_no_provider_for_kind() {
        let chain = UnderstandingChain::new(vec![]);
        let result = chain.process(&image_attachment(0)).await;
        assert!(result.is_err());
    }

    #[test]
    fn from_config_disabled_returns_empty() {
        let config = frankclaw_core::config::MediaUnderstandingConfig::default();
        let providers = UnderstandingChain::from_config(&config);
        assert!(providers.is_empty());
    }

    #[test]
    fn from_config_ollama_needs_no_api_key() {
        let config = frankclaw_core::config::MediaUnderstandingConfig {
            enabled: true,
            vision_provider: "ollama".into(),
            vision_model: Some("llava".into()),
            ..Default::default()
        };
        let providers = UnderstandingChain::from_config(&config);
        assert_eq!(providers.len(), 1);
        assert_eq!(providers[0].id(), "ollama-vision");
    }
}
