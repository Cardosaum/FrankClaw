use async_trait::async_trait;
use futures_util::StreamExt;
use reqwest::Client;
use secrecy::{ExposeSecret, SecretString};
use std::collections::BTreeMap;
use tracing::debug;

use frankclaw_core::error::{FrankClawError, Result};
use frankclaw_core::model::*;

use crate::sse::SseDecoder;

/// OpenAI-compatible completions provider.
///
/// Works with OpenAI, Azure OpenAI, OpenRouter, and any OpenAI-compatible API.
pub struct OpenAiProvider {
    id: String,
    client: Client,
    base_url: String,
    api_key: SecretString,
    models: Vec<String>,
}

impl OpenAiProvider {
    pub fn new(
        id: impl Into<String>,
        base_url: impl Into<String>,
        api_key: SecretString,
        models: Vec<String>,
    ) -> Self {
        let client = Client::builder()
            .timeout(std::time::Duration::from_secs(120))
            .build()
            .expect("failed to build HTTP client");

        Self {
            id: id.into(),
            client,
            base_url: base_url.into(),
            api_key,
            models,
        }
    }
}

#[async_trait]
impl ModelProvider for OpenAiProvider {
    fn id(&self) -> &str {
        &self.id
    }

    async fn complete(
        &self,
        request: CompletionRequest,
        stream_tx: Option<tokio::sync::mpsc::Sender<StreamDelta>>,
    ) -> Result<CompletionResponse> {
        let mut body = build_request_body(&request);
        if stream_tx.is_some() {
            body["stream"] = serde_json::json!(true);
            body["stream_options"] = serde_json::json!({
                "include_usage": true,
            });
        }

        let url = format!("{}/chat/completions", self.base_url.trim_end_matches('/'));
        debug!(url, model = %request.model_id, "sending completion request");

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
            let body = response.text().await.unwrap_or_default();
            return Err(FrankClawError::ModelProvider {
                msg: format!("HTTP {status}: {body}"),
            });
        }

        if let Some(stream_tx) = stream_tx {
            let mut decoder = SseDecoder::default();
            let mut state = OpenAiStreamState::default();
            let mut stream = response.bytes_stream();
            while let Some(chunk) = stream.next().await {
                let chunk = chunk.map_err(|e| FrankClawError::ModelProvider {
                    msg: format!("failed to read streaming response: {e}"),
                })?;
                for event in decoder.push(chunk.as_ref()) {
                    for delta in apply_stream_event(&mut state, &event.data)? {
                        let _ = stream_tx.send(delta).await;
                    }
                    if state.done {
                        break;
                    }
                }
                if state.done {
                    break;
                }
            }
            if !state.done {
                if let Some(event) = decoder.finish() {
                    for delta in apply_stream_event(&mut state, &event.data)? {
                        let _ = stream_tx.send(delta).await;
                    }
                }
            }
            let response = state.finish()?;
            let _ = stream_tx.send(StreamDelta::Done {
                usage: Some(response.usage.clone()),
            }).await;
            return Ok(response);
        }

        let data: serde_json::Value = response.json().await.map_err(|e| FrankClawError::ModelProvider {
            msg: format!("invalid response: {e}"),
        })?;
        parse_completion_response(&data)
    }

    async fn list_models(&self) -> Result<Vec<ModelDef>> {
        // Return configured models (not fetching from API for now).
        Ok(self
            .models
            .iter()
            .map(|id| ModelDef {
                id: id.clone(),
                name: id.clone(),
                api: ModelApi::OpenaiCompletions,
                reasoning: false,
                input: vec![InputModality::Text],
                cost: ModelCost::default(),
                context_window: 128_000,
                max_output_tokens: 4096,
                compat: ModelCompat {
                    supports_tools: true,
                    supports_streaming: true,
                    supports_system_message: true,
                    ..Default::default()
                },
            })
            .collect())
    }

    async fn health(&self) -> bool {
        let url = format!("{}/models", self.base_url.trim_end_matches('/'));
        self.client
            .get(&url)
            .header("Authorization", format!("Bearer {}", self.api_key.expose_secret()))
            .send()
            .await
            .map(|r| r.status().is_success())
            .unwrap_or(false)
    }
}

fn build_request_body(request: &CompletionRequest) -> serde_json::Value {
    let messages: Vec<serde_json::Value> = {
        let mut msgs = Vec::new();
        if let Some(system) = &request.system {
            msgs.push(serde_json::json!({
                "role": "system",
                "content": system,
            }));
        }
        for msg in &request.messages {
            msgs.push(serde_json::json!({
                "role": msg.role,
                "content": msg.content,
            }));
        }
        msgs
    };

    let mut body = serde_json::json!({
        "model": request.model_id,
        "messages": messages,
    });

    if let Some(max_tokens) = request.max_tokens {
        body["max_tokens"] = serde_json::json!(max_tokens);
    }
    if let Some(temp) = request.temperature {
        body["temperature"] = serde_json::json!(temp);
    }
    if !request.tools.is_empty() {
        let tools: Vec<serde_json::Value> = request
            .tools
            .iter()
            .map(|t| {
                serde_json::json!({
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                    }
                })
            })
            .collect();
        body["tools"] = serde_json::json!(tools);
    }

    body
}

fn parse_completion_response(data: &serde_json::Value) -> Result<CompletionResponse> {
    let choice = data["choices"]
        .get(0)
        .ok_or_else(|| FrankClawError::ModelProvider {
            msg: "no choices in response".into(),
        })?;

    let content = choice["message"]["content"]
        .as_str()
        .unwrap_or("")
        .to_string();
    let usage = parse_usage(data);
    let tool_calls = choice["message"]["tool_calls"]
        .as_array()
        .map(|arr| {
            arr.iter()
                .filter_map(|tc| {
                    Some(ToolCallResponse {
                        id: tc["id"].as_str()?.to_string(),
                        name: tc["function"]["name"].as_str()?.to_string(),
                        arguments: tc["function"]["arguments"]
                            .as_str()
                            .unwrap_or("{}")
                            .to_string(),
                    })
                })
                .collect()
        })
        .unwrap_or_default();

    Ok(CompletionResponse {
        content,
        tool_calls,
        usage,
        finish_reason: parse_finish_reason(choice["finish_reason"].as_str()),
    })
}

#[derive(Debug)]
struct OpenAiStreamState {
    content: String,
    tool_calls: BTreeMap<usize, StreamingToolCall>,
    usage: Usage,
    finish_reason: FinishReason,
    done: bool,
}

impl Default for OpenAiStreamState {
    fn default() -> Self {
        Self {
            content: String::new(),
            tool_calls: BTreeMap::new(),
            usage: Usage::default(),
            finish_reason: FinishReason::Stop,
            done: false,
        }
    }
}

#[derive(Debug, Default)]
struct StreamingToolCall {
    id: String,
    name: String,
    arguments: String,
    started: bool,
    ended: bool,
}

impl OpenAiStreamState {
    fn finish(mut self) -> Result<CompletionResponse> {
        let mut tool_calls = Vec::with_capacity(self.tool_calls.len());
        for tool_call in self.tool_calls.values_mut() {
            if !tool_call.ended && tool_call.started {
                tool_call.ended = true;
            }
            if tool_call.id.trim().is_empty() || tool_call.name.trim().is_empty() {
                return Err(FrankClawError::ModelProvider {
                    msg: "streamed tool call missing id or name".into(),
                });
            }
            tool_calls.push(ToolCallResponse {
                id: tool_call.id.clone(),
                name: tool_call.name.clone(),
                arguments: tool_call.arguments.clone(),
            });
        }
        Ok(CompletionResponse {
            content: self.content,
            tool_calls,
            usage: self.usage,
            finish_reason: self.finish_reason,
        })
    }
}

fn apply_stream_event(
    state: &mut OpenAiStreamState,
    data: &str,
) -> Result<Vec<StreamDelta>> {
    if data.trim() == "[DONE]" {
        state.done = true;
        return Ok(Vec::new());
    }

    let payload: serde_json::Value = serde_json::from_str(data).map_err(|err| FrankClawError::ModelProvider {
        msg: format!("invalid streaming response chunk: {err}"),
    })?;
    let mut deltas = Vec::new();

    if payload["choices"].as_array().map(|choices| choices.is_empty()).unwrap_or(false) {
        state.usage = parse_usage(&payload);
        return Ok(deltas);
    }

    for choice in payload["choices"].as_array().into_iter().flatten() {
        if let Some(text) = choice["delta"]["content"].as_str() {
            if !text.is_empty() {
                state.content.push_str(text);
                deltas.push(StreamDelta::Text(text.to_string()));
            }
        }

        if let Some(tool_calls) = choice["delta"]["tool_calls"].as_array() {
            for tool_call in tool_calls {
                let index = tool_call["index"].as_u64().unwrap_or(0) as usize;
                let entry = state.tool_calls.entry(index).or_default();
                if let Some(id) = tool_call["id"].as_str() {
                    entry.id = id.to_string();
                }
                if let Some(name) = tool_call["function"]["name"].as_str() {
                    entry.name = name.to_string();
                }
                if !entry.started && !entry.id.is_empty() && !entry.name.is_empty() {
                    entry.started = true;
                    deltas.push(StreamDelta::ToolCallStart {
                        id: entry.id.clone(),
                        name: entry.name.clone(),
                    });
                }
                if let Some(arguments_delta) = tool_call["function"]["arguments"].as_str() {
                    if !arguments_delta.is_empty() {
                        entry.arguments.push_str(arguments_delta);
                        if entry.started {
                            deltas.push(StreamDelta::ToolCallDelta {
                                id: entry.id.clone(),
                                arguments: arguments_delta.to_string(),
                            });
                        }
                    }
                }
            }
        }

        let finish_reason = parse_finish_reason(choice["finish_reason"].as_str());
        if choice["finish_reason"].as_str().is_some() {
            state.finish_reason = finish_reason;
            if matches!(finish_reason, FinishReason::ToolUse) {
                for tool_call in state.tool_calls.values_mut() {
                    if tool_call.started && !tool_call.ended {
                        tool_call.ended = true;
                        deltas.push(StreamDelta::ToolCallEnd {
                            id: tool_call.id.clone(),
                        });
                    }
                }
            }
        }
    }

    Ok(deltas)
}

fn parse_usage(data: &serde_json::Value) -> Usage {
    Usage {
        input_tokens: data["usage"]["prompt_tokens"].as_u64().unwrap_or(0) as u32,
        output_tokens: data["usage"]["completion_tokens"].as_u64().unwrap_or(0) as u32,
        ..Default::default()
    }
}

fn parse_finish_reason(reason: Option<&str>) -> FinishReason {
    match reason {
        Some("stop") => FinishReason::Stop,
        Some("length") => FinishReason::MaxTokens,
        Some("tool_calls") => FinishReason::ToolUse,
        Some("content_filter") => FinishReason::ContentFilter,
        _ => FinishReason::Stop,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn apply_stream_event_accumulates_text_and_usage() {
        let mut state = OpenAiStreamState::default();

        let deltas = apply_stream_event(
            &mut state,
            r#"{"choices":[{"delta":{"content":"hel"},"finish_reason":null}]}"#,
        )
        .expect("chunk should parse");
        assert_eq!(deltas, vec![StreamDelta::Text("hel".into())]);

        let deltas = apply_stream_event(
            &mut state,
            r#"{"choices":[{"delta":{"content":"lo"},"finish_reason":"stop"}],"usage":{"prompt_tokens":4,"completion_tokens":2}}"#,
        )
        .expect("chunk should parse");
        assert_eq!(deltas, vec![StreamDelta::Text("lo".into())]);
        state.usage = parse_usage(&serde_json::json!({
            "usage": { "prompt_tokens": 4, "completion_tokens": 2 }
        }));

        let response = state.finish().expect("response should build");
        assert_eq!(response.content, "hello");
        assert_eq!(response.finish_reason, FinishReason::Stop);
    }

    #[test]
    fn apply_stream_event_accumulates_tool_calls() {
        let mut state = OpenAiStreamState::default();

        let deltas = apply_stream_event(
            &mut state,
            r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"lookup","arguments":"{\"q\":\"op"}}]}}]}"#,
        )
        .expect("chunk should parse");
        assert_eq!(
            deltas,
            vec![
                StreamDelta::ToolCallStart {
                    id: "call_1".into(),
                    name: "lookup".into(),
                },
                StreamDelta::ToolCallDelta {
                    id: "call_1".into(),
                    arguments: "{\"q\":\"op".into(),
                }
            ]
        );

        let deltas = apply_stream_event(
            &mut state,
            r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"enai\"}"}}],"content":""},"finish_reason":"tool_calls"}]}"#,
        )
        .expect("chunk should parse");
        assert_eq!(
            deltas,
            vec![
                StreamDelta::ToolCallDelta {
                    id: "call_1".into(),
                    arguments: "enai\"}".into(),
                },
                StreamDelta::ToolCallEnd {
                    id: "call_1".into(),
                }
            ]
        );

        let response = state.finish().expect("response should build");
        assert_eq!(response.tool_calls.len(), 1);
        assert_eq!(response.tool_calls[0].arguments, "{\"q\":\"openai\"}");
        assert_eq!(response.finish_reason, FinishReason::ToolUse);
    }
}
