//! Terminal UI for FrankClaw chat using ratatui.
//!
//! Launched by `frankclaw chat --tui`. Provides a scrollable chat log,
//! multi-line input area, and status bar showing model/session/state.

#![forbid(unsafe_code)]

use std::io;
use std::sync::Arc;

use crossterm::ExecutableCommand;
use crossterm::event::{self, Event, KeyCode, KeyModifiers};
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Direction, Layout};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph, Wrap};
use rust_i18n::t;
use tokio::sync::mpsc;

use frankclaw_core::model::StreamDelta;
use frankclaw_core::types::{AgentId, SessionKey};
use frankclaw_runtime::{ChatRequest, Runtime};

/// How often to poll for input events (ms).
const POLL_INTERVAL_MS: u64 = 50;

/// Scroll speed for PageUp/PageDown.
const PAGE_SCROLL: u16 = 10;

/// A message in the chat log.
#[derive(Debug, Clone)]
pub enum ChatMessage {
    User(String),
    Assistant(String),
    System(String),
    ToolCall(String),
}

impl ChatMessage {
    fn style(&self) -> Style {
        match self {
            Self::User(_) => Style::default().fg(Color::Cyan),
            Self::Assistant(_) => Style::default().fg(Color::White),
            Self::System(_) => Style::default().fg(Color::Yellow),
            Self::ToolCall(_) => Style::default().fg(Color::DarkGray),
        }
    }

    fn prefix(&self) -> &str {
        match self {
            Self::User(_) => "you> ",
            Self::Assistant(_) => "assistant> ",
            Self::System(_) => "system> ",
            Self::ToolCall(_) => "tool> ",
        }
    }

    fn text(&self) -> &str {
        match self {
            Self::User(s) | Self::Assistant(s) | Self::System(s) | Self::ToolCall(s) => s,
        }
    }
}

/// TUI application state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TuiStatus {
    Idle,
    Streaming,
    Error,
}

/// Parse a slash command from input text.
///
/// Returns `Some((command, args))` if the input starts with `/`, else `None`.
pub fn parse_slash_command(input: &str) -> Option<(&str, &str)> {
    let trimmed = input.trim();
    let cmd = trimmed.strip_prefix('/')?;
    match cmd.split_once(char::is_whitespace) {
        Some((c, a)) => Some((c, a.trim())),
        None => Some((cmd, "")),
    }
}

/// Configuration for the TUI session.
pub struct TuiConfig {
    pub agent_id: Option<AgentId>,
    pub session_key: Option<SessionKey>,
    pub model_id: Option<String>,
    pub thinking_budget: Option<u32>,
}

/// Run the TUI chat interface.
pub async fn run_tui(runtime: Arc<Runtime>, config: TuiConfig) -> anyhow::Result<()> {
    std::future::ready(()).await;
    enable_raw_mode()?;
    io::stdout().execute(EnterAlternateScreen)?;

    let backend = CrosstermBackend::new(io::stdout());
    let mut terminal = Terminal::new(backend)?;

    let result = run_tui_loop(&mut terminal, &runtime, config);

    disable_raw_mode()?;
    io::stdout().execute(LeaveAlternateScreen)?;

    result
}

type TuiTerminal = Terminal<CrosstermBackend<io::Stdout>>;

fn drain_stream_results(
    messages: &mut Vec<ChatMessage>,
    delta_rx: &mut mpsc::Receiver<StreamResult>,
    session_key: &mut Option<SessionKey>,
    status: &mut TuiStatus,
) {
    while let Ok(result) = delta_rx.try_recv() {
        match result {
            StreamResult::Delta(delta) => match delta {
                StreamDelta::Text(text) => {
                    if let Some(ChatMessage::Assistant(s)) = messages.last_mut() {
                        s.push_str(&text);
                    }
                }
                StreamDelta::ToolCallStart { name, .. } => {
                    messages.push(ChatMessage::ToolCall(format!("{name}...")));
                }
                StreamDelta::Done { .. } | StreamDelta::Error(_) => {}
                StreamDelta::ToolCallDelta { .. } | StreamDelta::ToolCallEnd { .. } => {}
            },
            StreamResult::Finished { session_key: sk } => {
                *session_key = Some(sk);
                *status = TuiStatus::Idle;
            }
            StreamResult::Failed(msg) => {
                messages.push(ChatMessage::System(msg));
                *status = TuiStatus::Error;
            }
        }
    }
}

fn render_tui(
    terminal: &mut TuiTerminal,
    messages: &[ChatMessage],
    input: &str,
    scroll_offset: u16,
    session_key: Option<&SessionKey>,
    model_id: Option<&str>,
    status: TuiStatus,
) -> anyhow::Result<()> {
    terminal.draw(|f| {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Min(5),    // chat log
                Constraint::Length(3), // input
                Constraint::Length(1), // status bar
            ])
            .split(f.area());
        let (Some(chat_area), Some(input_area), Some(status_area)) =
            (chunks.first(), chunks.get(1), chunks.get(2))
        else {
            return;
        };

        let mut lines: Vec<Line> = Vec::new();
        for msg in messages {
            let prefix = Span::styled(msg.prefix(), msg.style().add_modifier(Modifier::BOLD));
            let content = Span::styled(msg.text(), msg.style());
            lines.push(Line::from(vec![prefix, content]));
        }

        let chat = Paragraph::new(lines)
            .block(Block::default().borders(Borders::ALL).title("Chat"))
            .wrap(Wrap { trim: false })
            .scroll((scroll_offset, 0));
        f.render_widget(chat, *chat_area);

        let input_display = if input.is_empty() {
            t!("tui.input_placeholder").to_string()
        } else {
            input.to_string()
        };
        let input_widget = Paragraph::new(input_display.as_str())
            .block(Block::default().borders(Borders::ALL).title("Input"))
            .style(if input.is_empty() {
                Style::default().fg(Color::DarkGray)
            } else {
                Style::default().fg(Color::White)
            });
        f.render_widget(input_widget, *input_area);

        let model_label = model_id.unwrap_or("default");
        let session_label = session_key.map_or_else(|| "new".to_string(), ToString::to_string);
        let status_label = match status {
            TuiStatus::Idle => t!("tui.status_idle").to_string(),
            TuiStatus::Streaming => t!("tui.status_streaming").to_string(),
            TuiStatus::Error => t!("tui.status_error").to_string(),
        };
        let status_line = Line::from(vec![
            Span::styled(
                format!(" {status_label} "),
                Style::default().fg(Color::Black).bg(match status {
                    TuiStatus::Idle => Color::Green,
                    TuiStatus::Streaming => Color::Yellow,
                    TuiStatus::Error => Color::Red,
                }),
            ),
            Span::raw(format!(" model: {model_label}  session: {session_label} ")),
        ]);
        let status_bar = Paragraph::new(status_line);
        f.render_widget(status_bar, *status_area);
    })?;

    Ok(())
}

fn handle_slash_command(
    cmd: &str,
    args: &str,
    messages: &mut Vec<ChatMessage>,
    session_key: &mut Option<SessionKey>,
    model_id: &mut Option<String>,
    thinking_budget: &mut Option<u32>,
) -> bool {
    match cmd {
        "quit" | "exit" => return true,
        "clear" => {
            *session_key = None;
            messages.clear();
            messages.push(ChatMessage::System(t!("repl.session_cleared").to_string()));
        }
        "help" => {
            let help_text =
                "/quit  /clear  /help  /session  /model [id]  /think [n|off]".to_string();
            messages.push(ChatMessage::System(help_text));
        }
        "session" => {
            let msg = match session_key {
                Some(key) => format!("{}: {key}", t!("repl.current_session")),
                None => t!("repl.no_session").to_string(),
            };
            messages.push(ChatMessage::System(msg));
        }
        "model" => {
            if args.is_empty() {
                let msg = match model_id {
                    Some(id) => format!("{}: {id}", t!("repl.current_model")),
                    None => t!("repl.default_model").to_string(),
                };
                messages.push(ChatMessage::System(msg));
            } else {
                *model_id = Some(args.to_string());
                messages.push(ChatMessage::System(format!(
                    "{}: {args}",
                    t!("repl.model_set")
                )));
            }
        }
        "think" => match args {
            "" => {
                let msg = match thinking_budget {
                    Some(budget) => format!("{}: {budget}", t!("repl.thinking_budget")),
                    None => t!("repl.thinking_disabled").to_string(),
                };
                messages.push(ChatMessage::System(msg));
            }
            "off" | "0" => {
                *thinking_budget = None;
                messages.push(ChatMessage::System(
                    t!("repl.thinking_disabled").to_string(),
                ));
            }
            _ => {
                if let Ok(budget) = args.parse::<u32>() {
                    *thinking_budget = Some(budget);
                    messages.push(ChatMessage::System(format!(
                        "{}: {budget}",
                        t!("repl.thinking_budget_set")
                    )));
                }
            }
        },
        _ => {
            messages.push(ChatMessage::System(format!(
                "{}: /{cmd}",
                t!("repl.unknown_command")
            )));
        }
    }

    false
}

fn spawn_chat_request(
    runtime: &Arc<Runtime>,
    agent_id: Option<AgentId>,
    session_key: Option<SessionKey>,
    message: String,
    model_id: Option<String>,
    thinking_budget: Option<u32>,
    result_tx: mpsc::Sender<StreamResult>,
) {
    let runtime = runtime.clone();
    let (stream_tx, mut stream_rx) = mpsc::channel::<StreamDelta>(64);
    let request = ChatRequest {
        agent_id,
        session_key,
        message,
        attachments: Vec::new(),
        model_id,
        max_tokens: None,
        temperature: None,
        stream_tx: Some(stream_tx),
        thinking_budget,
        channel_id: None,
        channel_capabilities: None,
        canvas: Some(frankclaw_gateway::canvas::CanvasStore::new()),
        cancel_token: None,
        approval_tx: None,
    };

    tokio::spawn(async move {
        let forward_tx = result_tx.clone();
        let forward_handle = tokio::spawn(async move {
            while let Some(delta) = stream_rx.recv().await {
                let _ = forward_tx.send(StreamResult::Delta(delta)).await;
            }
        });

        match runtime.chat(request).await {
            Ok(resp) => {
                let _ = forward_handle.await;
                let _ = result_tx
                    .send(StreamResult::Finished {
                        session_key: resp.session_key,
                    })
                    .await;
            }
            Err(e) => {
                let _ = forward_handle.await;
                let _ = result_tx.send(StreamResult::Failed(e.to_string())).await;
            }
        }
    });
}

fn run_tui_loop(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    runtime: &Arc<Runtime>,
    config: TuiConfig,
) -> anyhow::Result<()> {
    let mut messages: Vec<ChatMessage> = Vec::new();
    let mut input = String::new();
    let mut scroll_offset: u16 = 0;
    let mut session_key = config.session_key;
    let mut model_id = config.model_id;
    let mut thinking_budget = config.thinking_budget;
    let agent_id = config.agent_id;
    let mut status = TuiStatus::Idle;

    // Channel for streaming tokens from runtime.
    let (delta_tx, mut delta_rx) = mpsc::channel::<StreamResult>(64);

    messages.push(ChatMessage::System(t!("repl.welcome").to_string()));

    loop {
        drain_stream_results(&mut messages, &mut delta_rx, &mut session_key, &mut status);
        render_tui(
            terminal,
            &messages,
            &input,
            scroll_offset,
            session_key.as_ref(),
            model_id.as_deref(),
            status,
        )?;

        // Poll for input events.
        if event::poll(std::time::Duration::from_millis(POLL_INTERVAL_MS))?
            && let Event::Key(key) = event::read()?
        {
            match (key.code, key.modifiers) {
                (KeyCode::Char('c' | 'd'), KeyModifiers::CONTROL) => {
                    break;
                }
                (KeyCode::Enter, _) => {
                    let trimmed = input.trim().to_string();
                    if trimmed.is_empty() {
                        continue;
                    }

                    if let Some((cmd, args)) = parse_slash_command(&trimmed) {
                        let should_exit = handle_slash_command(
                            cmd,
                            args,
                            &mut messages,
                            &mut session_key,
                            &mut model_id,
                            &mut thinking_budget,
                        );
                        input.clear();
                        if should_exit {
                            break;
                        }
                        continue;
                    }

                    messages.push(ChatMessage::User(trimmed.clone()));
                    messages.push(ChatMessage::Assistant(String::new()));
                    status = TuiStatus::Streaming;
                    spawn_chat_request(
                        runtime,
                        agent_id.clone(),
                        session_key.clone(),
                        trimmed,
                        model_id.clone(),
                        thinking_budget,
                        delta_tx.clone(),
                    );

                    input.clear();
                    scroll_offset = 0;
                }
                (KeyCode::Char(c), _) => {
                    input.push(c);
                }
                (KeyCode::Backspace, _) => {
                    input.pop();
                }
                (KeyCode::Up, _) => {
                    scroll_offset = scroll_offset.saturating_sub(1);
                }
                (KeyCode::Down, _) => {
                    scroll_offset = scroll_offset.saturating_add(1);
                }
                (KeyCode::PageUp, _) => {
                    scroll_offset = scroll_offset.saturating_sub(PAGE_SCROLL);
                }
                (KeyCode::PageDown, _) => {
                    scroll_offset = scroll_offset.saturating_add(PAGE_SCROLL);
                }
                (KeyCode::Esc, _) => {
                    input.clear();
                }
                _ => {}
            }
        }
    }

    Ok(())
}

/// Internal result type for stream channel communication.
enum StreamResult {
    Delta(StreamDelta),
    Finished { session_key: SessionKey },
    Failed(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_slash_command_basic() {
        assert_eq!(parse_slash_command("/quit"), Some(("quit", "")));
        assert_eq!(
            parse_slash_command("/model gpt-4o"),
            Some(("model", "gpt-4o"))
        );
        assert_eq!(parse_slash_command("/think 1024"), Some(("think", "1024")));
        assert_eq!(parse_slash_command("  /help  "), Some(("help", "")));
    }

    #[test]
    fn parse_slash_command_not_slash() {
        assert_eq!(parse_slash_command("hello"), None);
        assert_eq!(parse_slash_command(""), None);
    }

    #[test]
    fn chat_message_prefix() {
        assert_eq!(ChatMessage::User("hi".into()).prefix(), "you> ");
        assert_eq!(ChatMessage::Assistant("ok".into()).prefix(), "assistant> ");
        assert_eq!(ChatMessage::System("info".into()).prefix(), "system> ");
        assert_eq!(ChatMessage::ToolCall("bash".into()).prefix(), "tool> ");
    }

    #[test]
    fn chat_message_text() {
        let msg = ChatMessage::User("hello world".into());
        assert_eq!(msg.text(), "hello world");
    }

    #[test]
    fn scroll_bounds() {
        let offset: u16 = 0;
        assert_eq!(offset.saturating_sub(1), 0);
        assert_eq!(offset.saturating_add(PAGE_SCROLL), PAGE_SCROLL);
    }

    #[test]
    fn tui_status_variants() {
        assert_ne!(TuiStatus::Idle, TuiStatus::Streaming);
        assert_ne!(TuiStatus::Streaming, TuiStatus::Error);
        assert_eq!(TuiStatus::Idle, TuiStatus::Idle);
    }
}
