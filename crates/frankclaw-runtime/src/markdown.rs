//! Markdown IR and ANSI terminal rendering.
//!
//! Parses CommonMark into a flat text representation with style annotations,
//! then renders to ANSI escape sequences for terminal display.

use pulldown_cmark::{Event, Options, Parser, Tag, TagEnd};

/// Inline style types that can be applied to text spans.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MarkdownStyle {
    Bold,
    Italic,
    Strikethrough,
    Code,
    CodeBlock,
    Blockquote,
}

/// A styled span over byte offsets in the flat text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StyleSpan {
    pub start: usize,
    pub end: usize,
    pub style: MarkdownStyle,
}

/// A link annotation over byte offsets in the flat text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LinkSpan {
    pub start: usize,
    pub end: usize,
    pub href: String,
}

/// Intermediate representation of parsed markdown.
///
/// Contains flat text (no markdown syntax) plus style and link annotations
/// as byte-offset spans. Can be rendered to ANSI, plain text, or TUI styles.
#[derive(Debug, Clone)]
pub struct MarkdownIR {
    pub text: String,
    pub styles: Vec<StyleSpan>,
    pub links: Vec<LinkSpan>,
}

impl MarkdownIR {
    /// Return the plain text content (no formatting).
    pub fn plain(&self) -> &str {
        &self.text
    }
}

/// Parse a markdown string into a `MarkdownIR`.
pub fn parse_markdown(input: &str) -> MarkdownIR {
    let mut opts = Options::empty();
    opts.insert(Options::ENABLE_STRIKETHROUGH);
    opts.insert(Options::ENABLE_TABLES);

    let parser = Parser::new_ext(input, opts);

    let mut text = String::new();
    let mut styles: Vec<StyleSpan> = Vec::new();
    let mut links: Vec<LinkSpan> = Vec::new();

    // Track open spans as (start_byte_offset, style).
    let mut style_stack: Vec<(usize, MarkdownStyle)> = Vec::new();
    // Track open link as (start_byte_offset, href).
    let mut link_stack: Vec<(usize, String)> = Vec::new();

    let mut in_code_block = false;
    let mut list_depth: u32 = 0;
    let mut at_block_start = true;

    for event in parser {
        match event {
            Event::Start(tag) => match tag {
                Tag::Emphasis => {
                    style_stack.push((text.len(), MarkdownStyle::Italic));
                }
                Tag::Strong => {
                    style_stack.push((text.len(), MarkdownStyle::Bold));
                }
                Tag::Strikethrough => {
                    style_stack.push((text.len(), MarkdownStyle::Strikethrough));
                }
                Tag::CodeBlock(_) => {
                    if !at_block_start {
                        text.push('\n');
                    }
                    in_code_block = true;
                    style_stack.push((text.len(), MarkdownStyle::CodeBlock));
                }
                Tag::BlockQuote(_) => {
                    style_stack.push((text.len(), MarkdownStyle::Blockquote));
                }
                Tag::Link { dest_url, .. } => {
                    link_stack.push((text.len(), dest_url.to_string()));
                }
                Tag::List(_) => {
                    list_depth += 1;
                }
                Tag::Item => {
                    if !at_block_start {
                        text.push('\n');
                    }
                    for _ in 1..list_depth {
                        text.push_str("  ");
                    }
                    text.push_str("- ");
                    at_block_start = false;
                }
                Tag::Paragraph => {
                    if !at_block_start {
                        text.push_str("\n\n");
                    }
                }
                Tag::Heading { .. } => {
                    if !at_block_start {
                        text.push_str("\n\n");
                    }
                    style_stack.push((text.len(), MarkdownStyle::Bold));
                }
                _ => {}
            },
            Event::End(tag_end) => match tag_end {
                TagEnd::Emphasis => {
                    if let Some((start, style)) = pop_style(&mut style_stack, MarkdownStyle::Italic)
                    {
                        if start < text.len() {
                            styles.push(StyleSpan {
                                start,
                                end: text.len(),
                                style,
                            });
                        }
                    }
                }
                TagEnd::Strong => {
                    if let Some((start, style)) = pop_style(&mut style_stack, MarkdownStyle::Bold) {
                        if start < text.len() {
                            styles.push(StyleSpan {
                                start,
                                end: text.len(),
                                style,
                            });
                        }
                    }
                }
                TagEnd::Strikethrough => {
                    if let Some((start, style)) =
                        pop_style(&mut style_stack, MarkdownStyle::Strikethrough)
                    {
                        if start < text.len() {
                            styles.push(StyleSpan {
                                start,
                                end: text.len(),
                                style,
                            });
                        }
                    }
                }
                TagEnd::CodeBlock => {
                    in_code_block = false;
                    if let Some((start, style)) =
                        pop_style(&mut style_stack, MarkdownStyle::CodeBlock)
                    {
                        // Trim trailing newline from code block content.
                        let end = if text.ends_with('\n') {
                            text.len() - 1
                        } else {
                            text.len()
                        };
                        if start < end {
                            styles.push(StyleSpan { start, end, style });
                        }
                    }
                }
                TagEnd::BlockQuote(_) => {
                    if let Some((start, style)) =
                        pop_style(&mut style_stack, MarkdownStyle::Blockquote)
                    {
                        if start < text.len() {
                            styles.push(StyleSpan {
                                start,
                                end: text.len(),
                                style,
                            });
                        }
                    }
                }
                TagEnd::Link => {
                    if let Some((start, href)) = link_stack.pop() {
                        if start < text.len() {
                            links.push(LinkSpan {
                                start,
                                end: text.len(),
                                href,
                            });
                        }
                    }
                }
                TagEnd::List(_) => {
                    list_depth = list_depth.saturating_sub(1);
                }
                TagEnd::Item => {}
                TagEnd::Paragraph => {
                    at_block_start = false;
                }
                TagEnd::Heading(_) => {
                    if let Some((start, style)) = pop_style(&mut style_stack, MarkdownStyle::Bold) {
                        if start < text.len() {
                            styles.push(StyleSpan {
                                start,
                                end: text.len(),
                                style,
                            });
                        }
                    }
                    at_block_start = false;
                }
                _ => {}
            },
            Event::Text(s) => {
                text.push_str(&s);
                at_block_start = false;
            }
            Event::Code(s) => {
                let start = text.len();
                text.push_str(&s);
                if start < text.len() {
                    styles.push(StyleSpan {
                        start,
                        end: text.len(),
                        style: MarkdownStyle::Code,
                    });
                }
                at_block_start = false;
            }
            Event::SoftBreak | Event::HardBreak => {
                if in_code_block {
                    text.push('\n');
                } else {
                    text.push('\n');
                }
            }
            Event::Rule => {
                if !at_block_start {
                    text.push('\n');
                }
                text.push_str("---");
                at_block_start = false;
            }
            _ => {}
        }
    }

    MarkdownIR {
        text,
        styles,
        links,
    }
}

/// Pop a specific style from the stack (not necessarily the top, for robustness).
fn pop_style(
    stack: &mut Vec<(usize, MarkdownStyle)>,
    target: MarkdownStyle,
) -> Option<(usize, MarkdownStyle)> {
    if let Some(pos) = stack.iter().rposition(|(_, s)| *s == target) {
        Some(stack.remove(pos))
    } else {
        None
    }
}

/// ANSI SGR escape codes.
const BOLD_ON: &str = "\x1b[1m";
const ITALIC_ON: &str = "\x1b[3m";
const STRIKETHROUGH_ON: &str = "\x1b[9m";
const CODE_ON: &str = "\x1b[36m"; // cyan
const CODE_BLOCK_ON: &str = "\x1b[36m"; // cyan
const BLOCKQUOTE_ON: &str = "\x1b[2m"; // dim
const RESET: &str = "\x1b[0m";

/// Render a `MarkdownIR` to an ANSI-escaped string.
pub fn render_ansi(ir: &MarkdownIR) -> String {
    if ir.styles.is_empty() {
        return ir.text.clone();
    }

    // Collect boundary events: (byte_offset, is_close, sgr_code).
    let mut events: Vec<(usize, bool, &str)> = Vec::new();
    for span in &ir.styles {
        let on = match span.style {
            MarkdownStyle::Bold => BOLD_ON,
            MarkdownStyle::Italic => ITALIC_ON,
            MarkdownStyle::Strikethrough => STRIKETHROUGH_ON,
            MarkdownStyle::Code => CODE_ON,
            MarkdownStyle::CodeBlock => CODE_BLOCK_ON,
            MarkdownStyle::Blockquote => BLOCKQUOTE_ON,
        };
        events.push((span.start, false, on));
        events.push((span.end, true, RESET));
    }
    // Sort: by position, then closes before opens at same position.
    events.sort_by(|a, b| a.0.cmp(&b.0).then(b.1.cmp(&a.1)));

    let text_bytes = ir.text.as_bytes();
    let mut result = String::with_capacity(ir.text.len() + events.len() * 8);
    let mut pos = 0;

    for (offset, _is_close, code) in &events {
        if *offset > pos {
            // Safety: offsets come from string positions built during parsing.
            result.push_str(&ir.text[pos..*offset]);
        }
        result.push_str(code);
        if pos < *offset {
            pos = *offset;
        }
    }

    // Remaining text after last event.
    if pos < text_bytes.len() {
        result.push_str(&ir.text[pos..]);
    }

    result
}

/// Convenience: parse markdown and render directly to ANSI.
pub fn render_markdown_to_ansi(input: &str) -> String {
    let ir = parse_markdown(input);
    render_ansi(&ir)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_bold() {
        let ir = parse_markdown("**hello**");
        assert_eq!(ir.text, "hello");
        assert_eq!(ir.styles.len(), 1);
        assert_eq!(ir.styles[0].style, MarkdownStyle::Bold);
        assert_eq!(ir.styles[0].start, 0);
        assert_eq!(ir.styles[0].end, 5);
    }

    #[test]
    fn parse_italic() {
        let ir = parse_markdown("*world*");
        assert_eq!(ir.text, "world");
        assert_eq!(ir.styles.len(), 1);
        assert_eq!(ir.styles[0].style, MarkdownStyle::Italic);
    }

    #[test]
    fn parse_inline_code() {
        let ir = parse_markdown("use `println!`");
        assert_eq!(ir.text, "use println!");
        assert_eq!(ir.styles.len(), 1);
        assert_eq!(ir.styles[0].style, MarkdownStyle::Code);
        assert_eq!(&ir.text[ir.styles[0].start..ir.styles[0].end], "println!");
    }

    #[test]
    fn parse_code_block() {
        let ir = parse_markdown("```\nfn main() {}\n```");
        assert!(ir.text.contains("fn main() {}"));
        assert!(
            ir.styles
                .iter()
                .any(|s| s.style == MarkdownStyle::CodeBlock)
        );
    }

    #[test]
    fn parse_nested_bold_italic() {
        let ir = parse_markdown("***bold and italic***");
        assert_eq!(ir.text, "bold and italic");
        let has_bold = ir.styles.iter().any(|s| s.style == MarkdownStyle::Bold);
        let has_italic = ir.styles.iter().any(|s| s.style == MarkdownStyle::Italic);
        assert!(has_bold || has_italic, "should have at least one style");
    }

    #[test]
    fn parse_link() {
        let ir = parse_markdown("[click here](https://example.com)");
        assert_eq!(ir.text, "click here");
        assert_eq!(ir.links.len(), 1);
        assert_eq!(ir.links[0].href, "https://example.com");
        assert_eq!(ir.links[0].start, 0);
        assert_eq!(ir.links[0].end, 10);
    }

    #[test]
    fn parse_strikethrough() {
        let ir = parse_markdown("~~removed~~");
        assert_eq!(ir.text, "removed");
        assert_eq!(ir.styles.len(), 1);
        assert_eq!(ir.styles[0].style, MarkdownStyle::Strikethrough);
    }

    #[test]
    fn render_bold_ansi() {
        let output = render_markdown_to_ansi("**bold**");
        assert!(output.contains("\x1b[1m"));
        assert!(output.contains("\x1b[0m"));
        assert!(output.contains("bold"));
    }

    #[test]
    fn render_italic_ansi() {
        let output = render_markdown_to_ansi("*italic*");
        assert!(output.contains("\x1b[3m"));
    }

    #[test]
    fn render_code_ansi() {
        let output = render_markdown_to_ansi("`code`");
        assert!(output.contains("\x1b[36m"));
    }

    #[test]
    fn empty_input() {
        let ir = parse_markdown("");
        assert_eq!(ir.text, "");
        assert!(ir.styles.is_empty());
        assert!(ir.links.is_empty());
    }

    #[test]
    fn plain_text_no_styles() {
        let ir = parse_markdown("just plain text");
        assert_eq!(ir.text, "just plain text");
        assert!(ir.styles.is_empty());
    }

    #[test]
    fn plain_accessor() {
        let ir = parse_markdown("**hello** world");
        assert_eq!(ir.plain(), "hello world");
    }

    #[test]
    fn heading_parsed_as_bold() {
        let ir = parse_markdown("# Title");
        assert_eq!(ir.text, "Title");
        assert!(ir.styles.iter().any(|s| s.style == MarkdownStyle::Bold));
    }

    #[test]
    fn blockquote() {
        let ir = parse_markdown("> quoted text");
        assert!(ir.text.contains("quoted text"));
        assert!(
            ir.styles
                .iter()
                .any(|s| s.style == MarkdownStyle::Blockquote)
        );
    }

    #[test]
    fn list_items() {
        let ir = parse_markdown("- one\n- two\n- three");
        assert!(ir.text.contains("- one"));
        assert!(ir.text.contains("- two"));
        assert!(ir.text.contains("- three"));
    }

    #[test]
    fn ansi_render_preserves_plain_text() {
        let output = render_markdown_to_ansi("no formatting here");
        assert_eq!(output, "no formatting here");
    }
}
