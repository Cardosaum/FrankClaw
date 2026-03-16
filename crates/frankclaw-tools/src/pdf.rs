//! PDF text extraction tool.

use std::fmt::Write as _;

use async_trait::async_trait;

use frankclaw_core::error::{AgentRuntime, FrankClawError, Internal, InvalidRequest, Result};
use frankclaw_core::model::{ToolDef, ToolRiskLevel};

use crate::file::validate_workspace_path;
use crate::{Tool, ToolContext};

/// Maximum PDF file size (10 MB).
const MAX_PDF_BYTES: u64 = 10 * 1024 * 1024;

/// Maximum pages to extract (0 = all).
const DEFAULT_MAX_PAGES: usize = 20;

/// Maximum output characters.
const MAX_OUTPUT_CHARS: usize = 200_000;

fn agent_runtime_err(msg: impl Into<String>) -> FrankClawError {
    AgentRuntime { msg: msg.into() }.build()
}

fn internal_err(msg: impl Into<String>) -> FrankClawError {
    Internal { msg: msg.into() }.build()
}

fn invalid_request_err(msg: impl Into<String>) -> FrankClawError {
    InvalidRequest { msg: msg.into() }.build()
}

// --------------------------------------------------------------------------
// pdf.read
// --------------------------------------------------------------------------

pub struct PdfReadTool;

#[async_trait]
impl Tool for PdfReadTool {
    fn definition(&self) -> ToolDef {
        ToolDef {
            name: "pdf_read".into(),
            description: "Extract text content from a PDF file in the workspace. \
                Returns the extracted text with page markers."
                .into(),
            parameters: serde_json::json!({
                "type": "object",
                "required": ["path"],
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path to a PDF file within the workspace."
                    },
                    "pages": {
                        "type": "string",
                        "description": "Page range to extract (e.g. '1-5', '1,3,7'). Default: first 20 pages."
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
            .map(str::trim)
            .filter(|v| !v.is_empty())
            .ok_or_else(|| invalid_request_err("pdf.read requires a path"))?;

        let resolved = validate_workspace_path(workspace, path_str)?;

        // Verify the file exists and is a PDF.
        let metadata = tokio::fs::metadata(&resolved)
            .await
            .map_err(|e| agent_runtime_err(format!("failed to read '{path_str}': {e}")))?;

        if metadata.len() > MAX_PDF_BYTES {
            return Err(invalid_request_err(format!(
                "PDF file exceeds {} MB limit",
                MAX_PDF_BYTES / (1024 * 1024)
            )));
        }

        // Parse page ranges.
        let page_filter = args
            .get("pages")
            .and_then(|v| v.as_str())
            .map(parse_page_ranges)
            .transpose()?;

        let max_pages = page_filter.as_ref().map_or(DEFAULT_MAX_PAGES, Vec::len);

        // Read the PDF file.
        let pdf_bytes = tokio::fs::read(&resolved)
            .await
            .map_err(|e| agent_runtime_err(format!("failed to read PDF '{path_str}': {e}")))?;

        // Extract text (blocking operation, offload to thread pool).
        let text = tokio::task::spawn_blocking(move || {
            pdf_extract::extract_text_from_mem(&pdf_bytes)
                .map_err(|e| agent_runtime_err(format!("failed to extract text from PDF: {e}")))
        })
        .await
        .map_err(|e| internal_err(format!("PDF extraction task failed: {e}")))??;

        // Split into pages (pdf-extract separates pages with form-feed chars).
        let pages: Vec<&str> = text.split('\u{0C}').collect();
        let total_pages = pages.len();

        // Apply page filter or default limit.
        let selected_pages: Vec<(usize, &str)> = if let Some(ref indices) = page_filter {
            indices
                .iter()
                .filter(|&&i| i <= total_pages && i > 0)
                .map(|&i| (i, pages[i - 1]))
                .collect()
        } else {
            pages
                .iter()
                .enumerate()
                .take(max_pages)
                .map(|(i, p)| (i + 1, *p))
                .collect()
        };

        // Build output with page markers.
        let mut output = String::new();
        for (page_num, page_text) in &selected_pages {
            let trimmed = page_text.trim();
            if !trimmed.is_empty() {
                let _ = writeln!(output, "--- Page {page_num} ---");
                let _ = writeln!(output, "{trimmed}\n");
            }
        }

        let truncated = output.len() > MAX_OUTPUT_CHARS;
        if truncated {
            output.truncate(MAX_OUTPUT_CHARS);
            output.push_str("\n... [truncated]");
        }

        Ok(serde_json::json!({
            "path": path_str,
            "total_pages": total_pages,
            "pages_extracted": selected_pages.len(),
            "text": output,
            "truncated": truncated,
        }))
    }
}

/// Parse page range strings like "1-5", "1,3,7-9" into a sorted list of page numbers.
fn parse_page_ranges(input: &str) -> Result<Vec<usize>> {
    let mut pages = Vec::new();
    for part in input.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        if let Some((start, end)) = part.split_once('-') {
            let start: usize = start
                .trim()
                .parse()
                .map_err(|_| invalid_request_err(format!("invalid page range: '{input}'")))?;
            let end: usize = end
                .trim()
                .parse()
                .map_err(|_| invalid_request_err(format!("invalid page range: '{input}'")))?;
            if start == 0 || end == 0 || start > end || end > 10000 {
                return Err(invalid_request_err(format!("invalid page range: '{part}'")));
            }
            pages.extend(start..=end);
        } else {
            let page: usize = part
                .parse()
                .map_err(|_| invalid_request_err(format!("invalid page number: '{part}'")))?;
            if page == 0 || page > 10000 {
                return Err(invalid_request_err(format!(
                    "invalid page number: '{part}'"
                )));
            }
            pages.push(page);
        }
    }
    pages.sort_unstable();
    pages.dedup();
    Ok(pages)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_single_pages() {
        let pages = parse_page_ranges("1,3,7").expect("single pages should parse");
        assert_eq!(pages, vec![1, 3, 7]);
    }

    #[test]
    fn parse_page_range() {
        let pages = parse_page_ranges("1-5").expect("simple range should parse");
        assert_eq!(pages, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn parse_mixed_ranges() {
        let pages = parse_page_ranges("1-3,7,10-12").expect("mixed ranges should parse");
        assert_eq!(pages, vec![1, 2, 3, 7, 10, 11, 12]);
    }

    #[test]
    fn parse_deduplicates() {
        let pages = parse_page_ranges("1-3,2-4").expect("overlapping ranges should parse");
        assert_eq!(pages, vec![1, 2, 3, 4]);
    }

    #[test]
    fn parse_rejects_zero_page() {
        parse_page_ranges("0").expect_err("page zero should be rejected");
    }

    #[test]
    fn parse_rejects_invalid_range() {
        parse_page_ranges("5-3").expect_err("descending range should be rejected");
    }

    #[test]
    fn pdf_read_definition_is_valid() {
        let tool = PdfReadTool;
        let def = tool.definition();
        assert_eq!(def.name, "pdf_read");
        assert_eq!(def.risk_level, ToolRiskLevel::ReadOnly);
    }
}
