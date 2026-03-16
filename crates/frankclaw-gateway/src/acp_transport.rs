//! ACP transport: NDJSON (newline-delimited JSON) over stdin/stdout.

use std::io::{self, BufRead, Write};

use serde::{Deserialize, Serialize};

/// Maximum line size for NDJSON input (2 MB to match MAX_PROMPT_BYTES).
pub const MAX_LINE_BYTES: usize = 2 * 1024 * 1024;

/// JSON-RPC 2.0 request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcRequest {
    pub jsonrpc: String,
    pub id: Option<serde_json::Value>,
    pub method: String,
    #[serde(default)]
    pub params: serde_json::Value,
}

/// JSON-RPC 2.0 response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcResponse {
    pub jsonrpc: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<JsonRpcError>,
}

/// JSON-RPC 2.0 error object.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcError {
    pub code: i64,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<serde_json::Value>,
}

// Standard JSON-RPC error codes.
pub const PARSE_ERROR: i64 = -32700;
pub const INVALID_REQUEST: i64 = -32600;
pub const METHOD_NOT_FOUND: i64 = -32601;
pub const INVALID_PARAMS: i64 = -32602;
pub const INTERNAL_ERROR: i64 = -32603;
// Application-specific.
pub const RATE_LIMITED: i64 = -32000;
pub const SESSION_NOT_FOUND: i64 = -32001;

impl JsonRpcResponse {
    pub fn success(id: Option<serde_json::Value>, result: serde_json::Value) -> Self {
        Self {
            jsonrpc: "2.0".into(),
            id,
            result: Some(result),
            error: None,
        }
    }

    pub fn error(id: Option<serde_json::Value>, code: i64, message: impl Into<String>) -> Self {
        Self {
            jsonrpc: "2.0".into(),
            id,
            result: None,
            error: Some(JsonRpcError {
                code,
                message: message.into(),
                data: None,
            }),
        }
    }
}

/// Parse a single NDJSON line into a `JsonRpcRequest`.
pub fn parse_request(line: &str) -> Result<JsonRpcRequest, JsonRpcResponse> {
    let req: JsonRpcRequest = serde_json::from_str(line)
        .map_err(|e| JsonRpcResponse::error(None, PARSE_ERROR, format!("parse error: {e}")))?;

    if req.jsonrpc != "2.0" {
        return Err(JsonRpcResponse::error(
            req.id.clone(),
            INVALID_REQUEST,
            "jsonrpc must be \"2.0\"",
        ));
    }

    if req.method.is_empty() {
        return Err(JsonRpcResponse::error(
            req.id.clone(),
            INVALID_REQUEST,
            "method must not be empty",
        ));
    }

    Ok(req)
}

/// Write a JSON-RPC response as NDJSON to stdout.
pub fn write_response(response: &JsonRpcResponse) -> io::Result<()> {
    let line = serde_json::to_string(response)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    let stdout = io::stdout();
    let mut out = stdout.lock();
    out.write_all(line.as_bytes())?;
    out.write_all(b"\n")?;
    out.flush()
}

/// Read NDJSON lines from a reader, applying size limits.
pub fn read_lines(reader: impl BufRead) -> impl Iterator<Item = io::Result<String>> {
    reader.lines().map(|line_result| {
        let line = line_result?;
        if line.len() > MAX_LINE_BYTES {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("line exceeds maximum size ({} bytes)", MAX_LINE_BYTES),
            ));
        }
        Ok(line)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_valid_request() {
        let json = r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}"#;
        let req = parse_request(json).expect("should parse");
        assert_eq!(req.method, "initialize");
        assert_eq!(req.id, Some(serde_json::json!(1)));
    }

    #[test]
    fn parse_invalid_jsonrpc_version() {
        let json = r#"{"jsonrpc":"1.0","id":1,"method":"test"}"#;
        let err = parse_request(json).expect_err("should fail");
        assert_eq!(err.error.unwrap().code, INVALID_REQUEST);
    }

    #[test]
    fn parse_empty_method() {
        let json = r#"{"jsonrpc":"2.0","id":1,"method":""}"#;
        let err = parse_request(json).expect_err("should fail");
        assert_eq!(err.error.unwrap().code, INVALID_REQUEST);
    }

    #[test]
    fn parse_malformed_json() {
        let err = parse_request("not json").expect_err("should fail");
        assert_eq!(err.error.unwrap().code, PARSE_ERROR);
    }

    #[test]
    fn response_success() {
        let resp =
            JsonRpcResponse::success(Some(serde_json::json!(1)), serde_json::json!({"ok": true}));
        assert!(resp.result.is_some());
        assert!(resp.error.is_none());
        let json = serde_json::to_string(&resp).expect("serialize");
        assert!(json.contains("\"result\""));
        assert!(!json.contains("\"error\""));
    }

    #[test]
    fn response_error() {
        let resp =
            JsonRpcResponse::error(Some(serde_json::json!(2)), METHOD_NOT_FOUND, "not found");
        assert!(resp.result.is_none());
        assert!(resp.error.is_some());
        assert_eq!(resp.error.as_ref().unwrap().code, METHOD_NOT_FOUND);
    }

    #[test]
    fn read_lines_respects_limit() {
        let big_line = "a".repeat(MAX_LINE_BYTES + 1);
        let reader = io::Cursor::new(big_line);
        let results: Vec<_> = read_lines(io::BufReader::new(reader)).collect();
        assert_eq!(results.len(), 1);
        assert!(results[0].is_err());
    }

    #[test]
    fn notification_has_no_id() {
        let json = r#"{"jsonrpc":"2.0","method":"notify","params":{}}"#;
        let req = parse_request(json).expect("should parse");
        assert!(req.id.is_none());
    }
}
