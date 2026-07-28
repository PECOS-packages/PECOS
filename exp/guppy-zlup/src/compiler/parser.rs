//! JSON IR parser.

use crate::ir::GuppyIR;

/// Parse Guppy IR from JSON string.
pub fn parse_ir(json: &str) -> Result<GuppyIR, ParseError> {
    serde_json::from_str(json).map_err(ParseError::Json)
}

/// Parse Guppy IR from a file.
pub fn parse_ir_file(path: &str) -> Result<GuppyIR, ParseError> {
    let json = std::fs::read_to_string(path).map_err(ParseError::Io)?;
    parse_ir(&json)
}

/// Parse error.
#[derive(Debug, thiserror::Error)]
pub enum ParseError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON parse error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Invalid IR version: expected {expected}, got {actual}")]
    InvalidVersion { expected: String, actual: String },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_minimal() {
        let json = r#"{"version": "0.1.0", "functions": []}"#;
        let ir = parse_ir(json).unwrap();
        assert_eq!(ir.version, "0.1.0");
        assert!(ir.functions.is_empty());
    }

    #[test]
    fn test_parse_function() {
        let json = r#"{
            "version": "0.1.0",
            "functions": [
                {
                    "name": "test",
                    "params": [],
                    "body": []
                }
            ]
        }"#;
        let ir = parse_ir(json).unwrap();
        assert_eq!(ir.functions.len(), 1);
        assert_eq!(ir.functions[0].name, "test");
    }

    #[test]
    fn test_parse_invalid_json() {
        let json = "not valid json";
        let result = parse_ir(json);
        assert!(result.is_err());
    }
}
