// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0

//! Recognize QIR at the QIS loading boundary, without optional LLVM bindings.
//!
//! This is a dialect guard, not an LLVM verifier. LLVM still validates syntax.
//! Comments, strings, symbol names, function signatures and attribute lists are
//! distinguished so data containing QIR terminology does not change its dialect.

#[derive(Debug, PartialEq, Eq)]
enum Token {
    Word(String),
    String(String),
    Global(String),
    Local,
    Punct(u8),
}

impl Token {
    fn word(&self, value: &str) -> bool {
        matches!(self, Self::Word(word) if word == value)
    }
}

fn quoted(bytes: &[u8], cursor: &mut usize) -> String {
    *cursor += 1;
    let mut value = Vec::new();
    while *cursor < bytes.len() && bytes[*cursor] != b'"' {
        if bytes[*cursor] == b'\\' && *cursor + 2 < bytes.len() {
            let high = char::from(bytes[*cursor + 1]).to_digit(16);
            let low = char::from(bytes[*cursor + 2]).to_digit(16);
            if let (Some(high), Some(low)) = (high, low) {
                value.push(u8::try_from(high * 16 + low).expect("two hex digits fit in a byte"));
                *cursor += 3;
                continue;
            }
        }
        value.push(bytes[*cursor]);
        *cursor += 1;
    }
    *cursor = (*cursor + 1).min(bytes.len());
    String::from_utf8_lossy(&value).into_owned()
}

fn is_name(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'.' | b'$' | b'-')
}

fn lex(ir: &str) -> Vec<Token> {
    let bytes = ir.as_bytes();
    let mut tokens = Vec::new();
    let mut cursor = 0;
    while cursor < bytes.len() {
        match bytes[cursor] {
            b';' => {
                while cursor < bytes.len() && bytes[cursor] != b'\n' {
                    cursor += 1;
                }
            }
            byte if byte.is_ascii_whitespace() => cursor += 1,
            b'"' => tokens.push(Token::String(quoted(bytes, &mut cursor))),
            prefix @ (b'@' | b'%') => {
                cursor += 1;
                let name = if bytes.get(cursor) == Some(&b'"') {
                    quoted(bytes, &mut cursor)
                } else {
                    let start = cursor;
                    while cursor < bytes.len() && is_name(bytes[cursor]) {
                        cursor += 1;
                    }
                    String::from_utf8_lossy(&bytes[start..cursor]).into_owned()
                };
                tokens.push(if prefix == b'@' {
                    Token::Global(name)
                } else {
                    Token::Local
                });
            }
            byte if is_name(byte) => {
                let start = cursor;
                while cursor < bytes.len() && is_name(bytes[cursor]) {
                    cursor += 1;
                }
                tokens.push(Token::Word(
                    String::from_utf8_lossy(&bytes[start..cursor]).into_owned(),
                ));
            }
            byte => {
                tokens.push(Token::Punct(byte));
                cursor += 1;
            }
        }
    }
    tokens
}

/// Find the closing delimiter, including nested parameter attributes/types.
fn closing(tokens: &[Token], start: usize, open: u8, close: u8) -> Option<usize> {
    let mut depth = 0usize;
    for (index, token) in tokens.iter().enumerate().skip(start) {
        if *token == Token::Punct(open) {
            depth += 1;
        } else if *token == Token::Punct(close) {
            depth = depth.checked_sub(1)?;
            if depth == 0 {
                return Some(index);
            }
        }
    }
    None
}

fn qir_attribute(tokens: &[Token]) -> Option<String> {
    let mut cursor = 0;
    while cursor < tokens.len() {
        // These function-header clauses take strings that are not attributes.
        // `c"..."` is an LLVM byte-array constant, including prefix/prologue
        // data; its contents must not be interpreted as attribute names.
        if tokens[cursor].word("section")
            || tokens[cursor].word("partition")
            || tokens[cursor].word("gc")
            || tokens[cursor].word("c")
        {
            cursor += 2;
            continue;
        }
        if let Token::String(name) = &tokens[cursor] {
            if matches!(
                name.as_str(),
                "entry_point" | "qir_profiles" | "required_num_results"
            ) {
                return Some(format!("QIR function attribute {name:?}"));
            }
            if tokens.get(cursor + 1) == Some(&Token::Punct(b'=')) {
                // Attribute values may themselves contain a QIR marker.
                cursor += 3;
                continue;
            }
        }
        cursor += 1;
    }
    None
}

fn shared_intrinsic(name: &str) -> bool {
    if matches!(
        name,
        "__quantum__rt__qubit_allocate"
            | "__quantum__rt__qubit_release"
            | "__quantum__rt__result_allocate"
            | "__quantum__rt__result_get_one"
    ) {
        return true;
    }
    let Some(gate) = name
        .strip_prefix("__quantum__qis__")
        .and_then(|name| name.strip_suffix("__body"))
    else {
        return false;
    };
    // These QIS intrinsics take integer handles (and rotation angles), never
    // pointers. Runtime result_record_output deliberately is NOT in this set:
    // real QIS programs use its pointer ABI too.
    matches!(
        gate,
        "h" | "x"
            | "y"
            | "z"
            | "s"
            | "sdg"
            | "t"
            | "tdg"
            | "cx"
            | "cnot"
            | "cy"
            | "cz"
            | "ch"
            | "rx"
            | "ry"
            | "rz"
            | "rzz"
            | "r1xy"
            | "crz"
            | "ccx"
            | "zz"
            | "m"
            | "mz"
            | "reset"
    )
}

fn has_pointer(tokens: &[Token]) -> bool {
    tokens
        .iter()
        .any(|token| token.word("ptr") || *token == Token::Punct(b'*'))
}

fn skip_delimited(tokens: &[Token], start: usize) -> Option<usize> {
    let close = match tokens.get(start)? {
        Token::Punct(b'(') => b')',
        Token::Punct(b'[') => b']',
        Token::Punct(b'{') => b'}',
        Token::Punct(b'<') => b'>',
        _ => return None,
    };
    let Token::Punct(open) = tokens[start] else {
        return None;
    };
    closing(tokens, start, open, close).map(|end| end + 1)
}

/// Skip a prefix/prologue/personality typed constant. In particular, aggregate
/// data braces in a function header are not the start of its instruction body.
fn skip_header_constant(tokens: &[Token], start: usize) -> Option<usize> {
    let mut cursor = match tokens.get(start)? {
        Token::Word(_) | Token::Local => start + 1,
        _ => skip_delimited(tokens, start)?,
    };
    // LLVM type suffixes: function types, address spaces and typed pointers.
    loop {
        match tokens.get(cursor) {
            Some(Token::Punct(b'*')) => cursor += 1,
            Some(Token::Punct(b'(')) => cursor = skip_delimited(tokens, cursor)?,
            Some(token) if token.word("addrspace") => {
                cursor = skip_delimited(tokens, cursor + 1)?;
            }
            _ => break,
        }
    }
    match tokens.get(cursor)? {
        token if token.word("c") => {
            matches!(tokens.get(cursor + 1), Some(Token::String(_))).then_some(cursor + 2)
        }
        Token::Punct(b'[' | b'{' | b'<') => skip_delimited(tokens, cursor),
        Token::Word(_) if tokens.get(cursor + 1) == Some(&Token::Punct(b'(')) => {
            skip_delimited(tokens, cursor + 1)
        }
        token if token.word("getelementptr") => {
            // Constant GEP may carry `inbounds`, `nuw` and `nusw` flags.
            cursor += 1;
            while tokens.get(cursor).is_some_and(|token| {
                token.word("inbounds") || token.word("nuw") || token.word("nusw")
            }) {
                cursor += 1;
            }
            skip_delimited(tokens, cursor)
        }
        token if token.word("dso_local_equivalent") || token.word("no_cfi") => {
            matches!(tokens.get(cursor + 1), Some(Token::Global(_))).then_some(cursor + 2)
        }
        Token::Word(_) | Token::Global(_) | Token::Local => Some(cursor + 1),
        _ => None,
    }
}

/// Return a recognizable QIR marker, or `None` for input without one.
pub(crate) fn qir_reason(ir: &str) -> Option<String> {
    let tokens = lex(ir);
    let mut cursor = 0;
    while cursor < tokens.len() {
        if tokens[cursor].word("attributes") {
            if let Some(open) =
                (cursor + 1..tokens.len()).find(|&index| tokens[index] == Token::Punct(b'{'))
                && let Some(end) = closing(&tokens, open, b'{', b'}')
            {
                if let Some(reason) = qir_attribute(&tokens[open + 1..end]) {
                    return Some(reason);
                }
                cursor = end + 1;
                continue;
            }
        } else if tokens[cursor].word("declare") || tokens[cursor].word("define") {
            let start = cursor;
            cursor += 1;
            while cursor < tokens.len() && !matches!(tokens[cursor], Token::Global(_)) {
                cursor += 1;
            }
            let Some(Token::Global(name)) = tokens.get(cursor) else {
                break;
            };
            let open = cursor + 1;
            if tokens.get(open) != Some(&Token::Punct(b'(')) {
                continue;
            }
            let Some(end) = closing(&tokens, open, b'(', b')') else {
                break;
            };
            if shared_intrinsic(name)
                && (has_pointer(&tokens[start + 1..cursor]) || has_pointer(&tokens[open + 1..end]))
            {
                return Some(format!("QIR pointer signature for {name}"));
            }
            cursor = end + 1;
            let attr_start = cursor;
            while cursor < tokens.len()
                && !tokens[cursor].word("declare")
                && !tokens[cursor].word("define")
                && !tokens[cursor].word("attributes")
                && !matches!(
                    tokens[cursor],
                    Token::Global(_) | Token::Local | Token::Punct(b'{' | b'!')
                )
            {
                if (tokens[cursor].word("prefix")
                    || tokens[cursor].word("prologue")
                    || tokens[cursor].word("personality"))
                    && let Some(end) = skip_header_constant(&tokens, cursor + 1)
                {
                    cursor = end;
                    continue;
                }
                cursor += 1;
            }
            if let Some(reason) = qir_attribute(&tokens[attr_start..cursor]) {
                return Some(reason);
            }
            // Function bodies may contain strings or instructions named after
            // attributes. Only their signature/header can establish dialect.
            if tokens.get(cursor) == Some(&Token::Punct(b'{')) {
                cursor = closing(&tokens, cursor, b'{', b'}').map_or(tokens.len(), |end| end + 1);
            }
            continue;
        }
        cursor += 1;
    }
    None
}

#[cfg(test)]
#[path = "qir_detection_tests.rs"]
mod tests;
