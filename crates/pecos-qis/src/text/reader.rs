//! Shared token spans, module entities and type readers. No LLVM dependency.

#[derive(Clone, Copy, Debug)]
pub(super) struct Token<'a> {
    pub text: &'a str,
    pub start: usize,
    pub end: usize,
    character_start: usize,
}

fn identifier(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || b"_.$-".contains(&byte)
}

pub(super) fn lex(text: &str) -> Vec<Token<'_>> {
    let bytes = text.as_bytes();
    let mut index = 0;
    let mut character_index = 0;
    let mut tokens = Vec::new();
    while index < bytes.len() {
        let start = index;
        let character = text[index..].chars().next().expect("remaining character");
        if character.is_whitespace() {
            index += character.len_utf8();
            character_index += 1;
            continue;
        }
        if bytes[index] == b';' {
            while index < bytes.len() && bytes[index] != b'\n' {
                index += 1;
            }
        } else {
            if b"@%!".contains(&bytes[index]) && bytes.get(index + 1) == Some(&b'"') {
                index += 1;
            }
            if bytes[index] == b'"' {
                index += 1;
                while index < bytes.len() {
                    let byte = bytes[index];
                    index += 1;
                    if byte == b'"' {
                        break;
                    }
                    if byte == b'\\' && index < bytes.len() {
                        index += 1;
                    }
                }
            } else {
                if b"@%$!#".contains(&bytes[index])
                    && bytes.get(index + 1).is_some_and(|b| identifier(*b))
                {
                    index += 1;
                }
                if identifier(bytes[index]) {
                    while index < bytes.len() && identifier(bytes[index]) {
                        index += 1;
                    }
                } else {
                    index += character.len_utf8();
                }
            }
        }
        tokens.push(Token {
            text: &text[start..index],
            start,
            end: index,
            character_start: character_index,
        });
        character_index += text[start..index].chars().count();
    }
    tokens
}

pub(super) fn tokens(text: &str) -> Vec<Token<'_>> {
    lex(text)
        .into_iter()
        .filter(|token| !token.text.starts_with(';'))
        .collect()
}

pub(super) fn symbol(lexeme: &str) -> String {
    let Some(quoted) = lexeme.strip_prefix("@\"").and_then(|s| s.strip_suffix('"')) else {
        return lexeme.get(1..).unwrap_or_default().to_owned();
    };
    let mut decoded = String::new();
    let mut chars = quoted.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch == '\\' {
            let mut lookahead = chars.clone();
            if let (Some(first), Some(second)) = (lookahead.next(), lookahead.next())
                && let (Some(first), Some(second)) = (first.to_digit(16), second.to_digit(16))
            {
                decoded.push(char::from_u32(first * 16 + second).expect("byte character"));
                chars = lookahead;
                continue;
            }
        }
        decoded.push(ch);
    }
    decoded
}

pub(super) fn nesting(token: &str) -> i32 {
    match token {
        "(" | "[" | "{" | "<" => 1,
        ")" | "]" | "}" | ">" => -1,
        _ => 0,
    }
}

pub(super) fn matching_paren(tokens: &[Token<'_>], start: usize) -> Result<usize, String> {
    let mut depth = 0;
    for (index, token) in tokens.iter().enumerate().skip(start) {
        match token.text {
            "(" => depth += 1,
            ")" => {
                depth -= 1;
                if depth == 0 {
                    return Ok(index);
                }
            }
            _ => {}
        }
    }
    Err(format!(
        "Unterminated function parameter list at offset {}",
        tokens[start].character_start
    ))
}

pub(super) fn split_parameters(text: &str) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut start = 0;
    let mut depth = 0;
    for token in tokens(text) {
        if token.text == "," && depth == 0 {
            parts.push(text[start..token.start].trim());
            start = token.end;
        }
        depth += nesting(token.text);
    }
    if !text.trim().is_empty() {
        parts.push(text[start..].trim());
    }
    parts
}

pub(super) fn parameter_type(parameter: &str) -> Result<&str, String> {
    let tokens = tokens(parameter);
    if tokens.is_empty() {
        return Ok("");
    }
    let mut index = 1;
    if matches!(tokens[0].text, "[" | "{" | "<") {
        let mut depth = 1;
        while index < tokens.len() && depth != 0 {
            depth += nesting(tokens[index].text);
            index += 1;
        }
    }
    while index < tokens.len() {
        match tokens[index].text {
            "*" => index += 1,
            "(" => index = matching_paren(&tokens, index)? + 1,
            "addrspace" if tokens.get(index + 1).is_some_and(|t| t.text == "(") => {
                index = matching_paren(&tokens, index + 1)? + 1;
            }
            _ => break,
        }
    }
    Ok(&parameter[..tokens[index - 1].end])
}

pub(super) fn return_type(prefix: &str) -> Result<&str, String> {
    for token in tokens(prefix) {
        let candidate = &prefix[token.start..];
        if parameter_type(candidate)? == candidate {
            return Ok(candidate);
        }
    }
    Ok(prefix)
}

pub(super) fn joined(tokens: &[Token<'_>]) -> String {
    tokens
        .iter()
        .map(|token| token.text)
        .collect::<Vec<_>>()
        .join(" ")
}

#[derive(Debug)]
pub(super) struct Function {
    pub keyword: usize,
    pub name: usize,
    pub parameters_end: usize,
    pub end: usize,
    pub body: Option<(usize, usize)>,
}

fn module_start(tokens: &[Token<'_>], index: usize) -> bool {
    let text = tokens[index].text;
    if text == "target" {
        return tokens
            .get(index + 1)
            .is_some_and(|t| matches!(t.text, "triple" | "datalayout"));
    }
    matches!(
        text,
        "declare"
            | "define"
            | "attributes"
            | "source_filename"
            | "module"
            | "uselistorder"
            | "uselistorder_bb"
    ) || (text.starts_with(['@', '%', '$', '!'])
        && tokens.get(index + 1).is_some_and(|t| t.text == "="))
}

fn entity_end(tokens: &[Token<'_>], start: usize) -> (usize, Option<(usize, usize)>) {
    let mut depth = 0;
    let mut body_start = None;
    let mut body = None;
    let mut index = start;
    while index < tokens.len() {
        let text = tokens[index].text;
        if depth == 0 && module_start(tokens, index) {
            break;
        }
        if text == "{" && depth == 0 {
            body_start = Some(index);
        }
        depth += nesting(text);
        if text == "}"
            && depth == 0
            && let Some(start) = body_start
        {
            body = Some((start, index));
        }
        index += 1;
    }
    (index, body)
}

pub(super) fn functions(tokens: &[Token<'_>]) -> Result<Vec<Function>, String> {
    let mut functions = Vec::new();
    let mut index = 0;
    while index < tokens.len() {
        let keyword = index;
        if !matches!(tokens[index].text, "declare" | "define") {
            index += 1;
            continue;
        }
        index += 1;
        while index < tokens.len() && !tokens[index].text.starts_with('@') {
            if module_start(tokens, index) {
                break;
            }
            index += 1;
        }
        if index + 1 >= tokens.len()
            || !tokens[index].text.starts_with('@')
            || tokens[index + 1].text != "("
        {
            continue;
        }
        let name = index;
        let parameters_end = matching_paren(tokens, name + 1)?;
        let (end, body) = entity_end(tokens, parameters_end + 1);
        index = end;
        functions.push(Function {
            keyword,
            name,
            parameters_end,
            end,
            body,
        });
    }
    Ok(functions)
}
