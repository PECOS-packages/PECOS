//! Small lexical reader for canonical LLVM text. Nesting and quoted identifiers
//! are handled before interpreting instructions; attributes are not operands.
use super::{Result, error};

pub fn tokens(text: &str) -> Vec<String> {
    let b = text.as_bytes();
    let mut i = 0;
    let mut out = Vec::new();
    while i < b.len() {
        if b[i].is_ascii_whitespace() {
            i += 1;
            continue;
        }
        if b[i] == b';' {
            break;
        }
        let start = i;
        if b[i] == b'"'
            || ((b[i] == b'%' || b[i] == b'@' || b[i] == b'c') && b.get(i + 1) == Some(&b'"'))
        {
            if b[i] != b'"' {
                i += 1;
            }
            i += 1;
            while i < b.len() {
                if b[i] == b'"' {
                    i += 1;
                    break;
                }
                if b[i] == b'\\' {
                    i += 1;
                }
                i += 1;
            }
        } else if b"()[]{}<>,=:*".contains(&b[i]) {
            i += 1;
        } else {
            while i < b.len() && !b[i].is_ascii_whitespace() && !b"()[]{}<>,=:*;".contains(&b[i]) {
                i += 1;
            }
        }
        out.push(text[start..i].to_string());
    }
    out
}
pub fn text(t: &[String]) -> String {
    t.join(" ")
}
pub fn matching(t: &[String], start: usize) -> Result<usize> {
    let close = match t.get(start).map(String::as_str) {
        Some("(") => ")",
        Some("[") => "]",
        Some("{") => "}",
        Some("<") => ">",
        _ => return Err(error("expected opening delimiter")),
    };
    let mut depth = 0;
    for (i, token) in t.iter().enumerate().skip(start) {
        if token == &t[start] {
            depth += 1;
        }
        if token == close {
            depth -= 1;
            if depth == 0 {
                return Ok(i);
            }
        }
    }
    Err(error("unclosed LLVM delimiter"))
}
pub fn split(t: &[String]) -> Vec<Vec<String>> {
    let mut depth = 0;
    let mut start = 0;
    let mut out = Vec::new();
    for (i, token) in t.iter().enumerate() {
        match token.as_str() {
            "(" | "[" | "{" | "<" => depth += 1,
            ")" | "]" | "}" | ">" => depth -= 1,
            "," if depth == 0 => {
                out.push(t[start..i].to_vec());
                start = i + 1;
            }
            _ => (),
        }
    }
    if start < t.len() {
        out.push(t[start..].to_vec());
    }
    out
}
pub fn type_end(t: &[String], start: usize) -> Result<usize> {
    let v = t.get(start).ok_or_else(|| error("missing LLVM type"))?;
    if matches!(v.as_str(), "[" | "{" | "<") {
        return Ok(matching(t, start)? + 1);
    }
    if v == "ptr" {
        if t.get(start + 1).is_some_and(|s| s == "addrspace") {
            return Ok(matching(t, start + 2)? + 1);
        }
        return Ok(start + 1);
    }
    if v.starts_with('%')
        || matches!(
            v.as_str(),
            "void"
                | "half"
                | "bfloat"
                | "float"
                | "double"
                | "fp128"
                | "x86_fp80"
                | "ppc_fp128"
                | "label"
        )
        || v.strip_prefix('i')
            .is_some_and(|n| !n.is_empty() && n.bytes().all(|b| b.is_ascii_digit()))
    {
        return Ok(start + 1);
    }
    Err(error(format!("expected LLVM type, got {v}")))
}
pub fn typed(t: &[String]) -> Result<(String, String)> {
    let end = type_end(t, 0)?;
    let ty = text(&t[..end]);
    let mut i = end;
    while i < t.len() {
        let s = t[i].as_str();
        if matches!(
            s,
            "noundef"
                | "nonnull"
                | "nocapture"
                | "readonly"
                | "readnone"
                | "writeonly"
                | "immarg"
                | "inreg"
                | "signext"
                | "zeroext"
                | "noalias"
                | "nest"
                | "returned"
                | "swiftself"
                | "swifterror"
                | "nofree"
        ) {
            i += 1;
        } else if s == "align" {
            i += 2;
        } else if t.get(i + 1).is_some_and(|s| s == "(")
            && matches!(
                s,
                "dereferenceable"
                    | "dereferenceable_or_null"
                    | "byval"
                    | "sret"
                    | "byref"
                    | "preallocated"
                    | "elementtype"
                    | "alignstack"
                    | "range"
                    | "captures"
                    | "nofpclass"
            )
        {
            i = matching(t, i + 1)? + 1;
        } else {
            break;
        }
    }
    Ok((ty, text(&t[i..])))
}
#[derive(Clone)]
pub struct Call {
    pub symbol: String,
    pub ty: String,
    pub args: Vec<(String, String)>,
}
pub fn call(t: &[String]) -> Result<Call> {
    let at = t
        .iter()
        .position(|s| s.starts_with('@'))
        .ok_or_else(|| error("indirect call or inline asm"))?;
    let symbol = t[at].clone();
    if t.get(at + 1).is_none_or(|s| s != "(") {
        return Err(error("unsupported call signature"));
    }
    let mut ty = None;
    for start in 0..at {
        if type_end(t, start).is_ok_and(|end| end == at) {
            ty = Some(text(&t[start..at]));
            break;
        }
    }
    let end = matching(t, at + 1)?;
    let args = split(&t[at + 2..end])
        .iter()
        .map(|a| typed(a))
        .collect::<Result<_>>()?;
    Ok(Call {
        symbol,
        ty: ty.ok_or_else(|| error("missing call return type"))?,
        args,
    })
}
pub fn allocation_size(ty: &str) -> String {
    format!("ptrtoint (ptr getelementptr ({ty}, ptr null, i64 1) to i64)")
}

/// Scalar and fixed-vector representation bytes differ from ABI allocation
/// stride (i24 and <3 x i8> each store three bytes, not four). Aggregate offsets
/// and every alloca extent remain LLVM expressions, independent of these widths.
pub fn size(ty: &str) -> String {
    fn bits(ty: &str) -> Option<usize> {
        ty.strip_prefix('i')
            .and_then(|v| v.parse::<usize>().ok())
            .or(match ty {
                "half" | "bfloat" => Some(16),
                "float" => Some(32),
                "double" => Some(64),
                "x86_fp80" => Some(80),
                "fp128" | "ppc_fp128" => Some(128),
                _ => None,
            })
    }
    let t = tokens(ty);
    if t.first().is_some_and(|s| s == "<") && t.get(2).is_some_and(|s| s == "x") {
        let element = text(&t[3..t.len() - 1]);
        if let Some(total) = t[1]
            .parse::<usize>()
            .ok()
            .zip(bits(&element))
            .and_then(|(count, width)| count.checked_mul(width))
        {
            return total.div_ceil(8).to_string();
        }
        if element.starts_with("ptr") {
            return format!(
                "ptrtoint (ptr getelementptr ({element}, ptr null, i64 {}) to i64)",
                t[1]
            );
        }
    }
    bits(ty).map_or_else(|| allocation_size(ty), |bits| bits.div_ceil(8).to_string())
}

/// Instrumented functions allocate shadow vectors, synchronize on the context,
/// and access memory outside their original arguments. Relax obsolete effect
/// promises while preserving calling conventions and value/parameter attributes.
pub fn relax_effects(line: &str) -> Result<String> {
    let t = tokens(line);
    let mut result = Vec::new();
    let mut i = 0;
    while i < t.len() {
        if t[i] == "memory" && t.get(i + 1).is_some_and(|s| s == "(") {
            i = matching(&t, i + 1)? + 1;
        } else if matches!(
            t[i].as_str(),
            "readnone"
                | "readonly"
                | "writeonly"
                | "argmemonly"
                | "inaccessiblememonly"
                | "inaccessiblemem_or_argmemonly"
                | "nosync"
                | "nofree"
                | "speculatable"
                | "nocallback"
        ) {
            i += 1;
        } else {
            result.push(t[i].clone());
            i += 1;
        }
    }
    Ok(text(&result))
}
