// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0

//! Recognize QIR in `llvm-dis` output. LLVM canonicalization makes line-based
//! detection sound: signatures occupy one top-level line, function string
//! attributes live in attribute groups, and pointer types use `ptr`. Symbol
//! escapes are normalized, and body instructions are indented.

fn shared_intrinsic(name: &str) -> bool {
    // QIS gates never take pointers. Only these runtime functions share that
    // restriction: real QIS uses pointer-taking result_record_output and others.
    name.starts_with("__quantum__qis__")
        || matches!(
            name,
            "__quantum__rt__qubit_allocate"
                | "__quantum__rt__qubit_release"
                | "__quantum__rt__result_allocate"
                | "__quantum__rt__result_get_one"
        )
}

fn has_pointer(signature: &str) -> bool {
    signature
        .split(|c: char| {
            !(c.is_ascii_alphanumeric() || matches!(c, '_' | '.' | '$' | '-' | '%' | '@'))
        })
        .any(|token| token == "ptr")
}

/// Return a recognizable QIR marker, or `None` for canonical IR without one.
pub(crate) fn qir_reason(ir: &str) -> Option<String> {
    for line in ir.lines() {
        if line.starts_with("attributes #") {
            // LLVM escapes embedded quotes as hex, so quote pairs delimit strings.
            let mut parts = line.split('"');
            while let (Some(before), Some(key)) = (parts.next(), parts.next()) {
                if !before.trim_end().ends_with('=')
                    && matches!(key, "entry_point" | "qir_profiles" | "required_num_results")
                {
                    return Some(format!("QIR function attribute {key:?}"));
                }
            }
        } else if let Some(header) = line
            .strip_prefix("declare ")
            .or_else(|| line.strip_prefix("define "))
        {
            // Mask strings while preserving offsets, so parameter attributes and
            // quoted identifiers cannot masquerade as type tokens or delimiters.
            let unquoted: String = header
                .split('"')
                .enumerate()
                .map(|(i, part)| {
                    if i % 2 == 0 {
                        part.to_owned()
                    } else {
                        " ".repeat(part.len())
                    }
                })
                .collect::<Vec<_>>()
                .join(" ");
            let Some(at) = unquoted.find('@') else {
                continue;
            };
            let after = &header[at + 1..];
            let (name, end) = if let Some(quoted) = after.strip_prefix('"') {
                let Some(end) = quoted.find('"') else {
                    continue;
                };
                (&quoted[..end], at + end + 3)
            } else {
                let Some(end) = after.find('(') else { continue };
                (&after[..end], at + end + 1)
            };
            if !shared_intrinsic(name) {
                continue;
            }
            let mut depth = 0;
            let close = unquoted[end..].char_indices().find_map(|(i, c)| {
                match c {
                    '(' => depth += 1,
                    ')' => {
                        depth -= 1;
                        if depth == 0 {
                            return Some(end + i);
                        }
                    }
                    _ => {}
                }
                None
            });
            if let Some(close) = close
                && (has_pointer(&unquoted[..at]) || has_pointer(&unquoted[end..close]))
            {
                return Some(format!("QIR pointer signature for {name}"));
            }
        }
    }
    None
}

#[cfg(test)]
#[path = "qir_detection_tests.rs"]
mod tests;
