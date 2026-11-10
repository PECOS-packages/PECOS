//! Certify ordering and narrow, same-function scalar measurement provenance.

mod cfg;

use super::reader;
use cfg::FunctionBody;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

/// Sorted tags with execution ordinals (or unsupported holes), and measurement count.
pub type ResultTagMeasurements = (BTreeMap<String, Vec<Option<usize>>>, usize);

/// The lowered program is outside the supported certification subset.
///
/// This includes the malformed text explicitly classified as certification
/// failures by the original Python reader, but not compiler or runtime errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResultTagCertificationError(String);

impl fmt::Display for ResultTagCertificationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Cannot certify measurement order from the lowered program: {}",
            self.0
        )
    }
}
impl std::error::Error for ResultTagCertificationError {}
impl From<String> for ResultTagCertificationError {
    fn from(reason: String) -> Self {
        Self(reason)
    }
}
impl From<&str> for ResultTagCertificationError {
    fn from(reason: &str) -> Self {
        Self(reason.to_owned())
    }
}
type Result<T> = std::result::Result<T, ResultTagCertificationError>;

fn print_kind(name: &str) -> Option<(&'static str, &'static str)> {
    let name = name.strip_prefix("print_")?;
    for (kind, prefix, scalar) in [
        ("bool", "BOOL", "i1"),
        ("int", "INT", "i64"),
        ("uint", "UINT", "i64"),
        ("float", "FLOAT", "double"),
    ] {
        if let Some(suffix) = name.strip_prefix(kind)
            && matches!(suffix, "" | "_arr" | "_selene" | "_arr_selene")
        {
            return Some((prefix, scalar));
        }
    }
    None
}
fn relevant(name: &str) -> bool {
    matches!(
        name,
        "___lazy_measure"
            | "___lazy_measure_leaked"
            | "__quantum__rt__result_allocate"
            | "__quantum__qis__m__body"
            | "__quantum__qis__mz__body"
    ) || print_kind(name).is_some()
}
fn runtime(name: &str) -> bool {
    relevant(name) || name == "___read_future_bool"
}

fn decode_bytes(literal: &str) -> Result<Vec<u8>> {
    let bytes = literal.as_bytes();
    let mut result = Vec::new();
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] != b'\\' {
            result.push(bytes[index]);
            index += 1;
            continue;
        }
        if bytes.get(index + 1) == Some(&b'\\') {
            result.push(b'\\');
            index += 2;
            continue;
        }
        let escape = bytes
            .get(index + 1..index + 3)
            .and_then(|b| std::str::from_utf8(b).ok())
            .and_then(|s| u8::from_str_radix(s, 16).ok());
        result.push(escape.ok_or("invalid LLVM string escape")?);
        index += 3;
    }
    Ok(result)
}

#[derive(Clone, Debug)]
struct Call {
    target: String,
    result: Option<String>,
    return_type: String,
    args: Vec<String>,
}

impl Call {
    fn check(&self, return_type: &str, arg_types: &[&str]) -> Result<()> {
        let actual = self
            .args
            .iter()
            .map(|arg| reader::parameter_type(arg))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        if self.return_type != return_type || actual != arg_types {
            return Err(format!("unsupported signature for @{}", self.target).into());
        }
        if self.args.iter().any(|arg| reader::tokens(arg).len() < 2) {
            return Err(format!("missing operand for @{}", self.target).into());
        }
        Ok(())
    }

    fn parse(instruction: &str) -> Result<Option<Self>> {
        let tokens = reader::tokens(instruction);
        let mut start = cfg::instruction_start(&tokens)?;
        if matches!(tokens[start].text, "tail" | "musttail" | "notail") {
            start += 1;
            if start >= tokens.len() {
                return Err("incomplete tail call".into());
            }
        }
        if tokens[start].text != "call" {
            return Ok(None);
        }
        let mut depth = 0;
        let mut opening = None;
        for index in start + 1..tokens.len().saturating_sub(1) {
            let text = tokens[index].text;
            if depth == 0 && text.starts_with(['@', '%']) && tokens[index + 1].text == "(" {
                opening = Some(index + 1);
            }
            depth += reader::nesting(text);
        }
        let opening = opening
            .filter(|index| tokens[index - 1].text.starts_with('@'))
            .ok_or("indirect call")?;
        let closing = reader::matching_paren(&tokens, opening).map_err(|_| "unterminated call")?;
        let prefix = reader::joined(&tokens[start + 1..opening - 1]);
        let return_type = reader::return_type(&prefix)?
            .split(" (")
            .next()
            .unwrap_or_default()
            .to_owned();
        let args = reader::joined(&tokens[opening + 1..closing]);
        Ok(Some(Self {
            target: reader::symbol(tokens[opening - 1].text),
            result: (start >= 2 && tokens[1].text == "=").then(|| tokens[0].text.to_owned()),
            return_type,
            args: reader::split_parameters(&args)
                .into_iter()
                .map(str::to_owned)
                .collect(),
        }))
    }

    fn tag(&self, globals: &BTreeMap<String, Vec<u8>>) -> Result<String> {
        let (prefix, scalar_type) = print_kind(&self.target).expect("print call");
        let array = self.target.contains("_arr");
        let selene = self.target.ends_with("_selene");
        let mut types = vec!["ptr", "i64", if array { "ptr" } else { scalar_type }];
        if array && selene {
            types.push("i64");
        }
        self.check("void", &types)?;
        let pointer = operand(&self.args[0]);
        if !pointer.starts_with('@') {
            return Err("result tag operand is not a module global".into());
        }
        let name = reader::symbol(pointer);
        let data = globals
            .get(&name)
            .ok_or_else(|| format!("missing tag global @{name}"))?;
        let length = reader::tokens(&self.args[1]);
        if length.len() != 2 || !length[1].text.bytes().all(|b| b.is_ascii_digit()) {
            return Err("result tag length is not a non-negative constant".into());
        }
        // Python rejects decimal conversions above its default 4300-digit cap.
        // Larger-than-usize values below that cap can only exceed the global.
        if length[1].text.len() > 4300 {
            return Err("result tag length exceeds the supported integer range".into());
        }
        let size = length[1]
            .text
            .parse::<usize>()
            .map_err(|_| "tag global is shorter than the runtime label length")?;
        let offset = usize::from(!selene);
        let end = offset
            .checked_add(size)
            .ok_or("tag global is shorter than the runtime label length")?;
        let label = data
            .get(offset..end)
            .ok_or("tag global is shorter than the runtime label length")?;
        let label = std::str::from_utf8(label).map_err(|_| "invalid UTF-8 tag label")?;
        for kind in [prefix.to_owned(), format!("{prefix}ARR")] {
            if let Some(tag) = label.strip_prefix(&format!("USER:{kind}:")) {
                return Ok(tag.to_owned());
            }
        }
        Err("tag label does not match USER:<TYPE>:<tag>".into())
    }
}

fn operand(arg: &str) -> &str {
    reader::tokens(arg).last().map_or("", |token| token.text)
}

struct Analysis<'a> {
    declarations: BTreeSet<String>,
    bodies: BTreeMap<String, &'a str>,
    globals: BTreeMap<String, Vec<u8>>,
    functions: BTreeMap<String, FunctionBody>,
}

impl<'a> Analysis<'a> {
    fn parse(text: &'a str) -> Result<Self> {
        let tokens = reader::tokens(text);
        let functions = reader::functions(&tokens)?;
        let mut declarations = BTreeSet::new();
        let mut bodies = BTreeMap::new();
        for function in functions {
            let name = reader::symbol(tokens[function.name].text);
            if let Some((start, end)) = function.body {
                bodies.insert(name, &text[tokens[start].end..tokens[end].start]);
            } else {
                declarations.insert(name);
            }
        }
        let mut definitions: BTreeSet<_> = bodies.keys().cloned().collect();
        for pair in tokens.windows(2) {
            if pair[0].text.starts_with('@') && pair[1].text == "=" {
                definitions.insert(reader::symbol(pair[0].text));
            }
        }
        if let Some(name) = definitions.iter().find(|name| runtime(name)) {
            return Err(format!("module defines runtime call @{name}").into());
        }
        let mut globals = BTreeMap::new();
        for line in text.lines() {
            let tokens = reader::tokens(line);
            if tokens.len() < 10 || !tokens[0].text.starts_with('@') || tokens[1].text != "=" {
                continue;
            }
            let mut index = 2;
            while index < tokens.len()
                && tokens[index].text != "constant"
                && tokens[index]
                    .text
                    .bytes()
                    .all(|b| b.is_ascii_lowercase() || b == b'_')
            {
                index += 1;
            }
            let Some(parts) = tokens.get(index..index + 8) else {
                continue;
            };
            if parts[0].text == "constant"
                && parts[1].text == "["
                && parts[2].text.bytes().all(|b| b.is_ascii_digit())
                && parts[3].text == "x"
                && parts[4].text == "i8"
                && parts[5].text == "]"
                && parts[6].text == "c"
                && let Some(literal) = parts[7]
                    .text
                    .strip_prefix('"')
                    .and_then(|s| s.strip_suffix('"'))
            {
                globals.insert(reader::symbol(tokens[0].text), decode_bytes(literal)?);
            }
        }
        Ok(Self {
            declarations,
            bodies,
            globals,
            functions: BTreeMap::new(),
        })
    }

    fn discover(&mut self) -> Result<()> {
        let mut pending = vec![("qmain".to_owned(), false)];
        let mut active = BTreeSet::new();
        while let Some((name, finished)) = pending.pop() {
            if finished {
                active.remove(&name);
                continue;
            }
            if active.contains(&name) {
                return Err(format!("recursive call to @{name}").into());
            }
            if self.functions.contains_key(&name) {
                continue;
            }
            // Retain the legacy certification boundary without using the Rust stack.
            if active.len() >= 1000 {
                return Err("call graph exceeds Python's recursion limit".into());
            }
            let body = self
                .bodies
                .get(&name)
                .ok_or_else(|| format!("missing definition of @{name}"))?;
            let function = FunctionBody::parse(body)?;
            active.insert(name.clone());
            pending.push((name.clone(), true));
            for call in function.calls.iter().flatten() {
                if self.bodies.contains_key(&call.target) {
                    pending.push((call.target.clone(), false));
                } else if !self.declarations.contains(&call.target) && !runtime(&call.target) {
                    return Err(format!(
                        "callee @{} is neither a function definition nor a declaration",
                        call.target
                    )
                    .into());
                }
            }
            self.functions.insert(name, function);
        }
        Ok(())
    }

    fn ordered(&self) -> Result<BTreeMap<String, Vec<usize>>> {
        let mut relevant_functions = BTreeSet::new();
        loop {
            let added: BTreeSet<_> = self
                .functions
                .iter()
                .filter(|(_, function)| {
                    function.calls.iter().flatten().any(|call| {
                        relevant(&call.target) || relevant_functions.contains(&call.target)
                    })
                })
                .map(|(name, _)| name.clone())
                .collect();
            let changed = !added.is_subset(&relevant_functions);
            relevant_functions.extend(added);
            if !changed {
                break;
            }
        }
        self.functions
            .iter()
            .map(|(name, function)| {
                Ok((name.clone(), function.ordered_blocks(&relevant_functions)?))
            })
            .collect()
    }

    fn walk(&self, ordered: &BTreeMap<String, Vec<usize>>) -> Result<ResultTagMeasurements> {
        let mut occurrences = BTreeMap::<String, Vec<Option<usize>>>::new();
        let mut count = 0;
        let mut stack = vec![Frame::new(&self.functions["qmain"], &ordered["qmain"])?];
        while let Some(frame) = stack.last_mut() {
            let Some(call) = frame.calls.get(frame.index) else {
                stack.pop();
                continue;
            };
            frame.index += 1;
            match call.target.as_str() {
                "__quantum__rt__result_allocate" => {
                    return Err(format!("result-allocating call @{}", call.target).into());
                }
                "__quantum__qis__m__body" | "__quantum__qis__mz__body" => {
                    return Err(format!("unsupported measurement call @{}", call.target).into());
                }
                "___lazy_measure" | "___lazy_measure_leaked" => {
                    call.check("i64", &["i64"])?;
                    if call.target == "___lazy_measure"
                        && let Some(result) = &call.result
                    {
                        frame.futures.insert(result.clone(), count);
                    }
                    count += 1;
                }
                name if print_kind(name).is_some() => {
                    let tag = call.tag(&self.globals)?;
                    let ordinal = if name == "print_bool" {
                        frame
                            .reads
                            .get(operand(&call.args[2]))
                            .and_then(|future| frame.futures.get(future))
                            .copied()
                    } else {
                        None
                    };
                    occurrences.entry(tag).or_default().push(ordinal);
                }
                name => {
                    if let Some(function) = self.functions.get(name) {
                        stack.push(Frame::new(function, &ordered[name])?);
                    }
                }
            }
        }
        Ok((occurrences, count))
    }
}

struct Frame<'a> {
    calls: Vec<&'a Call>,
    index: usize,
    reads: BTreeMap<String, String>,
    futures: BTreeMap<String, usize>,
}
impl<'a> Frame<'a> {
    fn new(function: &'a FunctionBody, order: &[usize]) -> Result<Self> {
        let mut reads = BTreeMap::new();
        for call in function.calls.iter().flatten() {
            if call.target == "___read_future_bool" {
                call.check("i1", &["i64"])?;
                if let Some(result) = &call.result {
                    reads.insert(result.clone(), operand(&call.args[0]).to_owned());
                }
            }
        }
        Ok(Self {
            calls: order
                .iter()
                .flat_map(|block| function.calls[*block].iter())
                .collect(),
            index: 0,
            reads,
            futures: BTreeMap::new(),
        })
    }
}

/// Certify execution order from `qmain` and return sorted result tags and count.
/// Only scalar `print_bool <- ___read_future_bool <- ___lazy_measure` in the
/// same function carries provenance; other runtime prints contribute holes.
///
/// # Errors
/// Returns a certification error for unsupported CFGs, calls, tags or runtime
/// overrides. Dead code has no measurements, but discovered block boundaries
/// are validated even when unreachable.
pub fn extract_result_tag_measurements(qis: &str) -> Result<ResultTagMeasurements> {
    let mut analysis = Analysis::parse(qis)?;
    analysis.discover()?;
    analysis.walk(&analysis.ordered()?)
}

#[cfg(test)]
mod tests;
