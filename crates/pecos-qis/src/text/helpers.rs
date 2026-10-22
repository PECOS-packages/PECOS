//! Public helper symbol normalization at the QIS text boundary.

use super::reader::{self, Function, Token};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

/// Runtime ABIs matching the extern signatures in `pecos-qis-ffi/src/ffi.rs`.
pub const PECOS_HELPER_ABIS: [(&str, &str); 4] = [
    ("pecos_qis_trace_metadata_hugr", "void (ptr, ptr)"),
    ("pecos_qis_trace_metadata_qubit_hugr", "i64 (i64, ptr, ptr)"),
    ("pecos_qis_runtime_barrier_qubit_hugr", "i64 (i64)"),
    (
        "pecos_qis_runtime_barrier_qubits2_hugr",
        "{ i64, i64 } (i64, i64)",
    ),
];

/// A helper declaration, use or mangled symbol violates the runtime ABI.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HelperNormalizationError(String);

impl fmt::Display for HelperNormalizationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for HelperNormalizationError {}
impl From<String> for HelperNormalizationError {
    fn from(reason: String) -> Self {
        Self(reason)
    }
}

fn abi(name: &str) -> Option<&'static str> {
    PECOS_HELPER_ABIS
        .iter()
        .find_map(|(helper, abi)| (*helper == name).then_some(*abi))
}

fn canonical(name: &str) -> Option<&str> {
    if abi(name).is_some() {
        return Some(name);
    }
    let private = name.strip_prefix("__hugr__.")?;
    let mut parts = private.rsplit('.');
    let mut last = parts.next()?;
    if !last.is_empty() && last.bytes().all(|b| b.is_ascii_digit()) {
        last = parts.next()?;
    }
    abi(last).map(|_| last)
}

fn rename(ir: &str) -> String {
    let mut result = String::new();
    let mut previous = 0;
    for token in reader::tokens(ir) {
        if token.text.starts_with('@')
            && let Some(name) = canonical(&reader::symbol(token.text))
        {
            result.push_str(&ir[previous..token.start]);
            result.push('@');
            result.push_str(name);
            previous = token.end;
        }
    }
    result.push_str(&ir[previous..]);
    result
}

fn without_comments(text: &str) -> String {
    let mut output = String::new();
    let mut previous = 0;
    for token in reader::lex(text) {
        if token.text.starts_with(';') {
            output.push_str(&text[previous..token.start]);
            output.push(' ');
            previous = token.end;
        }
    }
    output.push_str(&text[previous..]);
    output
}

fn signature(ir: &str, tokens: &[Token<'_>], function: &Function) -> Result<String, String> {
    let prefix = without_comments(&ir[tokens[function.keyword].end..tokens[function.name].start]);
    let parameters =
        without_comments(&ir[tokens[function.name + 1].end..tokens[function.parameters_end].start]);
    let types = reader::split_parameters(&parameters)
        .into_iter()
        .map(reader::parameter_type)
        .collect::<Result<Vec<_>, _>>()?;
    Ok(format!(
        "{} ({})",
        reader::return_type(prefix.trim())?,
        types.join(", ")
    ))
}

fn instruction_uses(tokens: &[Token<'_>], functions: &[Function]) -> BTreeSet<usize> {
    const INSTRUCTIONS: &str = "ret br switch indirectbr invoke callbr resume catchswitch catchret cleanupret unreachable fneg add fadd sub fsub mul fmul udiv sdiv fdiv urem srem frem shl lshr ashr and or xor extractelement insertelement shufflevector extractvalue insertvalue alloca load store fence cmpxchg atomicrmw getelementptr trunc zext sext fptrunc fpext fptoui fptosi uitofp sitofp ptrtoint inttoptr bitcast addrspacecast icmp fcmp phi select freeze call va_arg landingpad catchpad cleanuppad";
    let mut uses = BTreeSet::new();
    for function in functions {
        if tokens[function.keyword].text != "define" {
            continue;
        }
        let Some((start, end)) = function.body else {
            continue;
        };
        let mut instruction = false;
        for index in start + 1..end {
            let text = tokens[index].text;
            if index + 1 < end && matches!(tokens[index + 1].text, ":" | "=") {
                instruction = false;
            } else if INSTRUCTIONS.split_whitespace().any(|opcode| opcode == text) {
                instruction = true;
            } else if text.starts_with('@') && instruction {
                uses.insert(index);
            }
        }
    }
    uses
}

fn check_calling_convention(
    prefix: &[Token<'_>],
    helper: &str,
) -> Result<(), HelperNormalizationError> {
    // LLVM prints unnamed conventions as `ccN`; source also accepts `cc N`.
    // Named conventions generally end in `cc`.
    for (index, token) in prefix.iter().enumerate() {
        let convention = token.text;
        if (convention == "cc"
            && prefix
                .get(index + 1)
                .is_none_or(|t| t.text.parse::<u32>() != Ok(0)))
            || convention.strip_prefix("cc").is_some_and(|number| {
                !number.is_empty()
                    && number.bytes().all(|byte| byte.is_ascii_digit())
                    && number.parse::<u32>() != Ok(0)
            })
            || (convention.ends_with("cc") && !matches!(convention, "ccc" | "cc"))
            || matches!(
                convention,
                "aarch64_vector_pcs"
                    | "aarch64_sve_vector_pcs"
                    | "aarch64_sme_preservemost_from_x0"
                    | "aarch64_sme_preservemost_from_x1"
                    | "aarch64_sme_preservemost_from_x2"
                    | "amdgpu_cs_chain"
                    | "amdgpu_cs_chain_preserve"
                    | "hhvm_ccc"
                    | "ptx_kernel"
                    | "ptx_device"
                    | "spir_func"
                    | "spir_kernel"
                    | "intel_ocl_bicc"
                    | "amdgpu_vs"
                    | "amdgpu_gs"
                    | "amdgpu_ps"
                    | "amdgpu_cs"
                    | "amdgpu_kernel"
                    | "amdgpu_es"
                    | "amdgpu_ls"
                    | "amdgpu_hs"
                    | "amdgpu_gfx"
                    | "msp430_intr"
                    | "avr_intr"
                    | "avr_signal"
                    | "x86_stdcallcc"
                    | "x86_fastcallcc"
                    | "x86_thiscallcc"
                    | "x86_vectorcallcc"
                    | "x86_intr"
            )
        {
            return Err(format!(
                "runtime helper {helper:?} has an incompatible ABI: calling convention {convention}"
            )
            .into());
        }
    }
    Ok(())
}

/// Rename private helper tokens, merge equivalent declarations and validate ABIs.
/// Strings and comments retain their original bytes. Public helper symbols must
/// occur in declarations or instruction operands, never definitions or aliases.
///
/// # Errors
/// Returns an error for incompatible types or calling conventions, definitions,
/// conflicting declarations, unsupported placements or unknown private mangling.
pub fn normalize_pecos_helper_symbols(ir: &str) -> Result<String, HelperNormalizationError> {
    let renamed = rename(ir);
    let tokens = reader::tokens(&renamed);
    let functions = reader::functions(&tokens)?;
    let mut declarations = BTreeMap::<String, (String, &str)>::new();
    let mut declared_names = BTreeSet::new();
    let mut duplicates = Vec::new();
    for function in &functions {
        let helper = reader::symbol(tokens[function.name].text);
        let Some(expected) = abi(&helper) else {
            continue;
        };
        if tokens[function.keyword].text == "define" {
            let line = renamed[..tokens[function.name].start]
                .bytes()
                .filter(|b| *b == b'\n')
                .count()
                + 1;
            return Err(format!(
                "PECOS helper '{helper}' must be declared, not defined (line {line})"
            )
            .into());
        }
        check_calling_convention(&tokens[function.keyword + 1..function.name], &helper)?;
        let signature = signature(&renamed, &tokens, function)?;
        declared_names.insert(function.name);
        if let Some((previous, _)) = declarations.get(&helper) {
            if reader::joined(&reader::tokens(previous))
                != reader::joined(&reader::tokens(&signature))
            {
                return Err(format!("PECOS helper '{helper}' has conflicting signatures: '{previous}' and '{signature}'").into());
            }
            duplicates.push((tokens[function.keyword].start, tokens[function.end - 1].end));
        } else {
            declarations.insert(helper, (signature, expected));
        }
    }
    for (helper, (actual, expected)) in &declarations {
        if reader::joined(&reader::tokens(actual)) != reader::joined(&reader::tokens(expected)) {
            return Err(format!(
                "PECOS helper '{helper}' expects '{expected}', actual signature is '{actual}'"
            )
            .into());
        }
    }
    let uses = instruction_uses(&tokens, &functions);
    for (index, token) in tokens.iter().enumerate() {
        if !token.text.starts_with('@') {
            continue;
        }
        let symbol = reader::symbol(token.text);
        let line = renamed[..token.start]
            .bytes()
            .filter(|b| *b == b'\n')
            .count()
            + 1;
        if symbol.starts_with("__hugr__.") && symbol.split('.').any(|part| abi(part).is_some()) {
            return Err(
                format!("Unrecognized PECOS helper symbol '@{symbol}' on line {line}").into(),
            );
        }
        if abi(&symbol).is_some() && !declared_names.contains(&index) && !uses.contains(&index) {
            return Err(format!(
                "PECOS helper '{symbol}' has unsupported placement on line {line}"
            )
            .into());
        }
    }
    let mut output = String::new();
    let mut previous = 0;
    for (start, end) in duplicates {
        output.push_str(&renamed[previous..start]);
        for token in reader::lex(&renamed[start..end]) {
            if token.text.starts_with(';') {
                output.push_str(token.text);
                output.push('\n');
            }
        }
        previous = end;
    }
    output.push_str(&renamed[previous..]);
    Ok(output)
}

#[cfg(test)]
mod tests;
