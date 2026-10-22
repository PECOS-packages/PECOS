// Copyright 2026 The PECOS Developers
// Licensed under the Apache License, Version 2.0

//! Native HUGR lowering, isolated behind the `hugr` feature.

use crate::DemBuildError;
use compiler::inkwell::{AddressSpace, context::Context, module::Module};
use pecos_programs::Qis;
use selene_hugr_qis_compiler as compiler;

fn preparation(error: impl std::fmt::Display) -> DemBuildError {
    DemBuildError::Preparation(format!("HUGR compilation: {error}"))
}

pub(super) fn lower(bytes: &[u8]) -> Result<Qis, DemBuildError> {
    let (_, package) =
        compiler::hugr::envelope::read_envelope(bytes, &compiler::REGISTRY).map_err(preparation)?;
    let mut modules = package.modules.into_iter();
    let mut hugr = modules
        .next()
        .ok_or_else(|| preparation("expected one HUGR module"))?;
    if modules.next().is_some() {
        return Err(preparation("expected exactly one HUGR module"));
    }
    compiler::validate(&hugr).map_err(preparation)?;
    let context = Context::create();
    let level = compiler::get_opt_level(2).map_err(preparation)?;
    let target = compiler::get_native_target_machine(level).map_err(preparation)?;
    let platform = compiler::get_platform("helios").map_err(preparation)?;
    let args = compiler::CompileArgs::new(&"pecos_dem", &target, level, platform, &context, false);
    let module = compiler::compile(&args, &context, &mut hugr).map_err(preparation)?;
    normalize_helpers(&module, &context)?;
    module.verify().map_err(preparation)?;
    Ok(Qis::from_string(module.to_string()))
}

/// Resolve the same public PECOS helper ABIs as the Python compiler boundary.
/// Work on LLVM values, preserving comments/strings and validating real types.
fn normalize_helpers(module: &Module<'_>, context: &Context) -> Result<(), DemBuildError> {
    let pointer = context.ptr_type(AddressSpace::default());
    let integer = context.i64_type();
    let pair = context.struct_type(&[integer.into(), integer.into()], false);
    let helpers = [
        (
            "pecos_qis_trace_metadata_hugr",
            context
                .void_type()
                .fn_type(&[pointer.into(), pointer.into()], false),
        ),
        (
            "pecos_qis_trace_metadata_qubit_hugr",
            integer.fn_type(&[integer.into(), pointer.into(), pointer.into()], false),
        ),
        (
            "pecos_qis_runtime_barrier_qubit_hugr",
            integer.fn_type(&[integer.into()], false),
        ),
        (
            "pecos_qis_runtime_barrier_qubits2_hugr",
            pair.fn_type(&[integer.into(), integer.into()], false),
        ),
    ];
    let functions: Vec<_> = module.get_functions().collect();
    for function in functions {
        let name = function.get_name().to_str().map_err(preparation)?;
        let suffix = if let Some(name) = name.strip_prefix("__hugr__.") {
            let mut parts: Vec<_> = name.split('.').collect();
            if parts
                .last()
                .is_some_and(|s| !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit()))
            {
                parts.pop();
            }
            parts.last().copied().unwrap_or_default()
        } else {
            name
        };
        let Some((canonical, expected)) =
            helpers.iter().find(|(canonical, _)| *canonical == suffix)
        else {
            continue;
        };
        if function.get_type() != *expected
            || function.count_basic_blocks() != 0
            || function.get_call_conventions() != 0
        {
            return Err(preparation(format!(
                "runtime helper {name:?} has an incompatible ABI or a user-defined body"
            )));
        }
        if name == *canonical {
            continue;
        }
        if let Some(existing) = module.get_function(canonical) {
            if existing.get_type() != *expected
                || existing.count_basic_blocks() != 0
                || existing.get_call_conventions() != 0
            {
                return Err(preparation(format!(
                    "runtime helper {canonical:?} has an incompatible ABI"
                )));
            }
            function.replace_all_uses_with(existing);
        } else {
            function.as_global_value().set_name(canonical);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use compiler::inkwell::memory_buffer::MemoryBuffer;

    #[test]
    fn helper_normalization_changes_symbols_and_preserves_strings() {
        let context = Context::create();
        let ir = r#"
            @payload = private constant [48 x i8] c"__hugr__.pecos_qis_runtime_barrier_qubit_hugr.1\00"
            declare i64 @__hugr__.pecos_qis_runtime_barrier_qubit_hugr.1(i64)
            define i64 @test(i64 %q) {
                %r = call i64 @__hugr__.pecos_qis_runtime_barrier_qubit_hugr.1(i64 %q)
                ret i64 %r
            }
        "#;
        let module = context
            .create_module_from_ir(MemoryBuffer::create_from_memory_range_copy(
                std::ffi::CString::new(ir).unwrap().as_bytes_with_nul(),
                "helpers",
            ))
            .unwrap();
        normalize_helpers(&module, &context).unwrap();
        module.verify().unwrap();
        let normalized = module.to_string();
        assert!(normalized.contains("call i64 @pecos_qis_runtime_barrier_qubit_hugr(i64 %q)"));
        assert!(normalized.contains(r#"c"__hugr__.pecos_qis_runtime_barrier_qubit_hugr.1\00""#));
    }

    #[test]
    fn helper_normalization_rejects_wrong_abi_and_bodies() {
        let context = Context::create();
        for ir in [
            "declare i32 @__hugr__.pecos_qis_runtime_barrier_qubit_hugr.1(i64)",
            "define i64 @__hugr__.pecos_qis_runtime_barrier_qubit_hugr.1(i64 %q) { ret i64 %q }",
        ] {
            let module = context
                .create_module_from_ir(MemoryBuffer::create_from_memory_range_copy(
                    std::ffi::CString::new(ir).unwrap().as_bytes_with_nul(),
                    "bad_helper",
                ))
                .unwrap();
            assert!(normalize_helpers(&module, &context).is_err());
        }
    }
}
