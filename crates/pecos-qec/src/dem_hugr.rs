// Copyright 2026 The PECOS Developers
// Licensed under the Apache License, Version 2.0

//! Native HUGR lowering, isolated behind the `hugr` feature.

use crate::DemBuildError;
use compiler::hugr::{Hugr, HugrView, ops::OpType};
use compiler::inkwell::context::Context;
use pecos_programs::Qis;
use pecos_qis::text::normalize_pecos_helper_symbols;
use selene_hugr_qis_compiler as compiler;

fn preparation(error: impl std::fmt::Display) -> DemBuildError {
    DemBuildError::Preparation(format!("HUGR compilation: {error}"))
}

/// Inspect every node, including function bodies outside the entrypoint.
pub(crate) fn has_nontrivial_control_flow(hugr: &Hugr) -> bool {
    hugr.nodes().any(|node| match hugr.get_optype(node) {
        OpType::Conditional(_)
        | OpType::TailLoop(_)
        | OpType::FuncDecl(_)
        | OpType::CallIndirect(_) => true,
        OpType::DataflowBlock(block) => block.sum_rows.len() > 1,
        _ => false,
    })
}

/// Return the sole module and its nontrivial-control-flow verdict before lowering.
pub(crate) fn parse(bytes: &[u8]) -> Result<(Hugr, bool), DemBuildError> {
    let (_, package) =
        compiler::hugr::envelope::read_envelope(bytes, &compiler::REGISTRY).map_err(preparation)?;
    let mut modules = package.modules.into_iter();
    let hugr = modules
        .next()
        .ok_or_else(|| preparation("expected one HUGR module"))?;
    if modules.next().is_some() {
        return Err(preparation("expected exactly one HUGR module"));
    }
    compiler::validate(&hugr).map_err(preparation)?;
    let nontrivial = has_nontrivial_control_flow(&hugr);
    Ok((hugr, nontrivial))
}

pub(super) fn lower(
    bytes: &[u8],
    platform: crate::QSystemPlatform,
    level: crate::OptimizationLevel,
) -> Result<Qis, DemBuildError> {
    let (mut hugr, _nontrivial) = parse(bytes)?;
    let context = Context::create();
    let target = compiler::get_native_target_machine(level).map_err(preparation)?;
    let args = compiler::CompileArgs::new(&"pecos_dem", &target, level, platform, &context, false);
    let module = compiler::compile(&args, &context, &mut hugr).map_err(preparation)?;
    module.verify().map_err(preparation)?;
    let normalized = normalize_pecos_helper_symbols(&module.to_string()).map_err(preparation)?;
    Ok(Qis::from_string(normalized))
}

#[cfg(test)]
mod tests {
    use super::*;
    use compiler::hugr::{envelope::EnvelopeConfig, package::Package};
    use compiler::inkwell::memory_buffer::MemoryBuffer;

    #[test]
    fn control_flow_matches_shared_corpus() {
        let directory = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/hugr_control_flow");
        let verdicts: std::collections::BTreeMap<String, bool> =
            serde_json::from_slice(&std::fs::read(directory.join("verdicts.json")).unwrap())
                .unwrap();
        for (file, expected) in verdicts {
            let bytes = std::fs::read(directory.join(&file)).unwrap();
            let (hugr, nontrivial) = parse(&bytes).unwrap();
            assert_eq!(nontrivial, expected, "{file}");
            assert_eq!(has_nontrivial_control_flow(&hugr), expected, "{file}");
        }
    }

    #[test]
    fn parse_requires_exactly_one_module() {
        let (hugr, _) = parse(include_bytes!(
            "../tests/fixtures/hugr_control_flow/straight_line.hugr"
        ))
        .unwrap();
        for (modules, message) in [
            (vec![], "expected one HUGR module"),
            (vec![hugr.clone(), hugr], "expected exactly one HUGR module"),
        ] {
            let mut bytes = Vec::new();
            Package::new(modules)
                .store(&mut bytes, EnvelopeConfig::default())
                .unwrap();
            assert!(parse(&bytes).unwrap_err().to_string().contains(message));
        }
    }

    #[test]
    fn lowers_parities_for_both_platforms() {
        for platform in [crate::QSystemPlatform::Helios, crate::QSystemPlatform::Sol] {
            let qis = lower(
                include_bytes!("../tests/fixtures/dem_programs/parities.hugr"),
                platform,
                crate::OptimizationLevel::Default,
            )
            .unwrap();
            let pecos_programs::QisContent::Ir(ir) = qis.content else {
                panic!("HUGR lowering must produce QIS IR");
            };
            assert!(ir.contains("define"));
        }
    }

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
        module.verify().unwrap();
        let normalized = normalize_pecos_helper_symbols(&module.to_string()).unwrap();
        let normalized_module = context
            .create_module_from_ir(MemoryBuffer::create_from_memory_range_copy(
                std::ffi::CString::new(normalized.as_str())
                    .unwrap()
                    .as_bytes_with_nul(),
                "normalized",
            ))
            .unwrap();
        normalized_module.verify().unwrap();
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
            assert!(normalize_pecos_helper_symbols(&module.to_string()).is_err());
        }
    }

    #[test]
    fn helper_text_validation_agrees_with_llvm_values() {
        let context = Context::create();
        let expected = context
            .i64_type()
            .fn_type(&[context.i64_type().into()], false);
        for declaration in [
            "declare i64 @NAME(i64)",
            "declare ccc i64 @NAME(i64)",
            "declare cc 0 i64 @NAME(i64)",
            "declare fastcc i64 @NAME(i64)",
            "declare cc 1023 i64 @NAME(i64)",
            "declare i32 @NAME(i64)",
            "declare i64 @NAME(i32)",
            "declare i64 @NAME(i64, ...)",
            "declare i64 @NAME(ptr addrspace(1))",
            "define i64 @NAME(i64 %q) { ret i64 %q }",
        ] {
            for name in [
                "pecos_qis_runtime_barrier_qubit_hugr",
                "__hugr__.scope.pecos_qis_runtime_barrier_qubit_hugr.1",
            ] {
                let ir = declaration.replace("NAME", name);
                let module = context
                    .create_module_from_ir(MemoryBuffer::create_from_memory_range_copy(
                        std::ffi::CString::new(ir.as_str())
                            .unwrap()
                            .as_bytes_with_nul(),
                        "abi",
                    ))
                    .unwrap();
                let function = module.get_function(name).unwrap();
                let accepted = function.get_type() == expected
                    && function.count_basic_blocks() == 0
                    && function.get_call_conventions() == 0;
                assert_eq!(
                    normalize_pecos_helper_symbols(&ir).is_ok(),
                    accepted,
                    "{ir}"
                );
                assert_eq!(
                    normalize_pecos_helper_symbols(&module.to_string()).is_ok(),
                    accepted,
                    "{ir}"
                );
            }
        }
    }
}
