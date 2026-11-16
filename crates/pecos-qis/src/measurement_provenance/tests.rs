use super::*;
use pecos_engines::{ClassicalControlEngineBuilder, ClassicalEngine};

#[test]
fn capture_gate_preserves_plain_bytes_and_control_manifest() {
    let plain = "; spacing is intentional\n define void @main() {\n ret void\n}\n";
    assert_eq!(
        crate::engine_builder::prepare_program(plain, false, false)
            .unwrap()
            .0,
        plain
    );
    assert!(
        crate::engine_builder::prepare_program(plain, false, true)
            .unwrap()
            .0
            .contains("call void @pecos_prov_begin_v1()")
    );
    let source = include_str!("../../tests/fixtures/runtime_decode_native.ll")
        .replace("@label =", "@ctx = private constant [2 x i8] c\"{}\"\n@label =")
        .replace("declare i64 @pecos_runtime_decode(i64, i64, ptr, i64)", "declare i64 @pecos_runtime_decode_observed_v1(i64, i64, ptr, i64, ptr, i64)")
        .replace("@pecos_runtime_decode(i64 7, i64 %instance, ptr %data, i64 1)", "@pecos_runtime_decode_observed_v1(i64 7, i64 %instance, ptr %data, i64 1, ptr @ctx, i64 2)");
    let (_, plain_manifest) = crate::engine_builder::prepare_program(&source, true, false).unwrap();
    let (traced, traced_manifest) =
        crate::engine_builder::prepare_program(&source, true, true).unwrap();
    assert_eq!(
        serde_json::to_value(plain_manifest).unwrap(),
        serde_json::to_value(traced_manifest).unwrap()
    );
    assert!(traced.contains("call void @pecos_prov_begin_v1()"));
    assert!(traced.contains("call void @pecos_runtime_control_v1"));
}

#[test]
fn late_trace_sink_reports_unavailable_provenance() {
    use pecos_engines::{ControlEngine, Engine, EngineStage, StateVecEngine};
    let source = r#"@label = private constant [6 x i8] c"filled"

declare i64 @___qalloc()
declare i64 @___lazy_measure(i64)
declare i1 @___read_future_bool(i64)
declare void @___qfree(i64)
declare void @llvm.memset.p0.i64(ptr nocapture writeonly, i8, i64, i1 immarg)
declare void @print_bool_selene(ptr, i64, i1)

define i64 @qmain(i64 %shot) {
entry:
  %q = call i64 @___qalloc()
  %future = call i64 @___lazy_measure(i64 %q)
  %m = call i1 @___read_future_bool(i64 %future)
  %byte = zext i1 %m to i8
  %buffer = alloca i8
  call void @llvm.memset.p0.i64(ptr %buffer, i8 %byte, i64 1, i1 false)
  %filled = load i1, ptr %buffer
  call void @print_bool_selene(ptr @label, i64 6, i1 %filled)
  call void @___qfree(i64 %q)
  ret i64 0
}
"#;
    let mut engine = crate::qis_engine()
        .interface(crate::helios_interface_builder())
        .runtime(crate::selene_simple_runtime().unwrap())
        .program_ir(source)
        .build()
        .unwrap();
    let trace = crate::OperationTraceStore::default();
    engine.set_operation_trace_collector(trace.clone());
    engine.set_num_qubits_hint(1);
    let mut quantum = StateVecEngine::new(1);
    let mut stage = engine.start(()).unwrap();
    loop {
        stage = match stage {
            EngineStage::NeedsProcessing(commands) => engine
                .continue_processing(quantum.process(commands).unwrap())
                .unwrap(),
            EngineStage::Complete(_) => break,
        };
    }
    let trace = trace.lock().unwrap();
    let records: Vec<_> = trace.iter().flat_map(|c| &c.named_result_traces).collect();
    assert_eq!(records.len(), 1);
    assert!(records[0].provenance.is_none());
    assert!(records[0].control_dependent.is_none());
    assert_eq!(records[0].result_ids, [] as [usize; 0]);
    assert_eq!(records[0].values, [false]);
    assert_eq!(
        trace
            .last()
            .unwrap()
            .measurement_results
            .values()
            .copied()
            .collect::<Vec<_>>(),
        [0]
    );
}

#[test]
fn parser_accepts_musttail_attributes_and_switch() {
    let source = "define i1 @helper(i1 noundef %v) {\nentry:\n ret i1 %v\n}\ndefine i1 @main(i1 %a) {\nentry:\n %v = musttail call i1 @helper(i1 noundef %a)\n ret i1 %v\n}";
    let ir = instrument(source).unwrap();
    assert_eq!(ir, instrument(source).unwrap());
    assert!(ir.contains("@pecos_prov_op_v1"));
    assert!(!ir.contains("musttail"));
    let switch = "define void @main() {\nentry:\n switch i32 0, label %end [\n i32 1, label %end\n ]\nend:\n ret void\n}";
    assert!(instrument(switch).unwrap().contains("switch i32"));
    // Tail modifiers must never rewrite the contents of quoted symbols.
    let quoted = source.replace("@helper", "@\"tail call helper\"");
    assert!(
        instrument(&quoted)
            .unwrap()
            .contains("@\"tail call helper\"")
    );
    let conversion =
        "@address = constant ptr inttoptr (i64 4 to ptr)\ndefine void @main() { ret void }";
    assert!(
        instrument(conversion)
            .unwrap_err()
            .to_string()
            .contains("integer/pointer conversion")
    );
}

#[test]
fn zero_argument_quantum_call_needs_no_control_operation() {
    let source = "declare i64 @___qalloc()\ndefine void @main() {\n %q = call i64 @___qalloc()\n ret void\n}";
    let ir = instrument(source).unwrap();
    assert!(!ir.contains(&format!("@pecos_prov_op_v1(i64 {},", op::CONTROL)));
}

#[test]
fn integer_vector_copy_does_not_admit_vector_computation() {
    let prefix =
        "define void @main() {\nentry:\n%p = alloca <2 x i64>\n%v = load <2 x i64>, ptr %p\n";
    for instruction in [
        "%w = add <2 x i64> %v, %v",
        "%w = extractelement <2 x i64> %v, i32 0",
        "%w = insertelement <2 x i64> %v, i64 0, i32 0",
        "%w = shufflevector <2 x i64> %v, <2 x i64> %v, <2 x i32> zeroinitializer",
        "%w = select i1 true, <2 x i64> %v, <2 x i64> %v",
        "%w = icmp eq <2 x i64> %v, %v",
        "%w = bitcast <2 x i64> %v to i128",
        "call void @consume(<2 x i64> %v)",
    ] {
        let source =
            format!("declare void @consume(<2 x i64>)\n{prefix}{instruction}\nret void\n}}");
        assert!(instrument(&source).is_err(), "{instruction}");
    }
    let returns = "define <2 x i64> @copy(ptr %p) {\nentry:\n%v = load <2 x i64>, ptr %p\nret <2 x i64> %v\n}\ndefine void @main() {\nret void\n}";
    assert!(instrument(returns).is_err());
    let phi =
        format!("{prefix}br label %next\nnext:\n%w = phi <2 x i64> [%v, %entry]\nret void\n}}");
    assert!(instrument(&phi).is_err());
}

#[test]
fn integer_vector_copies_and_constant_operands_are_admitted() {
    let copies = "define void @main() {\nentry:\n%p = alloca <2 x i64>\n%q = alloca <2 x i64>\nstore <2 x i64> <i64 1, i64 0>, ptr %p\n%v = load <2 x i64>, ptr %p\nstore <2 x i64> %v, ptr %q\nret void\n}";
    assert!(instrument(copies).is_ok());
    for source in [
        "declare void @consume(<2 x i64>)\ndefine void @main() {\nentry:\ncall void @consume(<2 x i64> <i64 1, i64 0>)\nret void\n}",
        "define <2 x i64> @constant() {\nentry:\nret <2 x i64> <i64 1, i64 0>\n}\ndefine void @main() {\nentry:\nret void\n}",
        "define void @main() {\nentry:\n%a = insertvalue {i64, <2 x i64>} zeroinitializer, <2 x i64> <i64 1, i64 0>, 1\nret void\n}",
    ] {
        let outcome = instrument(source);
        assert!(outcome.is_ok(), "{source}: {outcome:?}");
    }
}

#[test]
fn rejected_vector_operands_are_named() {
    for instruction in [
        "call void @consume(<2 x i64> %actual)",
        "%a = insertvalue {<2 x i64>} zeroinitializer, <2 x i64> %actual, 0",
        "ret <2 x i64> %actual",
    ] {
        let end = if instruction.starts_with("ret ") {
            ""
        } else {
            "ret <2 x i64> zeroinitializer"
        };
        let source = format!(
            "declare void @consume(<2 x i64>)\ndefine <2 x i64> @main() {{\nentry:\n%p = alloca <2 x i64>\n%actual = load <2 x i64>, ptr %p\n{instruction}\n{end}\n}}"
        );
        let message = instrument(&source).unwrap_err().to_string();
        assert!(
            message.contains("non-constant vector value %actual"),
            "{message}"
        );
    }
}

#[test]
fn memory_vectors_require_fixed_integer_elements() {
    for ty in ["<vscale x 2 x i64>", "<2 x ptr>", "<2 x float>"] {
        assert!(reject_memory_vector(ty, "SSA").is_err(), "{ty}");
    }
    for ty in ["<2 x i64>", "<2 x i8>", "<3 x i8>"] {
        assert!(reject_memory_vector(ty, "SSA").is_ok(), "{ty}");
    }
}

#[test]
fn literal_vector_store_remains_constant() {
    let source = "define void @main() {\nentry:\n%p = alloca [2 x double]\nstore <2 x double> <double 1.0, double 2.0>, ptr %p, align 8\nret void\n}";
    assert!(instrument(source).is_ok());
}
