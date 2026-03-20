/// Integration tests that parse real .ll files from the repository.
use pecos_phir::qis_parser::parse_qis_llvm_ir;

#[test]
fn test_parse_bell_ll() {
    let ir = include_str!("../../../examples/llvm/bell.ll");
    let module = parse_qis_llvm_ir(ir).expect("bell.ll should parse");
    assert!(!module.body.blocks.is_empty());
}

#[test]
fn test_parse_qprog_ll() {
    let ir = include_str!("../../../examples/llvm/qprog.ll");
    let module = parse_qis_llvm_ir(ir).expect("qprog.ll should parse");
    // This file has control flow (br i1 / labels)
    assert!(module.body.blocks.len() >= 3);
}

#[test]
fn test_parse_bell_final_ll() {
    let ir = include_str!("../../../examples/bell_final.ll");
    let module = parse_qis_llvm_ir(ir).expect("bell_final.ll should parse");
    assert!(!module.body.blocks.is_empty());
}

#[test]
fn test_parse_hugr_bell_state_ll() {
    let ir = include_str!("../../../crates/pecos/tests/test_data/hugr/bell_state.ll");
    let module = parse_qis_llvm_ir(ir).expect("bell_state.ll should parse");
    assert!(!module.body.blocks.is_empty());
}

#[test]
fn test_parse_arithmetic_ops_ll() {
    let ir = include_str!(
        "../../../python/quantum-pecos/tests/pecos/integration/ll/ArithmeticOps.Targeted.ll"
    );
    let module = parse_qis_llvm_ir(ir).expect("ArithmeticOps.Targeted.ll should parse");
    // This file has many blocks with phi nodes
    assert!(module.body.blocks.len() >= 10);
}

#[test]
fn test_parse_integer_support_ll() {
    let ir = include_str!(
        "../../../python/quantum-pecos/tests/pecos/integration/ll/IntegerSupport.TargetedAlt.ll"
    );
    let module = parse_qis_llvm_ir(ir).expect("IntegerSupport.TargetedAlt.ll should parse");
    assert!(module.body.blocks.len() >= 5);
}
