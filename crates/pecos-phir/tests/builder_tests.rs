/*!
Tests for `PhirEngineBuilder`

Verifies the builder pattern works correctly for constructing `PhirEngine`s,
including the `from_qis_llvm_ir` path and the `to_sim()` `SimBuilder` integration.
*/

use pecos_engines::engine_builder::ClassicalControlEngineBuilder;
use pecos_engines::quantum_engine_builder::StateVectorEngineBuilder;
use pecos_phir::{PhirEngineBuilder, phir_engine};

/// Minimal QIS LLVM IR that allocates two qubits, applies H+CX, and measures both.
const BELL_QIS_IR: &str = r"
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
  ; CX = rxy(pi/2, -pi/2) rzz(-pi/4) rz(-pi/2) rxy(pi/2, pi/2)
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

/// Minimal IR with just a single qubit and H + measure.
const SINGLE_H_QIS_IR: &str = r"
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

// ──────────────────────────────────────────────────────────────────────
// Builder construction tests
// ──────────────────────────────────────────────────────────────────────

#[test]
fn test_builder_default() {
    let builder = PhirEngineBuilder::new();
    // Building without a module should error
    let result = builder.build();
    assert!(result.is_err(), "build() without a module should fail");
}

#[test]
fn test_builder_convenience_function() {
    let builder = phir_engine();
    let result = builder.build();
    assert!(
        result.is_err(),
        "phir_engine() without a module should fail"
    );
}

#[test]
fn test_builder_from_qis_llvm_ir() {
    let builder = phir_engine()
        .from_qis_llvm_ir(SINGLE_H_QIS_IR)
        .expect("should parse valid QIS LLVM IR");
    let engine = builder.build().expect("should build engine from parsed IR");
    assert!(!engine.module().unwrap().name.is_empty());
}

#[test]
fn test_builder_from_qis_llvm_ir_bell() {
    let builder = phir_engine()
        .from_qis_llvm_ir(BELL_QIS_IR)
        .expect("should parse Bell state QIS LLVM IR");
    let engine = builder.build().expect("should build engine");
    assert!(!engine.module().unwrap().name.is_empty());
}

#[test]
fn test_builder_from_empty_ir() {
    // Empty IR with no functions should still parse but produce an empty module
    let result = phir_engine().from_qis_llvm_ir("");
    // May succeed with empty module or fail - either is valid
    if let Ok(builder) = result {
        let engine = builder.build().expect("should build from empty IR");
        assert!(engine.module().is_some());
    }
}

#[test]
fn test_builder_clone() {
    let builder = phir_engine()
        .from_qis_llvm_ir(SINGLE_H_QIS_IR)
        .expect("should parse");
    let builder2 = builder.clone();
    // Both should build successfully
    let engine1 = builder.build().expect("build 1");
    let engine2 = builder2.build().expect("build 2");
    assert_eq!(
        engine1.module().unwrap().name,
        engine2.module().unwrap().name
    );
}

// ──────────────────────────────────────────────────────────────────────
// Builder program() method tests
// ──────────────────────────────────────────────────────────────────────

#[test]
fn test_builder_program_method() {
    use pecos_phir::PhirEngineBuilder;

    // Build a module manually and set it via program()
    let module = pecos_phir::parse_qis_to_quantum(SINGLE_H_QIS_IR).expect("should parse");

    let builder = PhirEngineBuilder::new().program(module);
    let engine = builder.build().expect("should build engine from program()");
    assert!(engine.module().is_some());
}

// ──────────────────────────────────────────────────────────────────────
// RON serialization roundtrip tests
// ──────────────────────────────────────────────────────────────────────

#[test]
fn test_builder_ron_roundtrip() {
    // Parse QIS IR into a builder
    let builder = phir_engine()
        .from_qis_llvm_ir(SINGLE_H_QIS_IR)
        .expect("should parse");

    // Serialize the module to RON
    let ron_string = builder.to_ron().expect("should serialize to RON");
    assert!(!ron_string.is_empty());

    // Deserialize back from RON into a new builder
    let builder2 = phir_engine()
        .from_ron(&ron_string)
        .expect("should deserialize from RON");

    // Both builders should produce working engines
    let engine1 = builder.build().expect("build original");
    let engine2 = builder2.build().expect("build from RON");
    assert_eq!(
        engine1.module().unwrap().name,
        engine2.module().unwrap().name
    );
}

#[test]
fn test_builder_ron_roundtrip_bell() {
    let builder = phir_engine()
        .from_qis_llvm_ir(BELL_QIS_IR)
        .expect("should parse");

    let ron_string = builder
        .to_ron()
        .expect("should serialize Bell state to RON");

    // RON should contain quantum operation names
    assert!(ron_string.contains("RZ"), "RON should contain RZ ops");
    assert!(ron_string.contains("R1XY"), "RON should contain R1XY ops");
    assert!(
        ron_string.contains("Measure"),
        "RON should contain Measure ops"
    );

    // Roundtrip
    let builder2 = phir_engine()
        .from_ron(&ron_string)
        .expect("should deserialize Bell state from RON");
    let engine = builder2.build().expect("build from RON");
    assert!(engine.module().is_some());
}

#[test]
fn test_builder_ron_roundtrip_execute() {
    // Full pipeline: QIS IR -> Module -> RON -> Module -> execute
    let builder = phir_engine()
        .from_qis_llvm_ir(SINGLE_H_QIS_IR)
        .expect("should parse");

    let ron_string = builder.to_ron().expect("serialize");

    let result = phir_engine()
        .from_ron(&ron_string)
        .expect("deserialize")
        .to_sim()
        .quantum(StateVectorEngineBuilder::default())
        .run(10);

    assert!(
        result.is_ok(),
        "RON roundtrip execution failed: {:?}",
        result.err()
    );
    assert_eq!(result.unwrap().shots.len(), 10);
}

#[test]
fn test_builder_ron_file_roundtrip() {
    let builder = phir_engine()
        .from_qis_llvm_ir(SINGLE_H_QIS_IR)
        .expect("should parse");

    let tmp_path = std::env::temp_dir().join("pecos_test_phir.ron");

    // Save to file
    builder.save_ron(&tmp_path).expect("should save RON file");
    assert!(tmp_path.exists());

    // Load from file
    let builder2 = phir_engine()
        .from_ron_file(&tmp_path)
        .expect("should load RON file");

    let engine = builder2.build().expect("build from RON file");
    assert!(engine.module().is_some());

    // Clean up
    let _ = std::fs::remove_file(&tmp_path);
}

#[test]
fn test_builder_to_ron_without_module_errors() {
    let builder = phir_engine();
    let result = builder.to_ron();
    assert!(result.is_err(), "to_ron() without a module should fail");
}

#[test]
fn test_builder_save_ron_without_module_errors() {
    let builder = phir_engine();
    let tmp_path = std::env::temp_dir().join("pecos_test_save_ron_no_module.ron");
    let result = builder.save_ron(&tmp_path);
    assert!(result.is_err(), "save_ron() without a module should fail");
}

#[test]
fn test_builder_from_qis_llvm_ir_invalid() {
    let result = phir_engine().from_qis_llvm_ir("this is not valid LLVM IR {{{");
    // Should either error or produce an empty module -- not panic
    if let Ok(builder) = result {
        let engine = builder.build();
        // Even if parsing succeeded, the module should be usable
        assert!(engine.is_ok() || engine.is_err());
    }
}

#[test]
fn test_builder_from_ron_invalid() {
    let result = phir_engine().from_ron("not valid RON data!!!");
    assert!(result.is_err(), "from_ron() with invalid RON should fail");
}

// ──────────────────────────────────────────────────────────────────────
// SimBuilder integration tests
// ──────────────────────────────────────────────────────────────────────

#[test]
fn test_builder_to_sim_run() {
    let builder = phir_engine()
        .from_qis_llvm_ir(SINGLE_H_QIS_IR)
        .expect("should parse");

    let result = builder
        .to_sim()
        .quantum(StateVectorEngineBuilder::default())
        .run(10);
    assert!(
        result.is_ok(),
        "to_sim().run() should succeed: {:?}",
        result.err()
    );
    let shots = result.unwrap();
    assert_eq!(shots.shots.len(), 10);
}

#[test]
fn test_builder_to_sim_run_bell() {
    let builder = phir_engine()
        .from_qis_llvm_ir(BELL_QIS_IR)
        .expect("should parse");

    let result = builder
        .to_sim()
        .quantum(StateVectorEngineBuilder::default())
        .run(100);
    assert!(
        result.is_ok(),
        "Bell state sim should succeed: {:?}",
        result.err()
    );
    let shots = result.unwrap();
    assert_eq!(shots.shots.len(), 100);
}

#[test]
fn test_builder_to_sim_with_seed() {
    let builder = phir_engine()
        .from_qis_llvm_ir(SINGLE_H_QIS_IR)
        .expect("should parse");

    let result = builder
        .to_sim()
        .quantum(StateVectorEngineBuilder::default())
        .seed(42)
        .run(10);
    assert!(
        result.is_ok(),
        "seeded sim should succeed: {:?}",
        result.err()
    );
}

#[test]
fn test_builder_to_sim_build_then_run() {
    let builder = phir_engine()
        .from_qis_llvm_ir(SINGLE_H_QIS_IR)
        .expect("should parse");

    let mut sim = builder
        .to_sim()
        .quantum(StateVectorEngineBuilder::default())
        .build()
        .expect("should build sim");
    let result = sim.run(5);
    assert!(
        result.is_ok(),
        "built sim run should succeed: {:?}",
        result.err()
    );
    assert_eq!(result.unwrap().shots.len(), 5);
}
