//! Tests for the unified engine builder API

#[test]
fn test_qasm_engine_builder_api() {
    use pecos_engines::sim_builder;
    use pecos_qasm::qasm_engine;

    // Test that the builder has all the expected methods
    let builder = qasm_engine()
        .qasm("OPENQASM 2.0; include \"qelib1.inc\"; qreg q[1]; h q[0];")
        .with_virtual_include("custom.inc", "gate custom a { h a; }")
        .with_include_path("/tmp/includes")
        .allow_complex_conditionals(true);

    // Test that it converts to SimBuilder properly
    let _sim_builder = sim_builder().classical(builder);
}

#[test]
fn test_qasm_engine_builder_from_file() {
    use pecos_engines::ClassicalControlEngineBuilder;
    use pecos_qasm::qasm_engine;
    use std::io::Write;
    use tempfile::NamedTempFile;

    // Create a temporary QASM file
    let mut temp_file = NamedTempFile::new().unwrap();
    writeln!(temp_file, "OPENQASM 2.0;").unwrap();
    writeln!(temp_file, "include \"qelib1.inc\";").unwrap();
    writeln!(temp_file, "qreg q[1];").unwrap();
    writeln!(temp_file, "h q[0];").unwrap();
    temp_file.flush().unwrap();

    // Test building from file
    let builder = qasm_engine().qasm_file(temp_file.path());

    // Test that it can be built
    let engine = builder.build();
    assert!(engine.is_ok());
}

#[cfg(feature = "wasm")]
#[test]
fn test_qasm_engine_builder_with_wasm() {
    use pecos_qasm::qasm_engine;

    // Test that WASM method exists and compiles
    let _builder = qasm_engine()
        .qasm("OPENQASM 2.0; qreg q[1]; custom_gate q[0];")
        .wasm("custom_gates.wasm");

    // Note: We can't actually build this without a real WASM file,
    // but at least we verify the API exists
}

#[test]
fn test_engine_specific_vs_common_methods() {
    use pecos_engines::{ClassicalControlEngineBuilder, DepolarizingNoise, state_vector};
    use pecos_programs::Qasm;
    use pecos_qasm::qasm_engine;

    // Engine-specific methods on QasmEngineBuilder
    let engine_builder = qasm_engine()
        .program(Qasm::from_string("OPENQASM 2.0; qreg q[1];")) // Common: unified program input
        .allow_complex_conditionals(true); // Engine-specific: parser option

    // Common simulation methods on TypedSimBuilder
    let _sim_builder = engine_builder
        .to_sim()
        .seed(42) // Common: random seed
        .workers(4) // Common: parallelization
        .noise(DepolarizingNoise { p: 0.01 }) // Common: noise model
        .quantum(state_vector().qubits(1)); // Common: quantum backend
}
