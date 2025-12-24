//! Demonstration of the unified simulation API
//!
//! This example shows how to use both the base `sim_builder` from pecos-engines
//! and the convenience `sim()` from the pecos meta-crate.

use pecos::sim;
use pecos_engines::{DepolarizingNoise, sim_builder, sparse_stabilizer};
use pecos_programs::Qasm;
use pecos_qasm::qasm_engine;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Example 1: Using base sim_builder with explicit classical engine
    println!("Example 1: Base sim_builder with explicit .classical()");
    let qasm = Qasm::from_string(
        r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        creg c[2];
        h q[0];
        cx q[0], q[1];
        measure q -> c;
    "#,
    );

    let results = sim_builder()
        .classical(qasm_engine().program(qasm))
        .seed(42)
        .quantum(sparse_stabilizer())
        .noise(DepolarizingNoise { p: 0.001 })
        .run(1000)?;

    println!("  Ran {} shots", results.len());

    // Example 2: Using convenience sim() with auto-selection
    println!("\nExample 2: Convenience sim() with auto-selection");
    let qasm2 = Qasm::from_string(
        r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[3];
        creg c[3];
        h q[0];
        h q[1];
        cx q[0], q[2];
        cx q[1], q[2];
        measure q -> c;
    "#,
    );

    let results2 = sim(qasm2)
        .seed(123)
        .workers(4)
        .quantum(sparse_stabilizer())
        .run(500)?;

    println!("  Ran {} shots", results2.len());

    // Example 3: Override auto-selection with different engine
    println!("\nExample 3: Override auto-selection");
    let qasm3 = Qasm::from_string(
        r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        creg c[1];
        x q[0];
        measure q -> c;
    "#,
    );

    // Even though we provide a QASM program, we can override to use a different engine
    // (though in practice you'd use the auto-selected one)
    let results3 = sim(qasm3.clone())
        .classical(qasm_engine().program(qasm3))
        .verbose(true)
        .run(100)?;

    println!("  Ran {} shots", results3.len());

    Ok(())
}
