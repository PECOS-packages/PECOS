//! Final simplified `sim()` API examples

use pecos::prelude::*;
use pecos::qis_engine;
use pecos::{sim, sim_builder};
use pecos_engines::{DepolarizingNoise, sparse_stab, state_vector};
use pecos_programs::{Qasm, Qis};
use pecos_qasm::qasm_engine;

fn main() -> Result<(), PecosError> {
    println!("PECOS Simplified Simulation API Examples\n");

    // The primary API: sim(program)
    println!("1. Primary API - sim(program) with automatic engine selection:");

    let qasm_prog = Qasm::from_string(
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

    let results = sim(qasm_prog)
        .quantum(state_vector())
        .noise(DepolarizingNoise { p: 0.01 })
        .seed(42)
        .workers(4)
        .run(1000)?;

    println!(
        "   Bell state simulation: {} shots completed",
        results.len()
    );

    // Build once, run multiple times
    println!("\n2. Build once, run multiple times pattern:");

    let qasm_prog = Qasm::from_string(
        r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        creg c[1];
        h q[0];
        measure q[0] -> c[0];
    "#,
    );

    let mut engine = sim(qasm_prog)
        .seed(42)
        .workers(4) // Default workers
        .build()?;

    let batch1 = engine.run(100)?;
    let batch2 = engine.run(500)?;
    let batch3 = engine.run_with_workers(1000, 8)?; // Override workers

    println!("   Batch 1: {} shots", batch1.len());
    println!("   Batch 2: {} shots", batch2.len());
    println!("   Batch 3: {} shots with 8 workers", batch3.len());

    // Manual configuration with sim_builder()
    println!("\n3. Manual configuration with sim_builder():");

    let results = sim_builder()
        .classical(qasm_engine().qasm(
            r#"
            OPENQASM 2.0;
            include "qelib1.inc";
            qreg q[3];
            creg c[3];
            h q[0];
            cx q[0], q[1];
            cx q[1], q[2];
            measure q -> c;
        "#,
        ))
        .quantum(sparse_stab())
        .auto_workers()
        .run(200)?;

    println!("   GHZ state simulation: {} shots", results.len());

    // Override automatic engine selection
    println!("\n4. Override automatic engine selection:");

    let qasm_prog = Qasm::from_string("OPENQASM 2.0; qreg q[1];");
    let llvm_prog = Qis::from_string(
        r#"
        define void @main() #0 { ret void }
        attributes #0 = { "EntryPoint" }
    "#,
    );

    // QASM program but use LLVM engine
    let results = sim(qasm_prog)
        .classical(qis_engine().program(llvm_prog))
        .qubits(1)
        .run(10)?;

    println!("   Override engine: {} shots", results.len());

    println!("\nAll examples completed successfully!");
    Ok(())
}
