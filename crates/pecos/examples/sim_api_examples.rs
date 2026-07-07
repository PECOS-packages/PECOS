//! Examples of using the `sim()` API for quantum simulations

use pecos::prelude::*;
use pecos::qis_engine;
use pecos::{sim, sim_builder};
use pecos_engines::{DepolarizingNoise, sim as sim_from, sparse_stab, state_vector};
use pecos_programs::{Qasm, Qis};
use pecos_qasm::qasm_engine;

fn main() -> Result<(), PecosError> {
    // Example 1: Using sim(program) for automatic engine selection
    println!("Example 1: Automatic engine selection");
    let qasm_prog = Qasm::from_string("OPENQASM 2.0; qreg q[1]; h q[0]; measure q[0] -> c[0];");
    let results = sim(qasm_prog)
        .quantum(state_vector())
        .noise(DepolarizingNoise { p: 0.01 })
        .seed(42)
        .shots(50)
        .run()?;
    println!("  Results: {} shots", results.len());

    // Example 2: Different program types
    println!("\nExample 2: Different program types");

    // QASM program
    let qasm_prog = Qasm::from_string("OPENQASM 2.0; qreg q[2]; h q[0]; cx q[0],q[1];");
    let results = sim(qasm_prog)
        .quantum(sparse_stab())
        .seed(42)
        .shots(100)
        .run()?;
    println!("  QASM: {} shots", results.len());

    // LLVM program
    let llvm_prog = Qis::from_string(
        r#"
        declare void @__quantum__qis__h__body(i64)

        define void @main() #0 {
            call void @__quantum__qis__h__body(i64 0)
            ret void
        }

        attributes #0 = { "EntryPoint" }
        "#,
    );
    let results = sim(llvm_prog)
        .qubits(1) // LLVM programs need explicit qubit count
        .shots(50)
        .run()?;
    println!("  LLVM: {} shots", results.len());

    // Example 3: Using sim_builder() for empty builder
    println!("\nExample 3: Empty builder with sim_builder()");
    let results = sim_builder()
        .classical(qasm_engine().qasm("OPENQASM 2.0; qreg q[1]; h q[0];"))
        .run(10)?;
    println!("  Results: {} shots", results.len());

    // Example 4: Override automatic engine selection
    println!("\nExample 4: Override engine selection");
    let qasm_prog = Qasm::from_string("OPENQASM 2.0; qreg q[1]; h q[0];");
    let llvm_prog = Qis::from_string(
        r#"
        declare void @__quantum__qis__h__body(i64)
        declare i32 @__quantum__qis__m__body(i64, i64)
        declare void @__quantum__rt__result_record_output(i64, i8*)

        @.str.result = constant [7 x i8] c"result\00"

        define void @main() #0 {
            call void @__quantum__qis__h__body(i64 0)
            %result = call i32 @__quantum__qis__m__body(i64 0, i64 0)
            call void @__quantum__rt__result_record_output(i64 0, i8* getelementptr inbounds ([7 x i8], [7 x i8]* @.str.result, i32 0, i32 0))
            ret void
        }

        attributes #0 = { "EntryPoint" }
        "#,
    );

    // QASM program but use LLVM engine
    let results = sim(qasm_prog)
        .classical(qis_engine().program(llvm_prog))
        .shots(20)
        .run()?;
    println!("  Results: {} shots", results.len());

    // Example 5: Build once, run multiple times
    println!("\nExample 5: Build once, run multiple");
    let llvm_prog = Qis::from_string(
        r#"
        declare void @__quantum__qis__h__body(i64)

        define void @main() #0 {
            call void @__quantum__qis__h__body(i64 0)
            ret void
        }

        attributes #0 = { "EntryPoint" }
        "#,
    );

    let mut engine = sim(llvm_prog)
        .workers(4) // Default to 4 workers
        .build()?;

    // Run with default workers
    let batch1 = engine.run(100)?;
    println!("  Batch 1: {} shots with default workers", batch1.len());

    // Run with custom worker count
    let batch2 = engine.run_with_workers(200, 8)?;
    println!("  Batch 2: {} shots with 8 workers", batch2.len());

    // Example 6: Using auto_workers()
    println!("\nExample 6: Auto workers");
    let qasm_prog =
        Qasm::from_string("OPENQASM 2.0; qreg q[3]; h q[0]; cx q[0],q[1]; cx q[1],q[2];");
    let results = sim(qasm_prog)
        .auto_workers() // Use all available CPU cores
        .shots(1000)
        .run()?;
    println!("  Results: {} shots with auto workers", results.len());

    // Example 7: Using engine builder with sim_from()
    println!("\nExample 7: Engine builder with sim_from()");
    let results = sim_from(qasm_engine().qasm("OPENQASM 2.0; qreg q[2]; h q[0]; cx q[0],q[1];"))
        .quantum(sparse_stab())
        .seed(42)
        .run(100)?;
    println!("  Results: {} shots", results.len());

    Ok(())
}
