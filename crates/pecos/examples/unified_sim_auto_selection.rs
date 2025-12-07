//! Demonstration of automatic engine selection based on program type
//!
//! This example shows how the pecos `sim()` function automatically selects
//! the appropriate classical engine based on the program type.

use pecos::sim;
use pecos_engines::{sparse_stabilizer, state_vector};
use pecos_programs::{Hugr, Qasm, Qis};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Example 1: QASM program automatically uses QASM engine
    println!("Example 1: QASM program -> QASM engine (automatic)");
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

    let results = sim(qasm_prog).seed(42).quantum(state_vector()).run(100)?;

    println!("  Ran {} shots for QASM program", results.len());

    // Example 2: LLVM program automatically uses LLVM engine
    println!("\nExample 2: LLVM program -> LLVM engine (automatic)");
    // Note: LLVM programs require specific format with EntryPoint attribute
    // For this demo, we'll use bitcode instead
    let _llvm_prog = Qis::from_bitcode(vec![0x42, 0x43]); // BC magic number

    // Note: Since this is not valid bitcode, this would fail at runtime.
    // In a real scenario, you'd use proper LLVM bitcode.
    println!("  (Skipping LLVM execution - would use LLVM engine automatically)");

    // Example 3: HUGR program automatically uses Selene engine
    println!("\nExample 3: HUGR program -> Selene engine (automatic)");
    // Note: HUGR programs use serialized HUGR format
    let _hugr_prog = Hugr::from_bytes(vec![0x48, 0x55, 0x47, 0x52]);

    // Note: Since this is not valid HUGR, this would fail at runtime.
    // In a real scenario, you'd use proper HUGR serialization.
    println!("  (Skipping HUGR execution - would use Selene engine automatically)");

    // Example 4: Demonstrating configuration propagation
    println!("\nExample 4: All configuration options work with auto-selection");
    let qasm_prog2 = Qasm::from_string(
        r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        creg c[1];
        h q[0];
        measure q -> c;
    "#,
    );

    let results4 = sim(qasm_prog2)
        .seed(789)
        .workers(2)
        .verbose(false)
        .quantum(sparse_stabilizer())
        .run(200)?;

    println!("  Ran {} shots with custom configuration", results4.len());

    println!("\nAll examples completed successfully!");
    Ok(())
}
