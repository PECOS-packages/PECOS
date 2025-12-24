//! Example demonstrating AST-like PHIR that progressively lowers to SSA form
//!
//! This example shows how we can represent quantum programs at different
//! abstraction levels within the same MLIR framework.

fn main() {
    println!("=== AST to SSA Progressive Lowering Example ===\n");

    // Show the same quantum teleportation circuit at different lowering stages
    demonstrate_progressive_lowering();
}

fn demonstrate_progressive_lowering() {
    println!("Quantum Teleportation Circuit - Progressive Lowering\n");

    // Stage 1: AST-like representation (what the parser might produce)
    println!("Stage 1: AST-like PHIR (Direct from Parser)");
    println!("=========================================");
    show_ast_representation();

    println!("\n\nStage 2: Resolved PHIR (Names and Types Resolved)");
    println!("=================================================");
    show_resolved_representation();

    println!("\n\nStage 3: SSA Form (Ready for Optimization)");
    println!("==========================================");
    show_ssa_representation();
}

/// Stage 1: AST-like representation with unresolved names and high-level constructs
fn show_ast_representation() {
    println!(
        r#"
module @quantum_teleportation {{
  // AST-like: Variable declarations with inferred types
  "parse.var_decl"() {{name = "alice_qubit", type = "qubit"}} : () -> ()
  "parse.var_decl"() {{name = "bob_qubit", type = "qubit"}} : () -> ()
  "parse.var_decl"() {{name = "msg_qubit", type = "qubit",
                      init = "parse.quantum_state"() {{state = "|ψ⟩"}}}} : () -> ()

  "parse.function_def"() {{
    name = "teleport",
    params = [],
    body = [{{
      // Create entangled pair (AST preserves high-level intent)
      "parse.quantum_protocol"() {{
        protocol = "bell_pair",
        args = ["alice_qubit", "bob_qubit"]
      }} : () -> ()

      // Alice's operations (structured, not CFG)
      "parse.scope_begin"() {{name = "alice_operations"}} : () -> ()

      "parse.quantum_gate"() {{
        gate = "CNOT",
        qubits = ["msg_qubit", "alice_qubit"]
      }} : () -> ()

      "parse.quantum_gate"() {{
        gate = "H",
        qubits = ["msg_qubit"]
      }} : () -> ()

      // Measurement with conditional (high-level if-then)
      %m1 = "parse.measurement"() {{
        qubit = "msg_qubit",
        basis = "computational"
      }} : () -> !parse.unknown

      %m2 = "parse.measurement"() {{
        qubit = "alice_qubit",
        basis = "computational"
      }} : () -> !parse.unknown

      "parse.scope_end"() : () -> ()

      // Bob's operations (AST-like conditional)
      "parse.if_else"(%m2) {{
        then = [{{
          "parse.quantum_gate"() {{gate = "X", qubits = ["bob_qubit"]}} : () -> ()
        }}],
        else = []
      }} : (!parse.unknown) -> ()

      "parse.if_else"(%m1) {{
        then = [{{
          "parse.quantum_gate"() {{gate = "Z", qubits = ["bob_qubit"]}} : () -> ()
        }}],
        else = []
      }} : (!parse.unknown) -> ()

      // Return teleported state
      "parse.return"() {{value = "bob_qubit"}} : () -> ()
    }}]
  }} : () -> ()
}}
"#
    );
}

/// Stage 2: After name resolution and type inference
fn show_resolved_representation() {
    println!(
        r#"
module @quantum_teleportation {{
  // Resolved: Concrete types and allocations
  %alice = quantum.alloc : !quantum.qubit
  %bob = quantum.alloc : !quantum.qubit
  %msg = quantum.alloc : !quantum.qubit
  quantum.init %msg, "|ψ⟩" : !quantum.qubit

  func @teleport() -> !quantum.qubit {{
    // Bell pair protocol expanded but still structured
    quantum.h %alice : !quantum.qubit
    quantum.cnot %alice, %bob : !quantum.qubit, !quantum.qubit

    // Alice's operations (still using high-level control flow)
    quantum.cnot %msg, %alice : !quantum.qubit, !quantum.qubit
    quantum.h %msg : !quantum.qubit

    %m1 = quantum.measure %msg : !quantum.qubit -> i1
    %m2 = quantum.measure %alice : !quantum.qubit -> i1

    // Structured control flow (scf dialect)
    scf.if %m2 {{
      quantum.x %bob : !quantum.qubit
    }}

    scf.if %m1 {{
      quantum.z %bob : !quantum.qubit
    }}

    return %bob : !quantum.qubit
  }}
}}
"#
    );
}

/// Stage 3: Fully lowered to SSA form with CFG
fn show_ssa_representation() {
    println!(
        r"
module @quantum_teleportation {{
  // SSA form: Explicit memory and control flow
  %0 = llvm.mlir.global @alice_qubit : !llvm.ptr
  %1 = llvm.mlir.global @bob_qubit : !llvm.ptr
  %2 = llvm.mlir.global @msg_qubit : !llvm.ptr

  func @teleport() -> !llvm.ptr {{
    // Allocate qubits
    %alice_ptr = call @__quantum__rt__qubit_allocate() : () -> !llvm.ptr
    %bob_ptr = call @__quantum__rt__qubit_allocate() : () -> !llvm.ptr
    %msg_ptr = call @__quantum__rt__qubit_allocate() : () -> !llvm.ptr

    // Initialize message qubit (would be more complex in practice)
    call @__quantum__rt__qubit_init(%msg_ptr) : (!llvm.ptr) -> ()

    // Create Bell pair
    call @__quantum__qis__h__body(%alice_ptr) : (!llvm.ptr) -> ()
    call @__quantum__qis__cnot__body(%alice_ptr, %bob_ptr) : (!llvm.ptr, !llvm.ptr) -> ()

    // Alice's operations
    call @__quantum__qis__cnot__body(%msg_ptr, %alice_ptr) : (!llvm.ptr, !llvm.ptr) -> ()
    call @__quantum__qis__h__body(%msg_ptr) : (!llvm.ptr) -> ()

    // Measurements
    %result1_ptr = call @__quantum__rt__result_get_zero() : () -> !llvm.ptr
    call @__quantum__qis__mz__body(%msg_ptr, %result1_ptr) : (!llvm.ptr, !llvm.ptr) -> ()
    %m1 = call @__quantum__qis__read_result__body(%result1_ptr) : (!llvm.ptr) -> i1

    %result2_ptr = call @__quantum__rt__result_get_zero() : () -> !llvm.ptr
    call @__quantum__qis__mz__body(%alice_ptr, %result2_ptr) : (!llvm.ptr, !llvm.ptr) -> ()
    %m2 = call @__quantum__qis__read_result__body(%result2_ptr) : (!llvm.ptr) -> i1

    // CFG for conditionals
    llvm.cond_br %m2, ^apply_x, ^check_z

  ^apply_x:
    call @__quantum__qis__x__body(%bob_ptr) : (!llvm.ptr) -> ()
    llvm.br ^check_z

  ^check_z:
    llvm.cond_br %m1, ^apply_z, ^done

  ^apply_z:
    call @__quantum__qis__z__body(%bob_ptr) : (!llvm.ptr) -> ()
    llvm.br ^done

  ^done:
    llvm.return %bob_ptr : !llvm.ptr
  }}
}}
"
    );
}
