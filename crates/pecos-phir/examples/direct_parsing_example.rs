//! Example of parsing directly to PMIR without a separate AST
//!
//! This shows how we handle various parsing challenges using MLIR-style ops

use pecos_phir::{
    builtin_ops::{FuncOp, ModuleOp},
    phir::{Block, Region},
    types::{FunctionType, IntWidth, Type},
};

fn main() {
    println!("=== Direct PMIR Parsing Example ===\n");

    // Example 1: Parsing a module with forward references
    example_forward_references();

    // Example 2: Gradual type inference
    example_type_inference();

    // Example 3: High-level control flow
    example_control_flow();
}

/// Example 1: Handle forward references during parsing
fn example_forward_references() {
    println!("1. Forward References Example");
    println!("-----------------------------");

    // Parsing this code:
    // ```
    // module @quantum_program {
    //   func @main() -> i32 {
    //     %result = call @helper() : () -> i32  // Forward reference!
    //     return %result : i32
    //   }
    //
    //   func @helper() -> i32 {
    //     %c42 = arith.constant 42 : i32
    //     return %c42 : i32
    //   }
    // }
    // ```

    let mut module = ModuleOp::new("quantum_program");

    // Phase 1: First pass - collect declarations
    let mut forward_decls = std::collections::BTreeMap::new();
    forward_decls.insert(
        "helper",
        FunctionType {
            inputs: vec![],
            outputs: vec![Type::Int(IntWidth::I32)],
            variadic: false,
        },
    );

    // Phase 2: Parse main function with unresolved call
    let main_func = {
        let func = FuncOp::new(
            "main",
            FunctionType {
                inputs: vec![],
                outputs: vec![Type::Int(IntWidth::I32)],
                variadic: false,
            },
        );

        // During parsing, we create a placeholder for the forward reference
        // In real implementation, this would be UnresolvedCall
        println!("  - Creating forward reference to @helper");

        func
    };

    // Phase 3: Parse helper function
    let helper_func = FuncOp::new("helper", forward_decls["helper"].clone());

    // Phase 4: Resolution pass - resolve all forward references
    println!("  - Resolving forward references...");
    module.add_function(main_func);
    module.add_function(helper_func);

    println!("  Successfully parsed with forward references\n");
}

#[derive(Debug)]
#[allow(dead_code)]
struct TypeVar(u32);

#[derive(Debug)]
#[allow(dead_code)]
enum InferredType {
    Known(Type),
    Unknown(TypeVar),
}

/// Example 2: Type inference during parsing
fn example_type_inference() {
    println!("2. Type Inference Example");
    println!("-------------------------");

    // Parsing code with type inference:
    // ```
    // func @infer_types(%x: ?) -> ? {
    //   %y = arith.constant 42        // Infer %y : i32
    //   %z = arith.addi %x, %y : ?    // Infer %x : i32, %z : i32
    //   return %z : ?                  // Infer return type i32
    // }
    // ```

    // During parsing, we create type variables
    let mut type_var_counter = 0;
    let mut new_type_var = || {
        let tv = TypeVar(type_var_counter);
        type_var_counter += 1;
        InferredType::Unknown(tv)
    };

    let x_type = new_type_var();
    let return_type = new_type_var();

    println!("  - Created type variables: {x_type:?}, {return_type:?}");

    // Collect constraints during parsing
    let mut constraints = vec![];

    // From: %y = arith.constant 42
    // y_type is implicitly i32 from the constant
    let _ = InferredType::Known(Type::Int(IntWidth::I32));

    // From: %z = arith.addi %x, %y
    constraints.push("x_type must equal i32 (from addi operation)");
    constraints.push("z_type must equal i32 (from addi operation)");

    // From: return %z
    constraints.push("return_type must equal z_type");

    println!("  - Collected constraints:");
    for c in &constraints {
        println!("    • {c}");
    }

    // Type inference solver would run here
    println!("  - Running type inference...");
    println!("  Inferred: %x : i32, return type: i32\n");
}

/// Example 3: High-level control flow
fn example_control_flow() {
    println!("3. High-Level Control Flow Example");
    println!("----------------------------------");

    // Parsing high-level control flow:
    // ```
    // func @quantum_loop(%n: i32) {
    //   for %i = 0 to %n {
    //     %q = quantum.alloc : !quantum.qubit
    //     quantum.h %q : !quantum.qubit
    //     quantum.measure %q : !quantum.qubit -> i1
    //   }
    // }
    // ```

    // During parsing, we create high-level loop operation
    println!("  - Parsing for-loop as high-level operation");

    // This would be represented as a ForLoop parsing op with:
    // - Induction variable: %i
    // - Range: 0 to %n
    // - Body region containing quantum operations

    let mut loop_region = Region::new(pecos_phir::region_kinds::RegionKind::SSACFG);
    let loop_body = Block::new(Some("loop.body".to_string()));

    // The loop body would contain the quantum operations
    println!("  - Loop body contains quantum operations");

    loop_region.add_block(loop_body);

    // Later lowering pass would convert to:
    // ```
    // ^entry:
    //   %c0 = arith.constant 0 : i32
    //   br ^loop.header(%c0 : i32)
    //
    // ^loop.header(%i: i32):
    //   %cond = arith.cmpi "slt", %i, %n : i32
    //   cond_br %cond, ^loop.body, ^loop.exit
    //
    // ^loop.body:
    //   // ... quantum operations ...
    //   %next_i = arith.addi %i, %c1 : i32
    //   br ^loop.header(%next_i : i32)
    //
    // ^loop.exit:
    //   return
    // ```

    println!("  - Will be lowered to CFG during optimization");
    println!("  Successfully represented high-level control flow\n");
}
