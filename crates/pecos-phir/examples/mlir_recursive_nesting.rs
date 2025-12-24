//! Example demonstrating MLIR-style recursive nesting of operations and regions
//!
//! Shows how operations can contain regions, which contain blocks, which contain
//! more operations - creating the recursive structure that makes MLIR powerful.

use pecos_phir::{
    ops::{ClassicalOp, ControlFlowOp, Operation, SSAValue, ValueRef},
    phir::{Block, BlockRef, Function, Instruction, Region, Terminator},
    region_kinds::RegionKind,
    types::{FunctionType, IntWidth, Type},
};

fn main() {
    println!("=== MLIR Recursive Nesting Example ===\n");

    // Create a function with nested control flow
    let func = create_nested_function();

    // Show the nested structure
    println!("Function with nested regions:");
    println!("{}", func.to_mlir_text());

    // Show functions as operations
    demonstrate_function_as_operation();
}

/// Create a function demonstrating nested regions
fn create_nested_function() -> Function {
    let signature = FunctionType {
        inputs: vec![Type::Int(IntWidth::I32), Type::Int(IntWidth::I32)],
        outputs: vec![Type::Int(IntWidth::I32)],
        variadic: false,
    };

    let mut func = Function::new_with_visibility(
        "nested_example",
        signature,
        pecos_phir::phir::Visibility::Public,
    );

    // Main function region
    let mut main_region = Region::new(RegionKind::SSACFG);
    let mut entry_block = Block::new(Some("entry".to_string()));

    // Create a conditional operation with nested regions
    let cond_op = create_conditional_with_regions();
    entry_block.add_instruction(cond_op);

    // Create a loop operation with nested regions
    let loop_op = create_loop_with_regions();
    entry_block.add_instruction(loop_op);

    entry_block.set_terminator(Terminator::Return {
        values: vec![SSAValue::new(100)],
    });

    main_region.add_block(entry_block);
    func.body.push(main_region);

    func
}

/// Create a conditional operation with nested regions (if-then-else)
fn create_conditional_with_regions() -> Instruction {
    // The conditional operation itself
    let cond_op = Operation::ControlFlow(ControlFlowOp::Branch(
        pecos_phir::ops::BranchType::Conditional {
            condition: ValueRef::SSA(SSAValue::new(1)),
            then_block: "then_block".to_string(),
            else_block: Some("else_block".to_string()),
        },
    ));

    // Create the "then" region
    let mut then_region = Region::new(RegionKind::SSACFG);
    let mut then_block = Block::new(Some("then_entry".to_string()));

    // Nested operation inside the then block
    let nested_op = Instruction::new(
        Operation::Classical(ClassicalOp::Add),
        vec![SSAValue::new(2), SSAValue::new(3)],
        vec![SSAValue::new(4)],
        vec![Type::Int(IntWidth::I32)],
    );
    then_block.add_instruction(nested_op);

    // The then block can even have another nested conditional!
    let inner_cond = create_inner_conditional();
    then_block.add_instruction(inner_cond);

    then_block.set_terminator(Terminator::Branch {
        target: BlockRef::Parent,
        args: vec![],
    });
    then_region.add_block(then_block);

    // Create the "else" region
    let mut else_region = Region::new(RegionKind::SSACFG);
    let mut else_block = Block::new(Some("else_entry".to_string()));

    let else_op = Instruction::new(
        Operation::Classical(ClassicalOp::Sub),
        vec![SSAValue::new(2), SSAValue::new(3)],
        vec![SSAValue::new(5)],
        vec![Type::Int(IntWidth::I32)],
    );
    else_block.add_instruction(else_op);

    else_block.set_terminator(Terminator::Branch {
        target: BlockRef::Parent,
        args: vec![],
    });
    else_region.add_block(else_block);

    // Create the instruction with both regions
    Instruction::with_regions(
        cond_op,
        vec![SSAValue::new(1)], // condition
        vec![],
        vec![],
        vec![then_region, else_region],
    )
}

/// Create an inner conditional to show deep nesting
fn create_inner_conditional() -> Instruction {
    let inner_cond_op = Operation::ControlFlow(ControlFlowOp::Branch(
        pecos_phir::ops::BranchType::Conditional {
            condition: ValueRef::SSA(SSAValue::new(10)),
            then_block: "inner_then".to_string(),
            else_block: None,
        },
    ));

    let mut inner_region = Region::new(RegionKind::SSACFG);
    let mut inner_block = Block::new(Some("inner_then".to_string()));

    // Even deeper nesting!
    let deep_op = Instruction::new(
        Operation::Classical(ClassicalOp::Mul),
        vec![SSAValue::new(11), SSAValue::new(12)],
        vec![SSAValue::new(13)],
        vec![Type::Int(IntWidth::I32)],
    );
    inner_block.add_instruction(deep_op);

    inner_block.set_terminator(Terminator::Branch {
        target: BlockRef::Parent,
        args: vec![],
    });
    inner_region.add_block(inner_block);

    Instruction::with_regions(
        inner_cond_op,
        vec![SSAValue::new(10)],
        vec![],
        vec![],
        vec![inner_region],
    )
}

/// Create a loop operation with nested regions
fn create_loop_with_regions() -> Instruction {
    let loop_op = Operation::ControlFlow(ControlFlowOp::Loop(pecos_phir::ops::LoopType::While {
        condition: ValueRef::SSA(SSAValue::new(20)),
        body_block: "loop_body".to_string(),
    }));

    // Create the loop body region
    let mut loop_region = Region::new(RegionKind::SSACFG);
    let mut loop_header = Block::new(Some("loop_header".to_string()));
    let mut loop_body = Block::new(Some("loop_body".to_string()));

    // Loop header checks condition
    loop_header.set_terminator(Terminator::ConditionalBranch {
        condition: SSAValue::new(20),
        true_target: BlockRef::by_label("loop_body"),
        true_args: vec![],
        false_target: BlockRef::Parent,
        false_args: vec![],
    });

    // Loop body contains operations
    let increment = Instruction::new(
        Operation::Classical(ClassicalOp::Add),
        vec![SSAValue::new(21), SSAValue::new(22)],
        vec![SSAValue::new(23)],
        vec![Type::Int(IntWidth::I32)],
    );
    loop_body.add_instruction(increment);

    // Loop body can contain more nested structures!
    let nested_in_loop = create_conditional_with_regions();
    loop_body.add_instruction(nested_in_loop);

    loop_body.set_terminator(Terminator::Branch {
        target: BlockRef::by_label("loop_header"),
        args: vec![],
    });

    loop_region.add_block(loop_header);
    loop_region.add_block(loop_body);

    Instruction::with_regions(loop_op, vec![], vec![], vec![], vec![loop_region])
}

/// Demonstrate that functions themselves can be viewed as operations
fn demonstrate_function_as_operation() {
    println!("\n=== Functions as Operations ===\n");

    // In pure MLIR style, a function is just an operation with regions
    let func_op = Operation::ControlFlow(ControlFlowOp::Call(pecos_phir::ops::FunctionCall {
        name: "my_function".to_string(),
        args: vec![],
    }));

    // The function body is a region
    let mut func_region = Region::new(RegionKind::SSACFG);

    // With entry block
    let mut entry = Block::new(Some("entry".to_string()));
    entry.set_terminator(Terminator::Return { values: vec![] });
    func_region.add_block(entry);

    let func_as_op = Instruction::with_regions(func_op, vec![], vec![], vec![], vec![func_region]);

    println!("Function represented as an operation with regions:");
    println!("{}", func_as_op.to_mlir_text(0));
}
