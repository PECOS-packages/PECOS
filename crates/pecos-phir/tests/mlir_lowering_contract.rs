use pecos_phir::{
    Module, PhirConfig,
    builtin_ops::FuncOp,
    mlir_lowering::phir_to_mlir,
    ops::{Operation, QuantumOp},
    phir::{Block, Instruction, SSAValue, Terminator},
    types::{FunctionType, Type},
};

fn instruction(op: QuantumOp, operands: u32, results: u32) -> Instruction {
    Instruction::new(
        Operation::Quantum(op),
        (0..operands).map(SSAValue::new).collect(),
        (10..10 + results).map(SSAValue::new).collect(),
        vec![Type::Qubit; results as usize],
    )
}
fn module(instructions: Vec<Instruction>) -> Module {
    let mut func = FuncOp::new(
        "main",
        FunctionType {
            inputs: vec![],
            outputs: vec![],
            variadic: false,
        },
    );
    func.body[0].blocks[0].operations = instructions;
    func.body[0].blocks[0].terminator = Some(Terminator::Return { values: vec![] });
    let mut module = Module::new("test");
    let mut block = Block::entry();
    block.operations.push(Instruction::new(
        func.to_operation(),
        vec![],
        vec![],
        vec![],
    ));
    module.body.blocks.push(block);
    module
}
#[test]
fn unsupported_operations_are_errors() {
    let program = module(vec![instruction(QuantumOp::X, 1, 0)]);
    let error = phir_to_mlir(&program, &PhirConfig::default())
        .expect_err("X is unsupported")
        .to_string();
    assert!(error.contains("quantum.x"), "{error}");
    let mut top_level = Module::new("top");
    let mut block = Block::entry();
    block.operations.push(instruction(QuantumOp::X, 1, 0));
    top_level.body.blocks.push(block);
    assert!(phir_to_mlir(&top_level, &PhirConfig::default()).is_err());
}
#[test]
fn supported_operations_reject_extra_operands() {
    for op in [QuantumOp::H, QuantumOp::CX, QuantumOp::Measure] {
        let name = op.name();
        let program = module(vec![instruction(op, 3, 1)]);
        let error = phir_to_mlir(&program, &PhirConfig::default())
            .expect_err("extra operands must fail")
            .to_string();
        assert!(error.contains(name) && error.contains('3'), "{error}");
    }
}
#[test]
fn supported_program_keeps_qir_calls() {
    let program = module(vec![
        instruction(QuantumOp::Alloc, 0, 1),
        instruction(QuantumOp::H, 1, 0),
        instruction(QuantumOp::CX, 2, 0),
        instruction(QuantumOp::Measure, 1, 1),
    ]);
    let text = phir_to_mlir(&program, &PhirConfig::default()).unwrap();
    for call in [
        "%10 = call @__quantum__rt__qubit_allocate()",
        "call @__quantum__qis__h__body(%0)",
        "call @__quantum__qis__cx__body(%0, %1)",
        "%10 = call @__quantum__qis__m__body(%0, %910)",
        "    return",
    ] {
        assert!(text.contains(call), "{text}");
    }
    assert!(!text.contains("TODO"));
}
