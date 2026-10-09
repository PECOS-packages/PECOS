use pecos_core::errors::PecosError;
use pecos_engines::{ClassicalControlEngine, Engine};
use pecos_phir::PhirEngine;
use pecos_phir::ops::{ClassicalOp, Operation, QuantumOp};
use pecos_phir::phir::{AttributeValue, Block, Instruction, Module, SSAValue};
use pecos_phir::types::{IntWidth, Type};

fn classical_engine() -> Box<dyn ClassicalControlEngine> {
    let mut module = Module::new("boxed_classical");
    let value = SSAValue { id: 0, version: 0 };
    let mut block = Block::entry();
    block.add_instruction(Instruction::new(
        Operation::Classical(ClassicalOp::ConstInt(42)),
        vec![],
        vec![value],
        vec![Type::Int(IntWidth::I64)],
    ));
    let mut export = Instruction::new(
        Operation::Classical(ClassicalOp::Result),
        vec![value],
        vec![],
        vec![],
    );
    export.attributes.insert(
        "export_name".to_string(),
        AttributeValue::String("answer".to_string()),
    );
    block.add_instruction(export);
    module.body.add_block(block);
    Box::new(PhirEngine::new(module).unwrap())
}

fn assert_completed(engine: &mut Box<dyn ClassicalControlEngine>) {
    let shot = engine.process(()).unwrap();
    assert_eq!(shot.data["answer"].as_u32(), Some(42));
    let concrete = engine.as_any().downcast_ref::<PhirEngine>().unwrap();
    assert!(concrete.finished);
    assert_eq!(concrete.current_op, 2);
}

#[test]
fn boxed_phir_completes_classical_export() {
    assert_completed(&mut classical_engine());
}

#[test]
fn boxed_phir_reset_reexecutes_classical_program() {
    let mut engine = classical_engine();
    assert_completed(&mut engine);
    Engine::reset(&mut engine).unwrap();
    let concrete = engine.as_any().downcast_ref::<PhirEngine>().unwrap();
    assert_eq!(concrete.current_op, 0);
    assert!(!concrete.finished);
    assert_completed(&mut engine);
}

#[test]
fn boxed_phir_rejects_quantum_commands() {
    let mut module = Module::new("boxed_quantum");
    let mut block = Block::entry();
    block.add_instruction(Instruction::new(
        Operation::Quantum(QuantumOp::H),
        vec![SSAValue { id: 0, version: 0 }],
        vec![],
        vec![],
    ));
    module.body.add_block(block);
    let mut engine: Box<dyn ClassicalControlEngine> = Box::new(PhirEngine::new(module).unwrap());
    let result = engine.process(());
    let Err(PecosError::Processing(message)) = result else {
        panic!("expected quantum processing error, got {result:?}");
    };
    assert_eq!(
        message,
        "Box<dyn ClassicalControlEngine>::process(()) cannot execute quantum commands; use start()/continue_processing() with a quantum engine, or the simulation builder."
    );
}
