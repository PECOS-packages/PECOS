use pecos_phir::ops::{ClassicalOp, Operation};
use pecos_phir::phir::{Instruction, SSAValue};
use pecos_phir::types::{IntWidth, Type};
use pecos_phir_json::phir_json_to_module;
use serde_json::json;

fn qubit_constant(id: u32) -> Instruction {
    Instruction::new(
        Operation::Classical(ClassicalOp::ConstInt(i64::from(id))),
        vec![],
        vec![SSAValue::new(id)],
        vec![Type::Int(IntWidth::I64)],
    )
}

#[test]
fn large_register_defines_only_resolved_qubits() {
    // The small case detects eager emission before a regression can allocate
    // a large instruction list; the million-qubit case checks sparse conversion.
    for size in [10, 1_000_000] {
        let last = size - 1;
        let program = json!({
            "format": "PHIR/JSON", "version": "0.1.0", "ops": [
                {"data": "qvar_define", "variable": "q", "size": size},
                {"qop": "X", "args": [["q", last]], "returns": [["q", last]]},
                {"qop": "CX", "args": [["q", last], ["q", 7]], "returns": []},
                {"qop": "Measure", "args": [["q", 7]], "returns": []}
            ]
        });
        let module = phir_json_to_module(&program.to_string()).unwrap();
        let instructions = &module.body.blocks[0].operations;
        let constants: Vec<_> = instructions
            .iter()
            .filter(|instruction| {
                matches!(
                    instruction.operation,
                    Operation::Classical(ClassicalOp::ConstInt(_))
                )
            })
            .cloned()
            .collect();
        let expected = vec![qubit_constant(7), qubit_constant(last)];
        assert_eq!(constants, expected);
        assert_eq!(instructions[..2], expected);
        assert_eq!(instructions.len(), 6);
        // Constants reuse reserved ids; the register definition keeps the first free id.
        assert_eq!(instructions[2].results[0].id, size);
    }
}
