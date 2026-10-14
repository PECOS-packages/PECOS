use pecos_phir::{
    execution::{PhirProcessor, TypedValue},
    ops::{ClassicalOp, Operation, QuantumOp},
};
use pecos_phir_json::{phir_json_to_module, v0_1::operations::OperationProcessor};
use serde_json::json;

fn assert_measurement_value(size: usize, bits: &[usize], outcomes: &[u32], expected: u64) {
    assert_eq!(bits.len(), outcomes.len());
    let returns: Vec<_> = bits.iter().map(|&bit| ("m".to_string(), bit)).collect();
    let args: Vec<_> = (0..bits.len()).map(|qubit| json!(["q", qubit])).collect();
    let input = json!({
        "format": "PHIR/JSON", "version": "0.1.0",
        "ops": [
            {"data":"qvar_define", "data_type":"qubits", "variable":"q", "size":bits.len()},
            {"data":"cvar_define", "data_type":"u32", "variable":"m", "size":size},
            {"qop":"Measure", "args":args, "returns":returns},
            {"cop":"Result", "args":["m"], "returns":["result"]}
        ]
    });

    // Observe the JSON processor's register directly, without engine results.
    let mut processor = OperationProcessor::new();
    processor
        .handle_variable_definition("cvar_define", "u32", "m", size)
        .unwrap();
    processor
        .record_measurement_returns(bits.len(), &returns)
        .unwrap();
    processor.handle_measurements(outcomes, &[]).unwrap();
    assert_eq!(processor.environment.get_raw("m"), Some(expected));

    // The JSON processor does not execute the converter's output. Exercise that
    // output separately using the existing PHIR classical instruction executor.
    // Supply measurement SSA values directly to avoid engine outcome routing.
    let module = phir_json_to_module(&input.to_string()).unwrap();
    let mut classical = PhirProcessor::new();
    let mut outcomes = outcomes.iter();
    let mut register_value = None;
    for instruction in &module.body.blocks[0].operations {
        match &instruction.operation {
            Operation::Quantum(QuantumOp::Measure) => {
                for result in &instruction.results {
                    classical
                        .ssa_values
                        .insert(result.id, TypedValue::Bool(*outcomes.next().unwrap() != 0));
                }
            }
            Operation::Classical(ClassicalOp::Result) => {
                // Read m's value before export, not the engine's final results.
                register_value = Some(
                    classical.ssa_values[&instruction.operands[0].id]
                        .to_u64()
                        .unwrap(),
                );
            }
            Operation::Classical(op) => {
                classical
                    .process_classical_operation(op, instruction)
                    .unwrap();
            }
            _ => {}
        }
    }
    assert!(outcomes.next().is_none());
    assert_eq!(register_value, Some(expected), "converted register m");
}

#[test]
fn single_measurement_bit_one() {
    assert_measurement_value(2, &[1], &[1], 2);
}

#[test]
fn single_measurement_bit_zero() {
    assert_measurement_value(2, &[0], &[1], 1);
}

#[test]
fn single_measurement_bit_three() {
    assert_measurement_value(4, &[3], &[1], 8);
}

#[test]
fn multiple_measurements_bits_zero_and_two() {
    assert_measurement_value(3, &[0, 2], &[1, 1], 5);
    assert_measurement_value(3, &[2, 0], &[1, 1], 5);
}

#[test]
fn zero_measurement_bit_one() {
    assert_measurement_value(2, &[1], &[0], 0);
}
