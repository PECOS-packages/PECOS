use std::f64::consts::PI;

use pecos_core::errors::PecosError;
use pecos_engines::byte_message::GateType;
use pecos_engines::prelude::*;
use selene_core::runtime::{BatchOperation, Operation};
use selene_core::simulator::SimulatorInterface;

pub struct SeleneSimulator;

impl SeleneSimulator {
    fn error(context: &str, error: impl std::fmt::Display) -> PecosError {
        PecosError::Generic(format!("Selene simulator {context} failed: {error}"))
    }

    pub fn process(
        simulator: &mut dyn SimulatorInterface,
        message: &ByteMessage,
    ) -> Result<ByteMessage, PecosError> {
        let mut operations = Vec::new();
        let mut measurement_count = 0_usize;

        for gate in message.quantum_ops()? {
            let qubits = gate
                .qubits
                .iter()
                .map(|qubit| u64::try_from(usize::from(*qubit)))
                .collect::<Result<Vec<_>, _>>()
                .map_err(|error| Self::error("qubit conversion", error))?;
            let angle = |index: usize| gate.angles[index].to_radians_signed();
            match gate.gate_type {
                GateType::X => {
                    operations.extend(qubits.iter().map(|&qubit_id| Operation::RXYGate {
                        qubit_id,
                        theta: PI,
                        phi: 0.0,
                    }));
                }
                GateType::Y => {
                    operations.extend(qubits.iter().map(|&qubit_id| Operation::RXYGate {
                        qubit_id,
                        theta: PI,
                        phi: PI / 2.0,
                    }));
                }
                GateType::Z => {
                    operations.extend(qubits.iter().map(|&qubit_id| Operation::RZGate {
                        qubit_id,
                        theta: PI,
                    }));
                }
                GateType::H => {
                    operations.extend(qubits.iter().map(|&qubit_id| Operation::RXYGate {
                        qubit_id,
                        theta: PI / 2.0,
                        phi: -PI / 2.0,
                    }));
                }
                GateType::RX => {
                    operations.extend(qubits.iter().map(|&qubit_id| Operation::RXYGate {
                        qubit_id,
                        theta: angle(0),
                        phi: 0.0,
                    }));
                }
                GateType::RY => {
                    operations.extend(qubits.iter().map(|&qubit_id| Operation::RXYGate {
                        qubit_id,
                        theta: angle(0),
                        phi: PI / 2.0,
                    }));
                }
                GateType::RZ => {
                    operations.extend(qubits.iter().map(|&qubit_id| Operation::RZGate {
                        qubit_id,
                        theta: angle(0),
                    }));
                }
                GateType::R1XY => {
                    operations.extend(qubits.iter().map(|&qubit_id| Operation::RXYGate {
                        qubit_id,
                        theta: angle(0),
                        phi: angle(1),
                    }));
                }
                GateType::RZZ => {
                    operations.extend(qubits.chunks_exact(2).map(|pair| Operation::RZZGate {
                        qubit_id_1: pair[0],
                        qubit_id_2: pair[1],
                        theta: angle(0),
                    }));
                }
                GateType::SZZ => {
                    operations.extend(qubits.chunks_exact(2).map(|pair| Operation::RZZGate {
                        qubit_id_1: pair[0],
                        qubit_id_2: pair[1],
                        theta: PI / 2.0,
                    }));
                }
                GateType::SZZdg => {
                    operations.extend(qubits.chunks_exact(2).map(|pair| Operation::RZZGate {
                        qubit_id_1: pair[0],
                        qubit_id_2: pair[1],
                        theta: -PI / 2.0,
                    }));
                }
                GateType::SZ => {
                    operations.extend(qubits.iter().map(|&qubit_id| Operation::RZGate {
                        qubit_id,
                        theta: PI / 2.0,
                    }));
                }
                GateType::SZdg => {
                    operations.extend(qubits.iter().map(|&qubit_id| Operation::RZGate {
                        qubit_id,
                        theta: -PI / 2.0,
                    }));
                }
                GateType::T => {
                    operations.extend(qubits.iter().map(|&qubit_id| Operation::RZGate {
                        qubit_id,
                        theta: PI / 4.0,
                    }));
                }
                GateType::Tdg => {
                    operations.extend(qubits.iter().map(|&qubit_id| Operation::RZGate {
                        qubit_id,
                        theta: -PI / 4.0,
                    }));
                }
                GateType::MZ | GateType::MeasureLeaked => {
                    for qubit_id in qubits {
                        let result_id = u64::try_from(measurement_count)
                            .map_err(|error| Self::error("measurement ID conversion", error))?;
                        operations.push(Operation::Measure {
                            qubit_id,
                            result_id,
                        });
                        measurement_count += 1;
                    }
                }
                GateType::PZ => {
                    operations.extend(qubits.iter().map(|&qubit_id| Operation::Reset { qubit_id }));
                }
                GateType::Idle
                | GateType::MeasCrosstalkGlobalPayload
                | GateType::MeasCrosstalkLocalPayload => {}
                unsupported => {
                    return Err(PecosError::Generic(format!(
                        "PECOS general-noise bridge produced unsupported gate {unsupported}"
                    )));
                }
            }
        }

        let mut outcomes = vec![None; measurement_count];
        if !operations.is_empty() {
            let results = simulator
                .handle_operations(BatchOperation::error_model(operations))
                .map_err(|error| Self::error("operation batch", error))?;
            if !results.u64_results.is_empty() {
                return Err(PecosError::Generic(
                    "Selene simulator returned leakage results for boolean PECOS measurements"
                        .to_string(),
                ));
            }
            for result in results.bool_results {
                let index = usize::try_from(result.result_id)
                    .map_err(|error| Self::error("measurement result ID conversion", error))?;
                let Some(slot) = outcomes.get_mut(index) else {
                    return Err(PecosError::Generic(format!(
                        "Selene simulator returned unexpected measurement result ID {}",
                        result.result_id
                    )));
                };
                if slot.replace(result.value).is_some() {
                    return Err(PecosError::Generic(format!(
                        "Selene simulator returned duplicate measurement result ID {}",
                        result.result_id
                    )));
                }
            }
        }

        let outcomes = outcomes
            .into_iter()
            .enumerate()
            .map(|(index, outcome)| {
                outcome.map(usize::from).ok_or_else(|| {
                    PecosError::Generic(format!(
                        "Selene simulator omitted measurement result ID {index}"
                    ))
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut builder = ByteMessage::outcomes_builder();
        builder.add_outcomes(&outcomes);
        Ok(builder.build())
    }
}
