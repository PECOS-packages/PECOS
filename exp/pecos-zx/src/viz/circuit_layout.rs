// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Circuit layout engine: ASAP scheduling and grid placement.
//!
//! Converts a [`DagCircuit`] or [`TickCircuit`] into a [`CircuitLayout`] grid
//! suitable for rendering as SVG or ASCII.

use pecos_core::ClassicalBitId;
use pecos_core::gate_type::GateType;
use pecos_quantum::{Circuit, DagCircuit, TickCircuit};

/// A gate placed in the circuit grid.
#[derive(Debug, Clone)]
pub struct GateSlot {
    /// The gate type.
    pub gate_type: GateType,
    /// Display label (e.g., "H", "X", "RZ(0.5)").
    pub label: String,
    /// Which qubit rows this gate spans (sorted).
    pub qubits: Vec<usize>,
    /// For controlled gates: is this the control dot position?
    pub is_control: bool,
    /// Whether this gate has a classical condition.
    pub has_condition: bool,
    /// Classical bit for condition or measurement target.
    pub cbit: Option<usize>,
    /// For measurement: the classical bit target.
    pub meas_cbit: Option<usize>,
}

/// Grid-based layout of a quantum circuit.
///
/// The grid has qubits as rows and time steps as columns. Each cell may
/// contain a [`GateSlot`] or be empty.
#[derive(Debug, Clone)]
pub struct CircuitLayout {
    /// Grid positions: `grid[qubit][time_step] = Option<GateSlot>`.
    grid: Vec<Vec<Option<GateSlot>>>,
    /// Number of qubit wires.
    pub num_qubits: usize,
    /// Number of time steps.
    pub num_steps: usize,
    /// Number of classical bit wires.
    pub num_cbits: usize,
}

impl CircuitLayout {
    /// Access the gate at a given qubit and time step.
    #[must_use]
    pub fn get(&self, qubit: usize, step: usize) -> Option<&GateSlot> {
        self.grid
            .get(qubit)
            .and_then(|row| row.get(step))
            .and_then(|cell| cell.as_ref())
    }

    /// Access the full grid.
    #[must_use]
    pub fn grid(&self) -> &Vec<Vec<Option<GateSlot>>> {
        &self.grid
    }
}

/// Build a gate label from gate type and angles.
fn gate_label(gate_type: GateType, angles: &[pecos_core::Angle64]) -> String {
    let base = match gate_type {
        GateType::I => "I",
        GateType::X => "X",
        GateType::Y => "Y",
        GateType::Z => "Z",
        GateType::H => "H",
        GateType::SX => "SX",
        GateType::SXdg => "SX+",
        GateType::SY => "SY",
        GateType::SYdg => "SY+",
        GateType::SZ => "S",
        GateType::SZdg => "S+",
        GateType::T => "T",
        GateType::Tdg => "T+",
        GateType::RX => "RX",
        GateType::RY => "RY",
        GateType::RZ => "RZ",
        GateType::U => "U",
        GateType::R1XY => "R1XY",
        GateType::CX => "CX",
        GateType::CY => "CY",
        GateType::CZ => "CZ",
        GateType::CH => "CH",
        GateType::SZZ => "SZZ",
        GateType::SZZdg => "SZZ+",
        GateType::SWAP => "SW",
        GateType::CRZ => "CRZ",
        GateType::RXX => "RXX",
        GateType::RYY => "RYY",
        GateType::RZZ => "RZZ",
        GateType::CCX => "CCX",
        GateType::MZ => "M",
        GateType::MeasureFree => "MF",
        GateType::PZ => "P",
        GateType::QAlloc => "QA",
        GateType::QFree => "QF",
        GateType::Idle => "ID",
        _ => "?",
    };

    if angles.is_empty() {
        base.to_string()
    } else {
        let params: Vec<String> = angles
            .iter()
            .map(|a| format!("{:.2}", a.to_radians()))
            .collect();
        format!("{base}({})", params.join(","))
    }
}

/// Compute a [`CircuitLayout`] from a [`DagCircuit`] using ASAP scheduling.
///
/// Each gate is placed at the earliest time step where all its qubits are free.
#[must_use]
pub fn layout_from_dag(dag: &DagCircuit) -> CircuitLayout {
    let num_qubits = if dag.gate_count() > 0 {
        dag.max_qubit() + 1
    } else {
        0
    };
    let num_cbits = dag.num_cbits();

    // Two-pass scheduling to handle classical dependencies correctly.
    // Pass 1: schedule non-conditional gates, record measurement positions.
    // Pass 2: schedule conditional gates after the measurements they depend on.

    let mut qubit_next_free = vec![0usize; num_qubits];
    let mut cbit_ready_at = vec![0usize; num_cbits]; // step after which each cbit is available
    let mut slots: Vec<(usize, GateSlot)> = Vec::new();

    // Collect all nodes with their gate info
    struct NodeInfo {
        qubit_indices: Vec<usize>,
        condition: Option<(ClassicalBitId, bool)>,
        meas_target: Option<ClassicalBitId>,
        gate_type: GateType,
        label: String,
    }

    let mut non_conditional: Vec<NodeInfo> = Vec::new();
    let mut conditional: Vec<NodeInfo> = Vec::new();

    for node in dag.topological_order() {
        let Some(gate) = dag.gate(node) else {
            continue;
        };
        let qubit_indices: Vec<usize> = gate.qubits.iter().map(|q| q.index()).collect();
        let condition = dag.condition(node);
        let meas_target = dag.measurement_target(node);
        let info = NodeInfo {
            qubit_indices,
            condition,
            meas_target,
            gate_type: gate.gate_type,
            label: gate_label(gate.gate_type, &gate.angles),
        };
        if condition.is_some() {
            conditional.push(info);
        } else {
            non_conditional.push(info);
        }
    }

    // Pass 1: schedule non-conditional gates
    for info in &non_conditional {
        let step = info
            .qubit_indices
            .iter()
            .map(|&q| qubit_next_free.get(q).copied().unwrap_or(0))
            .max()
            .unwrap_or(0);

        for &q in &info.qubit_indices {
            if q < qubit_next_free.len() {
                qubit_next_free[q] = step + 1;
            }
        }

        if let Some(cbit) = info.meas_target {
            let c = cbit.index();
            if c >= cbit_ready_at.len() {
                cbit_ready_at.resize(c + 1, 0);
            }
            cbit_ready_at[c] = step + 1;
        }

        let slot = GateSlot {
            gate_type: info.gate_type,
            label: info.label.clone(),
            qubits: info.qubit_indices.clone(),
            is_control: false,
            has_condition: false,
            cbit: None,
            meas_cbit: info.meas_target.map(ClassicalBitId::index),
        };
        slots.push((step, slot));
    }

    // Pass 2: schedule conditional gates (must come after their conditioning measurement)
    for info in &conditional {
        let mut step = info
            .qubit_indices
            .iter()
            .map(|&q| qubit_next_free.get(q).copied().unwrap_or(0))
            .max()
            .unwrap_or(0);

        if let Some((cbit, _)) = info.condition {
            let c = cbit.index();
            if c < cbit_ready_at.len() {
                step = step.max(cbit_ready_at[c]);
            }
        }

        for &q in &info.qubit_indices {
            if q < qubit_next_free.len() {
                qubit_next_free[q] = step + 1;
            }
        }

        let slot = GateSlot {
            gate_type: info.gate_type,
            label: info.label.clone(),
            qubits: info.qubit_indices.clone(),
            is_control: false,
            has_condition: info.condition.is_some(),
            cbit: info.condition.map(|(c, _)| c.index()),
            meas_cbit: info.meas_target.map(ClassicalBitId::index),
        };
        slots.push((step, slot));
    }

    let num_steps = qubit_next_free.iter().copied().max().unwrap_or(0);

    // Build the grid
    let mut grid = vec![vec![None; num_steps]; num_qubits];
    for (step, slot) in slots {
        if let Some(&first_qubit) = slot.qubits.first()
            && first_qubit < grid.len()
            && step < num_steps
        {
            grid[first_qubit][step] = Some(slot);
        }
    }

    CircuitLayout {
        grid,
        num_qubits,
        num_steps,
        num_cbits,
    }
}

/// Compute a [`CircuitLayout`] from a [`TickCircuit`].
///
/// Each tick becomes a time step. Gates within a tick are placed in parallel.
#[must_use]
pub fn layout_from_tick_circuit(tc: &TickCircuit) -> CircuitLayout {
    let all_qubits = tc.all_qubits();
    let num_qubits = all_qubits.iter().map(|q| q.index() + 1).max().unwrap_or(0);
    let num_cbits = 0; // TODO: TickCircuit classical bit support
    let num_steps = tc.num_ticks();

    let mut grid = vec![vec![None; num_steps]; num_qubits];

    for (tick_idx, tick) in tc.ticks().iter().enumerate() {
        if tick_idx >= num_steps {
            break;
        }
        for gate in tick.gates().iter() {
            let qubit_indices: Vec<usize> = gate.qubits.iter().map(|q| q.index()).collect();

            // TODO: TickCircuit classical bit/condition support
            let slot = GateSlot {
                gate_type: gate.gate_type,
                label: gate_label(gate.gate_type, &gate.angles),
                qubits: qubit_indices.clone(),
                is_control: false,
                has_condition: false,
                cbit: None,
                meas_cbit: None,
            };

            if let Some(&first_qubit) = qubit_indices.first()
                && first_qubit < grid.len()
            {
                grid[first_qubit][tick_idx] = Some(slot);
            }
        }
    }

    CircuitLayout {
        grid,
        num_qubits,
        num_steps,
        num_cbits,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layout_single_gate() {
        let mut dag = DagCircuit::new();
        dag.h(&[0]);

        let layout = layout_from_dag(&dag);
        assert_eq!(layout.num_qubits, 1);
        assert_eq!(layout.num_steps, 1);

        let slot = layout.get(0, 0).unwrap();
        assert_eq!(slot.gate_type, GateType::H);
        assert_eq!(slot.label, "H");
    }

    #[test]
    fn test_layout_two_qubit_gate() {
        let mut dag = DagCircuit::new();
        dag.h(&[0]);
        dag.cx(&[(0, 1)]);

        let layout = layout_from_dag(&dag);
        assert_eq!(layout.num_qubits, 2);
        assert_eq!(layout.num_steps, 2);

        // H at step 0, CX at step 1
        assert_eq!(layout.get(0, 0).unwrap().gate_type, GateType::H);
        assert_eq!(layout.get(0, 1).unwrap().gate_type, GateType::CX);
    }

    #[test]
    fn test_layout_parallel_gates() {
        let mut dag = DagCircuit::new();
        dag.h(&[0]);
        dag.x(&[1]);

        let layout = layout_from_dag(&dag);
        assert_eq!(layout.num_qubits, 2);
        // Both gates can be at step 0 since they're on different qubits
        assert_eq!(layout.num_steps, 1);

        assert_eq!(layout.get(0, 0).unwrap().gate_type, GateType::H);
        assert_eq!(layout.get(1, 0).unwrap().gate_type, GateType::X);
    }

    #[test]
    fn test_layout_with_measurement() {
        let mut dag = DagCircuit::new();
        dag.h(&[0]);
        dag.mz(&[0]);

        let layout = layout_from_dag(&dag);

        let meas_slot = layout.get(0, 1).unwrap();
        assert_eq!(meas_slot.gate_type, GateType::MZ);
    }

    // TODO: requires DagCircuit classical bit API
    // #[test]
    // fn test_layout_with_condition() {
    //     let mut dag = DagCircuit::new();
    //     dag.set_num_cbits(1);
    //     dag.h(&[0]);
    //     dag.mz_to(0, pecos_core::ClassicalBitId::new(0));
    //     dag.if_bit(pecos_core::ClassicalBitId::new(0), true).x(&[1]);
    //
    //     let layout = layout_from_dag(&dag);
    //
    //     // Find the conditional X gate
    //     let x_slot = layout.get(1, 2).unwrap();
    //     assert_eq!(x_slot.gate_type, GateType::X);
    //     assert!(x_slot.has_condition);
    //     assert_eq!(x_slot.cbit, Some(0));
    // }

    #[test]
    fn test_layout_empty_circuit() {
        let dag = DagCircuit::new();
        let layout = layout_from_dag(&dag);
        assert_eq!(layout.num_qubits, 0);
        assert_eq!(layout.num_steps, 0);
    }

    #[test]
    fn test_layout_from_tick_circuit() {
        let mut tc = TickCircuit::new();
        tc.tick().h(&[0]).x(&[1]);
        tc.tick().cx(&[(0, 1)]);

        let layout = layout_from_tick_circuit(&tc);
        assert_eq!(layout.num_qubits, 2);
        assert_eq!(layout.num_steps, 2);

        assert_eq!(layout.get(0, 0).unwrap().gate_type, GateType::H);
        assert_eq!(layout.get(1, 0).unwrap().gate_type, GateType::X);
        assert_eq!(layout.get(0, 1).unwrap().gate_type, GateType::CX);
    }

    #[test]
    fn test_gate_label_parameterized() {
        let label = gate_label(
            GateType::RZ,
            &[pecos_core::Angle64::from_radians(
                std::f64::consts::FRAC_PI_2,
            )],
        );
        assert!(label.starts_with("RZ("));
    }

    #[test]
    fn test_gate_label_simple() {
        assert_eq!(gate_label(GateType::H, &[]), "H");
        assert_eq!(gate_label(GateType::X, &[]), "X");
        assert_eq!(gate_label(GateType::CX, &[]), "CX");
        assert_eq!(gate_label(GateType::MZ, &[]), "M");
    }
}
