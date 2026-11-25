// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under
// the License.

//! Physical seepage shared by the single- and two-qubit gate channels.

use super::NoiseResponse;
use crate::command::{GateCommand, GateType};
use pecos_core::QubitId;
use pecos_random::{PecosRng, rng_ext::RngProbabilityExt};
use smallvec::{SmallVec, smallvec};

/// Discard each selected qubit's state with an independent uniform Pauli twirl.
/// Identity needs no injected gate. Selection probabilities belong to the caller;
/// no random numbers are consumed for qubits that did not seep.
pub(super) fn seep_response(qubits: SmallVec<[QubitId; 4]>, rng: &mut PecosRng) -> NoiseResponse {
    if qubits.is_empty() {
        return NoiseResponse::None;
    }
    let mut gates = SmallVec::new();
    for &qubit in &qubits {
        let gate_type = match rng.random_index_4() {
            0 => continue,
            1 => GateType::X,
            2 => GateType::Y,
            _ => GateType::Z,
        };
        gates.push(GateCommand::new(gate_type, smallvec![qubit]));
    }
    let response = NoiseResponse::MarkUnleaked(qubits);
    if gates.is_empty() {
        response
    } else {
        response.combine(NoiseResponse::inject_gates(gates))
    }
}
