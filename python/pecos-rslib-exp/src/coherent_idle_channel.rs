// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file
// except in compliance with the License. You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the
// License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
// express or implied. See the License for the specific language governing permissions and
// limitations under the License.

//! Coherent idle noise channel: RZ(angle) on both qubits after each CX gate.

use pecos_core::Angle64;
use pecos_neo::command::{GateCommand, GateType};
use pecos_neo::noise::{NoiseChannel, NoiseContext, NoiseEvent, NoiseResponse};
use pecos_random::PecosRng;
use smallvec::SmallVec;

/// Applies coherent RZ rotation after each two-qubit gate on both qubits.
///
/// Models uncompensated phase accumulation during idle time between gates.
/// The rotation angle represents the coherent Z-phase acquired per gate
/// application.
#[derive(Clone)]
pub struct CoherentIdleChannel {
    angle: Angle64,
}

impl CoherentIdleChannel {
    /// Create a coherent idle channel with the given RZ angle (radians).
    pub fn new(angle_radians: f64) -> Self {
        Self {
            angle: Angle64::from_radians(angle_radians),
        }
    }
}

impl NoiseChannel for CoherentIdleChannel {
    fn responds_to(&self, event: &NoiseEvent<'_>) -> bool {
        matches!(
            event,
            NoiseEvent::AfterGate {
                gate_type: GateType::CX | GateType::CZ | GateType::CY,
                ..
            }
        )
    }

    fn apply(
        &self,
        event: &NoiseEvent<'_>,
        _ctx: &mut NoiseContext,
        _rng: &mut PecosRng,
    ) -> NoiseResponse {
        if let NoiseEvent::AfterGate { qubits, .. } = event {
            let mut gates = SmallVec::new();
            for &q in *qubits {
                gates.push(GateCommand::rz(q, self.angle));
            }
            NoiseResponse::inject_gates(gates)
        } else {
            NoiseResponse::None
        }
    }

    fn name(&self) -> &'static str {
        "CoherentIdleRZ"
    }

    fn clone_box(&self) -> Box<dyn NoiseChannel> {
        Box::new(self.clone())
    }
}
