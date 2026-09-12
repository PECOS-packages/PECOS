// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
// in compliance with the License.You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under
// the License.

use pecos_core::errors::PecosError;
use pecos_engines::ByteMessage;
use pecos_engines::monte_carlo::engine::{ExternalClassicalEngine, MonteCarloEngine};
use pecos_engines::shot_results::Data;

#[test]
fn malformed_circuit_returns_input_error() {
    let circuit = ByteMessage::new(&[1, 2, 3]);
    let mut engine = MonteCarloEngine::new_with_depolarizing_noise(
        Box::new(ExternalClassicalEngine::new_with_circuit(circuit)),
        0.0,
    );

    let err = engine.run_with_workers(1, 1).unwrap_err();
    assert!(matches!(
        err,
        PecosError::Input(ref message) if message == "Message too small for batch header"
    ));
}

#[test]
fn valid_empty_circuits_still_complete_successfully() {
    for controller in [
        ExternalClassicalEngine::new(),
        ExternalClassicalEngine::new_with_circuit(ByteMessage::create_empty()),
        ExternalClassicalEngine::new_with_circuit(ByteMessage::new(&[])),
    ] {
        let mut engine = MonteCarloEngine::new_with_depolarizing_noise(Box::new(controller), 0.0);
        let results = engine.run_with_workers(4, 2).unwrap();
        assert_eq!(results.len(), 4);
        for shot in results.shots {
            assert_eq!(shot.data.get("result"), Some(&Data::U32(0)));
        }
    }
}
