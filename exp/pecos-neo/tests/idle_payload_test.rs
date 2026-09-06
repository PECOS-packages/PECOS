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

use pecos_core::{Angle64, Gate, QubitId, TimeUnits};
use pecos_neo::adapter::{
    byte_message_to_command_queue, command_queue_to_gates, gates_to_command_queue,
};
use pecos_neo::extensible::{
    AngleSnapper, CommandQueueValidation, SnapPolicy, is_clifford_circuit, snap_command_queue,
};
use pecos_neo::{
    CircuitRunner, CommandBuilder, CommandQueue, GateCommand, GateCommandError, GatePayload,
    NoiseResponse,
};
use pecos_quantum::TickCircuit;
use pecos_simulators::SparseStab;
use std::sync::{Arc, Mutex};

const DURATIONS: [u64; 6] = [0, 1, 23, 1 << 53, (1 << 53) + 1, u64::MAX];

#[test]
fn idle_payload_snap_preserves_duration() {
    let snapper = AngleSnapper::clifford(1e-9);
    for policy in [
        SnapPolicy::Exact,
        SnapPolicy::SnapOrKeep { tolerance: 1e-9 },
        SnapPolicy::SnapOrFail { tolerance: 1e-9 },
    ] {
        let results: Vec<_> = DURATIONS
            .map(|duration| {
                let commands = CommandBuilder::new().idle(&[0], duration).build();
                let snapped = snap_command_queue(&commands, &policy, &snapper);
                let actual = snapped
                    .as_ref()
                    .ok()
                    .and_then(|queue| queue.as_slice()[0].get_idle_duration());
                println!("snap {policy:?}: {duration} -> {actual:?}");
                (TimeUnits::new(duration), actual)
            })
            .into();
        for (expected, actual) in results {
            assert_eq!(actual, Some(expected), "{policy:?}");
        }
    }
}

#[test]
fn idle_payload_clifford_verdict_is_duration_independent() {
    for duration in DURATIONS {
        let commands = CommandBuilder::new()
            .pz(&[0])
            .h(&[0])
            .idle(&[0], duration)
            .mz(&[0])
            .build();
        let verdict = is_clifford_circuit(&commands);
        println!("Clifford duration {duration}: {verdict}");
        assert!(verdict, "duration {duration}");
        let non_clifford = CommandBuilder::new().t(&[0]).idle(&[0], duration).build();
        assert!(!is_clifford_circuit(&non_clifford), "duration {duration}");
    }
}

#[test]
fn idle_payload_has_no_angles() {
    for duration in DURATIONS {
        let idle = GateCommand::idle(QubitId(0), TimeUnits::new(duration));
        assert!(idle.angles().is_empty(), "duration {duration}");
        assert_eq!(idle.get_idle_duration(), Some(TimeUnits::new(duration)));
        assert_eq!(
            idle.payload,
            GatePayload::Duration(TimeUnits::new(duration))
        );
        assert_eq!(idle.clone(), idle);
        let commands = CommandBuilder::new().idle(&[0], duration).build();
        assert!(commands.to_gate_validations()[0].angles.is_empty());
    }
}

#[test]
fn idle_payload_round_trips_through_core_and_wire() {
    // Native durations stay exact; core Gate and ByteMessage use f64, so
    // conversion must reject durations that would lose precision.
    for duration in DURATIONS {
        let commands = CommandBuilder::new()
            .h(&[0])
            .rz(&[0], Angle64::QUARTER_TURN)
            .idle(&[0], duration)
            .build();
        if duration == (1 << 53) + 1 || duration == u64::MAX {
            assert_eq!(
                command_queue_to_gates(&commands),
                Err(GateCommandError::IdleDurationNotRepresentable { duration })
            );
            assert!(TickCircuit::try_from(&commands).is_err());
            assert!(TickCircuit::try_from(commands.clone()).is_err());
            continue;
        }
        let expected = &commands;
        let gates = command_queue_to_gates(&commands).unwrap();
        assert_eq!(
            gates[2],
            Gate::idle(TimeUnits::new(duration).as_f64(), vec![QubitId(0)])
        );
        assert_eq!(
            gates_to_command_queue(&gates).unwrap().as_slice(),
            expected.as_slice()
        );
        assert_eq!(
            GateCommand::try_from(&gates[2]).unwrap(),
            expected.as_slice()[2]
        );
        assert_eq!(
            GateCommand::try_from(gates[2].clone()).unwrap(),
            expected.as_slice()[2]
        );

        let circuit = TickCircuit::try_from(&commands).unwrap();
        assert_eq!(
            CommandQueue::try_from(&circuit).unwrap().as_slice(),
            expected.as_slice()
        );
        assert_eq!(
            CommandQueue::try_from(TickCircuit::try_from(commands.clone()).unwrap())
                .unwrap()
                .as_slice(),
            expected.as_slice()
        );

        let message = pecos_engines::ByteMessage::quantum_operations_builder()
            .add_gate_commands(&gates)
            .build();
        assert_eq!(
            byte_message_to_command_queue(&message).unwrap().as_slice(),
            expected.as_slice()
        );
    }
}

#[test]
fn idle_payload_reaches_dispatch_unchanged() {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let captured = Arc::clone(&seen);
    let mut runner = CircuitRunner::<SparseStab>::new().with_seed(42);
    runner.on_idle(move |ctx| {
        assert!(ctx.angles.is_empty());
        captured
            .lock()
            .unwrap()
            .push(ctx.duration.unwrap().as_u64());
        NoiseResponse::None
    });
    let commands = DURATIONS
        .map(|duration| GateCommand::idle(QubitId(0), TimeUnits::new(duration)))
        .into_iter()
        .collect();
    runner
        .apply_circuit(&mut SparseStab::with_seed(1, 42), &commands)
        .unwrap();
    assert_eq!(*seen.lock().unwrap(), DURATIONS);
}
