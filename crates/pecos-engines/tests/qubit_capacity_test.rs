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

//! Integration tests for the per-command qubit-capacity guard.
//!
//! A command targeting a qubit index at or beyond the simulator's capacity
//! (e.g. dynamic allocation past the configured qubit count) must fail with
//! the op and the capacity named -- through BOTH shared dispatchers
//! (`SparseStabEngine` -> Clifford dispatch, `StabVecEngine` -> general
//! dispatch), and INCLUDING commands consumed by the MZ-batching lookahead,
//! which bypasses the top-of-loop check.
//!
//! (`StateVecEngine` is NOT covered: it has its own dispatch loop that
//! auto-grows the simulator instead of rejecting -- pre-existing behavior
//! for static-circuit flows whose qubit count is inferred as 0.)

use pecos_engines::Engine;
use pecos_engines::byte_message::ByteMessageBuilder;
use pecos_engines::quantum::{SparseStabEngine, StabVecEngine};

fn build_message(build: impl FnOnce(&mut ByteMessageBuilder)) -> pecos_engines::ByteMessage {
    let mut builder = ByteMessageBuilder::new();
    let _ = builder.for_quantum_operations();
    build(&mut builder);
    builder.build()
}

fn assert_capacity_error(result: Result<pecos_engines::ByteMessage, impl std::fmt::Display>) {
    match result {
        Err(e) => {
            let msg = e.to_string();
            assert!(
                msg.contains("2 qubits"),
                "error should name the capacity: {msg}"
            );
            assert!(
                msg.contains("qubit 2"),
                "error should name the offending qubit: {msg}"
            );
        }
        Ok(_) => panic!("expected out-of-capacity command to fail"),
    }
}

#[test]
fn clifford_dispatch_rejects_gate_beyond_capacity() {
    let mut engine = SparseStabEngine::new(2);
    let msg = build_message(|b| {
        b.h(&[2]);
    });
    assert_capacity_error(engine.process(msg));
}

#[test]
fn general_dispatch_rejects_gate_beyond_capacity() {
    let mut engine = StabVecEngine::new(2);
    let msg = build_message(|b| {
        b.h(&[2]);
    });
    assert_capacity_error(engine.process(msg));
}

#[test]
fn clifford_dispatch_rejects_mz_lookahead_beyond_capacity() {
    // Two consecutive MZ commands batch via lookahead; the SECOND one is
    // consumed inside the batching loop, not at the top of the dispatch
    // loop, and must still be guarded.
    let mut engine = SparseStabEngine::new(2);
    let msg = build_message(|b| {
        b.mz(&[0]);
        b.mz(&[2]);
    });
    assert_capacity_error(engine.process(msg));
}

#[test]
fn general_dispatch_rejects_mz_lookahead_beyond_capacity() {
    let mut engine = StabVecEngine::new(2);
    let msg = build_message(|b| {
        b.mz(&[0]);
        b.mz(&[2]);
    });
    assert_capacity_error(engine.process(msg));
}

#[test]
fn in_range_commands_still_process() {
    let mut engine = SparseStabEngine::new(2);
    let msg = build_message(|b| {
        b.h(&[0]);
        b.cx(&[(0, 1)]);
        b.mz(&[0]);
        b.mz(&[1]);
    });
    let outcomes = engine
        .process(msg)
        .expect("in-range circuit should process")
        .outcomes()
        .expect("outcomes should parse");
    assert_eq!(outcomes.len(), 2);
    assert_eq!(outcomes[0], outcomes[1], "Bell pair outcomes must agree");
}
