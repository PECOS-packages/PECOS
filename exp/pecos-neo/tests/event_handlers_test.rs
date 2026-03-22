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

//! Tests for `EventHandlers` passed through `sim_neo()`.

use pecos_core::impl_signal;
use pecos_neo::prelude::*;
use pecos_neo::tool::sim_neo;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

#[derive(Copy, Clone, Debug)]
struct RoundMarker(pub u32);
impl_signal!(RoundMarker);

#[test]
fn event_handlers_on_before_gate_fires_through_sim_neo() {
    let counter = Arc::new(AtomicUsize::new(0));
    let c = counter.clone();

    let handlers = EventHandlers::new().on_before_gate(move |_ctx| {
        c.fetch_add(1, Ordering::Relaxed);
        NoiseResponse::None
    });

    // H gate is the only "gate" here (PZ is preparation, MZ is measurement)
    let circuit = CommandBuilder::new().pz(0).h(0).mz(0).build();

    let results = sim_neo(circuit)
        .event_handlers(handlers)
        .shots(10)
        .seed(42)
        .run();

    assert_eq!(results.len(), 10);
    // H gate fires on_before_gate once per shot
    assert_eq!(counter.load(Ordering::Relaxed), 10);
}

#[test]
fn event_handlers_parallel_workers_fire_per_worker() {
    let counter = Arc::new(AtomicUsize::new(0));
    let c = counter.clone();

    let handlers = EventHandlers::new().on_before_gate(move |_ctx| {
        c.fetch_add(1, Ordering::Relaxed);
        NoiseResponse::None
    });

    let circuit = CommandBuilder::new().pz(0).h(0).mz(0).build();

    let results = sim_neo(circuit)
        .event_handlers(handlers)
        .workers(2)
        .shots(20)
        .seed(42)
        .run();

    assert_eq!(results.len(), 20);
    // Each shot has 1 H gate -> 20 on_before_gate calls total across workers
    assert_eq!(counter.load(Ordering::Relaxed), 20);
}

#[test]
fn event_handlers_empty_is_noop() {
    let handlers = EventHandlers::new();
    assert!(handlers.is_empty());

    // Should not change results compared to no handlers
    let circuit = CommandBuilder::new().pz(0).x(0).mz(0).build();

    let results_with = sim_neo(circuit.clone())
        .event_handlers(handlers)
        .shots(10)
        .seed(42)
        .run();

    let results_without = sim_neo(circuit).shots(10).seed(42).run();

    assert_eq!(results_with.len(), results_without.len());
    for (a, b) in results_with
        .outcomes
        .iter()
        .zip(results_without.outcomes.iter())
    {
        assert_eq!(
            a.get_bit(QubitId(0)),
            b.get_bit(QubitId(0)),
            "Empty EventHandlers should not change results"
        );
    }
}

#[test]
fn event_handlers_multiple_handler_types() {
    let gate_count = Arc::new(AtomicUsize::new(0));
    let meas_count = Arc::new(AtomicUsize::new(0));
    let prep_count = Arc::new(AtomicUsize::new(0));

    let gc = gate_count.clone();
    let mc = meas_count.clone();
    let pc = prep_count.clone();

    let handlers = EventHandlers::new()
        .on_before_gate(move |_ctx| {
            gc.fetch_add(1, Ordering::Relaxed);
            NoiseResponse::None
        })
        .on_after_measurement(move |_ctx| {
            mc.fetch_add(1, Ordering::Relaxed);
            NoiseResponse::None
        })
        .on_after_preparation(move |_ctx| {
            pc.fetch_add(1, Ordering::Relaxed);
            NoiseResponse::None
        });

    // PZ (prep), H (gate), MZ (measurement)
    let circuit = CommandBuilder::new().pz(0).h(0).mz(0).build();

    let results = sim_neo(circuit)
        .event_handlers(handlers)
        .shots(5)
        .seed(42)
        .run();

    assert_eq!(results.len(), 5);
    assert_eq!(gate_count.load(Ordering::Relaxed), 5); // 1 H per shot
    assert_eq!(meas_count.load(Ordering::Relaxed), 5); // 1 MZ per shot
    assert_eq!(prep_count.load(Ordering::Relaxed), 5); // 1 PZ per shot
}

#[test]
fn event_handlers_on_runner_directly() {
    let counter = Arc::new(AtomicUsize::new(0));
    let c = counter.clone();

    let handlers = EventHandlers::new().on_after_gate(move |_ctx| {
        c.fetch_add(1, Ordering::Relaxed);
        NoiseResponse::None
    });

    let circuit = CommandBuilder::new().pz(0).h(0).mz(0).build();

    let mut state = pecos_simulators::SparseStab::new(1);
    let mut runner = CircuitRunner::<pecos_simulators::SparseStab>::new()
        .with_event_handlers(handlers)
        .with_seed(42);

    let _outcomes = runner.apply_circuit(&mut state, &circuit).unwrap();
    assert_eq!(counter.load(Ordering::Relaxed), 1); // 1 H gate
}

#[test]
fn signal_handler_fires_through_sim_neo() {
    let counter = Arc::new(AtomicUsize::new(0));
    let c = counter.clone();

    let handlers = EventHandlers::new().on_signal(move |sig: &RoundMarker| {
        assert_eq!(sig.0, 42);
        c.fetch_add(1, Ordering::Relaxed);
    });

    let circuit = CommandBuilder::new()
        .pz(0)
        .signal(RoundMarker(42))
        .h(0)
        .mz(0)
        .build();

    let results = sim_neo(circuit)
        .event_handlers(handlers)
        .shots(5)
        .seed(42)
        .run();

    assert_eq!(results.len(), 5);
    assert_eq!(counter.load(Ordering::Relaxed), 5);
}

#[test]
fn signal_handler_fires_through_sim_neo_parallel() {
    let counter = Arc::new(AtomicUsize::new(0));
    let c = counter.clone();

    let handlers = EventHandlers::new().on_signal(move |_sig: &RoundMarker| {
        c.fetch_add(1, Ordering::Relaxed);
    });

    let circuit = CommandBuilder::new()
        .pz(0)
        .signal(RoundMarker(1))
        .h(0)
        .signal(RoundMarker(2))
        .mz(0)
        .build();

    let results = sim_neo(circuit)
        .event_handlers(handlers)
        .workers(2)
        .shots(10)
        .seed(42)
        .run();

    assert_eq!(results.len(), 10);
    // 2 signals per shot, 10 shots = 20 handler calls
    assert_eq!(counter.load(Ordering::Relaxed), 20);
}
