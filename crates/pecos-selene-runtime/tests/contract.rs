use pecos_selene_runtime::{PecosRuntime, RuntimeConfig};
use selene_core::runtime::{BatchOperation, Operation, RuntimeInterface};

fn runtime(config: RuntimeConfig) -> PecosRuntime {
    let mut r = PecosRuntime::new(3, 100.into(), config).unwrap();
    r.shot_start(0, 7).unwrap();
    for _ in 0..3 {
        r.qalloc().unwrap();
    }
    r
}
fn drain(r: &mut PecosRuntime) -> Vec<BatchOperation> {
    let mut out = Vec::new();
    while let Some(batch) = r.get_next_operations().unwrap() {
        out.push(batch);
    }
    out
}
#[test]
fn forcing_both_measurement_kinds_releases_only_the_needed_prefix() {
    for leaked in [false, true] {
        let mut r = runtime(RuntimeConfig::default());
        r.rxy_gate(0, 1.0, 0.0).unwrap();
        let id = if leaked {
            r.measure_leaked(0)
        } else {
            r.measure(0)
        }
        .unwrap();
        r.rxy_gate(1, 2.0, 0.0).unwrap();
        assert!(drain(&mut r).is_empty());
        assert_eq!(r.get_u64_result(id).unwrap(), None);
        r.force_result(id).unwrap();
        let ops: Vec<_> = drain(&mut r).into_iter().flatten().collect();
        assert_eq!(ops.len(), 2);
        assert!(matches!(ops[0], Operation::RXYGate { qubit_id: 0, .. }));
        assert_eq!(r.get_u64_result(id).unwrap(), None);
        r.set_u64_result(id, if leaked { 2 } else { 1 }).unwrap();
        r.force_result(id).unwrap();
        assert!(drain(&mut r).is_empty());
        assert_eq!(
            r.get_u64_result(id).unwrap(),
            Some(if leaked { 2 } else { 1 })
        );
        if leaked {
            assert!(r.get_bool_result(id).is_err());
        }
        r.global_barrier(0).unwrap();
        assert_eq!(drain(&mut r).len(), 1);
        r.shot_end().unwrap();
    }
}
#[test]
fn virtual_z_changes_pulse_axis_without_emitting_an_rz() {
    let mut r = runtime(RuntimeConfig::default());
    r.rz_gate(0, 0.25).unwrap();
    r.rz_gate(0, 0.5).unwrap();
    r.rxy_gate(0, 1.5, 1.0).unwrap();
    r.global_barrier(0).unwrap();
    let ops: Vec<_> = drain(&mut r).into_iter().flatten().collect();
    assert_eq!(
        ops,
        [Operation::RXYGate {
            qubit_id: 0,
            theta: 1.5,
            phi: 0.25
        }]
    );
}
#[test]
fn batching_respects_dependencies_and_sealed_prefixes() {
    let mut r = runtime(RuntimeConfig {
        max_batch_size: 3,
        ..Default::default()
    });
    let a = r.measure(0).unwrap();
    let b = r.measure(1).unwrap();
    r.force_result(a).unwrap();
    let c = r.measure(2).unwrap(); // must not be appended to the sealed measurement batch
    let batches = drain(&mut r);
    assert_eq!(batches.len(), 1);
    assert_eq!(batches[0].len(), 2);
    r.set_bool_result(a, false).unwrap();
    r.set_bool_result(b, true).unwrap();
    assert_eq!(r.get_bool_result(c).unwrap(), None);
    r.force_result(c).unwrap();
    assert_eq!(drain(&mut r)[0].len(), 1);
}
#[test]
fn barriers_and_delays_have_checked_timing_and_continue_between_shots() {
    let mut r = runtime(RuntimeConfig {
        duration_ns_rxy: 10,
        ..Default::default()
    });
    r.rxy_gate(0, 1.0, 0.0).unwrap();
    r.local_barrier(&[0], 25).unwrap();
    let first = drain(&mut r);
    assert_eq!(u64::from(first[0].runtime_source().unwrap().start()), 100);
    r.rxy_gate(0, 1.0, 0.0).unwrap();
    r.global_barrier(20).unwrap();
    let second = drain(&mut r);
    assert_eq!(u64::from(second[0].runtime_source().unwrap().start()), 135);
    r.rxy_gate(1, 1.0, 0.0).unwrap();
    r.global_barrier(0).unwrap();
    assert_eq!(
        u64::from(drain(&mut r)[0].runtime_source().unwrap().start()),
        165
    );
    r.shot_end().unwrap();
    r.shot_start(1, 9).unwrap();
    let q = r.qalloc().unwrap();
    r.rxy_gate(q, 1.0, 0.0).unwrap();
    r.global_barrier(0).unwrap();
    assert_eq!(
        u64::from(drain(&mut r)[0].runtime_source().unwrap().start()),
        175
    );
    assert!(r.simulate_delay(u64::MAX).is_err());
}
#[test]
fn local_sleep_does_not_delay_an_unrelated_qubit() {
    let mut r = runtime(RuntimeConfig {
        duration_ns_rxy: 10,
        ..Default::default()
    });
    r.local_barrier(&[0], 1000).unwrap();
    r.rxy_gate(1, 1.0, 0.0).unwrap();
    r.global_barrier(0).unwrap();
    assert_eq!(
        u64::from(drain(&mut r)[0].runtime_source().unwrap().start()),
        100
    );
}
#[test]
fn retirement_before_execution_still_accepts_feedback_but_not_reads() {
    let mut r = runtime(RuntimeConfig {
        max_live_results: Some(1),
        ..Default::default()
    });
    let a = r.measure_leaked(0).unwrap();
    r.decrement_future_refcount(a).unwrap();
    assert!(r.force_result(a).is_err());
    assert!(r.measure(1).is_err());
    r.global_barrier(0).unwrap();
    drain(&mut r);
    r.set_u64_result(a, 2).unwrap();
    assert!(r.get_u64_result(a).is_err());
    let b = r.measure(1).unwrap();
    assert_ne!(a, b);
}
#[test]
fn queue_pressure_releases_work_and_rejects_undrained_growth() {
    let mut r = runtime(RuntimeConfig {
        max_queued_batches: 2,
        ..Default::default()
    });
    r.rxy_gate(0, 1.0, 0.0).unwrap();
    assert!(drain(&mut r).is_empty());
    r.rxy_gate(0, 1.0, 0.0).unwrap();
    assert!(r.rxy_gate(0, 1.0, 0.0).is_err());
    assert_eq!(drain(&mut r).len(), 2);
    r.rxy_gate(0, 1.0, 0.0).unwrap();
}
#[test]
fn allocation_reuse_preserves_order_and_clears_virtual_phase() {
    let mut r = runtime(RuntimeConfig {
        max_batch_size: 3,
        ..Default::default()
    });
    r.rz_gate(0, 0.8).unwrap();
    r.rxy_gate(0, 1.0, 0.0).unwrap();
    let a = r.measure(0).unwrap();
    r.qfree(0).unwrap();
    assert!(r.qfree(0).is_err());
    assert_eq!(r.qalloc().unwrap(), 0);
    r.reset(0).unwrap();
    r.rxy_gate(0, 1.0, 0.0).unwrap();
    r.global_barrier(0).unwrap();
    let ops: Vec<_> = drain(&mut r).into_iter().flatten().collect();
    assert!(matches!(ops[0], Operation::RXYGate { phi: -0.8, .. }));
    assert!(matches!(ops[1], Operation::Measure { .. }));
    assert!(matches!(ops[2], Operation::Reset { .. }));
    assert!(matches!(ops[3], Operation::RXYGate { phi: 0.0, .. }));
    r.set_bool_result(a, false).unwrap();
}
#[test]
fn invalid_feedback_and_unfinished_shots_are_rejected() {
    let mut r = runtime(RuntimeConfig::default());
    let a = r.measure(0).unwrap();
    assert!(r.set_bool_result(a, true).is_err());
    assert!(r.shot_start(1, 1).is_err());
    r.force_result(a).unwrap();
    drain(&mut r);
    assert!(r.set_u64_result(a, 2).is_err());
    r.set_bool_result(a, true).unwrap();
    assert!(r.set_bool_result(a, false).is_err());
    r.shot_end().unwrap();
    r.exit().unwrap();
    assert!(r.qalloc().is_err());
}

#[test]
fn selene_shutdown_flushes_unread_measurements_before_next_shot() {
    let mut r = runtime(RuntimeConfig::default());
    let a = r.measure(0).unwrap();
    r.rxy_gate(1, 1.0, 0.0).unwrap();
    assert!(drain(&mut r).is_empty());
    r.shot_end().unwrap();
    assert!(r.shot_start(1, 0).is_err());
    assert_eq!(drain(&mut r).len(), 2);
    assert!(r.shot_start(1, 0).is_err());
    r.set_bool_result(a, false).unwrap();
    r.shot_start(1, 0).unwrap();
    assert_eq!(r.qalloc().unwrap(), 0);
}

#[test]
fn independent_measurements_never_pass_earlier_measurements() {
    let mut r = runtime(RuntimeConfig {
        max_batch_size: 2,
        ..Default::default()
    });
    let a = r.measure(2).unwrap();
    r.rxy_gate(0, 1.0, 0.0).unwrap();
    let b = r.measure(0).unwrap();
    let c = r.measure(1).unwrap();
    r.shot_end().unwrap();
    let ids: Vec<_> = drain(&mut r)
        .into_iter()
        .flatten()
        .filter_map(|op| {
            if let Operation::Measure { result_id, .. } = op {
                Some(result_id)
            } else {
                None
            }
        })
        .collect();
    assert_eq!(ids, [a, b, c]);
}

#[test]
fn shot_clock_preserves_barriers_without_emitted_work() {
    for global in [false, true] {
        let mut r = runtime(RuntimeConfig::default());
        if global {
            r.global_barrier(1000).unwrap();
        } else {
            r.local_barrier(&[0], 1000).unwrap();
            r.qfree(0).unwrap();
        }
        assert!(drain(&mut r).is_empty());
        r.shot_end().unwrap();
        r.shot_start(1, 9).unwrap();
        let q = r.qalloc().unwrap();
        r.rxy_gate(q, 1.0, 0.0).unwrap();
        r.global_barrier(0).unwrap();
        assert_eq!(
            u64::from(drain(&mut r)[0].runtime_source().unwrap().start()),
            1100
        );
    }
}

#[test]
fn default_retains_readable_futures_beyond_the_former_limit() {
    let mut r = runtime(RuntimeConfig::default());
    let first = r.measure(0).unwrap();
    r.force_result(first).unwrap();
    drain(&mut r);
    r.set_bool_result(first, true).unwrap();
    for _ in 0..65_536 {
        let id = r.measure(0).unwrap();
        r.force_result(id).unwrap();
        drain(&mut r);
        r.set_bool_result(id, false).unwrap();
    }
    assert_eq!(r.get_bool_result(first).unwrap(), Some(true));
    assert_eq!(r.get_bool_result(first).unwrap(), Some(true));
}

#[test]
fn explicit_result_limit_preserves_live_handles_and_reclaims_retired_ones() {
    let mut r = runtime(RuntimeConfig {
        max_live_results: Some(1),
        ..Default::default()
    });
    let id = r.measure(0).unwrap();
    r.force_result(id).unwrap();
    drain(&mut r);
    r.set_bool_result(id, true).unwrap();
    r.increment_future_refcount(id).unwrap();
    r.decrement_future_refcount(id).unwrap();
    assert!(r.measure(1).is_err());
    assert_eq!(r.get_bool_result(id).unwrap(), Some(true));
    r.decrement_future_refcount(id).unwrap();
    assert!(r.get_bool_result(id).is_err());
    assert!(r.measure(1).is_ok());
}

#[test]
fn two_qubit_dependencies_survive_independent_pulse_packing() {
    for max_batch_size in [1, 2, 8] {
        let mut r = runtime(RuntimeConfig {
            max_batch_size,
            duration_ns_rxy: 10,
            duration_ns_rzz: 30,
            ..Default::default()
        });
        r.rxy_gate(0, 0.1, 0.0).unwrap();
        r.rzz_gate(0, 1, 0.2).unwrap();
        r.rxy_gate(1, 0.3, 0.0).unwrap();
        r.rxy_gate(2, 0.4, 0.0).unwrap();
        r.rzz_gate(1, 2, 0.5).unwrap();
        r.global_barrier(0).unwrap();
        let batches = drain(&mut r);
        let timings: Vec<_> = batches
            .iter()
            .map(|batch| {
                let source = batch.runtime_source().unwrap();
                (
                    u64::from(source.start()),
                    u64::from(source.end()),
                    batch.len(),
                )
            })
            .collect();
        if max_batch_size == 1 {
            assert_eq!(
                timings,
                [
                    (100, 110, 1),
                    (110, 140, 1),
                    (140, 150, 1),
                    (150, 160, 1),
                    (160, 190, 1)
                ]
            );
        } else {
            assert_eq!(
                timings,
                [(100, 110, 2), (110, 140, 1), (140, 150, 1), (150, 180, 1)]
            );
            assert_eq!(
                batches[0].iter_ops().cloned().collect::<Vec<_>>(),
                [
                    Operation::RXYGate {
                        qubit_id: 0,
                        theta: 0.1,
                        phi: 0.0
                    },
                    Operation::RXYGate {
                        qubit_id: 2,
                        theta: 0.4,
                        phi: 0.0
                    },
                ]
            );
        }
    }
}
