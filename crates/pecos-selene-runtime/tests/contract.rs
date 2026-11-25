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
        r.global_barrier(0).unwrap();
        assert_eq!(drain(&mut r).len(), 1);
        r.shot_end().unwrap();
    }
    // A leakage outcome cannot be read as a Boolean while the shot runs.
    let mut r = runtime(RuntimeConfig::default());
    let id = r.measure_leaked(0).unwrap();
    r.force_result(id).unwrap();
    drain(&mut r);
    r.set_u64_result(id, 2).unwrap();
    let message = r.get_bool_result(id).unwrap_err().to_string();
    assert!(message.contains("leakage outcome"), "{message}");
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
fn barriers_and_delays_have_checked_timing_and_reset_between_shots() {
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
        100
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
    let retired = || {
        let mut r = runtime(RuntimeConfig {
            max_live_results: Some(1),
            ..Default::default()
        });
        let a = r.measure_leaked(0).unwrap();
        r.decrement_future_refcount(a).unwrap();
        (r, a)
    };
    let (mut r, a) = retired();
    assert!(r.force_result(a).is_err());
    // The retired future still counts against the limit until its feedback arrives.
    let (mut r, _) = retired();
    assert!(r.measure(1).is_err());
    let (mut r, a) = retired();
    r.global_barrier(0).unwrap();
    drain(&mut r);
    r.set_u64_result(a, 2).unwrap();
    let b = r.measure(1).unwrap();
    assert_ne!(a, b);
    assert!(r.get_u64_result(a).is_err());
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
    // Reaching the limit releases the queue, so draining makes room.
    assert_eq!(drain(&mut r).len(), 2);
    r.rxy_gate(0, 1.0, 0.0).unwrap();

    let mut r = runtime(RuntimeConfig {
        max_queued_batches: 2,
        ..Default::default()
    });
    r.rxy_gate(0, 1.0, 0.0).unwrap();
    r.rxy_gate(0, 1.0, 0.0).unwrap();
    assert!(r.rxy_gate(0, 1.0, 0.0).is_err());
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

    let mut r = runtime(RuntimeConfig::default());
    r.qfree(0).unwrap();
    assert!(r.qfree(0).is_err());
}
#[test]
fn invalid_feedback_and_unfinished_shots_are_rejected() {
    let measured = || {
        let mut r = runtime(RuntimeConfig::default());
        let a = r.measure(0).unwrap();
        (r, a)
    };
    let emitted = || {
        let (mut r, a) = measured();
        r.force_result(a).unwrap();
        drain(&mut r);
        (r, a)
    };
    let (mut r, a) = measured();
    assert!(
        r.set_bool_result(a, true).is_err(),
        "feedback before emission"
    );
    let (mut r, _) = measured();
    assert!(r.shot_start(1, 1).is_err(), "unfinished shot");
    let (mut r, a) = emitted();
    assert!(
        r.set_u64_result(a, 2).is_err(),
        "leakage outcome for a plain measurement"
    );
    let (mut r, a) = emitted();
    r.set_bool_result(a, true).unwrap();
    assert!(r.set_bool_result(a, false).is_err(), "second delivery");
    let (mut r, a) = emitted();
    r.set_bool_result(a, true).unwrap();
    r.shot_end().unwrap();
    r.exit().unwrap();
    assert!(r.qalloc().is_err());
}

#[test]
fn selene_shutdown_flushes_unread_measurements_before_next_shot() {
    let ended = || {
        let mut r = runtime(RuntimeConfig::default());
        let a = r.measure(0).unwrap();
        r.rxy_gate(1, 1.0, 0.0).unwrap();
        assert!(drain(&mut r).is_empty());
        r.shot_end().unwrap();
        (r, a)
    };
    let (mut r, _) = ended();
    assert!(r.shot_start(1, 0).is_err(), "shutdown work not drained");
    let (mut r, _) = ended();
    assert_eq!(drain(&mut r).len(), 2);
    assert!(r.shot_start(1, 0).is_err(), "feedback not delivered");
    let (mut r, a) = ended();
    assert_eq!(drain(&mut r).len(), 2);
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
fn trailing_barriers_and_delays_do_not_reach_the_next_shot() {
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
            100
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
    assert_eq!(r.get_bool_result(id).unwrap(), Some(true));
    r.decrement_future_refcount(id).unwrap();
    assert!(r.measure(1).is_ok());
    assert!(r.get_bool_result(id).is_err());

    let mut r = runtime(RuntimeConfig {
        max_live_results: Some(1),
        ..Default::default()
    });
    let id = r.measure(0).unwrap();
    r.force_result(id).unwrap();
    drain(&mut r);
    r.set_bool_result(id, true).unwrap();
    assert!(r.measure(1).is_err(), "a live handle holds the only slot");
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

#[test]
fn large_cancelling_virtual_rotations_preserve_small_rotations() {
    for large in [2_f64.powi(54), 1e100, f64::MAX] {
        let mut r = runtime(RuntimeConfig::default());
        for theta in [large, 1.0, -large] {
            r.rz_gate(0, theta).unwrap();
        }
        r.rxy_gate(0, std::f64::consts::FRAC_PI_2, 0.0).unwrap();
        r.global_barrier(0).unwrap();
        let ops: Vec<_> = drain(&mut r).into_iter().flatten().collect();
        let Operation::RXYGate { phi, .. } = ops[0] else {
            panic!("expected pulse")
        };
        // The two large Z rotations cancel exactly, leaving a one-radian Z.
        assert!(
            (phi.sin() - (-1.0_f64).sin()).abs() < 1e-12,
            "large={large}, phi={phi}"
        );
        assert!((phi.cos() - 1.0_f64.cos()).abs() < 1e-12);
    }
}

#[test]
fn large_pulse_axis_preserves_small_virtual_frame() {
    for phi in [2_f64.powi(54), -1e100, f64::MAX] {
        let mut r = runtime(RuntimeConfig::default());
        r.rz_gate(0, 0.25).unwrap();
        r.rxy_gate(0, 1.0, phi).unwrap();
        r.global_barrier(0).unwrap();
        let ops: Vec<_> = drain(&mut r).into_iter().flatten().collect();
        let Operation::RXYGate { phi: actual, .. } = ops[0] else {
            panic!("expected pulse")
        };
        // Independent angle-subtraction identities avoid subtracting 0.25 from
        // a huge floating-point angle when constructing the expected axis.
        let (s, c) = phi.sin_cos();
        let (sf, cf) = 0.25_f64.sin_cos();
        assert!((actual.sin() - (s * cf - c * sf)).abs() < 1e-12);
        assert!((actual.cos() - (c * cf + s * sf)).abs() < 1e-12);
    }
}

#[test]
fn finite_rotations_do_not_overflow_the_virtual_frame() {
    let mut r = runtime(RuntimeConfig::default());
    for theta in [f64::MAX, f64::MAX, -f64::MAX, -f64::MAX] {
        r.rz_gate(0, theta).unwrap();
    }
    r.rxy_gate(0, 1.0, f64::MAX).unwrap();
    r.rxy_gate(0, 1.0, 0.0).unwrap();
    r.global_barrier(0).unwrap();
    let ops: Vec<_> = drain(&mut r).into_iter().flatten().collect();
    let Operation::RXYGate { phi, .. } = ops[1] else {
        panic!("expected pulse")
    };
    assert!(phi.abs() < 1e-12, "the frame drifted: {phi}");
    for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(
            runtime(RuntimeConfig::default())
                .rz_gate(0, invalid)
                .is_err()
        );
        assert!(
            runtime(RuntimeConfig::default())
                .rxy_gate(0, 1.0, invalid)
                .is_err()
        );
    }
}

#[test]
fn unallocatable_qubit_capacity_is_an_error_not_an_abort() {
    for n_qubits in [u64::MAX, 1 << 62] {
        let error = PecosRuntime::new(n_qubits, 0.into(), RuntimeConfig::default())
            .err()
            .expect("capacity cannot be allocated");
        assert!(error.to_string().contains(&n_qubits.to_string()), "{error}");
    }
}

#[test]
fn caller_errors_name_the_offending_values() {
    let r = || runtime(RuntimeConfig::default());
    let message = |error: anyhow::Error| error.to_string();
    assert!(message(r().rzz_gate(2, 2, 0.1).unwrap_err()).contains("got 2 twice"));
    assert!(message(r().rzz_gate(0, 1, f64::NAN).unwrap_err()).contains("theta=NaN"));
    assert!(message(r().rz_gate(1, f64::INFINITY).unwrap_err()).contains("qubit 1"));
    assert!(message(r().rxy_gate(0, 1.0, f64::NAN).unwrap_err()).contains("phi=NaN"));
    let limit = PecosRuntime::new(
        1,
        0.into(),
        RuntimeConfig {
            max_queued_batches: 0,
            ..Default::default()
        },
    );
    assert!(message(limit.err().unwrap()).contains("max_queued_batches"));
}

#[test]
fn an_error_leaves_the_runtime_able_only_to_exit() {
    let mut r = runtime(RuntimeConfig::default());
    assert!(r.rz_gate(0, f64::NAN).is_err());
    for later in [
        r.rz_gate(0, 0.5),
        r.global_barrier(0),
        r.shot_end(),
        r.shot_start(1, 0),
    ] {
        let message = later.unwrap_err().to_string();
        assert!(message.contains("earlier error"), "{message}");
        assert!(
            message.contains("theta=NaN"),
            "names the first error: {message}"
        );
    }
    assert!(r.qalloc().is_err());
    r.exit().unwrap();
}

#[test]
fn a_failed_exit_is_the_first_error() {
    let mut r = runtime(RuntimeConfig::default());
    r.exit().unwrap();
    assert!(r.exit().is_err());
    let message = r.custom_call(1, &[]).unwrap_err().to_string();
    assert!(message.contains("runtime has exited"), "{message}");
}

#[test]
fn shot_start_requires_the_running_shot_to_end() {
    let mut r = PecosRuntime::new(1, 0.into(), RuntimeConfig::default()).unwrap();
    r.shot_start(0, 0).unwrap();
    // Nothing is left over, but the shot was never ended.
    let q = r.qalloc().unwrap();
    r.qfree(q).unwrap();
    let message = r.shot_start(1, 0).unwrap_err().to_string();
    assert!(message.contains("shot_end"), "{message}");

    let mut r = PecosRuntime::new(1, 0.into(), RuntimeConfig::default()).unwrap();
    r.shot_start(0, 0).unwrap();
    r.shot_end().unwrap();
    r.shot_start(1, 0).unwrap();
}

#[test]
fn shot_start_rejects_state_created_before_the_first_shot() {
    let fresh = || PecosRuntime::new(1, 0.into(), RuntimeConfig::default()).unwrap();
    let mut r = fresh();
    r.qalloc().unwrap();
    let message = r.shot_start(0, 0).unwrap_err().to_string();
    assert!(message.contains("before the first shot"), "{message}");

    // A delivered but still referenced future would otherwise be discarded.
    let mut r = fresh();
    let q = r.qalloc().unwrap();
    let id = r.measure(q).unwrap();
    r.force_result(id).unwrap();
    drain(&mut r);
    r.set_bool_result(id, true).unwrap();
    r.qfree(q).unwrap();
    drain(&mut r);
    let message = r.shot_start(0, 0).unwrap_err().to_string();
    assert!(message.contains("before the first shot"), "{message}");
}

#[test]
fn unsupported_calls_also_retire_the_runtime() {
    let calls: [fn(&mut PecosRuntime) -> bool; 2] = [
        |r| r.custom_call(1, &[]).is_err(),
        |r| r.rpp_gate(0, 1, 0.1, 0.2).is_err(),
    ];
    for unsupported in calls {
        let mut r = runtime(RuntimeConfig::default());
        assert!(unsupported(&mut r));
        let message = r.qalloc().unwrap_err().to_string();
        assert!(message.contains("earlier error"), "{message}");
    }
    // `simulate_delay` is supported, and its overflow error retires the runtime too.
    let mut r = runtime(RuntimeConfig::default());
    r.simulate_delay(10).unwrap();
    assert!(r.simulate_delay(u64::MAX).is_err());
    let message = r.qalloc().unwrap_err().to_string();
    assert!(message.contains("earlier error"), "{message}");
}
