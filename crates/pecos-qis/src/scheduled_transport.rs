//! Admission of extracted native operations before mandatory transport encoding.
use crate::scheduled::{RuntimeScheduledOp as Op, ScheduledBatch};
use pecos_core::{Angle64, errors::PecosError};
use pecos_engines::scheduled_events::{
    ScheduledEventBatch, ScheduledEventOp, ScheduledResult, encode_event_batches,
};
use pecos_engines::scheduled_frame::{TimedBatch, encode_timed_batches};
use pecos_engines::{ByteMessage, Gate};
use std::collections::{BTreeMap, BTreeSet};
fn error(s: &str) -> PecosError {
    PecosError::Input(s.into())
}
fn angle(a: f64) -> Result<Angle64, PecosError> {
    if !a.is_finite() {
        return Err(error("nonfinite scheduled angle"));
    }
    Ok(Angle64::from_radians(a))
}
fn q(q: u64) -> Result<usize, PecosError> {
    usize::try_from(q).map_err(|_| error("scheduled target overflow"))
}
#[cfg(test)]
fn encode(
    batches: Vec<ScheduledBatch>,
    shot: u64,
) -> Result<(ByteMessage, Vec<usize>), PecosError> {
    encode_mode(batches, shot, false)
}
pub(crate) fn encode_mode(
    batches: Vec<ScheduledBatch>,
    shot: u64,
    events: bool,
) -> Result<(ByteMessage, Vec<usize>), PecosError> {
    if events {
        // Bound conversion allocations even for a custom QisRuntime implementation.
        let mut size = 16usize;
        for batch in &batches {
            if batch.operations.len() > pecos_engines::scheduled_events::MAX_BATCH_OPERATIONS
                || batch.measurements.len() > batch.operations.len()
            {
                return Err(error("scheduled event operation count limit"));
            }
            let mut payload = 0usize;
            size = size
                .checked_add(80 + batch.measurements.len() * 24)
                .ok_or_else(|| error("event transport overflow"))?;
            for op in &batch.operations {
                let bytes = if let Op::Custom { data, .. } = op {
                    payload = payload
                        .checked_add(data.len())
                        .ok_or_else(|| error("event payload overflow"))?;
                    if payload > pecos_engines::scheduled_events::MAX_BATCH_PAYLOAD {
                        return Err(error("event payload limit"));
                    }
                    24 + data.len()
                } else {
                    40
                };
                size = size
                    .checked_add(bytes)
                    .ok_or_else(|| error("event transport overflow"))?;
            }
            if size > pecos_engines::scheduled_frame::MAX_SCHEDULE_BYTES {
                return Err(error("event transport limit"));
            }
        }
    }
    let mut event_wire = Vec::new();
    let mut wire = Vec::new();
    let mut ids = Vec::new();
    let mut native = BTreeSet::new();
    let mut source = BTreeSet::new();
    for batch in batches {
        if batch.runtime_shot_id != shot {
            return Err(error("scheduled runtime shot mismatch"));
        }
        let mut mappings = BTreeMap::new();
        for m in &batch.measurements {
            if m.operation_index >= batch.operations.len()
                || mappings.insert(m.operation_index, m).is_some()
            {
                return Err(error("invalid scheduled measurement position"));
            }
        }
        let mut gates = Vec::new();
        let mut event_ops = Vec::new();
        let mut event_measurements = Vec::new();
        for (index, op) in batch.operations.iter().enumerate() {
            let gate = match op {
                Op::Rxy {
                    qubit_id,
                    theta,
                    phi,
                } => Gate::rxy1q(angle(*theta)?, angle(*phi)?, &[q(*qubit_id)?]),
                Op::Rz { qubit_id, theta } => Gate::rz(angle(*theta)?, &[q(*qubit_id)?]),
                Op::Rzz {
                    qubit_id_1,
                    qubit_id_2,
                    theta,
                } => Gate::rzz(angle(*theta)?, &[(q(*qubit_id_1)?, q(*qubit_id_2)?)]),
                Op::Rpp {
                    qubit_id_1,
                    qubit_id_2,
                    theta,
                    phi,
                } => Gate::rxyxy2q(
                    angle(*theta)?,
                    angle(*phi)?,
                    &[(q(*qubit_id_1)?, q(*qubit_id_2)?)],
                ),
                Op::Reset { qubit_id } => Gate::pz(&[q(*qubit_id)?]),
                Op::Measure {
                    qubit_id,
                    result_id,
                }
                | Op::MeasureLeaked {
                    qubit_id,
                    result_id,
                } => {
                    let m = mappings
                        .remove(&index)
                        .ok_or_else(|| error("missing scheduled measurement mapping"))?;
                    if m.runtime_result != *result_id
                        || !native.insert(*result_id)
                        || !source.insert(m.program_result)
                        || (matches!(op, Op::MeasureLeaked { .. }) && !m.leakage_aware)
                    {
                        return Err(error("invalid or duplicate scheduled measurement identity"));
                    }
                    ids.push(m.program_result);
                    if events {
                        event_measurements.push(ScheduledResult {
                            operation_index: index,
                            runtime_result: m.runtime_result,
                            program_result: m.program_result as u64,
                        });
                    }
                    if m.leakage_aware {
                        Gate::measure_leaked(&[q(*qubit_id)?])
                    } else {
                        Gate::mz(&[q(*qubit_id)?])
                    }
                }
                Op::Custom { tag, data } if events => {
                    event_ops.push(ScheduledEventOp::Custom {
                        tag: *tag as u64,
                        payload: data.clone(),
                    });
                    continue;
                }
                Op::Custom { .. } => {
                    return Err(error(
                        "scheduled idle transport does not support custom events",
                    ));
                }
            };
            if events {
                event_ops.push(ScheduledEventOp::Gate(Box::new(gate)));
            } else {
                gates.push(gate);
            }
        }
        if !mappings.is_empty() {
            return Err(error("mapping on scheduled non-measurement"));
        }
        if events {
            event_wire.push(ScheduledEventBatch {
                runtime_shot_id: batch.runtime_shot_id,
                batch_index: batch.batch_index as u64,
                start_nanos: batch.start_time_nanos,
                duration_nanos: batch.duration_nanos,
                operations: event_ops,
                measurements: event_measurements,
            });
        } else {
            wire.push(TimedBatch {
                runtime_shot_id: batch.runtime_shot_id,
                batch_index: batch.batch_index as u64,
                start_nanos: batch.start_time_nanos,
                duration_nanos: batch.duration_nanos,
                gates,
            });
        }
    }
    Ok((
        if events {
            encode_event_batches(&event_wire)?
        } else {
            encode_timed_batches(&wire)?
        },
        ids,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scheduled::ScheduledMeasurement;

    fn measured() -> ScheduledBatch {
        ScheduledBatch {
            runtime_shot_id: 7,
            batch_index: 0,
            start_time_nanos: 0,
            duration_nanos: 0,
            operations: vec![Op::Measure {
                qubit_id: 0,
                result_id: 23,
            }],
            measurements: vec![ScheduledMeasurement {
                operation_index: 0,
                runtime_result: 23,
                program_result: 91,
                leakage_aware: false,
            }],
        }
    }
    #[test]
    fn event_transport_retains_positions_and_executes_in_quantum_owner() {
        use pecos_engines::noise::IntoNoiseModel;
        use pecos_engines::runtime_frame::ShotContext;
        use pecos_engines::scheduled_events::{
            ScheduledBatchAdapter, ScheduledEventIdleZ, ScheduledGateBuffer, decode_event_batches,
        };
        use pecos_engines::scheduled_frame::ScheduledIdleZ;
        use pecos_engines::{Engine, StateVecEngine, quantum_system::QuantumSystem};
        struct Flip;
        impl ScheduledBatchAdapter for Flip {
            fn validate(&self, b: &ScheduledEventBatch) -> Result<(), PecosError> {
                for op in &b.operations {
                    if let ScheduledEventOp::Custom { tag, payload } = op
                        && (*tag != 901 || payload.as_slice() != [1])
                    {
                        return Err(error("unsupported synthetic event"));
                    }
                }
                Ok(())
            }
            fn translate(
                &mut self,
                b: &ScheduledEventBatch,
                out: &mut ScheduledGateBuffer<'_>,
            ) -> Result<(), PecosError> {
                for op in &b.operations {
                    out.push(match op {
                        ScheduledEventOp::Gate(g) => g.as_ref().clone(),
                        ScheduledEventOp::Custom { .. } => Gate::rxy1q(
                            Angle64::from_radians(std::f64::consts::PI),
                            Angle64::from_radians(0.0),
                            &[0],
                        ),
                    })?;
                }
                Ok(())
            }
        }
        let mut b = measured();
        b.batch_index = 0;
        b.operations.insert(
            0,
            Op::Custom {
                tag: 901,
                data: vec![1],
            },
        );
        b.measurements[0].operation_index = 1;
        assert!(encode(vec![b.clone()], 7).is_err());
        let (commands, ids) = encode_mode(vec![b], 7, true).unwrap();
        assert_eq!(ids, vec![91]);
        let decoded = decode_event_batches(&commands).unwrap();
        assert_eq!(decoded[0].measurements[0].runtime_result, 23);
        assert_eq!(decoded[0].measurements[0].program_result, 91);
        assert_eq!(decoded[0].measurements[0].operation_index, 1);
        let mut q = QuantumSystem::new(
            ScheduledEventIdleZ::new(ScheduledIdleZ::new(1, 0.0, 0.0, 0.0).unwrap(), |_| {
                Ok(Box::new(Flip))
            })
            .into_noise_model(),
            Box::new(StateVecEngine::new(1)),
        );
        q.begin_shot(ShotContext {
            run: 1,
            worker: 2,
            shot: 3,
        })
        .unwrap();
        assert_eq!(q.process(commands).unwrap().outcomes().unwrap(), vec![1]);
    }

    #[test]
    fn maps_distinct_namespaces_and_rejects_ambiguous_measurements() {
        assert_eq!(encode(vec![measured()], 7).unwrap().1, vec![91]);
        for case in 0..6 {
            let mut b = measured();
            match case {
                0 => b.measurements.clear(),
                1 => b.measurements[0].operation_index = 1,
                2 => b.measurements[0].runtime_result = 24,
                3 => b.measurements.push(b.measurements[0].clone()),
                4 => b.operations[0] = Op::Reset { qubit_id: 0 },
                _ => {
                    b.operations[0] = Op::MeasureLeaked {
                        qubit_id: 0,
                        result_id: 23,
                    }
                }
            }
            assert!(encode(vec![b], 7).is_err());
        }
        let a = measured();
        let mut b = measured();
        b.batch_index = 1;
        assert!(encode(vec![a.clone(), b.clone()], 7).is_err());
        b.operations[0] = Op::Measure {
            qubit_id: 0,
            result_id: 24,
        };
        b.measurements[0].runtime_result = 24;
        assert!(encode(vec![a, b], 7).is_err());
    }
    #[test]
    fn rejects_opaque_events_wrong_shots_and_nonfinite_angles() {
        assert!(encode(vec![measured()], 8).is_err());
        for op in [
            Op::Custom {
                tag: 3,
                data: vec![],
            },
            Op::Rz {
                qubit_id: 0,
                theta: f64::NAN,
            },
        ] {
            let mut b = measured();
            b.operations = vec![op];
            b.measurements.clear();
            assert!(encode(vec![b], 7).is_err());
        }
    }
}
