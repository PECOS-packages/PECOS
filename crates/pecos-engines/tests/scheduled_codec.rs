//! Independent wire fixtures for the v6 codec and nested v7 envelope.
use pecos_core::Angle64;
use pecos_engines::scheduled_events::{
    ScheduledEventBatch, ScheduledEventOp, ScheduledResult, decode_event_batches,
    encode_event_batches,
};
use pecos_engines::scheduled_frame::encode_timed_batches;
use pecos_engines::{ByteMessage, Gate};

fn words(out: &mut Vec<u8>, values: &[u64]) {
    for value in values {
        out.extend_from_slice(&value.to_le_bytes());
    }
}
fn header(version: u32, count: u32, size: usize) -> Vec<u8> {
    [
        pecos_engines::byte_message::protocol::BATCH_MAGIC,
        version,
        count,
        u32::try_from(size).unwrap(),
    ]
    .into_iter()
    .flat_map(u32::to_le_bytes)
    .collect()
}
fn v6(index: u64, records: &[[u64; 5]]) -> Vec<u8> {
    let mut out = header(6, 1, 56 + 40 * records.len());
    words(&mut out, &[17, index, 9, 2, records.len() as u64]);
    for record in records {
        words(&mut out, record);
    }
    out
}
fn fixture() -> (Vec<ScheduledEventBatch>, Vec<u8>, Vec<Vec<u8>>) {
    let gates = vec![
        Gate::rxy1q(Angle64::new(1), Angle64::new(u64::MAX), &[0]),
        Gate::rz(Angle64::new(u64::MAX), &[1]),
        Gate::rzz(Angle64::new(1 << 63), &[(0, 1)]),
        Gate::rxyxy2q(Angle64::new(3), Angle64::new(5), &[(1, 2)]),
        Gate::pz(&[2]),
        Gate::mz(&[0]),
        Gate::measure_leaked(&[1]),
    ];
    let records = [
        [1, 0, 0, 1, u64::MAX],
        [2, 1, 0, u64::MAX, 0],
        [3, 0, 1, 1 << 63, 0],
        [4, 1, 2, 3, 5],
        [5, 2, 0, 0, 0],
        [6, 0, 0, 0, 0],
        [7, 1, 0, 0, 0],
    ];
    let mut operations: Vec<_> = gates
        .into_iter()
        .map(|g| ScheduledEventOp::Gate(Box::new(g)))
        .collect();
    operations.insert(
        1,
        ScheduledEventOp::Custom {
            tag: 23,
            payload: vec![2, 3, 4],
        },
    );
    let batches = vec![
        ScheduledEventBatch {
            runtime_shot_id: 17,
            batch_index: 0,
            start_nanos: 9,
            duration_nanos: 2,
            operations: vec![ScheduledEventOp::Custom {
                tag: 19,
                payload: vec![1],
            }],
            measurements: vec![],
        },
        ScheduledEventBatch {
            runtime_shot_id: 17,
            batch_index: 1,
            start_nanos: 9,
            duration_nanos: 2,
            operations,
            measurements: vec![
                ScheduledResult {
                    operation_index: 6,
                    runtime_result: 8,
                    program_result: 31,
                },
                ScheduledResult {
                    operation_index: 7,
                    runtime_result: 9,
                    program_result: 31,
                },
            ],
        },
        ScheduledEventBatch {
            runtime_shot_id: 17,
            batch_index: 2,
            start_nanos: 9,
            duration_nanos: 2,
            operations: vec![],
            measurements: vec![],
        },
    ];
    let nested = vec![v6(0, &[]), v6(1, &records), v6(2, &[])];
    let mut out = header(7, 3, 0);
    words(&mut out, &[nested[0].len() as u64]);
    out.extend_from_slice(&nested[0]);
    words(&mut out, &[1, 0, 19, 1]);
    out.push(1);
    words(&mut out, &[0]);
    words(&mut out, &[nested[1].len() as u64]);
    out.extend_from_slice(&nested[1]);
    words(&mut out, &[1, 1, 23, 3]);
    out.extend_from_slice(&[2, 3, 4]);
    words(&mut out, &[2, 6, 8, 31, 7, 9, 31]);
    words(&mut out, &[nested[2].len() as u64]);
    out.extend_from_slice(&nested[2]);
    words(&mut out, &[0, 0]);
    let size = u32::try_from(out.len()).unwrap();
    out[12..16].copy_from_slice(&size.to_le_bytes());
    (batches, out, nested)
}

#[test]
fn codecs_match_independent_wire_with_odd_payloads_and_exact_angles() {
    let (batches, expected, nested) = fixture();
    assert_eq!(encode_event_batches(&batches).unwrap().as_bytes(), expected);
    for (batch, wire) in batches.iter().zip(&nested) {
        assert_eq!(
            encode_timed_batches(&[batch.source_gates()])
                .unwrap()
                .as_bytes(),
            wire
        );
    }
    let decoded = decode_event_batches(&ByteMessage::new(&expected)).unwrap();
    for (before, after) in batches.iter().zip(decoded) {
        assert_eq!(
            (
                after.runtime_shot_id,
                after.batch_index,
                after.start_nanos,
                after.duration_nanos
            ),
            (
                before.runtime_shot_id,
                before.batch_index,
                before.start_nanos,
                before.duration_nanos
            )
        );
        assert_eq!(after.measurements, before.measurements);
        assert_eq!(after.operations.len(), before.operations.len());
        for (a, b) in before.operations.iter().zip(after.operations) {
            match (a, b) {
                (ScheduledEventOp::Gate(a), ScheduledEventOp::Gate(b)) => {
                    assert_eq!(a.as_ref(), b.as_ref());
                }
                (
                    ScheduledEventOp::Custom {
                        tag: a,
                        payload: pa,
                    },
                    ScheduledEventOp::Custom {
                        tag: b,
                        payload: pb,
                    },
                ) => assert_eq!((*a, pa.as_slice()), (b, pb.as_slice())),
                _ => panic!("operation kind changed"),
            }
        }
    }
    assert_eq!(
        encode_event_batches(&[]).unwrap().as_bytes(),
        header(7, 0, 16)
    );
    assert_eq!(
        encode_timed_batches(&[]).unwrap().as_bytes(),
        header(6, 0, 16)
    );
}

#[test]
fn nested_frames_reject_truncated_and_reserved_data() {
    let (_, expected, _) = fixture();
    // Repair the outer length to force the nested parser to inspect each truncated body.
    for end in 16..expected.len() {
        let mut truncated = expected[..end].to_vec();
        truncated[12..16].copy_from_slice(&u32::try_from(end).unwrap().to_le_bytes());
        assert!(
            decode_event_batches(&ByteMessage::new(&truncated)).is_err(),
            "accepted prefix {end}"
        );
    }
    // The second frame is unaligned because the first custom payload has one byte.
    let second = 16 + 8 + 56 + 32 + 1 + 8 + 8;
    assert_ne!(second % 4, 0);
    for offset in [second + 5, second + 12, second + 56, second + 56 + 16] {
        let mut malformed = expected.clone();
        malformed[offset] ^= 0x80;
        assert!(
            decode_event_batches(&ByteMessage::new(&malformed)).is_err(),
            "accepted mutation {offset}"
        );
    }
}
