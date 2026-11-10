// Copyright 2026 The PECOS Developers
// Licensed under the Apache License, Version 2.0

use super::*;
use serde_json::json;

fn fixture() -> Value {
    json!([
        {
            "format": "pecos_qis_operation_trace_v1",
            "engine_trace_id": 1, "shot_index": 0, "chunk_index": 0,
            "stage": "execute", "num_operations": 2,
            "operations": [{"Quantum": {"Measure": [10, 7]}}, {"Quantum": {"MeasureLeaked": [11, 2]}}],
            "lowered_quantum_ops_complete": true,
            "lowered_quantum_ops": [
                {"gate_type": "H", "qubits": [3]},
                {"gate_type": "MZ", "qubits": [3, 4], "measurement_result_ids": [2, 7]}
            ],
            "named_result_traces": [{"name": "m", "values": [true, false], "result_ids": [2, 7]}]
        },
        {
            "format": "pecos_qis_operation_trace_v1",
            "engine_trace_id": 1, "shot_index": 0, "chunk_index": 1,
            "stage": "trace_complete", "num_operations": 0, "operations": [],
            "lowered_quantum_ops_complete": true, "lowered_quantum_ops": []
        }
    ])
}

fn decode(value: &Value) -> Vec<StoredTraceChunk> {
    parse_trace(&value.to_string()).unwrap()
}

#[test]
fn stored_replay_derives_physical_bound_and_preserves_source_order() {
    let shot = decode(&fixture());
    let circuit = replay(&shot, None, None).unwrap();
    assert_eq!(circuit.num_measurements(), 8);
    assert_eq!(
        circuit.get_meta("qis_source_measurement_ids"),
        Some(&Attribute::String("[7,2]".into()))
    );
    assert_eq!(named_result_traces(&shot)[0].result_ids, [2, 7]);
    assert!(replay(&shot, Some(4), None).is_err());
    assert!(replay(&shot, Some(5), None).is_ok());
}

#[test]
fn native_serialization_and_conversion_agree() {
    let chunk = OperationTraceChunk {
        format: "pecos_qis_operation_trace_v1",
        engine_trace_id: 1,
        shot_index: 0,
        chunk_index: 0,
        stage: "execute".into(),
        waiting_for_result_id: None,
        current_shot_seed: Some(5),
        simulated_op_count: 1,
        num_operations: 1,
        operations: vec![Operation::Quantum(QuantumOp::Measure(0, 7))],
        lowered_quantum_ops: vec![crate::LoweredQuantumGateTrace {
            gate_type: "MZ".into(),
            angles: vec![],
            params: vec![],
            qubits: vec![0],
            measurement_result_ids: vec![7],
            metadata: BTreeMap::from([("label".into(), "readout".into())]),
        }],
        lowered_quantum_ops_complete: true,
        runtime_lowering_deferred: false,
        runtime_events_complete: true,
        named_result_traces: vec![],
        named_results: BTreeMap::new(),
        measurement_results: BTreeMap::new(),
    };
    let terminal = OperationTraceChunk {
        chunk_index: 1,
        stage: "trace_complete".into(),
        operations: vec![],
        num_operations: 0,
        lowered_quantum_ops: vec![],
        ..chunk.clone()
    };
    let chunks = [chunk, terminal];
    let stored = parse_trace(&serde_json::to_string(&chunks).unwrap()).unwrap();
    let native = replay_in_memory(&chunks, 1).unwrap();
    let wire = replay(&stored, Some(1), None).unwrap();
    assert_eq!(native.num_ticks(), wire.num_ticks());
    for (native_tick, wire_tick) in native.ticks().iter().zip(wire.ticks()) {
        assert_eq!(native_tick.gate_batches(), wire_tick.gate_batches());
        assert_eq!(
            native_tick.get_gate_attr(0, "label"),
            wire_tick.get_gate_attr(0, "label")
        );
    }
    assert_eq!(
        native.get_meta("qis_source_measurement_ids"),
        wire.get_meta("qis_source_measurement_ids")
    );
}

#[test]
fn conversion_errors_identify_chunk_and_field() {
    for (field, value) in [
        ("engine_trace_id", json!(true)),
        ("shot_index", json!(-1)),
        ("num_operations", json!("2")),
        ("operations", json!([{"Quantum": {"Measure": [0, -1]}}])),
        ("named_result_traces", json!([{"name": "m"}])),
        (
            "lowered_quantum_ops",
            json!([{"gate_type": "H", "qubits": [false]}]),
        ),
    ] {
        let mut trace = fixture();
        trace[0][field] = value;
        let error = parse_trace(&trace.to_string()).unwrap_err().to_string();
        assert!(
            error.contains("chunk 0") && error.contains(field),
            "{error}"
        );
    }
    let mut trace = fixture();
    trace[1]
        .as_object_mut()
        .unwrap()
        .remove("lowered_quantum_ops_complete");
    let error = parse_trace(&trace.to_string()).unwrap_err().to_string();
    assert!(error.contains("chunk 1: field lowered_quantum_ops_complete"));
}

#[test]
fn every_chunk_requires_attestation_and_framing() {
    for index in 0..2 {
        let mut trace = decode(&fixture());
        trace[index].lowered_quantum_ops_complete = false;
        assert!(replay(&trace, None, None).is_err());
    }
    for corrupt in 0..6 {
        let mut trace = decode(&fixture());
        match corrupt {
            0 => trace[1].shot_index = 1,
            1 => trace[1].chunk_index = 2,
            2 => trace[0].num_operations = 0,
            3 => trace[0].stage = "trace_complete".into(),
            4 => trace[1].named_result_traces = trace[0].named_result_traces.clone(),
            _ => {
                trace.pop();
            }
        }
        assert!(replay(&trace, None, None).is_err());
    }
    let mut trace = decode(&fixture());
    trace[0]
        .named_results
        .insert("m".into(), NamedResult::Bool(vec![true]));
    let mut extra = trace[0].clone();
    extra.chunk_index = 1;
    trace.insert(1, extra);
    trace[2].chunk_index = 2;
    assert!(
        validate_shot(&trace)
            .unwrap_err()
            .to_string()
            .contains("more than one chunk")
    );
}

#[test]
fn terminal_chunk_rejects_each_payload_without_raw_operations() {
    for (field, value) in [
        (
            "lowered_quantum_ops",
            json!([{"gate_type": "H", "qubits": [0]}]),
        ),
        ("named_results", json!({"m": NamedResult::Bool(vec![true])})),
        (
            "named_result_traces",
            json!([{"name": "m", "values": [true], "result_ids": [7]}]),
        ),
    ] {
        let mut trace = fixture();
        trace[1][field] = value;
        let chunks = decode(&trace);
        assert!(chunks[1].operations.is_empty());
        assert_eq!(
            validate_shot(&chunks).unwrap_err().to_string(),
            "terminal trace chunk must contain no operations or named outputs",
            "{field}",
        );
    }
}

#[test]
fn strict_gate_and_measurement_validation() {
    for gate in [
        json!({"gate_type": "H", "qubits": [3], "angles": [0.5]}),
        json!({"gate_type": "H", "qubits": [3], "params": [0.5]}),
        json!({"gate_type": "H", "qubits": [3], "measurement_result_ids": [0]}),
        json!({"gate_type": "CRZ", "qubits": [3, 4], "angles": [0.5]}),
        json!({"gate_type": "H", "qubits": []}),
        json!({"gate_type": "H", "qubits": [3, 3]}),
        json!({"gate_type": "CX", "qubits": [3]}),
        json!({"gate_type": "RX", "qubits": [3], "angles": []}),
        json!({"gate_type": "RX", "qubits": [3], "angles": [0.5, 0.2]}),
    ] {
        let mut trace = fixture();
        trace[0]["lowered_quantum_ops"][0] = gate;
        assert!(replay(&decode(&trace), None, None).is_err());
    }
    for ids in [json!([]), json!([2]), json!([2, 2]), json!([2, 8])] {
        let mut trace = fixture();
        trace[0]["lowered_quantum_ops"][1]["measurement_result_ids"] = ids;
        assert!(replay(&decode(&trace), None, None).is_err());
    }
    let mut trace = decode(&fixture());
    trace[0].operations[1] = Operation::Quantum(QuantumOp::MeasureLeaked(11, 7));
    assert!(
        replay(&trace, None, None)
            .unwrap_err()
            .to_string()
            .contains("repeats measurement IDs")
    );
}

#[test]
fn idle_rounding_clamp_and_overflow() {
    for (seconds, expected) in [(0.0, 0.0), (1e-12, 1.0), (2.5e-9, 2.0), (3.5e-9, 4.0)] {
        let mut trace = fixture();
        trace[0]["lowered_quantum_ops"][0] =
            json!({"gate_type": "Idle", "qubits": [3], "params": [seconds]});
        let circuit = replay(&decode(&trace), None, None).unwrap();
        assert_eq!(circuit.ticks()[0].gate_batches()[0].params[0], expected);
    }
    for seconds in [-1.0, f64::INFINITY, f64::NAN, 2.0_f64.powi(64) / 1e9] {
        let mut trace = decode(&fixture());
        trace[0].lowered_quantum_ops[0].gate_type = "Idle".into();
        trace[0].lowered_quantum_ops[0].params = vec![seconds];
        assert!(replay(&trace, None, None).is_err());
    }
}

#[test]
fn global_payload_precedes_each_measurement_batch() {
    let trace = decode(&fixture());
    for topology in [None, Some("none"), Some("runtime_payloads")] {
        assert_eq!(replay(&trace, None, topology).unwrap().num_ticks(), 2);
    }
    let circuit = replay(&trace, None, Some("global_from_measurements")).unwrap();
    assert_eq!(circuit.num_ticks(), 3);
    let gates: Vec<_> = circuit
        .ticks()
        .iter()
        .flat_map(pecos_quantum::Tick::gate_batches)
        .collect();
    assert_eq!(gates[1].gate_type, GateType::MeasCrosstalkGlobalPayload);
    assert_eq!(gates[2].gate_type, GateType::MZ);
    assert_eq!(gates[1].qubits, gates[2].qubits);
    assert_eq!(
        replay(&trace, None, Some("invalid"))
            .unwrap_err()
            .to_string(),
        "measurement_crosstalk_topology must be None, 'runtime_payloads', or 'global_from_measurements'"
    );
}

#[test]
fn metadata_preserves_json_types_and_integer_precision() {
    let metadata = json!({"null": null, "bool": true, "int": -1, "float": 0.5,
        "uint": u64::MAX, "min": i64::MIN, "max": i64::MAX,
        "array": [1, "x"], "object": {"x": [null, u64::MAX, i64::MIN]}});
    let mut trace = fixture();
    trace[0]["lowered_quantum_ops"][0]["metadata"] = metadata.clone();
    let circuit = replay(&decode(&trace), None, None).unwrap();
    assert_eq!(
        circuit.ticks()[0].get_gate_attr(0, "min"),
        Some(&Attribute::Int(i64::MIN))
    );
    assert_eq!(
        circuit.ticks()[0].get_gate_attr(0, "max"),
        Some(&Attribute::Int(i64::MAX))
    );
    assert_eq!(
        circuit.ticks()[0].get_gate_attr(0, "uint"),
        Some(&Attribute::Json(Value::from(u64::MAX)))
    );
    for (key, value) in metadata.as_object().unwrap() {
        assert_eq!(
            circuit.ticks()[0].get_gate_attr(0, key),
            Some(&metadata_attribute(value))
        );
    }
}

#[test]
fn bookkeeping_only_trace_is_valid_but_unlowered_quantum_is_not() {
    let mut trace = decode(&fixture());
    trace[0].operations = vec![
        Operation::AllocateQubit { id: 100 },
        Operation::ReleaseQubit { id: 100 },
    ];
    trace[0].lowered_quantum_ops.clear();
    trace[0].named_result_traces.clear();
    let circuit = replay(&trace, Some(0), None).unwrap();
    assert_eq!(circuit.gate_count(), 0);
    assert_eq!(
        circuit.get_meta("qis_source_measurement_ids"),
        Some(&Attribute::String("[]".into()))
    );
    trace[0].operations[1] = Operation::Quantum(QuantumOp::H(100));
    assert!(
        replay(&trace, None, None)
            .unwrap_err()
            .to_string()
            .contains("no lowered_quantum_ops")
    );
}

#[test]
fn qubit_and_measurement_successor_overflow_are_rejected() {
    let mut trace = decode(&fixture());
    trace[0].lowered_quantum_ops[0].qubits = vec![usize::MAX];
    assert!(
        replay(&trace, None, None)
            .unwrap_err()
            .to_string()
            .contains("qubit bound overflows")
    );
    let mut trace = decode(&fixture());
    trace[0].operations[0] = Operation::Quantum(QuantumOp::Measure(10, usize::MAX));
    trace[0].lowered_quantum_ops[1].measurement_result_ids[1] = usize::MAX;
    assert!(
        replay(&trace, None, None)
            .unwrap_err()
            .to_string()
            .contains("leaves no room")
    );
}

#[test]
fn global_payload_is_inserted_for_repeated_measurement_batches() {
    let mut trace = decode(&fixture());
    trace[0]
        .operations
        .push(Operation::Quantum(QuantumOp::Measure(10, 9)));
    trace[0].num_operations += 1;
    trace[0].lowered_quantum_ops.push(StoredGate {
        gate_type: "MZ".into(),
        qubits: vec![3],
        measurement_result_ids: vec![9],
        angles: vec![],
        params: vec![],
        metadata: BTreeMap::new(),
    });
    let circuit = replay(&trace, None, Some("global_from_measurements")).unwrap();
    let kinds: Vec<_> = circuit
        .ticks()
        .iter()
        .flat_map(pecos_quantum::Tick::gate_batches)
        .map(|gate| gate.gate_type)
        .collect();
    assert_eq!(
        kinds,
        [
            GateType::H,
            GateType::MeasCrosstalkGlobalPayload,
            GateType::MZ,
            GateType::MeasCrosstalkGlobalPayload,
            GateType::MZ
        ]
    );
}

#[test]
fn metadata_integer_overflow_has_chunk_gate_and_key_context() {
    for token in [
        "18446744073709551616",
        "-9223372036854775809",
        "[1267650600228229401496703205376]",
        "{\"nested\":1267650600228229401496703205376}",
    ] {
        let mut trace = fixture();
        trace[0]["lowered_quantum_ops"][0]["metadata"] = json!({"value": "TOKEN"});
        let json = trace.to_string().replace("\"TOKEN\"", token);
        let error = parse_trace(&json).unwrap_err().to_string();
        for context in ["chunk 0", "gate 0", "key \"value\"", "losslessly"] {
            assert!(error.contains(context), "{error}");
        }
        assert!(serde_json::from_str::<Vec<StoredTraceChunk>>(&json).is_err());
    }
}

#[test]
fn native_measurement_ids_use_source_provenance() {
    for leaked in [false, true] {
        let mut value = fixture();
        value[0]["operations"][0] = json!({"RuntimeRequest": {
            "request": {"Native": {"Measure": {"qubit": 10, "result_id": 7, "leaked": leaked}}},
            "result_id": pecos_qis_ffi_types::RUNTIME_REQUEST_RESULT_ID,
        }});
        let shot = decode(&value);
        assert_eq!(source_ids(&shot).unwrap(), [7, 2]);
        assert!(replay(&shot, None, None).is_ok());
    }
}

#[test]
fn deferred_runtime_work_requires_complete_handled_drain() {
    let mut value = fixture();
    let mut lowered = value[1].clone();
    lowered["stage"] = json!("result_read");
    lowered["lowered_quantum_ops"] = value[0]["lowered_quantum_ops"].take();
    value[0]["lowered_quantum_ops"] = json!([]);
    value[0]["runtime_lowering_deferred"] = json!(true);
    value[1]["chunk_index"] = json!(2);
    value.as_array_mut().unwrap().insert(1, lowered);
    assert!(replay(&decode(&value), None, None).is_ok());
    for index in 0..3 {
        let mut unhandled = value.clone();
        unhandled[index]["runtime_events_complete"] = json!(false);
        assert!(
            replay(&decode(&unhandled), None, None)
                .unwrap_err()
                .to_string()
                .contains("unhandled runtime custom events")
        );
    }
    let mut missing_terminal = value.clone();
    missing_terminal.as_array_mut().unwrap().pop();
    assert!(replay(&decode(&missing_terminal), None, None).is_err());
    value[0]["runtime_lowering_deferred"] = json!(false);
    assert!(replay(&decode(&value), None, None).is_err());
    value[0]["operations"] = json!([{"RuntimeRequest": {
        "request": {"Custom": {"tag": 1, "data": []}}, "result_id": null,
    }}]);
    value[0]["num_operations"] = json!(1);
    assert!(validate_shot(&decode(&value)).is_err());
}

#[test]
fn hosted_replay_options_validate_compacted_metadata() {
    let mut value = fixture();
    value[0]["lowered_quantum_ops"][0]["metadata"] = json!({"host_id": "a"});
    value[0]["lowered_quantum_ops"][1]["metadata"] =
        json!({"host_id": "a", "local_role": "prefix"});
    let shot = decode(&value);
    assert!(
        replay_with_hosted_options(&shot, None, None, &HostedReplayOptions::default(), "test")
            .is_ok()
    );
    let options = HostedReplayOptions {
        require_hosted_operation_order: true,
        max_hosted_tick_separation: None,
    };
    let error = replay_with_hosted_options(&shot, None, None, &options, "test").unwrap_err();
    assert!(error.to_string().starts_with("test: hosted local gate MZ"));
    assert!(error.to_string().contains("ordering drift"));
    let options = HostedReplayOptions {
        require_hosted_operation_order: false,
        max_hosted_tick_separation: Some(0),
    };
    assert!(
        replay_with_hosted_options(&shot, None, None, &options, "test")
            .unwrap_err()
            .to_string()
            .contains("signed tick separation -1")
    );
    value[0]["lowered_quantum_ops"][1]["metadata"] = json!({"host_id": "a"});
    assert!(
        replay_with_hosted_options(&decode(&value), None, None, &options, "test")
            .unwrap_err()
            .to_string()
            .contains("appears on 2 host gates")
    );
}

#[test]
fn stored_crz_requires_runtime_lowering() {
    let mut trace = fixture();
    trace[0]["lowered_quantum_ops"][0] =
        json!({"gate_type": "CRZ", "qubits": [3, 4], "angles": [0.5]});
    assert_eq!(
        replay(&decode(&trace), None, None).unwrap_err().to_string(),
        "stored CRZ requires runtime lowering before trace replay"
    );
}
