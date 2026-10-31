// Copyright 2026 The PECOS Developers
// Licensed under the Apache License, Version 2.0

//! Export real inspector HTML for the DOM regression suite. Trace origins here
//! are synthetic; Python integration tests cover validated QIS trace replay.

use pecos_core::{Angle64, MeasId};
use pecos_quantum::{Attribute, DagCircuit, DiagramStyle, Gate, GateType, QubitId, TickCircuit};
use serde_json::json;
use std::{fs, path::PathBuf};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let directory = PathBuf::from(
        std::env::args_os()
            .nth(1)
            .ok_or("expected output directory")?,
    );
    fs::create_dir_all(&directory)?;
    let style = DiagramStyle::builder().show_measurement_ids(true).build();
    let save = |name: &str, circuit: &TickCircuit| {
        fs::write(
            directory.join(format!("{name}.html")),
            circuit.inspect(&style).to_html(),
        )
    };

    let mut nested = TickCircuit::new();
    nested.tick().h(&[0, 1, 2, 3]);
    nested.tick().cx(&[(0, 3), (1, 2)]).meta(
        "note",
        Attribute::String("</pre><script>globalThis.injected = true</script> & \"quoted\"".into()),
    );
    let mut measurement = Gate::mz(&[0, 1, 2, 3]);
    measurement.meas_ids = [42, 7, 13, 99].into_iter().map(MeasId::from_raw).collect();
    nested.tick().try_add_gate(measurement)?;
    save("nested", &nested)?;
    fs::write(
        directory.join("dag.html"),
        DagCircuit::try_from(&nested)?.inspect(&style).to_html(),
    )?;

    let base = json!({
        "format": "pecos_qis_lowered_origin_v1", "engine_trace_id": u64::MAX,
        "shot_index": 0, "chunk_index": 0, "lowered_op_index": 0,
    });
    let mut trace = TickCircuit::new();
    trace
        .tick()
        .rzz(Angle64::try_from_radians(-0.25)?, &[(0, 1)])
        .meta("_pecos_trace_origin", Attribute::Json(base.clone()));
    trace
        .tick()
        .rz(Angle64::try_from_radians(0.25)?, &[1])
        .meta("_pecos_trace_origin", Attribute::Json(base.clone()));
    save("trace", &trace)?;
    fs::write(
        directory.join("trace-range.html"),
        trace.inspect_range(1..2, &style)?.to_html(),
    )?;

    let mut origins = TickCircuit::new();
    let mut variants = vec![base.clone(), base.clone()];
    for (key, value) in [
        ("engine_trace_id", u64::MAX - 1),
        ("shot_index", 1),
        ("chunk_index", 1),
        ("lowered_op_index", 1),
    ] {
        let mut origin = base.clone();
        origin[key] = value.into();
        variants.push(origin);
    }
    variants.extend([json!({}), json!({"format": "bad"})]);
    for origin in variants {
        origins
            .tick()
            .h(&[0])
            .meta("_pecos_trace_origin", Attribute::Json(origin));
    }
    save("origins", &origins)?;
    save("empty", &TickCircuit::new())?;

    let mut no_symbol = TickCircuit::new();
    no_symbol
        .tick()
        .try_add_gate(Gate::simple(GateType::MeasCrosstalkGlobalPayload, vec![]))?;
    save("no-symbol", &no_symbol)?;
    let mut leakage = TickCircuit::new();
    for (kind, qubit, id) in [
        (GateType::MZ, 4, 42),
        (GateType::MeasureLeaked, 1, 7),
        (GateType::MZ, 3, 20),
    ] {
        let mut gate = Gate::simple(kind, vec![QubitId(qubit)]);
        gate.meas_ids = vec![MeasId::from_raw(id)].into();
        leakage.tick().try_add_gate(gate)?;
    }
    save("leakage", &leakage)?;

    let mut wide = TickCircuit::new();
    for tick in 0..80 {
        wide.tick().h(&[tick % 8]);
        if tick % 5 == 0 {
            wide.tick().cx(&[(tick % 8, (tick + 3) % 8)]);
        }
    }
    save("wide", &wide)?;
    Ok(())
}
