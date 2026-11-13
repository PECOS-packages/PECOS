// Copyright 2026 The PECOS Developers
// Licensed under the Apache License, Version 2.0.
//! Decode a flattened DEM and saved detector events, without simulating a circuit.
//!
//! `decode_dem full-dem MODEL.dem EVENTS.jsonl NEW_OUTPUT_DIR`
//! `decode_dem windowed MODEL.dem CONFIG.json EVENTS.jsonl NEW_OUTPUT_DIR`
//! `decode_dem resolve-windows MODEL.dem CONFIG.json NEW_OUTPUT_DIR`

use pecos_frontier::windowed::{
    WindowConfig, WindowPlan, WindowResult, WindowedDecoder, ZeroSyndromePolicy,
};
use pecos_frontier::{
    CommitteeDirection, FrontierBoundedProgress, FrontierCommittee, FrontierCommitteeResult,
    FrontierConfig, FrontierStatus, MetricMode, ObsMask, SparseDem, deadline_column_order,
};
use serde_json::{Value, json};
use std::collections::{BTreeMap, HashSet};
use std::error::Error;
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;

type AnyResult<T> = Result<T, Box<dyn Error>>;
const COLUMN_BUDGET: usize = 32;

fn settings() -> FrontierConfig {
    FrontierConfig {
        k: 512,
        delta: 1_000_000_000.0,
        score_alpha: 0.8,
        column_order: None,
        merge_indistinguishable: false,
        bp_score_iterations: 0,
        metric_mode: MetricMode::LogSumExpFloat,
        int_metric_scale: 1024,
    }
}

fn integer(value: &Value, field: &str) -> AnyResult<usize> {
    usize::try_from(
        value
            .as_u64()
            .ok_or_else(|| format!("{field} must be a nonnegative integer"))?,
    )
    .map_err(|_| format!("{field} exceeds usize").into())
}

fn window_config(raw: &Value) -> AnyResult<WindowConfig> {
    let object = raw
        .as_object()
        .ok_or("window configuration must be an object")?;
    let allowed = [
        "schema",
        "window_blocks",
        "stride_blocks",
        "detector_blocks",
        "zero_policy",
        "detector_free_error_owners",
    ];
    if object.keys().any(|key| !allowed.contains(&key.as_str())) {
        return Err("unknown window configuration field".into());
    }
    if raw["schema"] != "frontier-earliest-owner-joint-window.v1" {
        return Err("unsupported window configuration schema".into());
    }
    let blocks = raw["detector_blocks"]
        .as_array()
        .ok_or("detector_blocks must be an array")?
        .iter()
        .map(|block| {
            block
                .as_array()
                .ok_or_else(|| "each detector block must be an array".into())
                .and_then(|block| {
                    block
                        .iter()
                        .map(|id| u32::try_from(integer(id, "detector ID")?).map_err(Into::into))
                        .collect::<AnyResult<Vec<_>>>()
                })
        })
        .collect::<AnyResult<Vec<_>>>()?;
    let zero_policy = match raw["zero_policy"].as_str() {
        Some("decode") => ZeroSyndromePolicy::Decode,
        Some("skip_and_commit_zero") => ZeroSyndromePolicy::SkipAndCommitZero,
        _ => return Err("zero_policy must explicitly be decode or skip_and_commit_zero".into()),
    };
    let mut owners = BTreeMap::new();
    if let Some(value) = object.get("detector_free_error_owners") {
        for (index, owner) in value
            .as_object()
            .ok_or("detector_free_error_owners must be an object")?
        {
            let parsed = index.parse::<usize>()?;
            if parsed.to_string() != *index {
                return Err("owner keys must be canonical mechanism indices".into());
            }
            owners.insert(parsed, integer(owner, "owner block")?);
        }
    }
    Ok(WindowConfig {
        detector_blocks: blocks,
        window_blocks: integer(&raw["window_blocks"], "window_blocks")?,
        stride_blocks: integer(&raw["stride_blocks"], "stride_blocks")?,
        zero_policy,
        detector_free_error_owners: owners,
    })
}

fn bits(mask: &ObsMask, width: usize) -> Vec<u8> {
    (0..width).map(|bit| u8::from(mask.get(bit))).collect()
}

fn float(value: f64) -> Value {
    json!({"value": if value.is_finite() {Some(value)} else {None},
        "bits":format!("0x{:016x}",value.to_bits()),
        "nonfinite":if value.is_nan() {Some("NaN")} else if value==f64::INFINITY {Some("positive_infinity")}
            else if value==f64::NEG_INFINITY {Some("negative_infinity")} else {None}})
}

// This display ratio is approximate; the integer failure count and denominator
// are retained separately. It never influences a decoding decision.
#[allow(clippy::cast_precision_loss)]
fn failure_rate(failures: usize, scored: usize) -> Option<f64> {
    (scored > 0).then(|| failures as f64 / scored as f64)
}

fn telemetry(result: &FrontierCommitteeResult, width: usize) -> Value {
    let selected = &result.selected;
    let status = match selected.status {
        FrontierStatus::Exact => json!({"kind":"exact"}),
        FrontierStatus::Pruned {
            k_capped,
            delta_pruned,
        } => json!({"kind":"pruned","k_capped":k_capped,"delta_pruned":delta_pruned}),
    };
    json!({"prediction":bits(&selected.predicted,width),
        "direction":match result.direction {CommitteeDirection::Forward=>"forward",CommitteeDirection::Backward=>"backward"},
        "selected":{"status":status,"log_evidence":float(selected.log_evidence),"runner_up_gap":selected.runner_up_gap.map(float),
            "peak_retained_states":selected.peak_retained_states,"processed_columns":selected.processed_columns,
            "transitions":selected.transitions,"dropped_states":selected.dropped_states,"dropped_log_mass":float(selected.dropped_log_mass),
            "bp_seconds":float(selected.bp_seconds),"bp_runs":selected.bp_runs,"escalation_rungs_used":selected.escalation_rungs_used,
            "logical_masses":selected.logical_masses.iter().map(|mass|json!({"logical":bits(&mass.logical,width),"log_mass":float(mass.log_mass)})).collect::<Vec<_>>()},
        "forward":{"status":format!("{:?}",result.forward.status),"log_evidence":float(result.forward.log_evidence)},
        "backward":{"status":format!("{:?}",result.backward.status),"log_evidence":float(result.backward.log_evidence)}})
}

fn resolved_plan(plan: &WindowPlan, raw: &Value) -> AnyResult<Value> {
    let models = plan.models().iter().map(|model| {
        let order=deadline_column_order(model)?;
        Ok(json!({"model":{"num_detectors":model.num_detectors,"num_observables":model.num_observables,"mechanisms":model.mechanisms},
            "probability_f64_bits":model.mechanisms.iter().map(|(p,_,_)|format!("0x{:016x}",p.to_bits())).collect::<Vec<_>>(),
            "forward_column_order":order,"backward_column_order":order.iter().rev().copied().collect::<Vec<_>>()}))
    }).collect::<AnyResult<Vec<_>>>()?;
    let windows=plan.windows().iter().enumerate().map(|(index,window)|json!({"window_index":index,
        "start":window.start,"stop":window.stop,"commit_stop":window.commit_stop,"model_id":window.model_id,
        "local_to_global_detector":window.detector_ids,"class_future_detectors":window.future_detector_ids,
        "included_mechanism_indices":window.mechanism_indices,"committed_mechanism_indices":window.committed_mechanism_indices})).collect::<Vec<_>>();
    Ok(
        json!({"schema":"frontier-resolved-earliest-owner-joint-windows.v1","config":raw,
        "num_detectors":plan.num_detectors(),"num_observables":plan.num_observables(),
        "numeric_config":{"k":512,"delta":1_000_000_000.0,"score_alpha":0.8,"merge_indistinguishable":false,
            "bp_score_iterations":0,"metric_mode":"LogSumExpFloat","int_metric_scale":1024},
        "committee":"deadline_forward_and_plain_reverse","column_budget_per_step":COLUMN_BUDGET,
        "models":models,"windows":windows}),
    )
}

fn window_json(result: &WindowResult, plan: &WindowPlan) -> Value {
    let window = &plan.windows()[result.window_index];
    let width = plan.models()[window.model_id].num_observables;
    json!({"window_index":result.window_index,"model_id":window.model_id,"observed":result.observed,
        "selected_joint_class":bits(&result.selected_joint_class,width),
        "direction":result.committee.as_ref().map_or(0,|r|match r.direction{CommitteeDirection::Forward=>1,CommitteeDirection::Backward=>2}),
        "logical_abs_llrs":result.logical_abs_llrs.as_ref().map(|values|values.iter().copied().map(float).collect::<Vec<_>>()),
        "telemetry":result.committee.as_ref().map(|r|telemetry(r,width)),"bounded_step_calls":result.step_calls,"trace_hash":result.trace_hash})
}

fn bit_vector(value: &Value, width: usize, name: &str) -> AnyResult<Vec<u8>> {
    let input = value
        .as_array()
        .ok_or_else(|| format!("{name} must be an array"))?;
    if input.len() != width {
        return Err(format!("{name} requires exactly {width} bits").into());
    }
    input
        .iter()
        .map(|value| match value.as_u64() {
            Some(0) => Ok(0),
            Some(1) => Ok(1),
            _ => Err(format!("{name} entries must be integer0or1").into()),
        })
        .collect()
}

fn new_json(path: &Path, value: &Value) -> AnyResult<()> {
    let mut output = BufWriter::new(OpenOptions::new().write(true).create_new(true).open(path)?);
    serde_json::to_writer_pretty(&mut output, value)?;
    writeln!(output)?;
    output.flush()?;
    Ok(())
}

fn run_args(args: &[String]) -> AnyResult<i32> {
    let (model_path,event_path,config_path,output_path,resolve_only)=match args.first().map(String::as_str){
        Some("full-dem") if args.len()==4=>(&args[1],Some(&args[2]),None,&args[3],false),
        Some("windowed") if args.len()==5=>(&args[1],Some(&args[3]),Some(&args[2]),&args[4],false),
        Some("resolve-windows") if args.len()==4=>(&args[1],None,Some(&args[2]),&args[3],true),
        _=>return Err("usage: decode_dem full-dem MODEL EVENTS OUTPUT | windowed MODEL CONFIG EVENTS OUTPUT | resolve-windows MODEL CONFIG OUTPUT".into()),
    };
    let output = Path::new(output_path);
    std::fs::create_dir(output)?;
    let dem = SparseDem::from_dem_str(&std::fs::read_to_string(model_path)?)?;
    let raw = config_path
        .map(|path| -> AnyResult<Value> {
            Ok(serde_json::from_str(&std::fs::read_to_string(path)?)?)
        })
        .transpose()?;
    let plan = raw
        .as_ref()
        .map(|raw| -> AnyResult<WindowPlan> {
            Ok(WindowPlan::from_sparse_dem(&dem, window_config(raw)?)?)
        })
        .transpose()?;
    if let Some(plan) = &plan {
        new_json(
            &output.join("resolved_window_models.json"),
            &resolved_plan(plan, raw.as_ref().expect("window config"))?,
        )?;
    }
    if resolve_only {
        new_json(
            &output.join("summary.json"),
            &json!({"status":"RESOLVED_ONLY_NO_DECODER_EXECUTION","shots":0}),
        )?;
        return Ok(0);
    }
    let mut windowed = plan
        .map(|plan| WindowedDecoder::from_plan(plan, settings(), COLUMN_BUDGET))
        .transpose()?;
    let mut full = if windowed.is_none() {
        let mut config = settings();
        config.column_order = Some(deadline_column_order(&dem)?);
        Some(FrontierCommittee::from_sparse_dem(&dem, config)?.into_resumable())
    } else {
        None
    };
    let input = BufReader::new(File::open(event_path.expect("event path required"))?);
    let mut output_rows = BufWriter::new(
        OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(output.join("predictions.jsonl"))?,
    );
    let mut identities = HashSet::new();
    let (mut total, mut decoded, mut errors, mut known, mut scored, mut failures, mut known_errors) =
        (0usize, 0usize, 0usize, 0usize, 0usize, 0usize, 0usize);
    for (line_index, line) in input.lines().enumerate() {
        let text = line?;
        if text.trim().is_empty() {
            continue;
        }
        let row: Value = serde_json::from_str(&text)?;
        let object = row.as_object().ok_or("each event row must be an object")?;
        let id = object.get("shot_id").ok_or("shot_id is required")?;
        if !(id.as_str().is_some_and(|s| !s.is_empty()) || id.as_u64().is_some()) {
            return Err("shot_id must be a nonempty string or nonnegative integer".into());
        }
        if !identities.insert(serde_json::to_string(id)?) {
            return Err("duplicate shot_id".into());
        }
        let syndrome = bit_vector(&row["detectors"], dem.num_detectors, "detectors")?;
        let truth = object
            .get("truth")
            .map(|v| bit_vector(v, dem.num_observables, "truth"))
            .transpose()?;
        known += usize::from(truth.is_some());
        let (mut record, predicted) = if let Some(decoder) = &mut windowed {
            match decoder.decode(&syndrome) {
                Ok(result) => {
                    let predicted = bits(&result.predicted, dem.num_observables);
                    (
                        json!({"decoder_status":"success","prediction":predicted,"trace_hash":result.trace_hash,
                        "processed_windows":result.windows.len(),"decoded_windows":result.windows.iter().filter(|w|w.committee.is_some()).count(),
                        "windows":result.windows.iter().map(|w|window_json(w,decoder.plan())).collect::<Vec<_>>(),
                        "min_abs_llrs":result.min_abs_llrs.iter().copied().map(float).collect::<Vec<_>>(),"weakest_windows":result.weakest_windows}),
                        Some(predicted),
                    )
                }
                Err(error) => (
                    json!({"decoder_status":"error","error":error.to_string(),"failed_window":error.failed_window,
                    "prediction":null,"partial_windows":error.completed_windows.iter().map(|w|window_json(w,decoder.plan())).collect::<Vec<_>>()}),
                    None,
                ),
            }
        } else {
            let decoder = full.as_mut().expect("full decoder");
            let mut steps = 0;
            let result = (|| {
                decoder.begin(&syndrome)?;
                while !matches!(decoder.progress(), Some(FrontierBoundedProgress::Ready)) {
                    decoder.step(COLUMN_BUDGET)?;
                    steps += 1;
                }
                decoder.finish()
            })();
            match result {
                Ok(result) => {
                    let predicted = bits(&result.selected.predicted, dem.num_observables);
                    (
                        json!({"decoder_status":"success","telemetry":telemetry(&result,dem.num_observables),"bounded_step_calls":steps}),
                        Some(predicted),
                    )
                }
                Err(error) => {
                    decoder.cancel();
                    (
                        json!({"decoder_status":"error","error":error.to_string(),"prediction":null}),
                        None,
                    )
                }
            }
        };
        let failure = predicted
            .as_ref()
            .and_then(|p| truth.as_ref().map(|expected| p != expected));
        if predicted.is_some() {
            decoded += 1;
            if let Some(bad) = failure {
                scored += 1;
                failures += usize::from(bad);
            }
        } else {
            errors += 1;
            known_errors += usize::from(truth.is_some());
        }
        let record = record.as_object_mut().expect("result object");
        record.insert("shot_id".into(), id.clone());
        record.insert("source_line".into(), json!(line_index + 1));
        record.insert("truth".into(), json!(truth));
        record.insert("logical_failure".into(), json!(failure));
        serde_json::to_writer(&mut output_rows, &record)?;
        writeln!(output_rows)?;
        output_rows.flush()?;
        total += 1;
    }
    if total == 0 {
        return Err("no event rows".into());
    }
    new_json(
        &output.join("summary.json"),
        &json!({"status":if errors==0{"COMPLETED"}else{"COMPLETED_WITH_DECODER_ERRORS"},
        "shots":total,"decoded_shots":decoded,"decoder_error_shots":errors,"known_truth_shots":known,
        "known_truth_decoded_shots":scored,"known_truth_decoder_error_shots":known_errors,
        "logical_failures":if scored>0{Some(failures)}else{None},
        "logical_failure_rate_among_successful_known_truth_shots":failure_rate(failures,scored),
        "column_budget":COLUMN_BUDGET,"backend":"native","response_cache":false,"shot_deduplication":false}),
    )?;
    Ok(if errors == 0 { 0 } else { 2 })
}

fn main() {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    match run_args(&args) {
        Ok(code) => std::process::exit(code),
        Err(error) => {
            eprintln!("ERROR: {error}");
            std::process::exit(1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    static NEXT: AtomicUsize = AtomicUsize::new(0);

    #[test]
    fn parser_rejects_ambiguous_window_settings_and_nonbits() {
        let raw = json!({"schema":"frontier-earliest-owner-joint-window.v1","window_blocks":8,"stride_blocks":2,
            "detector_blocks":[[0]],"zero_policy":"skip_and_commit_zero"});
        assert!(window_config(&raw).is_ok());
        let mut bad = raw.clone();
        bad["window_bocks"] = json!(8);
        assert!(window_config(&bad).is_err());
        let mut bad = raw.clone();
        bad["zero_policy"] = json!("automatic");
        assert!(window_config(&bad).is_err());
        let mut bad = raw;
        bad["detector_blocks"] = json!([[true]]);
        assert!(window_config(&bad).is_err());
        assert!(bit_vector(&json!([true]), 1, "detectors").is_err());
        assert!(bit_vector(&json!([]), 1, "detectors").is_err());
        assert!(bit_vector(&Value::Null, 1, "truth").is_err());
    }

    fn folder() -> std::path::PathBuf {
        let directory = std::env::temp_dir().join(format!(
            "pecos-frontier-example-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&directory).unwrap();
        directory
    }

    #[test]
    fn window_cli_keeps_decoder_errors_and_truth_denominators() {
        let directory = folder();
        let dem = directory.join("model.dem");
        let config = directory.join("window.json");
        let events = directory.join("events.jsonl");
        let output = directory.join("output");
        std::fs::write(&dem, "error(0.1) D0 L0\ndetector D1\n").unwrap();
        std::fs::write(&config,json!({"schema":"frontier-earliest-owner-joint-window.v1","window_blocks":8,"stride_blocks":2,
            "detector_blocks":[[0],[1]],"zero_policy":"skip_and_commit_zero"}).to_string()).unwrap();
        std::fs::write(&events,"{\"shot_id\":\"error\",\"detectors\":[0,1],\"truth\":[0]}\n{\"shot_id\":\"valid\",\"detectors\":[1,0],\"truth\":[1]}\n{\"shot_id\":\"unknown\",\"detectors\":[0,0]}\n").unwrap();
        let args = vec![
            "windowed".into(),
            dem.to_string_lossy().into_owned(),
            config.to_string_lossy().into_owned(),
            events.to_string_lossy().into_owned(),
            output.to_string_lossy().into_owned(),
        ];
        assert_eq!(run_args(&args).unwrap(), 2);
        let summary: Value =
            serde_json::from_str(&std::fs::read_to_string(output.join("summary.json")).unwrap())
                .unwrap();
        assert_eq!(summary["shots"], 3);
        assert_eq!(summary["decoder_error_shots"], 1);
        assert_eq!(summary["known_truth_decoder_error_shots"], 1);
        assert_eq!(summary["known_truth_decoded_shots"], 1);
        assert_eq!(summary["logical_failures"], 0);
        let rows = std::fs::read_to_string(output.join("predictions.jsonl"))
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str::<Value>(line).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(rows[0]["prediction"], Value::Null);
        assert_eq!(rows[1]["prediction"], json!([1]));
        assert_eq!(rows[2]["truth"], Value::Null);
        assert_eq!(rows[2]["logical_failure"], Value::Null);
        assert!(run_args(&args).is_err()); // Existing output is preserved.
    }

    #[test]
    fn full_dem_cli_decodes_zero_syndrome_without_assuming_truth() {
        let directory = folder();
        let dem = directory.join("model.dem");
        let events = directory.join("events.jsonl");
        let output = directory.join("output");
        std::fs::write(&dem, "error(0.8) L0\n").unwrap();
        std::fs::write(&events, "{\"shot_id\":0,\"detectors\":[]}\n").unwrap();
        let args = vec![
            "full-dem".into(),
            dem.to_string_lossy().into_owned(),
            events.to_string_lossy().into_owned(),
            output.to_string_lossy().into_owned(),
        ];
        assert_eq!(run_args(&args).unwrap(), 0);
        let summary: Value =
            serde_json::from_str(&std::fs::read_to_string(output.join("summary.json")).unwrap())
                .unwrap();
        assert_eq!(summary["known_truth_shots"], 0);
        assert_eq!(summary["logical_failures"], Value::Null);
        let row: Value = serde_json::from_str(
            std::fs::read_to_string(output.join("predictions.jsonl"))
                .unwrap()
                .trim(),
        )
        .unwrap();
        assert_eq!(row["telemetry"]["prediction"], json!([1]));
    }
}
