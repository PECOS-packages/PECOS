// Copyright 2026 The PECOS Developers
// Licensed under the Apache License, Version 2.0

//! Replay an FWR1 hardware fixture through the native Frontier decoder.

use pecos_frontier::{FrontierConfig, FrontierDecoder, SparseDem, TrellisOrdering};
#[path = "../../pecos-frontier-wasm/src/replay.rs"]
mod replay;
use std::env;
use std::fs;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args_os().skip(1);
    let dem_path = args.next().ok_or("usage: replay_fwr MODEL.dem SHOTS.fwr")?;
    let replay_path = args.next().ok_or("usage: replay_fwr MODEL.dem SHOTS.fwr")?;
    if args.next().is_some() {
        return Err("usage: replay_fwr MODEL.dem SHOTS.fwr".into());
    }

    let dem_text = fs::read_to_string(dem_path)?;
    let dem = SparseDem::from_dem_str(&dem_text)?;
    let order = TrellisOrdering::Deadline.resolve(&dem)?;
    let mut decoder = FrontierDecoder::from_sparse_dem(
        &dem,
        FrontierConfig {
            column_order: order,
            ..FrontierConfig::default()
        },
    )?;

    let replay = fs::read(replay_path)?;
    let fixture = replay::Replay::parse(&replay)?;
    if fixture.detectors != dem.num_detectors || fixture.observables != dem.num_observables {
        return Err("fixture/model dimensions differ".into());
    }
    let shots = fixture.shots;
    let mut logical_errors = 0usize;
    for shot in 0..shots {
        let (syndrome, expected) = fixture.record(shot).ok_or("missing shot")?;
        let prediction = decoder.decode(&syndrome)?.predicted;
        let mut predicted_mask = 0u128;
        let actual_mask = expected
            .iter()
            .enumerate()
            .fold(0_u128, |mask, (word, value)| {
                mask | (u128::from(value.cast_unsigned()) << (word * 32))
            });
        for observable in prediction.iter_set_bits() {
            predicted_mask |= 1_u128 << observable;
        }
        logical_errors += usize::from(predicted_mask != actual_mask);
        println!("{predicted_mask}");
    }
    eprintln!("shots={shots} logical_errors={logical_errors}");
    Ok(())
}
