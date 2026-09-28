// Copyright 2026 The PECOS Developers
// Licensed under the Apache License, Version 2.0

//! Replay an FWR1 hardware fixture through the native Frontier decoder.

use pecos_frontier::{FrontierConfig, FrontierDecoder, deadline_column_order};
use pecos_trellis::SparseDem;
use std::env;
use std::fs;

const HEADER_BYTES: usize = 20;

fn read_u32(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(
        bytes[offset..offset + 4]
            .try_into()
            .expect("four-byte field"),
    )
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args_os().skip(1);
    let dem_path = args.next().ok_or("usage: replay_fwr MODEL.dem SHOTS.fwr")?;
    let replay_path = args.next().ok_or("usage: replay_fwr MODEL.dem SHOTS.fwr")?;
    if args.next().is_some() {
        return Err("usage: replay_fwr MODEL.dem SHOTS.fwr".into());
    }

    let dem_text = fs::read_to_string(dem_path)?;
    let dem = SparseDem::from_dem_str(&dem_text)?;
    let order = deadline_column_order(&dem)?;
    let mut decoder = FrontierDecoder::from_sparse_dem(
        &dem,
        FrontierConfig {
            column_order: Some(order),
            ..FrontierConfig::default()
        },
    )?;

    let replay = fs::read(replay_path)?;
    if replay.len() < HEADER_BYTES || &replay[..4] != b"FWR1" {
        return Err("invalid FWR1 fixture".into());
    }
    let version = read_u32(&replay, 4);
    let shots = read_u32(&replay, 8) as usize;
    let detectors = read_u32(&replay, 12) as usize;
    let observables = read_u32(&replay, 16) as usize;
    if version != 2 || detectors != dem.num_detectors || observables != dem.num_observables {
        return Err(format!(
            "fixture/model mismatch: version={version}, detectors={detectors}/{}, observables={observables}/{}",
            dem.num_detectors, dem.num_observables
        )
        .into());
    }

    let detector_words = detectors.div_ceil(32);
    let observable_words = observables.div_ceil(32);
    let record_words = detector_words + observable_words;
    let expected_bytes = HEADER_BYTES + shots * record_words * 4;
    if replay.len() != expected_bytes {
        return Err(format!(
            "invalid fixture length: {} != {expected_bytes}",
            replay.len()
        )
        .into());
    }

    let mut logical_errors = 0usize;
    for shot in 0..shots {
        let record_start = HEADER_BYTES + shot * record_words * 4;
        let mut syndrome = vec![0u8; detectors];
        for (detector, bit) in syndrome.iter_mut().enumerate() {
            let word = read_u32(&replay, record_start + (detector / 32) * 4);
            *bit = ((word >> (detector % 32)) & 1) as u8;
        }
        let prediction = decoder.decode(&syndrome)?.predicted;
        let mut predicted_mask = 0u128;
        let mut actual_mask = 0u128;
        for observable in 0..observables {
            if prediction.get(observable) {
                predicted_mask |= 1u128 << observable;
            }
            let word_offset = record_start + (detector_words + observable / 32) * 4;
            let word = read_u32(&replay, word_offset);
            actual_mask |= u128::from((word >> (observable % 32)) & 1) << observable;
        }
        logical_errors += usize::from(predicted_mask != actual_mask);
        println!("{predicted_mask}");
    }
    eprintln!("shots={shots} logical_errors={logical_errors}");
    Ok(())
}
