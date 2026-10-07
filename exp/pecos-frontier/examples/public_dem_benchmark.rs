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

//! Reproducible Frontier throughput probe for a public flattened Stim DEM.
//!
//! Usage: `public_dem_benchmark <model.dem> [shots] [repeats] [seed]`.

use pecos_decoder_core::dem::SparseDem;
use pecos_frontier::{FrontierConfig, FrontierDecoder};
use rand::{RngExt, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;
use std::time::Instant;

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args
        .next()
        .expect("usage: public_dem_benchmark <model.dem> [shots] [repeats] [seed]");
    let shots: usize = args
        .next()
        .map_or(Ok(256), |raw| raw.parse())
        .expect("shots must be an integer");
    let repeats: usize = args
        .next()
        .map_or(Ok(5), |raw| raw.parse())
        .expect("repeats must be an integer");
    let seed: u64 = args
        .next()
        .map_or(Ok(0x5eed_u64), |raw| raw.parse())
        .expect("seed must be an integer");
    assert!(
        shots > 0 && repeats > 0,
        "shots and repeats must be positive"
    );

    let dem_text = std::fs::read_to_string(&path).expect("read detector error model");
    let dem = SparseDem::from_dem_str(&dem_text).expect("parse flattened detector error model");
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    let mut syndromes = vec![vec![0_u8; dem.num_detectors]; shots];
    for syndrome in &mut syndromes {
        for &(probability, ref detectors, _) in &dem.mechanisms {
            if rng.random::<f64>() < probability {
                for &detector in detectors {
                    syndrome[detector as usize] ^= 1;
                }
            }
        }
    }

    let build_started = Instant::now();
    let mut decoder = FrontierDecoder::from_sparse_dem(&dem, FrontierConfig::default())
        .expect("construct Frontier decoder");
    let build_seconds = build_started.elapsed().as_secs_f64();
    for syndrome in syndromes.iter().take(shots.min(32)) {
        let _ = decoder.decode(syndrome);
    }

    let started = Instant::now();
    let mut checksum = 0_u64;
    let mut failures = 0_usize;
    for _ in 0..repeats {
        for syndrome in &syndromes {
            match decoder.decode(syndrome) {
                Ok(result) => {
                    for &word in result.predicted.words() {
                        checksum = checksum.rotate_left(7) ^ word;
                    }
                }
                Err(_) => failures += 1,
            }
        }
    }
    let decode_seconds = started.elapsed().as_secs_f64();
    let decoded = shots.checked_mul(repeats).expect("decode count overflow");
    let decoded = u32::try_from(decoded).expect("decode count exceeds u32");
    let microseconds_per_shot = decode_seconds * 1_000_000.0 / f64::from(decoded);
    println!(
        "model={path} detectors={} mechanisms={} shots={shots} repeats={repeats} seed={seed} \
         build_ms={:.3} decode_us_per_shot={microseconds_per_shot:.3} failures={failures} \
         checksum={checksum:016x}",
        dem.num_detectors,
        dem.mechanisms.len(),
        build_seconds * 1_000.0,
    );
}
