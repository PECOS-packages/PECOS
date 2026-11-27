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

//! End-to-end decode latency with and without discarded-mass diagnostics.

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use pecos_frontier::{
    FrontierCommittee, FrontierConfig, FrontierDecodeAttempt, FrontierDecoder, SparseDem,
};
use rand::{RngExt, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;
use std::hint::black_box;
use std::path::Path;

const SHOTS: u32 = 32;
const SEED: u64 = 0x5eed;
const MODELS: [&str; 2] = ["d5_z_stim", "d7_z_stim"];

fn load_shots(model: &str) -> (SparseDem, Vec<Vec<u8>>) {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../examples/surface_code_circuits")
        .join(format!("surface_code_{model}.dem"));
    eprintln!("model={} shots={SHOTS} seed={SEED:#x}", path.display());
    let text = std::fs::read_to_string(path).expect("read committed surface-code benchmark DEM");
    let dem = SparseDem::from_dem_str(&text).expect("parse surface-code benchmark DEM");
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(SEED);
    let mut syndromes = vec![vec![0_u8; dem.num_detectors]; SHOTS as usize];
    for syndrome in &mut syndromes {
        for &(probability, ref detectors, _) in &dem.mechanisms {
            if rng.random::<f64>() < probability {
                for &detector in detectors {
                    syndrome[detector as usize] ^= 1;
                }
            }
        }
    }
    (dem, syndromes)
}

fn decoder_stats(decoder: &mut FrontierDecoder, syndromes: &[Vec<u8>]) -> (u64, usize) {
    let mut dropped = 0;
    let mut errors = 0;
    for syndrome in syndromes {
        match decoder.decode(syndrome) {
            Ok(result) => dropped += result.dropped_states,
            Err(error) => {
                errors += 1;
                // decode discards no-path telemetry. Recover it outside timing
                // so the pruning total includes every shot, including failures.
                match decoder.decode_attempt(syndrome) {
                    FrontierDecodeAttempt::NoPath { dropped_states, .. } => {
                        dropped += dropped_states;
                    }
                    _ => panic!("benchmark setup expected a repeatable no-path error: {error:?}"),
                }
            }
        }
    }
    (dropped, errors)
}

fn bench_decoder(c: &mut Criterion) {
    let defaults = FrontierConfig::default();
    let regimes = [
        ("wide", defaults.k, defaults.delta),
        ("k-capped", 8, f64::INFINITY),
        ("delta-capped", 65_536, 2.0),
    ];
    let mut group = c.benchmark_group("decode");
    group.throughput(Throughput::Elements(u64::from(SHOTS)));
    for model in MODELS {
        let (dem, syndromes) = load_shots(model);
        for (regime, k, delta) in regimes {
            let mut cases = [false, true].map(|dropped_mass| {
                let config = FrontierConfig {
                    k,
                    delta,
                    dropped_mass,
                    ..FrontierConfig::default()
                };
                let mut decoder = FrontierDecoder::from_sparse_dem(&dem, config)
                    .expect("construct Frontier benchmark decoder");
                let (dropped, errors) = decoder_stats(&mut decoder, &syndromes);
                eprintln!(
                    "decode/{model}/{regime}/dropped_mass={dropped_mass}: \
                     k={k} delta={delta} dropped_states={dropped} decode_errors={errors}"
                );
                assert!(
                    regime == "wide" || dropped > 0,
                    "{model}/{regime}/dropped_mass={dropped_mass}: expected pruning to drop states on the sampled shots"
                );
                (dropped_mass, decoder, dropped, errors)
            });
            assert_eq!(
                cases[0].3, cases[1].3,
                "{model}/{regime}: flag changed decode-error count"
            );
            assert_eq!(
                cases[0].2, cases[1].2,
                "{model}/{regime}: flag changed dropped-state count"
            );
            for (dropped_mass, decoder, _, _) in &mut cases {
                group.bench_function(
                    BenchmarkId::new(
                        format!("{model}/{regime}"),
                        format!("dropped_mass={dropped_mass}"),
                    ),
                    |b| {
                        b.iter(|| {
                            for syndrome in &syndromes {
                                let _ = black_box(decoder.decode(black_box(syndrome)));
                            }
                        });
                    },
                );
            }
        }
    }
    group.finish();
}

fn bench_committee(c: &mut Criterion) {
    let model = MODELS[0];
    let (dem, syndromes) = load_shots(model);
    let mut cases = [false, true].map(|dropped_mass| {
        let config = FrontierConfig {
            dropped_mass,
            ..FrontierConfig::default()
        };
        let mut committee = FrontierCommittee::from_sparse_dem(&dem, config)
            .expect("construct Frontier benchmark committee");
        let mut dropped = 0;
        let mut errors = 0;
        for syndrome in &syndromes {
            match committee.decode(syndrome) {
                Ok(result) => dropped += result.selected.dropped_states,
                Err(_) => errors += 1,
            }
        }
        eprintln!(
            "committee/{model}/wide/dropped_mass={dropped_mass}: \
             selected_dropped_states={dropped} decode_errors={errors}"
        );
        (dropped_mass, committee, dropped, errors)
    });
    assert_eq!(
        cases[0].3, cases[1].3,
        "{model}/committee: flag changed decode-error count"
    );
    assert_eq!(
        cases[0].2, cases[1].2,
        "{model}/committee: flag changed selected dropped-state count"
    );
    let mut group = c.benchmark_group("committee");
    group.throughput(Throughput::Elements(u64::from(SHOTS)));
    for (dropped_mass, committee, _, _) in &mut cases {
        group.bench_function(
            BenchmarkId::new(
                format!("{model}/wide"),
                format!("dropped_mass={dropped_mass}"),
            ),
            |b| {
                b.iter(|| {
                    for syndrome in &syndromes {
                        let _ = black_box(committee.decode(black_box(syndrome)));
                    }
                });
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_decoder, bench_committee);
criterion_main!(benches);
