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

//! Differential oracle harness for the MWPF LP backend.
//!
//! MWPF solves small LP relaxations internally. This harness decodes a
//! deterministic stream of sampled-error syndromes on hyperedge-rich DEMs and
//! records the resulting observable masks, so two builds with different LP
//! backends (e.g. `HiGHS` vs a pure-Rust solver) can be compared shot-for-shot.
//!
//! Usage:
//!   dump:    `PECOS_MWPF_DIFF_OUT=/path/fixture.txt cargo test -p pecos-mwpf \
//!                --release -- --include-ignored differential`
//!   compare: `PECOS_MWPF_DIFF_IN=/path/fixture.txt cargo test -p pecos-mwpf \
//!                --release -- --include-ignored differential`

use std::fmt::Write as _;
use std::time::Instant;

use pecos_mwpf::{MwpfConfig, MwpfDecoder};

const SHOTS_PER_DEM: usize = 400;

/// Deterministic 64-bit LCG (MMIX constants) so the shot stream is identical
/// across builds without a rand dependency.
struct Lcg(u64);

impl Lcg {
    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0
    }

    /// Uniform f64 in [0, 1) from the top 53 bits.
    fn next_f64(&mut self) -> f64 {
        let bits = self.next_u64() >> 11;
        let high = u32::try_from(bits >> 32).expect("53-bit value's high word fits u32");
        let low = u32::try_from(bits & 0xFFFF_FFFF).expect("masked low word fits u32");
        (f64::from(high) * 4_294_967_296.0 + f64::from(low)) / 9_007_199_254_740_992.0
    }

    fn below(&mut self, n: usize) -> usize {
        let n_u64 = u64::try_from(n).expect("usize fits u64 on supported targets");
        usize::try_from(self.next_u64() % n_u64).expect("value below n fits usize")
    }
}

struct Mechanism {
    prob: f64,
    detectors: Vec<usize>,
    obs_mask: u64,
}

struct Dem {
    name: &'static str,
    num_detectors: usize,
    mechanisms: Vec<Mechanism>,
}

impl Dem {
    fn to_dem_text(&self) -> String {
        let mut out = String::new();
        for m in &self.mechanisms {
            write!(out, "error({})", m.prob).unwrap();
            for d in &m.detectors {
                write!(out, " D{d}").unwrap();
            }
            for bit in 0..64 {
                if m.obs_mask >> bit & 1 == 1 {
                    write!(out, " L{bit}").unwrap();
                }
            }
            out.push('\n');
        }
        out
    }
}

/// A matching backbone (weight-2 edges plus boundaries) with random 3- and
/// 4-detector hyperedges layered on top, so decoding forms clusters that
/// exercise the LP relaxation rather than pure matching.
fn random_hypergraph_dem(
    name: &'static str,
    num_detectors: usize,
    num_hyperedges: usize,
    seed: u64,
) -> Dem {
    let mut rng = Lcg(seed);
    let mut mechanisms = Vec::new();

    mechanisms.push(Mechanism {
        prob: 0.02,
        detectors: vec![0],
        obs_mask: 0,
    });
    mechanisms.push(Mechanism {
        prob: 0.02,
        detectors: vec![num_detectors - 1],
        obs_mask: 1,
    });
    for d in 0..num_detectors - 1 {
        mechanisms.push(Mechanism {
            prob: 0.01 + 0.03 * rng.next_f64(),
            detectors: vec![d, d + 1],
            obs_mask: u64::from(d % 3 == 0),
        });
    }

    for _ in 0..num_hyperedges {
        let arity = 3 + rng.below(2);
        let mut detectors = Vec::with_capacity(arity);
        while detectors.len() < arity {
            let d = rng.below(num_detectors);
            if !detectors.contains(&d) {
                detectors.push(d);
            }
        }
        detectors.sort_unstable();
        mechanisms.push(Mechanism {
            prob: 0.005 + 0.02 * rng.next_f64(),
            detectors,
            obs_mask: rng.next_u64() & 0b11,
        });
    }

    Dem {
        name,
        num_detectors,
        mechanisms,
    }
}

fn test_dems() -> Vec<Dem> {
    vec![
        random_hypergraph_dem("hyper24", 24, 16, 0xD5C0_DE01),
        random_hypergraph_dem("hyper48", 48, 36, 0xD5C0_DE02),
    ]
}

/// Sample each mechanism independently and XOR its detectors into the
/// syndrome, so every syndrome corresponds to a real error pattern.
fn sample_syndrome(dem: &Dem, rng: &mut Lcg) -> Vec<u8> {
    let mut syndrome = vec![0u8; dem.num_detectors];
    for m in &dem.mechanisms {
        if rng.next_f64() < m.prob * 4.0 {
            for &d in &m.detectors {
                syndrome[d] ^= 1;
            }
        }
    }
    syndrome
}

fn syndrome_hex(syndrome: &[u8]) -> String {
    let mut out = String::new();
    for chunk in syndrome.chunks(4) {
        let mut nibble = 0u8;
        for (i, &bit) in chunk.iter().enumerate() {
            nibble |= bit << i;
        }
        write!(out, "{nibble:x}").unwrap();
    }
    out
}

/// One line per shot: `<dem> <shot> <syndrome-hex> <mask-hex>`.
fn run_all() -> (Vec<String>, f64) {
    let mut lines = Vec::new();
    let mut decode_seconds = 0.0;
    for dem in test_dems() {
        let mut decoder = MwpfDecoder::from_dem(&dem.to_dem_text(), MwpfConfig::default())
            .expect("DEM should construct");
        let mut rng = Lcg(0xFEED_0000 ^ dem.num_detectors as u64);
        for shot in 0..SHOTS_PER_DEM {
            let syndrome = sample_syndrome(&dem, &mut rng);
            let start = Instant::now();
            let result = decoder
                .decode_syndrome(&syndrome)
                .expect("decode should succeed");
            decode_seconds += start.elapsed().as_secs_f64();
            lines.push(format!(
                "{} {} {} {:x}",
                dem.name,
                shot,
                syndrome_hex(&syndrome),
                result.observable_mask
            ));
        }
    }
    (lines, decode_seconds)
}

#[test]
#[ignore = "differential oracle harness; run explicitly with PECOS_MWPF_DIFF_OUT or PECOS_MWPF_DIFF_IN"]
fn differential() {
    let out_path = std::env::var("PECOS_MWPF_DIFF_OUT").ok();
    let in_path = std::env::var("PECOS_MWPF_DIFF_IN").ok();
    assert!(
        out_path.is_some() || in_path.is_some(),
        "set PECOS_MWPF_DIFF_OUT to dump a fixture or PECOS_MWPF_DIFF_IN to compare against one"
    );

    let (lines, decode_seconds) = run_all();
    println!("decoded {} shots in {decode_seconds:.3}s", lines.len());

    if let Some(path) = out_path {
        std::fs::write(&path, lines.join("\n") + "\n").expect("fixture should be writable");
        println!("fixture written to {path}");
    }

    if let Some(path) = in_path {
        let fixture = std::fs::read_to_string(&path).expect("fixture should be readable");
        let expected: Vec<&str> = fixture.lines().collect();
        assert_eq!(
            expected.len(),
            lines.len(),
            "shot count mismatch vs fixture"
        );
        let mut mismatches: Vec<(usize, &str, &str)> = Vec::new();
        for (i, (e, l)) in expected.iter().zip(lines.iter()).enumerate() {
            if *e != l.as_str() {
                mismatches.push((i, e, l));
            }
        }
        for (i, e, l) in &mismatches {
            println!("mismatch at shot {i}: fixture `{e}` vs current `{l}`");
        }
        println!(
            "{} / {} shots match fixture",
            lines.len() - mismatches.len(),
            lines.len()
        );
        assert!(
            mismatches.is_empty(),
            "{} shots differ from fixture; inspect above for LP-tie degeneracy vs real bugs",
            mismatches.len()
        );
    }
}
