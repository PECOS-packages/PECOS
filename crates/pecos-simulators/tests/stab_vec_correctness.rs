// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use
// this file except in compliance with the License. You may obtain a copy of the
// License at https://www.apache.org/licenses/LICENSE-2.0

//! Phase-zero correctness gate for `StabVecGeneric` performance work.
//!
//! `StateVecSoA` is the independent correctness oracle. The committed bit
//! patterns are a second, deliberately stricter gate: they make even a one-ULP
//! change visible when it remains well inside the oracle tolerance.

use num_complex::Complex64;
use pecos_core::{Angle64, BitSet, QubitId};
use pecos_random::{SeedableRng, TryRng};
use pecos_simulators::state_vector_test_utils::{
    assert_phase_exact_state_matches, normalized_z_projection,
};
use pecos_simulators::{
    ArbitraryRotationGateable, CliffordGateable, StabVec, StabVecGeneric, StateVecSoA,
};
use std::collections::BTreeMap;
use std::convert::Infallible;
use std::fmt::Write as _;

const ORACLE_TOLERANCE: f64 = 1e-12;
const REFERENCE_DATA: &str = include_str!("data/stab_vec_correctness_bits.txt");
const REGENERATE_CONFIRMATION: &str = "I_UNDERSTAND_THIS_REPLACES_PINNED_RESULTS";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct MeasurementPin {
    qubit: usize,
    outcome: bool,
    is_deterministic: bool,
}

#[derive(Debug)]
struct CaseResult {
    name: &'static str,
    amplitudes: Vec<Complex64>,
    measurements: Vec<MeasurementPin>,
}

#[derive(Debug)]
struct PinnedCase {
    amplitudes: Vec<(u64, u64)>,
    measurements: Vec<MeasurementPin>,
}

/// A fixed-word RNG for the exact `r == prob0` decision boundary.
#[derive(Clone, Debug)]
struct BoundaryRng {
    word: u64,
}

impl TryRng for BoundaryRng {
    type Error = Infallible;

    fn try_next_u32(&mut self) -> Result<u32, Self::Error> {
        Ok(u32::try_from(self.word >> 32).expect("upper half of u64 fits in u32"))
    }

    fn try_next_u64(&mut self) -> Result<u64, Self::Error> {
        Ok(self.word)
    }

    fn try_fill_bytes(&mut self, destination: &mut [u8]) -> Result<(), Self::Error> {
        let bytes = self.word.to_le_bytes();
        for (index, byte) in destination.iter_mut().enumerate() {
            *byte = bytes[index % bytes.len()];
        }
        Ok(())
    }
}

impl SeedableRng for BoundaryRng {
    type Seed = [u8; 8];

    fn from_seed(seed: Self::Seed) -> Self {
        Self {
            word: u64::from_le_bytes(seed),
        }
    }

    fn seed_from_u64(seed: u64) -> Self {
        Self { word: seed }
    }
}

fn qid(index: usize) -> [QubitId; 1] {
    [QubitId(index)]
}

fn measure(
    stab: &mut StabVec,
    dense_state: &mut Vec<Complex64>,
    qubit: usize,
    label: &str,
) -> MeasurementPin {
    let result = stab
        .mz(&qid(qubit))
        .into_iter()
        .next()
        .expect("single-qubit measurement returned no result");
    *dense_state = normalized_z_projection(dense_state, qubit, result.outcome, label);
    MeasurementPin {
        qubit,
        outcome: result.outcome,
        is_deterministic: result.is_deterministic,
    }
}

fn compare_to_oracle(stab: &mut StabVec, dense: &mut StateVecSoA, label: &str) -> Vec<Complex64> {
    let actual = stab.state_vector();
    let expected = dense.state();
    assert_phase_exact_state_matches(&actual, &expected, ORACLE_TOLERANCE, label);
    actual
}

fn compare_projected_to_oracle(
    stab: &mut StabVec,
    expected: &[Complex64],
    label: &str,
) -> Vec<Complex64> {
    let actual = stab.state_vector();
    assert_phase_exact_state_matches(&actual, expected, ORACLE_TOLERANCE, label);
    actual
}

fn exact_stab(num_qubits: usize, seed: u64) -> StabVec {
    StabVec::builder(num_qubits)
        .pruning_threshold(0.0)
        .seed(seed)
        .build()
}

fn clifford_only() -> CaseResult {
    const NAME: &str = "clifford_only";
    const SEED: u64 = 0xC11F_F04D;
    let mut stab = exact_stab(4, SEED);
    let mut dense = StateVecSoA::with_seed(4, SEED);

    stab.h(&qid(0)).h(&qid(1)).x(&qid(2)).y(&qid(3));
    dense.h(&qid(0)).h(&qid(1)).x(&qid(2)).y(&qid(3));
    stab.cx(&[(QubitId(0), QubitId(2))])
        .cz(&[(QubitId(1), QubitId(3))])
        .szdg(&qid(0));
    dense
        .cx(&[(QubitId(0), QubitId(2))])
        .cz(&[(QubitId(1), QubitId(3))])
        .szdg(&qid(0));

    assert_eq!(stab.num_terms(), 1, "{NAME}: Clifford gates made terms");
    let amplitudes = compare_to_oracle(&mut stab, &mut dense, NAME);
    CaseResult {
        name: NAME,
        amplitudes,
        measurements: Vec::new(),
    }
}

fn small_non_clifford() -> CaseResult {
    const NAME: &str = "small_non_clifford";
    const SEED: u64 = 0x5A11_C0DE;
    let mut stab = exact_stab(5, SEED);
    let mut dense = StateVecSoA::with_seed(5, SEED);

    for (qubit, radians) in [(0, 0.37), (1, -0.51), (2, 0.83)] {
        let angle = Angle64::from_radians(radians);
        stab.h(&qid(qubit)).rz(angle, &qid(qubit)).h(&qid(qubit));
        dense.h(&qid(qubit)).rz(angle, &qid(qubit)).h(&qid(qubit));
    }
    stab.cx(&[(QubitId(0), QubitId(3))]);
    dense.cx(&[(QubitId(0), QubitId(3))]);

    assert_eq!(stab.num_terms(), 8, "{NAME}: expected three doublings");
    assert!(
        !stab.has_shared_projection_structure(),
        "{NAME}: RZ-then-H must make the term structures diverge"
    );
    let amplitudes = compare_to_oracle(&mut stab, &mut dense, NAME);
    CaseResult {
        name: NAME,
        amplitudes,
        measurements: Vec::new(),
    }
}

fn crossover_case(t_count: usize) -> CaseResult {
    let (name, seed) = match t_count {
        8 => ("crossover_8t_256_terms", 0xC805_0256),
        10 => ("crossover_10t_1024_terms", 0xC10A_1024),
        12 => ("crossover_12t_4096_terms_mc", 0xC12B_4096),
        _ => panic!("unsupported crossover T count {t_count}"),
    };
    let mut stab = exact_stab(10, seed);
    let mut dense = StateVecSoA::with_seed(10, seed);

    for qubit in 0..10 {
        stab.h(&qid(qubit));
        dense.h(&qid(qubit));
    }

    // Materializing after each T prevents same-qubit rotations from fusing.
    // RZ decomposition itself changes gamma only, so the terms remain on the
    // shared-structure dispatch path while their count crosses 2^n.
    for step in 0..t_count {
        let target = step % 9;
        stab.t(&qid(target));
        dense.t(&qid(target));
        let checkpoint = format!("{name}/after_t_{}", step + 1);
        compare_to_oracle(&mut stab, &mut dense, &checkpoint);
    }

    let expected_terms = 1usize << t_count;
    assert_eq!(
        stab.num_terms(),
        expected_terms,
        "{name}: wrong crossover term count"
    );
    assert!(
        stab.has_shared_projection_structure(),
        "{name}: expected the shared-structure dispatch precondition"
    );

    let mut measurements = Vec::new();
    let amplitudes = if t_count == 12 {
        // 4096 > the default mc_threshold (2048). Qubit 9 is an unrotated
        // |+> factor, so this is nondeterministic and cannot take the earlier
        // deterministic shortcut. The selected result consumes the fixed RNG.
        let mut expected = dense.state();
        measurements.push(measure(
            &mut stab,
            &mut expected,
            9,
            "crossover_12t_4096_terms_mc/mz9",
        ));
        assert!(
            !measurements[0].is_deterministic,
            "{name}: Monte-Carlo measurement unexpectedly deterministic"
        );
        compare_projected_to_oracle(&mut stab, &expected, name)
    } else {
        compare_to_oracle(&mut stab, &mut dense, name)
    };

    CaseResult {
        name,
        amplitudes,
        measurements,
    }
}

fn shared_structure_measurements() -> CaseResult {
    const NAME: &str = "shared_structure_measurements";
    const SEED: u64 = 0x5A4E_ED01;
    let mut stab = exact_stab(7, SEED);
    let mut dense = StateVecSoA::with_seed(7, SEED);

    for qubit in 0..7 {
        stab.h(&qid(qubit));
        dense.h(&qid(qubit));
    }
    for qubit in 0..4 {
        stab.t(&qid(qubit));
        dense.t(&qid(qubit));
        compare_to_oracle(
            &mut stab,
            &mut dense,
            &format!("{NAME}/after_t_on_q{qubit}"),
        );
    }
    assert_eq!(stab.num_terms(), 16, "{NAME}: expected four doublings");
    assert!(
        stab.has_shared_projection_structure(),
        "{NAME}: expected shared projection structure"
    );

    let mut expected = dense.state();
    let measurements = vec![
        measure(&mut stab, &mut expected, 6, &format!("{NAME}/mz6")),
        measure(&mut stab, &mut expected, 6, &format!("{NAME}/repeat_mz6")),
        measure(&mut stab, &mut expected, 5, &format!("{NAME}/mz5")),
    ];
    assert!(!measurements[0].is_deterministic);
    assert!(measurements[1].is_deterministic);
    assert!(!measurements[2].is_deterministic);
    let amplitudes = compare_projected_to_oracle(&mut stab, &expected, NAME);

    CaseResult {
        name: NAME,
        amplitudes,
        measurements,
    }
}

fn divergent_structure_measurement() -> CaseResult {
    const NAME: &str = "divergent_structure_measurement";
    const SEED: u64 = 0xD1A3_6940;
    let mut stab = exact_stab(7, SEED);
    let mut dense = StateVecSoA::with_seed(7, SEED);

    stab.h(&qid(6));
    dense.h(&qid(6));
    for (qubit, radians) in [(0, 0.37), (1, 0.53), (2, 0.71)] {
        let angle = Angle64::from_radians(radians);
        stab.h(&qid(qubit)).rz(angle, &qid(qubit)).h(&qid(qubit));
        dense.h(&qid(qubit)).rz(angle, &qid(qubit)).h(&qid(qubit));
    }
    assert_eq!(stab.num_terms(), 8, "{NAME}: expected three doublings");
    assert!(
        !stab.has_shared_projection_structure(),
        "{NAME}: RZ-then-H must select the divergent dispatch"
    );

    let mut expected = dense.state();
    let measurements = vec![measure(&mut stab, &mut expected, 6, &format!("{NAME}/mz6"))];
    assert!(!measurements[0].is_deterministic);
    let amplitudes = compare_projected_to_oracle(&mut stab, &expected, NAME);

    CaseResult {
        name: NAME,
        amplitudes,
        measurements,
    }
}

fn pending_rz_repeated_measurements() -> CaseResult {
    const NAME: &str = "pending_rz_repeated_measurements";
    const SEED: u64 = 0x0EAD_BEEF;
    let mut stab = exact_stab(3, SEED);
    let mut dense = StateVecSoA::with_seed(3, SEED);
    let angle = Angle64::from_radians(0.37);

    stab.h(&qid(0)).rz(angle, &qid(0));
    dense.h(&qid(0)).rz(angle, &qid(0));
    assert_eq!(
        stab.num_terms(),
        1,
        "{NAME}: RZ was not pending immediately before measurement"
    );

    let mut expected = dense.state();
    let measurements = vec![
        measure(&mut stab, &mut expected, 0, &format!("{NAME}/mz0")),
        measure(&mut stab, &mut expected, 0, &format!("{NAME}/repeat_mz0")),
        measure(
            &mut stab,
            &mut expected,
            1,
            &format!("{NAME}/deterministic_mz1"),
        ),
    ];
    assert!(!measurements[0].is_deterministic);
    assert!(measurements[1].is_deterministic);
    assert!(measurements[2].is_deterministic);
    let amplitudes = compare_projected_to_oracle(&mut stab, &expected, NAME);

    CaseResult {
        name: NAME,
        amplitudes,
        measurements,
    }
}

fn measurement_probability_boundary() -> CaseResult {
    const NAME: &str = "measurement_probability_boundary";
    // StandardUniform maps the top 53 bits of this raw RNG word to exactly 0.5.
    const SEED: u64 = 1 << 63;
    let mut stab = StabVecGeneric::<BitSet, BoundaryRng>::with_seed(1, SEED);
    let mut dense = StateVecSoA::with_seed(1, SEED);
    stab.h(&qid(0));
    dense.h(&qid(0));

    let mut expected = dense.state();
    let result = stab
        .mz(&qid(0))
        .into_iter()
        .next()
        .expect("single-qubit measurement returned no result");
    // |+> has prob0 == 0.5 and the fixed RNG produces r == 0.5. The current
    // decision rule is outcome = r >= prob0, so equality selects one.
    assert!(
        result.outcome,
        "{NAME}: equality boundary did not select one"
    );
    assert!(!result.is_deterministic);
    expected = normalized_z_projection(&expected, 0, result.outcome, NAME);
    let amplitudes = stab.state_vector();
    assert_phase_exact_state_matches(&amplitudes, &expected, ORACLE_TOLERANCE, NAME);

    CaseResult {
        name: NAME,
        amplitudes,
        measurements: vec![MeasurementPin {
            qubit: 0,
            outcome: result.outcome,
            is_deterministic: result.is_deterministic,
        }],
    }
}

fn fast_corpus() -> Vec<CaseResult> {
    vec![
        clifford_only(),
        small_non_clifford(),
        crossover_case(8),
        crossover_case(10),
        shared_structure_measurements(),
        divergent_structure_measurement(),
        pending_rz_repeated_measurements(),
        measurement_probability_boundary(),
    ]
}

fn parse_bool(value: &str, line_number: usize) -> bool {
    match value {
        "0" => false,
        "1" => true,
        _ => panic!("reference line {line_number}: expected boolean bit, got {value}"),
    }
}

fn parse_reference() -> BTreeMap<String, PinnedCase> {
    let mut cases = BTreeMap::new();
    let mut current_name: Option<String> = None;
    let mut amplitudes = Vec::new();
    let mut measurements = Vec::new();

    for (line_index, raw_line) in REFERENCE_DATA.lines().enumerate() {
        let line_number = line_index + 1;
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let fields: Vec<_> = line.split_whitespace().collect();
        match fields.as_slice() {
            ["case", name] => {
                assert!(
                    current_name.replace((*name).to_owned()).is_none(),
                    "reference line {line_number}: nested case"
                );
            }
            ["amplitude", index, real, imag] => {
                let index: usize = index
                    .parse()
                    .unwrap_or_else(|error| panic!("reference line {line_number}: {error}"));
                assert_eq!(
                    index,
                    amplitudes.len(),
                    "reference line {line_number}: amplitudes are not contiguous"
                );
                let real = u64::from_str_radix(real, 16)
                    .unwrap_or_else(|error| panic!("reference line {line_number}: {error}"));
                let imag = u64::from_str_radix(imag, 16)
                    .unwrap_or_else(|error| panic!("reference line {line_number}: {error}"));
                amplitudes.push((real, imag));
            }
            ["measurement", qubit, outcome, deterministic] => {
                let qubit = qubit
                    .parse()
                    .unwrap_or_else(|error| panic!("reference line {line_number}: {error}"));
                measurements.push(MeasurementPin {
                    qubit,
                    outcome: parse_bool(outcome, line_number),
                    is_deterministic: parse_bool(deterministic, line_number),
                });
            }
            ["end"] => {
                let name = current_name
                    .take()
                    .unwrap_or_else(|| panic!("reference line {line_number}: end without case"));
                let case = PinnedCase {
                    amplitudes: std::mem::take(&mut amplitudes),
                    measurements: std::mem::take(&mut measurements),
                };
                assert!(
                    cases.insert(name.clone(), case).is_none(),
                    "reference line {line_number}: duplicate case {name}"
                );
            }
            _ => panic!("reference line {line_number}: malformed record {line:?}"),
        }
    }
    assert!(current_name.is_none(), "reference data ends inside a case");
    cases
}

fn assert_matches_reference(results: &[CaseResult]) {
    let mut reference = parse_reference();
    assert_eq!(
        reference.len(),
        9,
        "reference data must contain the complete nine-case corpus"
    );
    for result in results {
        let expected = reference
            .remove(result.name)
            .unwrap_or_else(|| panic!("{}: missing from reference data", result.name));
        assert_eq!(
            result.measurements, expected.measurements,
            "{}: measurement results changed",
            result.name
        );
        assert_eq!(
            result.amplitudes.len(),
            expected.amplitudes.len(),
            "{}: amplitude count changed",
            result.name
        );
        for (index, (actual, (expected_real, expected_imag))) in result
            .amplitudes
            .iter()
            .zip(expected.amplitudes)
            .enumerate()
        {
            assert_eq!(
                actual.re.to_bits(),
                expected_real,
                "{}: amplitude[{index}].re changed: actual={:016x}, expected={expected_real:016x}",
                result.name,
                actual.re.to_bits()
            );
            assert_eq!(
                actual.im.to_bits(),
                expected_imag,
                "{}: amplitude[{index}].im changed: actual={:016x}, expected={expected_imag:016x}",
                result.name,
                actual.im.to_bits()
            );
        }
    }
}

fn render_reference(results: &[CaseResult]) -> String {
    let mut output = String::from(
        "# StabVecGeneric phase-zero correctness reference.\n\
         # Generated by the ignored regenerate_reference_data test after every case passed\n\
         # its phase-exact StateVecSoA oracle comparison. Regenerate only deliberately with\n\
         # the confirmation command documented on that test; review this file's bit diff.\n\n",
    );
    for result in results {
        writeln!(output, "case {}", result.name).unwrap();
        for measurement in &result.measurements {
            writeln!(
                output,
                "measurement {} {} {}",
                measurement.qubit,
                u8::from(measurement.outcome),
                u8::from(measurement.is_deterministic)
            )
            .unwrap();
        }
        for (index, amplitude) in result.amplitudes.iter().enumerate() {
            writeln!(
                output,
                "amplitude {index} {:016x} {:016x}",
                amplitude.re.to_bits(),
                amplitude.im.to_bits()
            )
            .unwrap();
        }
        output.push_str("end\n\n");
    }
    output
}

#[test]
fn fast_corpus_matches_oracle_and_reference_bits() {
    assert_matches_reference(&fast_corpus());
}

#[test]
#[ignore = "4096-term crossover and Monte-Carlo measurement are intentionally slow"]
fn slow_corpus_matches_oracle_and_reference_bits() {
    assert_matches_reference(&[crossover_case(12)]);
}

/// Deliberate update procedure:
///
/// `PECOS_REGENERATE_STAB_VEC_CORRECTNESS=I_UNDERSTAND_THIS_REPLACES_PINNED_RESULTS \
/// cargo test -p pecos-simulators --release --test stab_vec_correctness \
/// regenerate_reference_data -- --ignored --exact`
///
/// The corpus constructors run the phase-exact `StateVecSoA` oracle checks
/// before this writes anything. Review the resulting hexadecimal bit diff.
#[test]
#[ignore = "reference regeneration must be explicit and reviewed"]
fn regenerate_reference_data() {
    assert_eq!(
        std::env::var("PECOS_REGENERATE_STAB_VEC_CORRECTNESS").as_deref(),
        Ok(REGENERATE_CONFIRMATION),
        "refusing to replace pinned results without the documented confirmation"
    );
    let mut results = fast_corpus();
    results.push(crossover_case(12));
    let rendered = render_reference(&results);
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data/stab_vec_correctness_bits.txt");
    std::fs::write(&path, rendered)
        .unwrap_or_else(|error| panic!("failed to write {}: {error}", path.display()));
}
