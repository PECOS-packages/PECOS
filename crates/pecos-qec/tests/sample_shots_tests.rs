// Copyright 2026 The PECOS Developers
// Licensed under the Apache License, Version 2.0

//! Rust batch sampling contracts, including the frozen Python geometric stream.
use pecos_core::pauli::X;
use pecos_decoders::batch::DecoderFactory;
use pecos_decoders::{DecodeModel, DecoderError, ExecutionTraits, ObsMask, ObservableDecoder};
use pecos_qec::fault_tolerance::InfluenceBuilder;
use pecos_qec::fault_tolerance::dem_builder::{
    DemOutput, DemSampler, DemSamplerBuilder, DetectorDef, DetectorErrorModel, FaultMechanism,
    ParsedDem, SampleBatch, SampleShotsError, SamplingEngine,
};
use pecos_quantum::DagCircuit;
use pecos_random::PecosRng;

const DEM: &str = "error(0.25) D0 L0\nerror(0.1) D0\n";
const SHOTS: [usize; 6] = [0, 1, 63, 64, 65, 1000];

fn model(tracked: bool) -> DetectorErrorModel {
    let mut dem = DetectorErrorModel::new();
    dem.add_detector(DetectorDef::new(0));
    dem.add_observable(DemOutput::new(0).with_label("first"));
    dem.add_observable(DemOutput::new(2).with_label("third"));
    if tracked {
        dem.add_tracked_pauli(DemOutput::new(4).with_pauli(X(0)).with_label("tracked"));
        dem.add_direct_contribution(
            FaultMechanism::from_unsorted_with_tracked_paulis([0], [0, 2], [4]),
            0.25,
        );
    } else {
        dem.add_direct_contribution(FaultMechanism::from_unsorted([0], [0, 2]), 0.25);
    }
    dem
}

fn check_columns(sampler: &DemSampler, detectors: usize, observables: usize) {
    for n in SHOTS {
        let mut actual_rng = PecosRng::seed_from_u64(91);
        let mut expected_rng = PecosRng::seed_from_u64(91);
        let actual = sampler.sample_shots(n, &mut actual_rng).unwrap();
        let (det, obs) = sampler.sample_batch_geometric(n, &mut expected_rng);
        assert_eq!(actual.num_shots(), n);
        assert_eq!(actual.num_detectors(), detectors);
        assert_eq!(actual.num_observables(), observables);
        assert_eq!(actual.det_columns(), det);
        assert_eq!(actual.obs_columns(), obs);
        assert_eq!(actual_rng.next_u64(), expected_rng.next_u64());
        // The validating constructor also checks every final word's padding.
        assert_eq!(SampleBatch::from_columnar(det, obs, n).unwrap(), actual);
    }
}

#[test]
fn frozen_python_stream() {
    // The Python test builds its sampler with DemSampler.from_dem_string, which
    // feeds SamplingEngine::from_mechanisms with each positive-probability
    // error line in order. Pin that route, and the ParsedDem route a Rust user
    // reaches for, against the same frozen values.
    let python_route = DemSampler::from_engine(SamplingEngine::from_mechanisms(
        [(0.25, vec![0], vec![0]), (0.1, vec![0], vec![])],
        1,
        1,
    ));
    let parsed_route = ParsedDem::parse(DEM).unwrap().to_dem_sampler();
    for sampler in [python_route, parsed_route] {
        let batch = sampler
            .sample_shots(12, &mut PecosRng::seed_from_u64(91))
            .unwrap();
        // test_dem_sampler_decode.py::test_sample_batch_stream_remains_frozen:
        // detectors at shots 0, 10, 11; observables at shots 0, 10.
        assert_eq!(batch.num_shots(), 12);
        assert_eq!(batch.det_columns(), &[vec![0xC01]]);
        assert_eq!(batch.obs_columns(), &[vec![0x401]]);
    }
}

#[test]
fn model_columns_exclude_tracked_paulis_and_keep_sparse_ids() {
    for tracked in [false, true] {
        let sampler = DemSampler::from_detector_error_model(&model(tracked));
        assert_eq!(sampler.num_tracked_paulis(), if tracked { 5 } else { 0 });
        check_columns(&sampler, 1, 3);
        let batch = sampler
            .sample_shots(1000, &mut PecosRng::seed_from_u64(91))
            .unwrap();
        assert_eq!(batch.obs_columns()[0], batch.obs_columns()[2]);
        assert!(batch.obs_columns()[0].iter().any(|&word| word != 0));
        assert!(batch.obs_columns()[1].iter().all(|&word| word == 0));
    }
}

#[test]
fn parsed_columns_cover_declarations_wide_ids_and_cancellation() {
    for (text, detectors, observables) in [
        (DEM, 1, 1),
        ("detector D2\nlogical_observable L3", 3, 4),
        ("error(0.25) D0 L64", 1, 65),
        ("error(0.25) D0 L0 ^ D0 D1 L0 L2", 2, 3),
    ] {
        let sampler = ParsedDem::parse(text).unwrap().to_dem_sampler();
        check_columns(&sampler, detectors, observables);
        let batch = sampler
            .sample_shots(1000, &mut PecosRng::seed_from_u64(91))
            .unwrap();
        if observables == 65 {
            assert_eq!(batch.det_columns()[0], batch.obs_columns()[64]);
            assert!(batch.obs_columns()[64].iter().any(|&word| word != 0));
            assert!(
                batch.obs_columns()[..64]
                    .iter()
                    .flatten()
                    .all(|&word| word == 0)
            );
        } else if text.contains('^') {
            assert!(batch.det_columns()[0].iter().all(|&word| word == 0));
            assert!(
                batch.obs_columns()[..2]
                    .iter()
                    .flatten()
                    .all(|&word| word == 0)
            );
            assert_eq!(batch.det_columns()[1], batch.obs_columns()[2]);
            assert!(batch.obs_columns()[2].iter().any(|&word| word != 0));
        }
    }
}

#[test]
fn raw_rejection_preserves_rng_even_for_zero_shots() {
    let mut circuit = DagCircuit::new();
    circuit.pz(&[0]);
    circuit.h(&[0]);
    circuit.mz(&[0]);
    let influence = InfluenceBuilder::new(&circuit).build().unwrap();
    let sampler = DemSamplerBuilder::new(&influence)
        .with_noise(0.1, 0.1, 0.1, 0.1)
        .raw_measurements()
        .build()
        .unwrap();
    for n in [0, 65] {
        let mut rng = PecosRng::seed_from_u64(91);
        let mut untouched = rng.clone();
        let result = sampler.sample_shots(n, &mut rng);
        for _ in 0..8 {
            assert_eq!(rng.next_u64(), untouched.next_u64());
        }
        assert_eq!(result, Err(SampleShotsError::RawMeasurements));
    }
}

#[test]
fn zero_shots_preserve_widths_and_rng() {
    let sampler = DemSampler::from_detector_error_model(&model(true));
    let mut rng = PecosRng::seed_from_u64(91);
    let mut untouched = rng.clone();
    let batch = sampler.sample_shots(0, &mut rng).unwrap();
    for _ in 0..8 {
        assert_eq!(rng.next_u64(), untouched.next_u64());
    }
    assert_eq!(batch.num_shots(), 0);
    assert_eq!(batch.det_columns(), &[Vec::<u64>::new()]);
    assert_eq!(batch.obs_columns(), vec![Vec::<u64>::new(); 3]);
}

#[test]
fn to_sampler_preserves_construction_and_metadata() {
    let dem = model(true);
    let actual = dem.to_sampler();
    let expected = DemSampler::from_detector_error_model(&dem);
    assert_eq!(actual.mode(), expected.mode());
    for (actual_outputs, expected_outputs) in [
        (&actual.labels().dem_outputs, &expected.labels().dem_outputs),
        (
            &actual.labels().tracked_paulis,
            &expected.labels().tracked_paulis,
        ),
    ] {
        assert_eq!(actual_outputs.len(), expected_outputs.len());
        for (actual_output, expected_output) in actual_outputs.iter().zip(expected_outputs) {
            assert_eq!(actual_output.is_some(), expected_output.is_some());
            if let (Some(a), Some(e)) = (actual_output, expected_output) {
                assert_eq!(a.id, e.id);
                assert_eq!(a.records, e.records);
                assert_eq!(a.kind, e.kind);
                assert_eq!(a.pauli, e.pauli);
                assert_eq!(a.label, e.label);
            }
        }
    }
    assert_eq!(
        actual.labels().dem_output_labels,
        expected.labels().dem_output_labels
    );
    assert_eq!(
        actual.labels().tracked_pauli_labels,
        expected.labels().tracked_pauli_labels
    );
    for n in SHOTS {
        assert_eq!(
            actual
                .sample_shots(n, &mut PecosRng::seed_from_u64(91))
                .unwrap(),
            expected
                .sample_shots(n, &mut PecosRng::seed_from_u64(91))
                .unwrap()
        );
    }
}

struct SyndromeDecoder;
impl ObservableDecoder for SyndromeDecoder {
    fn num_detectors(&self) -> Option<usize> {
        Some(1)
    }
    fn decode_obs(&mut self, syndrome: &[u8]) -> Result<ObsMask, DecoderError> {
        Ok(ObsMask::from(u64::from(syndrome[0])))
    }
}
impl DecoderFactory for SyndromeDecoder {
    type BuildError = DecoderError;
    fn execution_traits(&self) -> ExecutionTraits {
        ExecutionTraits::default()
    }
    fn native_batch_capable(&self) -> bool {
        false
    }
    fn build(&self, model: &DecodeModel) -> Result<Box<dyn ObservableDecoder>, DecoderError> {
        assert_eq!(model, &DecodeModel::SingleDem(DEM.to_string()));
        Ok(Box::new(Self))
    }
}

#[test]
fn sampled_batch_decodes_with_stub_factory() {
    let batch = ParsedDem::parse(DEM)
        .unwrap()
        .to_dem_sampler()
        .sample_shots(12, &mut PecosRng::seed_from_u64(91))
        .unwrap();
    let result = batch.decode(DEM, &SyndromeDecoder).unwrap();
    assert_eq!(result.num_shots, 12);
    // Predict L0 from D0: only shot 11 differs in the frozen stream.
    assert_eq!(result.num_errors, 1);
}
