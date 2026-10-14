//! Tesseract trellis-mode decoder through the PECOS wrap.
//!
//! The numeric expectations are upstream's own (`tesseract_trellis.test.cc`),
//! so the wrap is checked against the decoder it wraps, not against itself.

use ndarray::Array1;
use pecos_decoder_core::ObservableDecoder;
use pecos_tesseract::{
    TesseractConfig, TesseractDecoder, TesseractTrellisConfig, TesseractTrellisDecoder,
    TesseractTrellisRankingMode,
};

fn decode(
    decoder: &mut TesseractTrellisDecoder,
    fired: &[u64],
) -> pecos_tesseract::TesseractTrellisResult {
    decoder
        .decode_detections(&Array1::from_vec(fired.to_vec()).view())
        .unwrap()
}

#[test]
fn ambiguous_syndrome_reports_mass_ratio() {
    let dem = "error(0.1) D0\nerror(0.2) D0 L0\ndetector(0, 0, 0) D0\n";
    let mut decoder = TesseractTrellisDecoder::new(
        dem,
        TesseractTrellisConfig {
            beam_width: 16,
            ..TesseractTrellisConfig::default()
        },
    )
    .unwrap();
    assert_eq!(decoder.num_detectors(), 1);
    assert_eq!(decoder.num_errors(), 2);
    assert_eq!(decoder.num_observables(), 1);

    let fired = decode(&mut decoder, &[0]);
    assert!(!fired.low_confidence);
    assert_eq!(fired.observables_mask, 1);
    assert!((fired.observable_probability - 0.18 / 0.26).abs() < 1e-12);
    // Beam statistics are populated, not zero-filled: the model has one
    // detector and two mechanisms, so a shot expands at least one state,
    // keeps between one and `beam_width` states, and reaches frontier width 1.
    assert!(fired.num_states_expanded >= 1, "{fired:?}");
    assert!(fired.num_states_merged >= 1, "{fired:?}");
    assert!((1..=16).contains(&fired.max_beam_size_seen), "{fired:?}");
    assert_eq!(fired.max_frontier_width_seen, 1, "{fired:?}");

    let quiet = decode(&mut decoder, &[]);
    assert!(!quiet.low_confidence);
    assert_eq!(quiet.observables_mask, 0);
    assert!((quiet.observable_probability - 0.02 / 0.74).abs() < 1e-12);
}

#[test]
fn sums_mass_across_explanations_where_min_cost_would_not() {
    // Three two-error chains each flip L0; one single error does not. The
    // cheapest single explanation is the no-L0 branch, but the summed mass
    // of the chains wins.
    let dem = "error(0.2) D0 D1 L0\nerror(0.2) D1\nerror(0.2) D0 D2 L0\nerror(0.2) D2\n\
               error(0.2) D0 D3 L0\nerror(0.2) D3\nerror(0.1) D0\n\
               detector(0, 0, 0) D0\ndetector(1, 0, 0) D1\ndetector(2, 0, 0) D2\ndetector(3, 0, 0) D3\n";
    let mut decoder = TesseractTrellisDecoder::new(
        dem,
        TesseractTrellisConfig {
            beam_width: 64,
            ..TesseractTrellisConfig::default()
        },
    )
    .unwrap();
    let result = decode(&mut decoder, &[0]);

    let p_chain_edge: f64 = 0.2;
    let p_right: f64 = 0.1;
    let chain_present = p_chain_edge * p_chain_edge;
    let chain_absent = (1.0 - p_chain_edge) * (1.0 - p_chain_edge);
    let odd_chain_mass = 3.0 * chain_present * chain_absent * chain_absent + chain_present.powi(3);
    let even_chain_mass = chain_absent.powi(3) + 3.0 * chain_present * chain_present * chain_absent;
    let mass_l0 = odd_chain_mass * (1.0 - p_right);
    let mass_no_l0 = even_chain_mass * p_right;
    let expected = mass_l0 / (mass_l0 + mass_no_l0);
    assert!(expected > 0.5);

    assert!(!result.low_confidence);
    assert_eq!(result.observables_mask, 1);
    assert!((result.observable_probability - expected).abs() < 1e-12);

    // The A* decoder picks the single cheapest branch instead.
    let mut astar = TesseractDecoder::new(dem, TesseractConfig::default()).unwrap();
    let astar_result = astar
        .decode_detections(&Array1::from_vec(vec![0]).view())
        .unwrap();
    assert_eq!(astar_result.observables_mask, 0);
}

#[test]
fn unreachable_detector_is_low_confidence_with_nan_probability() {
    // D1 is declared but no mechanism touches it: upstream's low-confidence case.
    let dem = "error(0.1) D0 L0\ndetector(0, 0, 0) D0\ndetector(1, 0, 0) D1\n";
    let mut decoder = TesseractTrellisDecoder::new(dem, TesseractTrellisConfig::default()).unwrap();
    assert_eq!(decoder.num_detectors(), 2);
    let result = decode(&mut decoder, &[1]);
    assert!(result.low_confidence);
    assert!(result.observable_probability.is_nan());
}

#[test]
fn out_of_range_detector_is_an_input_error_like_astar() {
    let dem = "error(0.1) D0 L0\n";
    let mut trellis = TesseractTrellisDecoder::new(dem, TesseractTrellisConfig::default()).unwrap();
    let mut astar = TesseractDecoder::new(dem, TesseractConfig::default()).unwrap();
    let fired = Array1::from_vec(vec![1u64]);
    let error = trellis
        .decode_detections(&fired.view())
        .err()
        .unwrap()
        .to_string();
    assert!(error.contains("out of range"), "{error}");
    assert!(astar.decode_detections(&fired.view()).is_err());
    // The dense path sees the same guard: a syndrome wider than the model.
    let error = trellis.decode_obs(&[0, 1]).err().unwrap().to_string();
    assert!(error.contains("out of range"), "{error}");
}

#[test]
fn every_ranking_mode_decodes_the_ambiguous_syndrome() {
    let dem = "error(0.1) D0\nerror(0.2) D0 L0\ndetector(0, 0, 0) D0\n";
    for ranking_mode in [
        TesseractTrellisRankingMode::MassOnly,
        TesseractTrellisRankingMode::FutureDetcostRanked,
        TesseractTrellisRankingMode::FutureActiveDetcostRanked,
    ] {
        let mut decoder = TesseractTrellisDecoder::new(
            dem,
            TesseractTrellisConfig {
                beam_width: 16,
                ranking_mode,
                ..TesseractTrellisConfig::default()
            },
        )
        .unwrap();
        let result = decode(&mut decoder, &[0]);
        assert!(!result.low_confidence, "{ranking_mode:?}");
        assert_eq!(result.observables_mask, 1, "{ranking_mode:?}");
        assert!(
            (result.observable_probability - 0.18 / 0.26).abs() < 1e-12,
            "{ranking_mode:?}"
        );
    }
}

#[test]
fn merge_errors_combines_identical_mechanisms() {
    let dem = "error(0.1) D0 L0\nerror(0.1) D0 L0\nerror(0) D0\ndetector(0, 0, 0) D0\n";
    let mut merged = TesseractTrellisDecoder::new(dem, TesseractTrellisConfig::default()).unwrap();
    let mut unmerged = TesseractTrellisDecoder::new(
        dem,
        TesseractTrellisConfig {
            merge_errors: false,
            ..TesseractTrellisConfig::default()
        },
    )
    .unwrap();
    // Both count the three flattened-DEM mechanisms, like the A* wrap.
    assert_eq!(merged.num_errors(), 3);
    assert_eq!(unmerged.num_errors(), 3);
    let merged_result = decode(&mut merged, &[0]);
    let unmerged_result = decode(&mut unmerged, &[0]);
    for result in [&merged_result, &unmerged_result] {
        assert_eq!(result.observables_mask, 1);
        assert!((result.observable_probability - 1.0).abs() < 1e-12);
    }
    // Merging leaves one trellis layer instead of two, so fewer states expand.
    assert!(
        unmerged_result.num_states_expanded > merged_result.num_states_expanded,
        "{unmerged_result:?} vs {merged_result:?}"
    );
}

#[test]
fn repeated_detector_indices_are_rejected_by_both_decoders() {
    let dem = "error(0.1) D0\nerror(0.2) D0 L0\n";
    let mut trellis = TesseractTrellisDecoder::new(dem, TesseractTrellisConfig::default()).unwrap();
    let mut astar = TesseractDecoder::new(dem, TesseractConfig::default()).unwrap();
    let repeated = Array1::from_vec(vec![0u64, 0]);
    for error in [
        trellis
            .decode_detections(&repeated.view())
            .err()
            .unwrap()
            .to_string(),
        astar
            .decode_detections(&repeated.view())
            .err()
            .unwrap()
            .to_string(),
        astar
            .decode_with_order(&repeated.view(), 0)
            .err()
            .unwrap()
            .to_string(),
    ] {
        assert!(error.contains("repeated"), "{error}");
    }
    // The set form still decodes.
    assert_eq!(decode(&mut trellis, &[0]).observables_mask, 1);
}

#[test]
fn more_than_one_observable_is_rejected_at_construction() {
    let dem = "error(0.1) D0 L0\nerror(0.1) D1 L1\n";
    let error = TesseractTrellisDecoder::new(dem, TesseractTrellisConfig::default())
        .err()
        .unwrap()
        .to_string();
    assert!(error.contains("at most one observable"), "{error}");
}

#[test]
fn observable_decoder_matches_sparse_decode_and_astar_on_a_chain() {
    // D0's direct boundary flips L0 (mass 0.1*0.9*0.9); the detour through
    // D1 does not (mass 0.9*0.1*0.1). Both decoders must predict the flip,
    // and the trellis must report the 0.9 posterior.
    let dem = "error(0.1) D0 L0\nerror(0.1) D0 D1\nerror(0.1) D1\n";
    let mut trellis = TesseractTrellisDecoder::new(dem, TesseractTrellisConfig::default()).unwrap();
    let mut astar = TesseractDecoder::new(dem, TesseractConfig::default()).unwrap();
    for syndrome in [[0u8, 0], [1, 0], [0, 1], [1, 1]] {
        let fired: Vec<u64> = syndrome
            .iter()
            .enumerate()
            .filter(|(_, v)| **v != 0)
            .map(|(i, _)| i as u64)
            .collect();
        let sparse = decode(&mut trellis, &fired);
        let dense = trellis.decode_obs(&syndrome).unwrap();
        assert_eq!(
            dense.to_u64(),
            Some(sparse.observables_mask),
            "{syndrome:?}"
        );
        assert_eq!(
            astar.decode_obs(&syndrome).unwrap().to_u64(),
            Some(sparse.observables_mask),
            "{syndrome:?}"
        );
    }
    let result = decode(&mut trellis, &[0]);
    assert_eq!(result.observables_mask, 1);
    assert!((result.observable_probability - 0.9).abs() < 1e-12);
}

#[test]
fn dense_decode_accepts_strided_views_like_astar() {
    use pecos_decoder_core::Decoder;
    let dem = "error(0.1) D0 L0\nerror(0.1) D0 D1\nerror(0.1) D1\n";
    let mut trellis = TesseractTrellisDecoder::new(dem, TesseractTrellisConfig::default()).unwrap();
    let mut astar = TesseractDecoder::new(dem, TesseractConfig::default()).unwrap();
    // Every other element: the view reads [1, 0] without being contiguous.
    let backing = Array1::from_vec(vec![1u8, 9, 0, 9]);
    let view = backing.slice(ndarray::s![..;2]);
    assert!(view.as_slice().is_none());
    let trellis_result = trellis.decode(&view).unwrap();
    let astar_result = astar.decode(&view).unwrap();
    assert_eq!(trellis_result.observables_mask, 1);
    assert_eq!(astar_result.observables_mask, 1);
}
