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

//! Whole-component window fixtures from the streaming-window design packet.

use pecos_decoder_core::streaming::StreamingDecoder;
use pecos_decoder_core::window::StructuredDem;
use pecos_decoder_core::{DecoderError, EdgeDecoder, ObservableDecoder};
use pecos_pymatching::{PyMatchingDecoder, PyMatchingEdgeDecoder};
use pecos_uf_decoder::{StreamingWindowedDecoder, UfDecoder, UfDecoderConfig, WindowedConfig};
use std::fmt::Write as _;

fn strip_dem() -> String {
    let mut dem = String::new();
    let id = |x, t| 4 * t + x - 1;
    for t in 0..20 {
        for x in 1..=4 {
            writeln!(dem, "detector({x},0,{t}) D{}", id(x, t)).unwrap();
        }
    }
    let probability = |w: f64| 1.0 / (1.0 + w.exp());
    for t in 0..20 {
        for x in 1..=4 {
            if t + 1 < 20 {
                let w = if (x, t) == (1, 4) { 8.0 } else { 1.0 };
                writeln!(
                    dem,
                    "error({:?}) D{} D{}",
                    probability(w),
                    id(x, t),
                    id(x, t + 1)
                )
                .unwrap();
            }
            if x < 4 {
                let w = if t == 3 { 1.0 } else { 1.01 };
                writeln!(
                    dem,
                    "error({:?}) D{} D{}",
                    probability(w),
                    id(x, t),
                    id(x + 1, t)
                )
                .unwrap();
            }
        }
        writeln!(dem, "error({:?}) D{} L0", probability(2.0), id(1, t)).unwrap();
        writeln!(dem, "error({:?}) D{}", probability(2.0), id(4, t)).unwrap();
    }
    writeln!(dem, "logical_observable L0").unwrap();
    dem
}

fn model(errors: &str, times: &[u32]) -> String {
    let mut dem = errors.to_owned();
    for (d, t) in times.iter().enumerate() {
        writeln!(dem, "detector({d},0,{t}) D{d}").unwrap();
    }
    writeln!(dem, "logical_observable L0").unwrap();
    dem
}

fn matching(
    dem: &str,
    step: usize,
    buffer: usize,
    correlated: bool,
) -> StreamingWindowedDecoder<PyMatchingEdgeDecoder> {
    StreamingWindowedDecoder::from_dem(dem, WindowedConfig { step, buffer }, |w| {
        PyMatchingEdgeDecoder::from_commit_window(w, correlated)
    })
    .unwrap()
}

fn mono(dem: &str, syndrome: &[u8], correlated: bool) -> u64 {
    PyMatchingDecoder::from_dem_with_correlations(dem, correlated)
        .unwrap()
        .decode_to_observables(syndrome)
        .unwrap()
}

#[test]
fn test_1_straddling_strip() {
    let dem = strip_dem();
    let mut syndrome = vec![0; 80];
    for d in [13, 16, 18] {
        syndrome[d] = 1;
    }
    let expected = mono(&dem, &syndrome, false);
    let mut decoder = matching(&dem, 2, 1, false);
    assert_eq!(
        decoder.decode_to_observables(&syndrome).unwrap(),
        expected,
        "straddling strip must match monolithic matching"
    );
    assert!(decoder.residual().iter().all(|&s| s == 0));
}

#[test]
fn test_2_undercut() {
    // This pins the look-behind, rather than the component commit unit: it also
    // passes with per-edge components. Test 1 discriminates the component unit.
    let mut errors = String::new();
    for (weight, targets) in [
        (3.0f64, "D0 D1 L0"),
        (1.0, "D0 D2"),
        (0.5, "D1 D3"),
        (0.5, "D2 D4"),
        (4.0, "D3"),
        (4.0, "D4"),
    ] {
        writeln!(errors, "error({}) {targets}", 1.0 / (1.0 + weight.exp())).unwrap();
    }
    let dem = model(&errors, &[0, 1, 1, 2, 2]);
    let syndrome = [1, 1, 0, 0, 0];
    let expected = mono(&dem, &syndrome, false);
    assert_eq!(expected, 1, "undercut monolithic correction must flip L0");
    assert_eq!(
        matching(&dem, 1, 1, false)
            .decode_to_observables(&syndrome)
            .unwrap(),
        expected
    );
}

#[test]
fn fixture_v_future_veto() {
    let mut errors = String::new();
    for (weight, targets) in [
        (1.0f64, "D0 D1"),
        (0.5, "D1 L0"),
        (1.0, "D1 D2"),
        (1.2, "D2"),
        (10.0, "D0"),
    ] {
        writeln!(errors, "error({}) {targets}", 1.0 / (1.0 + weight.exp())).unwrap();
    }
    let dem = model(&errors, &[0, 1, 2]);
    let syndrome = [1, 0, 1];
    // Orchestrator-owned expected values from the reference prototype. The
    // merged boundary at U is future even though rep_any is the real boundary.
    assert_eq!(mono(&dem, &syndrome, false), 0);
    let mut decoder = matching(&dem, 1, 1, false);
    decoder.start_shot(&syndrome).unwrap();
    decoder.decode_next_window().unwrap();
    assert_eq!(
        decoder.residual()[0],
        1,
        "fixture V must defer A in window 0 because the merged edge is future"
    );
    assert_eq!(decoder.diagnostics().deferred_components, 1);
    assert!(decoder.diagnostics().committed_columns.is_empty());
    decoder.finish().unwrap();
    assert_eq!(
        decoder.accumulated_obs(),
        0,
        "fixture V windowed correction must be 0"
    );
}

#[test]
fn fixture_r_future_edge_representative() {
    let mut errors = String::new();
    for d in 0..4 {
        let observable = if d == 2 { " L0" } else { "" };
        writeln!(errors, "error(0.3) D{d} D{}{observable}", d + 1).unwrap();
    }
    for d in 0..5 {
        writeln!(errors, "error(0.00005) D{d}").unwrap();
    }
    let dem = model(&errors, &[0, 1, 2, 3, 4]);
    let syndrome = [1, 0, 0, 0, 0];
    // Orchestrator-owned expected values from the reference prototype. The
    // difference from monolithic is by design (section 3 limitations): the
    // representative is a valid lift, not a most-probable explanation.
    assert_eq!(mono(&dem, &syndrome, false), 0);
    let mut decoder = matching(&dem, 1, 1, false);
    assert_eq!(
        decoder.decode_to_observables(&syndrome).unwrap(),
        1,
        "fixture R must resolve the future edge to projected D2-D3 and flip L0"
    );
    assert_eq!(decoder.diagnostics().forced_future_components, 1);
    assert!(decoder.diagnostics().committed_columns.contains(&2));
    assert!(decoder.residual().iter().all(|&s| s == 0));
}

#[test]
fn feed_round_rejects_decoded_rows() {
    let mut decoder = matching(&projection_chain(), 1, 1, false);
    decoder.feed_round(0, &[(0, 1)]).unwrap();
    assert_eq!(decoder.next_window(), 0);
    decoder.feed_round(1, &[]).unwrap();
    assert_eq!(decoder.next_window(), 1);
    let residual = decoder.residual().to_vec();
    let diagnostics = decoder.diagnostics().clone();
    let accumulated = decoder.accumulated_obs();
    for (round, detectors) in [(0, vec![(0, 0)]), (1, vec![(1, 1)]), (1, vec![])] {
        let error = decoder.feed_round(round, &detectors).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("already been decoded through round 1")
        );
        assert_eq!(decoder.next_window(), 1);
        assert_eq!(decoder.residual(), residual);
        assert_eq!(decoder.diagnostics(), &diagnostics);
        assert_eq!(decoder.accumulated_obs(), accumulated);
    }
    decoder.feed_round(2, &[]).unwrap();
    decoder.finish().unwrap();
    assert!(
        decoder
            .feed_round(4, &[])
            .unwrap_err()
            .to_string()
            .contains("already been decoded through round 4")
    );
    decoder.reset();
    assert!(decoder.feed_round(0, &[(0, 1)]).is_ok());
}

fn projection_chain() -> String {
    let mut errors = String::new();
    for d in 0..4 {
        writeln!(errors, "error(0.3) D{d} D{}", d + 1).unwrap();
    }
    for d in 0..5 {
        writeln!(errors, "error(0.00005) D{d}").unwrap();
    }
    model(&errors, &[0, 1, 2, 3, 4])
}

#[test]
fn test_3_forcing_clears_retired_row() {
    let mut decoder = matching(&projection_chain(), 1, 1, false);
    decoder.start_shot(&[1, 0, 0, 0, 0]).unwrap();
    decoder.decode_next_window().unwrap();
    assert_eq!(
        decoder.residual()[0],
        1,
        "first window must defer the defect"
    );
    decoder.decode_next_window().unwrap();
    assert!(
        decoder.next_window() < decoder.num_windows(),
        "forcing must be exercised before the last window"
    );
    assert_eq!(
        decoder.residual()[0],
        0,
        "retired row 0 must clear in window 1 by forcing"
    );
    assert_eq!(decoder.diagnostics().forced_future_components, 1);
}

fn check_retired(dem: &str, syndrome: &[u8], step: usize, buffer: usize) {
    let parsed = StructuredDem::from_dem_str(dem).unwrap();
    let times = parsed.commit_detector_times().unwrap();
    let mut decoder = matching(dem, step, buffer, false);
    decoder.start_shot(syndrome).unwrap();
    for window in 0..decoder.num_windows() {
        decoder.decode_next_window().unwrap();
        for (row, &time) in times.iter().enumerate() {
            if (time as usize) < window * step {
                assert_eq!(
                    decoder.residual()[row],
                    0,
                    "retired row {row} is nonzero after window {window}"
                );
            }
        }
    }
    assert!(
        decoder.residual().iter().all(|&s| s == 0),
        "final syndrome must be satisfied"
    );
}

#[test]
fn test_4_retired_rows_strip() {
    let mut syndrome = vec![0; 80];
    for d in [13, 16, 18] {
        syndrome[d] = 1;
    }
    check_retired(&strip_dem(), &syndrome, 2, 1);
}

#[test]
fn test_4_retired_rows_200_surface_shots() {
    let dem = include_str!("data/window_surface_d5.dem");
    let parsed = StructuredDem::from_dem_str(dem).unwrap();
    let mut rng = pecos_random::PecosRng::seed_from_u64(1234);
    for _ in 0..200 {
        let mut syndrome = vec![0; parsed.num_detectors];
        for error in &parsed.errors {
            if rng.next_f64() < error.probability {
                for component in &error.components {
                    for &d in &component.detectors {
                        syndrome[d as usize] ^= 1;
                    }
                }
            }
        }
        check_retired(dem, &syndrome, 2, 1);
    }
}

#[test]
fn test_6_surface_code_is_solvable() {
    let dem = include_str!(
        "../../../examples/surface_code_circuits/surface_code_d3_z_stim_decomposed.dem"
    );
    let parsed = StructuredDem::from_dem_str(dem).unwrap();
    let minimum = pecos_decoder_core::window::min_buffer_rounds(&parsed).unwrap();
    let decoder = matching(dem, 1, minimum as usize, false);
    assert!(decoder.num_windows() > 0);
}

struct EmptyDecoder;
impl EdgeDecoder for EmptyDecoder {
    fn decode_to_edges(&mut self, _: &[u8]) -> Result<Vec<usize>, DecoderError> {
        Ok(Vec::new())
    }
}

#[test]
fn test_5_incidence_error_is_atomic() {
    let dem = model("error(0.1) D0 L0\n", &[0]);
    let mut decoder =
        StreamingWindowedDecoder::from_dem(&dem, WindowedConfig { step: 1, buffer: 0 }, |_| {
            Ok(EmptyDecoder)
        })
        .unwrap();
    decoder.start_shot(&[1]).unwrap();
    let before = decoder.diagnostics().clone();
    let error = decoder
        .decode_next_window()
        .expect_err("incomplete correction must fail the production incidence check");
    assert!(
        error
            .to_string()
            .contains("window 0: incidence mismatch at row 0")
    );
    assert_eq!(decoder.next_window(), 0);
    assert_eq!(decoder.residual(), [1]);
    assert_eq!(decoder.accumulated_obs(), 0);
    assert_eq!(decoder.diagnostics(), &before);
}

#[test]
fn test_6_solvability_rejects_boundary_free_and_isolated_rows() {
    for dem in [
        model(
            "error(0.1) D0 D1\nerror(0.2) D0 D2\nerror(0.2) D1 D3\nerror(0.2) D2 D4\n",
            &[0, 1, 1, 2, 2],
        ),
        model("error(0.1) D0\n", &[0, 0]),
    ] {
        let result =
            StreamingWindowedDecoder::from_dem(&dem, WindowedConfig { step: 1, buffer: 1 }, |w| {
                PyMatchingEdgeDecoder::from_commit_window(w, false)
            });
        let error = result
            .err()
            .expect("unsolvable model must fail construction");
        assert!(error.to_string().contains("window"));
        assert!(error.to_string().contains("solvability"));
    }
}

#[test]
fn test_7_completeness_buffer_bound() {
    let dem = model(
        "error(0.1) D0 D1\nerror(0.01) D0\nerror(0.01) D1\n",
        &[0, 2],
    );
    let parsed = StructuredDem::from_dem_str(&dem).unwrap();
    let minimum = pecos_decoder_core::window::min_buffer_rounds(&parsed).unwrap();
    let result = StreamingWindowedDecoder::from_dem(
        &dem,
        WindowedConfig {
            step: 1,
            buffer: (minimum - 1) as usize,
        },
        |w| PyMatchingEdgeDecoder::from_commit_window(w, false),
    );
    assert!(
        result
            .err()
            .unwrap()
            .to_string()
            .contains("max_forward_span")
    );
    matching(&dem, 1, minimum as usize, false);
}

#[test]
fn test_12_merged_tail_commits_tail_owned_column() {
    let dem = model(
        "error(0.1) D0\nerror(0.1) D1\nerror(0.1) D2\nerror(0.1) D3 L0\n",
        &[0, 1, 2, 3],
    );
    let syndrome = [0, 0, 0, 1];
    let expected = mono(&dem, &syndrome, false);
    let mut decoder = matching(&dem, 3, 0, false);
    assert_eq!(
        decoder.num_windows(),
        1,
        "merged tail must be constructed once"
    );
    assert_eq!(
        decoder.decode_to_observables(&syndrome).unwrap(),
        expected,
        "tail-owned observable column must be committed"
    );
    assert!(decoder.residual().iter().all(|&s| s == 0));
}

#[test]
fn test_13_forced_projection_global_carry() {
    let dem = projection_chain();
    let parsed = StructuredDem::from_dem_str(&dem).unwrap();
    let columns: Vec<_> = parsed.errors.iter().flat_map(|e| &e.components).collect();
    let raw = [1, 0, 0, 0, 0];
    let expected = mono(&dem, &raw, false);
    let mut decoder = matching(&dem, 1, 1, false);
    decoder.start_shot(&raw).unwrap();
    for _ in 0..decoder.num_windows() {
        decoder.decode_next_window().unwrap();
        let mut recomputed = raw;
        for &id in &decoder.diagnostics().committed_columns {
            for &detector in &columns[id].detectors {
                recomputed[detector as usize] ^= 1;
            }
        }
        assert_eq!(
            decoder.residual(),
            recomputed,
            "global committed-column incidence must reproduce residual independently"
        );
    }
    assert_eq!(decoder.accumulated_obs(), expected);
    assert_eq!(decoder.residual(), [0; 5]);
    assert_eq!(decoder.diagnostics().forced_future_components, 1);
}

#[test]
fn test_13_far_detector_does_not_join_local_components() {
    let mut errors =
        "error(0.3) D0 D2\nerror(0.3) D2 D4\nerror(0.3) D1 D3\nerror(0.3) D3 D4\n".to_string();
    for d in 0..5 {
        writeln!(errors, "error(0.00005) D{d}").unwrap();
    }
    let dem = model(&errors, &[0, 0, 1, 1, 2]);
    let raw = [1, 1, 0, 0, 0];
    let expected = mono(&dem, &raw, false);
    let mut decoder = matching(&dem, 1, 1, false);
    decoder.start_shot(&raw).unwrap();
    decoder.decode_next_window().unwrap();
    assert_eq!(decoder.residual(), raw);
    assert_eq!(
        decoder.diagnostics().deferred_components,
        2,
        "projected far detector must not join the two components"
    );
    assert!(decoder.diagnostics().committed_columns.is_empty());
    decoder.finish().unwrap();
    assert_eq!(decoder.accumulated_obs(), expected);
}

#[test]
fn test_13_cross_window_correlation_limitation() {
    let dem = model(
        "error(0.01) D0 D1 ^ D2 D3 L0\nerror(0.1) D2\nerror(0.1) D3\nerror(0.000001) D0\nerror(0.000001) D1\nerror(0.1) D4\n",
        &[0, 0, 2, 2, 4],
    );
    let raw = [1, 1, 1, 1, 0];
    let expected = mono(&dem, &raw, true);
    let mut decoder = matching(&dem, 1, 2, true);
    let actual = decoder.decode_to_observables(&raw).unwrap();
    assert_ne!(
        actual, expected,
        "correlation evidence is lost across windows in fixture C3"
    );
    assert_eq!(actual, mono(&dem, &raw, false));
    assert_eq!(decoder.residual(), [0; 5]);
}

#[test]
fn replacement_constructor_width_and_reset_contracts() {
    for width in [64, 65] {
        let dem = model(&format!("error(0.1) D0 L{}\n", width - 1), &[0]);
        let parsed = StructuredDem::from_dem_str(&dem).unwrap();
        let config = WindowedConfig { step: 1, buffer: 0 };
        let factory = |w: &_| UfDecoder::from_commit_window(w, UfDecoderConfig::fast());
        let text = StreamingWindowedDecoder::from_dem(&dem, config, factory);
        let direct = StreamingWindowedDecoder::from_structured_dem(&parsed, config, factory);
        if width == 65 {
            assert!(text.is_err());
            assert!(direct.is_err());
            continue;
        }
        let mut text = text.unwrap();
        let mut direct = direct.unwrap();
        let expected = UfDecoder::from_dem(&dem, UfDecoderConfig::fast())
            .unwrap()
            .decode_to_observables(&[1])
            .unwrap();
        assert_eq!(text.decode_to_observables(&[1]).unwrap(), expected);
        assert_eq!(direct.decode_to_observables(&[1]).unwrap(), expected);
        assert_eq!(text.num_windows(), direct.num_windows());
        assert!(text.start_shot(&[0, 0]).is_err());
        text.reset();
        assert_eq!(text.accumulated_obs(), 0);
        text.feed_round(0, &[(0, 1)]).unwrap();
        assert_eq!(text.finish().unwrap(), expected);
        assert_eq!(text.finish().unwrap(), 0);
    }
}
