# BP Trellis manual mutant checklist

The repository has no mutation runner. Apply each compiling edit separately,
run the named killer test(s), and restore the source before trying the next
row. “Equivalent” rows document mutations that intentionally have no killer.

| Mutant | Exact edit (quoted old → new) | Expected killer test(s) or disposition |
|---|---|---|
| `bptrellis_default_bp_off` | In `BpTrellisConfig::default`, replace `"bp_score_iterations: 5"` with `"bp_score_iterations: 0"`. | `bptrellis_defaults_enable_bp_merge_and_deadline_order` (including its `bp_seconds > 0.0` assertions) |
| `bptrellis_default_merge_off` | In `BpTrellisConfig::default`, replace `"merge_indistinguishable: true"` with `"merge_indistinguishable: false"`. | `bptrellis_defaults_enable_bp_merge_and_deadline_order` (including its `processed_columns` assertion for the planted duplicate pair) |
| `ladder_skips_accumulation` | In `decode_with_attempt`, delete `result.transitions += report.transitions;`. | `no_path_escalates_to_k16_and_accumulates_transitions`; executed below as `skip_success_transitions` |
| `terminal_evidence_fold_reversed` | In `exp/pecos-trellis/src/lib.rs`, in the binary float arm only, replace `terminal.iter().fold(f64::NEG_INFINITY, \|total, candidate\| {` with `terminal.iter().rev().fold(f64::NEG_INFINITY, \|total, candidate\| {`. | `bitwise_snapshot::decode_outputs_match_bitwise_snapshot`; verified failure in `binary/float/ordering_duplicates/unpruned/order=deadline/bp=0/merge=false/ladder=empty/syndrome=0x2`, field `log_evidence` (one bit). Applied and restored in an isolated workspace copy to preserve the source oracle. |
| `bp_residual_ignores_forced` | In the engine's `refresh_bp_suffix_values`, replace `(observed[word_index] ^ self.forced_syndrome[word_index]) & bit_mask` with `(observed[word_index] & bit_mask)`. | `bitwise_snapshot::decode_outputs_match_bitwise_snapshot`; verified `binary/float/forced_duplicate/default/order=deadline/bp=5/merge=true/ladder=empty/syndrome=0x41`, field `dropped_log_mass`. |

## Prepared-shot and outcome guards

Verified with compiling mutations in an isolated workspace copy, restoring each
source file before the next row. The final rerun uses a separate Cargo target
directory and loads Python mutants from a temporary module directory; it does
not install them into the working tree's environment. Each Rust killer exits
101 and each Python killer exits 1. Frozen fixtures were never regenerated.
The first failure column quotes the test output, including panic messages for
readiness guards. A test that expects a panic instead fails with “did not panic”.

Rust engine killers run with `cargo test --locked -p pecos-trellis --test prepared NAME`.
Facade killers run with `cargo test --locked -p pecos-bp-trellis --lib NAME`,
except the integration killers `no_path_escalates_to_k16_and_accumulates_transitions`,
`bptrellis_defaults_enable_bp_merge_and_deadline_order`,
`successful_base_decode_is_bit_identical_with_a_configured_ladder`, and
`exhausted_ladder_reports_the_attempted_rung_count`, which use
`--test bp_trellis`. Python killers run with `python -m pytest -n 0 FILE::NAME`
against the rebuilt mutant extension.

The old per-rung `rungs_skip_bp` mutation no longer has a construction site;
BP sharing is guarded by `prepare_again_each_rung` and the refresh counter.
Attempt counts use the private callable in `decode_with_attempt`, not decoder
counters or inferred telemetry.

| Mutant | Exact compiling edit | Killer | First failing line verbatim |
|---|---|---|---|
| `residual_attempt_anyway` | In `exp/pecos-bp-trellis/src/lib.rs`, replace `                report.cause = NoPathCause::Residual { detector };` with `                let params = self.inner.prune_params();\n                let _ = attempt(&mut self.inner, params);\n                report.cause = NoPathCause::Residual { detector };`. | `residual_skips_all_attempts_and_preserves_forced_mask` | `unexpected attempt call` |
| `zero_drops_are_exhausted` | In `exp/pecos-bp-trellis/src/lib.rs`, replace `if dropped_states == 0 {` with `if false && dropped_states == 0 {`. | `infeasible_base_skips_ladder` | `unexpected attempt call` |
| `classify_only_at_base` | In `exp/pecos-bp-trellis/src/lib.rs`, replace `if dropped_states == 0 {` with `if index == 0 && dropped_states == 0 {`. | `infeasible_rung_skips_remaining_rungs` | `unexpected attempt call` |
| `non_dominating_rungs_rejected` | In `exp/pecos-bp-trellis/src/lib.rs`, replace `            config.k = params.k;` with `            if params.k < self.k \|\| params.delta < self.delta { return Err(DecoderError::InvalidConfiguration("rung must dominate".into())); }\n            config.k = params.k;`. | `narrower_rung_recovers` | ``called `Result::unwrap()` on an `Err` value: InvalidConfiguration("rung must dominate")`` |
| `ladder_reuses_base_delta` | In `exp/pecos-bp-trellis/src/lib.rs within fn decode_with_attempt(`, insert `let base = self.inner.prune_params();` before `let params =`; replace `delta: rung.delta,` with `delta: base.delta,`. | `rung_delta_is_applied` | `rung must recover` |
| `ladder_reuses_base_k` | In `exp/pecos-bp-trellis/src/lib.rs within fn decode_with_attempt(`, insert `let base = self.inner.prune_params();` before `let params =`; replace `k: rung.k,` with `k: base.k,`. | `no_path_escalates_to_k16_and_accumulates_transitions` | ``called `Result::unwrap()` on an `Err` value: DecodingFailed("syndrome is unexplainable at the given pruning parameters after 1 escalation rung")`` |
| `prepare_again_each_rung` | In `exp/pecos-bp-trellis/src/lib.rs`, replace `            let outcome = attempt(&mut self.inner, params);` with `            if index > 0 { self.inner.prepare(syndrome)?; }\n            let outcome = attempt(&mut self.inner, params);`. | `bp_refresh_is_reused_across_the_ladder` | ``assertion `left == right` failed`` |
| `sum_bp_seconds_per_rung` | In `exp/pecos-bp-trellis/src/lib.rs`, replace `result.bp_seconds = report.bp_seconds;` with `result.bp_seconds += report.bp_seconds;`. | `bp_refresh_is_reused_across_the_ladder` | ``assertion `left == right` failed`` |
| `skip_no_path_transitions` | In `exp/pecos-bp-trellis/src/lib.rs`, replace `report.transitions += transitions;` with `report.transitions = transitions;`. | `exhausted_sums_every_attempt_exactly` | ``assertion `left == right` failed`` |
| `skip_success_transitions` | In `exp/pecos-bp-trellis/src/lib.rs`, replace `result.transitions += report.transitions;` with `result.transitions += 0;`. | `no_path_escalates_to_k16_and_accumulates_transitions` | `assertion failed: result.transitions > bare_result.transitions` |
| `accept_exact_base_ladder` | In `exp/pecos-bp-trellis/src/lib.rs`, replace `if !self.escalation.is_empty() && self.k == usize::MAX && self.delta.is_infinite()` with `if false && !self.escalation.is_empty() && self.k == usize::MAX && self.delta.is_infinite()`. | `rung_validation_has_no_dominance_rule` | ``called `Result::unwrap_err()` on an `Ok` value: ()`` |
| `skip_rung_validation` | In `exp/pecos-bp-trellis/src/lib.rs`, replace `config.k = params.k;` with `config.k = params.k.max(1);`. | `rung_validation_has_no_dominance_rule` | ``called `Result::unwrap_err()` on an `Ok` value: ()`` |
| `engine_error_becomes_no_path` | In `exp/pecos-bp-trellis/src/lib.rs`, replace `TrellisDecodeAttempt::Error(error) => return Err(error),` with `TrellisDecodeAttempt::Error(_) => return Ok(BpTrellisOutcome::NoPath(report)),`. | `attempt_errors_stay_errors_with_ladder` | `assertion failed: matches!(outcome, Err(DecoderError::InvalidConfiguration(_)))` |
| `prepare_error_becomes_no_path` | In `exp/pecos-bp-trellis/src/lib.rs`, replace `let prepared = self.inner.prepare(syndrome)?;` with `let prepared = self.inner.prepare(syndrome).unwrap_or(TrellisPrepared::Residual { detector: 0 });`. | `errors_stay_errors_with_ladder` | `assertion failed: matches!(decoder.decode_outcome(&[]),` |
| `placeholder_zero` | In `exp/pecos-bp-trellis/src/lib.rs`, replace `placeholder: self.inner.forced_observables(),` with `placeholder: ObsMask::from_words(&[]),`. | `residual_skips_all_attempts_and_preserves_forced_mask` | ``assertion `left == right` failed`` |
| `batch_outcomes_reversed` | In `exp/pecos-bp-trellis/src/lib.rs within pub fn decode_batch_outcomes(`, replace `            Self::decode_outcome,\n        )` with `            Self::decode_outcome,\n        ).map(\|mut outcomes\| { outcomes.reverse(); outcomes })`. | `mixed_batch_outcomes_and_strict_results_preserve_order` | `assertion failed: matches!(sequential[0], Ok(BpTrellisOutcome::Decoded(_)))` |
| `no_path_bp_runs_zero` | In `exp/pecos-bp-trellis/src/lib.rs`, replace `report.bp_runs = u32::try_from` with `report.bp_runs = 0 * u32::try_from`. | `no_path_bp_telemetry_counts_prepare_once` | ``assertion `left == right` failed`` |
| `infeasible_message_wrong` | In `exp/pecos-bp-trellis/src/lib.rs`, replace `syndrome is unexplainable under the detector error model` with `syndrome is unexplainable`. | `infeasible_base_skips_ladder` | ``assertion `left == right` failed`` |
| `residual_message_wrong` | In `exp/pecos-bp-trellis/src/lib.rs`, replace `has a residual no mechanism can change` with `is untouched`. | `residual_skips_all_attempts_and_preserves_forced_mask` | ``assertion `left == right` failed`` |
| `exhausted_message_wrong` | In `exp/pecos-bp-trellis/src/lib.rs`, replace `after {} escalation rung{}` with `after {} attempt{}`. | `exhausted_sums_every_attempt_exactly` | ``assertion `left == right` failed`` |
| `exhausted_message_always_plural` | In `exp/pecos-bp-trellis/src/lib.rs`, replace `if self.rungs_tried == 1 { "" } else { "s" }` with `"s"`. | `exhausted_ladder_reports_the_attempted_rung_count` (`--test bp_trellis`); `python/quantum-pecos/tests/qec/test_bp_trellis_decoder.py::test_no_path_reports_across_direct_methods` | ``assertion `left == right` failed``; `E   AssertionError: Regex pattern did not match.` |
| `bptrellis_deadline_ignored` | In `exp/pecos-bp-trellis/src/lib.rs`, replace `trellis_config.column_order = config.ordering.resolve(dem)?;` with `trellis_config.column_order = None;`. | `bptrellis_defaults_enable_bp_merge_and_deadline_order` | ``assertion `left == right` failed`` |
| `escalate_on_any_failure` | In `exp/pecos-bp-trellis/src/lib.rs`, replace `return Ok(BpTrellisOutcome::Decoded(result));` with `if matches!(result.status, pecos_trellis::TrellisStatus::Exact) { return Ok(BpTrellisOutcome::Decoded(result)); }`. | `successful_base_decode_is_bit_identical_with_a_configured_ladder` | ``called `Result::unwrap()` on an `Err` value: DecodingFailed("syndrome is unexplainable at the given pruning parameters after 0 escalation rungs")`` |
| `python_ladder_alias_wrong_delta` | In `python/pecos-rslib-exp/src/bp_trellis_bindings.rs within pub(crate) fn resolve_escalation(`, replace `EscalationRung { k, delta }` with `EscalationRung { k, delta: 50.0 }`. | `python/pecos-rslib-exp/tests/test_bp_trellis_batch_decode.py::test_resolved_rungs_and_strict_spec_route` | `E   assert bp_trellis(escalation=[(16, 50.0)]) == bp_trellis(escalation_ks=[16])` |
| `python_accept_both_ladders` | In `python/pecos-rslib-exp/src/bp_trellis_bindings.rs`, replace `Err(PyValueError::new_err(\n            "escalation_ks and escalation cannot both be supplied",\n        ))` with `Ok(Vec::new())`. | `python/pecos-rslib-exp/tests/test_bp_trellis_batch_decode.py::test_resolved_rungs_and_strict_spec_route` | `E   Failed: DID NOT RAISE <class 'ValueError'>` |
| `python_repr_infinite_rung_invalid` | In `python/pecos-rslib-exp/src/decoder_specs.rs within if !config.escalation.is_empty()`, replace `"float('inf')".into()` with `"inf".into()`. | `python/pecos-rslib-exp/tests/test_bp_trellis_batch_decode.py::test_public_spec_and_configuration` | `E   NameError: name 'inf' is not defined` |
| `python_invalid_policy_accepted` | In `python/pecos-rslib-exp/src/bp_trellis_bindings.rs`, replace `Err(PyValueError::new_err(\n            "on_no_path must be 'raise' or 'report'",\n        ))` with `Ok(true)`. | `python/quantum-pecos/tests/qec/test_bp_trellis_decoder.py::test_no_path_reports_across_direct_methods` | `E   Failed: DID NOT RAISE <class 'ValueError'>` |
| `python_wrong_residual` | In `python/pecos-rslib-exp/src/bp_trellis_bindings.rs within impl PyBpTrellisNoPath`, replace `=> "residual",` with `=> "unknown",`. | `python/quantum-pecos/tests/qec/test_bp_trellis_decoder.py::test_no_path_reports_across_direct_methods` | `E   AssertionError: assert 'unknown' == 'residual'` |
| `python_wrong_infeasible` | In `python/pecos-rslib-exp/src/bp_trellis_bindings.rs within impl PyBpTrellisNoPath`, replace `=> "infeasible",` with `=> "unknown",`. | `python/quantum-pecos/tests/qec/test_bp_trellis_decoder.py::test_no_path_reports_across_direct_methods` | `E   AssertionError: assert 'unknown' == 'infeasible'` |
| `python_wrong_exhausted` | In `python/pecos-rslib-exp/src/bp_trellis_bindings.rs within impl PyBpTrellisNoPath`, replace `=> "exhausted",` with `=> "unknown",`. | `python/quantum-pecos/tests/qec/test_bp_trellis_decoder.py::test_no_path_reports_across_direct_methods` | `E   AssertionError: assert 'unknown' == 'exhausted'` |
| `python_placeholder_zero` | In `python/pecos-rslib-exp/src/bp_trellis_bindings.rs`, replace `mask: self.inner.placeholder.clone(),` with `mask: ObsMask::from_words(&[]),`. | `python/quantum-pecos/tests/qec/test_bp_trellis_decoder.py::test_no_path_reports_across_direct_methods` | `E   assert 0 == (1 << 0)` |
| `python_no_path_aliases_observable_flips` | In `python/pecos-rslib-exp/src/bp_trellis_bindings.rs` within `impl PyBpTrellisNoPath`, insert `    #[getter]\n    fn observable_flips(&self) -> PyBpTrellisObservableFlips {\n        self.placeholder_flips()\n    }\n` after the `placeholder_flips` getter. | `python/quantum-pecos/tests/qec/test_bp_trellis_decoder.py::test_no_path_reports_across_direct_methods` | `E   AssertionError: assert not True` |
| `python_no_path_flag_false` | In `python/pecos-rslib-exp/src/bp_trellis_bindings.rs within impl PyBpTrellisNoPath`, replace `        true\n` with `        false\n`. | `python/quantum-pecos/tests/qec/test_bp_trellis_decoder.py::test_no_path_reports_across_direct_methods` | `E   AssertionError: assert False == (1 != 0)` |
| `python_result_flag_true` | In `python/pecos-rslib-exp/src/bp_trellis_bindings.rs within impl PyBpTrellisResult`, replace `        false\n` with `        true\n`. | `python/quantum-pecos/tests/qec/test_bp_trellis_decoder.py::test_no_path_reports_across_direct_methods` | `E   assert True == (0 != 0)` |
| `python_bp_runs_zero` | In `python/pecos-rslib-exp/src/bp_trellis_bindings.rs within impl PyBpTrellisResult`, replace `        self.inner.bp_runs\n` with `        0\n`. | `python/quantum-pecos/tests/qec/test_bp_trellis_decoder.py::test_bptrellis_defaults_and_decode_shapes` | `E   assert 0 == 1` |
| `python_batch_reports_reversed` | In `python/pecos-rslib-exp/src/bp_trellis_bindings.rs within fn decode_batch(`, replace `        results\n            .into_iter()` with `        results\n            .into_iter().rev()`. | `python/quantum-pecos/tests/qec/test_bp_trellis_decoder.py::test_no_path_reports_across_direct_methods` | `E   AssertionError: assert True == (0 != 0)` |
| `python_report_calls_strict_decode` | In `python/pecos-rslib-exp/src/bp_trellis_bindings.rs`, replace `self.inner.decode_outcome(&syndrome)` with `self.inner.decode(&syndrome).map(BpTrellisOutcome::Decoded)`. | `python/quantum-pecos/tests/qec/test_bp_trellis_decoder.py::test_no_path_reports_across_direct_methods` | `E   RuntimeError: Decoding failed: syndrome is unexplainable: detector 5 has a residual no mechanism can change` |
| `python_batch_report_calls_strict_decode` | In `python/pecos-rslib-exp/src/bp_trellis_bindings.rs`, replace `self.inner.decode_batch_outcomes(&shots, workers)` with `self.inner.decode_batch(&shots, workers).map(\|results\| results.into_iter().map(\|result\| result.map(BpTrellisOutcome::Decoded)).collect::<Vec<_>>())`. | `python/quantum-pecos/tests/qec/test_bp_trellis_decoder.py::test_no_path_reports_across_direct_methods` | `E   RuntimeError: shot 1: Decoding failed: syndrome is unexplainable: detector 5 has a residual no mechanism can change` |

## Review-round guards

Each edit below was compiled and run in an isolated workspace copy with a
separate Cargo target directory, then restored. Rust killers exited 101;
Python killers exited 1. The three Python telemetry getter edits were tested
independently, so zeroing one getter cannot be hidden by another failing getter.

| Mutant | Exact compiling edit | Killer | First failing line verbatim |
|---|---|---|---|
| `residual_runs_bp` | In `exp/pecos-trellis/src/lib.rs`, replace `            if residual != 0 {` with `            if residual != 0 {\n                self.model.refresh_bp_suffix_values(&mut self.scratch, &observed)?;`. Restrict the edit to `TrellisDecoder::prepare`. | `residual_report_has_no_bp_work_when_bp_enabled` | ``assertion `left == right` failed`` |
| `python_no_path_getters_zero` | In `impl PyBpTrellisNoPath` in `python/pecos-rslib-exp/src/bp_trellis_bindings.rs`, independently replace `self.inner.transitions` with `0`, `self.inner.bp_runs` with `0`, and `self.inner.bp_seconds` with `0.0`, restoring between variants. | `python/quantum-pecos/tests/qec/test_bp_trellis_decoder.py::test_no_path_exhausted_absolute_telemetry` | `transitions`: `E   AssertionError: assert 0 == 22`; `bp_runs`: `E   AssertionError: assert 0 == 1`; `bp_seconds`: `E   AssertionError: assert 0.0 > 0.0` |
| `python_no_path_repr_omits_detector` | In `python/pecos-rslib-exp/src/bp_trellis_bindings.rs`, replace `\|detector\| format!(", detector={detector}")` with `\|_\| String::new()`. | `python/quantum-pecos/tests/qec/test_bp_trellis_decoder.py::test_no_path_reports_across_direct_methods` | `E   assert 'detector=5' in "BpTrellisNoPath(cause='residual', rungs_tried=0, transitions=0)"` |
