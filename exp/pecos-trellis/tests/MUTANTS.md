# Trellis engine manual mutant checklist

The repository has no mutation runner. Apply each compiling edit separately,
run the named killer test(s), and restore the source before trying the next
row. "Equivalent" rows document mutations that intentionally have no killer.

Every edit below targets `exp/pecos-trellis/src/lib.rs`, and every killer named
below lives in this crate. The local `bitwise_snapshot` integration test freezes
every semantic result field except wall-clock timing, which is recorded only
as a ran/did-not-run bit, independently of the parity-port crate. `score_fold_restore`'s killers were re-derived by executing the mutant
against this crate's suite rather than assumed.

| Mutant | Exact edit (quoted old → new) | Expected killer test(s) or disposition |
|---|---|---|
| `merge_to_max` | `".and_modify(|mass| *mass = logaddexp(*mass, log_mass))"` → `".and_modify(|mass| *mass = (*mass).max(log_mass))"` | `degeneracy_mass_beats_the_single_most_likely_error` |
| `remove_close_check` | Delete the complete `"if state.active_syndrome.iter().zip(observed).zip(&column.close_mask).any(|((&accumulated, &expected), &closing)| (accumulated ^ expected) & closing != 0) { return; }"` block in `merge_branch`. | `fails_loud_for_untouched_and_unachievable_syndromes`; `unpruned_matches_independent_brute_force_on_seeded_random_dems` |
| `flip_wide_label_tie_break_to_little_endian` | `"compare_words_as_unsigned(&self.logical, &other.logical)"` → `"self.logical.cmp(&other.logical)"` | `wide_logical_ties_use_numeric_label_order` |
| `k_off_by_one` | `"let within_k = index < k;"` → `"let within_k = index <= k;"` | `width_pruning_accounts_for_the_discarded_state_and_mass`; `width_and_delta_pruning_can_change_the_logical_answer` |
| `remove_forced_seed` | `"let mut initial_syndrome = self.forced_syndrome.clone();"` → `"let mut initial_syndrome = vec![0; self.detector_words];"` | `forced_syndrome_shifts_shared_probabilistic_detector`; `unpruned_matches_independent_brute_force_on_seeded_random_dems` |
| `swap_suffix_rho_zero_and_one` | In `suffix_compatibility_score`, replace `"row.log_probability_zero"` with `"row.log_probability_one"` and the existing `"row.log_probability_one"` with `"row.log_probability_zero"` in the opposite branch. | `suffix_compatibility_changes_the_greedy_survivor` |
| `eta_off_by_one_apply_moment_before_snapshot` | Move the complete `"for detector in set_bits(&column.detector_toggle) { row_moments[detector] *= moment; }"` loop from after the assignment beginning `"*table = set_bits(&column.active_mask)"` and ending `".collect();"` to immediately before that assignment. | `suffix_compatibility_changes_the_greedy_survivor` |
| `remove_active_mask_projection` | Delete `"and_assign(&mut state.active_syndrome, &column.active_mask);"` from `merge_branch`. | **EQUIVALENT.** A branch survives the close check only when every closing detector equals the observed bit. Keeping those now-inactive bits changes the noncanonical key representation but cannot prevent equivalent surviving prefixes from merging: all survivors carry the same observed values in those positions. |
| `remove_transitions_increment` | Delete `"*transitions += 1;"` from `merge_branch`. | `transitions_count_every_candidate_branch_evaluation` |
| `skip_dropped_mass_accumulation` | Delete `"dropped_log_mass = logaddexp(dropped_log_mass, scored.candidate.log_mass);"` from the discarded-candidate branch in `prune`. | `width_pruning_accounts_for_the_discarded_state_and_mass` |
| `report_exact_after_pruning` | Replace the complete `"let status = if dropped_states == 0 { TrellisStatus::Exact } else { TrellisStatus::Pruned { k_capped, delta_pruned } };"` expression with `"let status = TrellisStatus::Exact;"`. | `width_pruning_accounts_for_the_discarded_state_and_mass`; `delta_pruning_reports_its_status_flag`; `one_prune_call_can_trigger_both_pruning_flags` |
| `merge_wrong_formula` | In `xor_combined_probability`, replace `"first * (1.0 - second) + second * (1.0 - first)"` with `"first * second"`. | `xor_probability_arithmetic_is_pinned`; `indistinguishable_merging_matches_unmerged_and_original_brute_force` |
| `merge_keeps_last_occurrence` | Change `merge_indistinguishable_columns` so a repeated symptom tuple replaces the first retained tuple at the duplicate's later sequence position instead of updating the first tuple in place. | `merging_keeps_first_ordered_occurrence_and_deletes_later_copy` |
| `merge_ignores_observables` | Change the `first_positions` key from the detector and observable word-mask pair to the detector word mask alone. | `merge_requires_matching_observable_sets` |
| `bp_tables_from_dem_priors` | In `bp_suffix_compatibility`, replace the call that builds tables from the BP-derived `moments` with clones of each column's static `suffix_compatibility` table. | `bp_scores_change_the_greedy_survivor_without_changing_its_mass` |
| `bp_clamp_removed` | In `bp_score_probability`, replace the complete `probability.clamp(BP_SCORE_PROBABILITY_MIN, 1.0 - BP_SCORE_PROBABILITY_MIN)` expression with `probability`. | `bp_score_probability_clamps_saturated_llrs` |
| `bp_runs_in_unpruned_fast_path` | In `TrellisDecoder::from_sparse_dem`, delete `&& !(config.k == usize::MAX && config.delta.is_infinite())` from the `bp_score` construction condition. | `bp_flag_is_bitwise_inert_on_the_unpruned_fast_path` |
| `maxlog_quantize_round_ties_even` | In `quantize_metric`, replace `"scaled.round()"` with `"scaled.round_ties_even()"`. | `integer_metric_quantization_saturates_at_its_boundaries` |
| `maxlog_dropped_mass_keeps_last_candidate` | In `prune_maxlog`, replace `"dropped_log_mass.max(scored.candidate.log_mass)"` with `"scored.candidate.log_mass"`. | `maxlog_dropped_mass_is_the_largest_discarded_route` |
| `maxlog_dropped_mass_keeps_last_column` | In binary max-log decode, replace `"dropped_log_mass.max(pruned.dropped_log_mass)"` with `"pruned.dropped_log_mass"`. | `maxlog_dropped_mass_is_the_largest_discarded_route` |
| `maxlog_delta_excludes_cutoff_tie` | In `prune_maxlog`, replace `"scored.score >= cutoff"` with `"scored.score > cutoff"`. | `maxlog_delta_retains_a_candidate_exactly_at_the_cutoff` |
| `bp_maxlog_builds_unquantized_suffix_tables` | In `bp_suffix_compatibility`, pass `None` instead of the selected integer scale. | `bp_scored_maxlog_changes_the_greedy_binary_survivor` |
| `bp_maxlog_uses_static_suffix_tables` | In binary max-log decode, ignore `bp_suffix_compatibility` and use `column.suffix_compatibility`. | `bp_scored_maxlog_changes_the_greedy_binary_survivor` |
| `maxlog_zero_alpha_scores_suffix` | Remove the `alpha_int == 0` short-circuit in `prune_maxlog`. | `maxlog_zero_alpha_skips_negative_infinite_suffix_scores` |
| `allow_maxlog_indistinguishable_merge` | Delete the `MetricMode::MaxLogInt` plus `merge_indistinguishable` rejection from `validate_config`. | `validates_probabilities_indices_order_and_pruning_configuration` |
| `maxlog_score_tie_ignores_log_mass` | Delete the log-mass comparator from `prune_maxlog`'s candidate ordering. | `maxlog_score_ties_prefer_the_higher_mass_state` |
| `terminal_evidence_fold_reversed` | In `exp/pecos-trellis/src/lib.rs`, in the binary float arm only, replace `terminal.iter().fold(f64::NEG_INFINITY, \|total, candidate\| {` with `terminal.iter().rev().fold(f64::NEG_INFINITY, \|total, candidate\| {`. | `bitwise_snapshot::decode_outputs_match_bitwise_snapshot`; verified failure in `binary/random_seed11/float/unpruned/bp=0/merge=false/order=input/syndrome=0x2`, field `log_evidence` (one bit). Applied and restored in an isolated workspace copy to preserve the source oracle. |
| `binary_branch_order_swapped` | In `process_binary_range`, swap the two `branch_context.emit(...)` calls inside the `for (index, &log_mass) in frontier.parent.masses.iter().enumerate()` loop, so the taken branch (`Some((&column.detector_toggle, &column.logical_toggle))`, `branch_base + column.log_odds`) is emitted before the not-taken branch (`None`, `branch_base`). | EQUIVALENT (verified 2026-09-24 on the post-#829 engine: `bitwise_snapshot` passes in both crates with the swap applied). Each merged state receives at most one taken and one not-taken route, and `frontier.merge(logaddexp)` combines two routes to the same bits in either order. |
| `bp_residual_ignores_forced` | In `bp_suffix_compatibility`, replace `(observed[word_index] ^ self.forced_syndrome[word_index]) & bit_mask` with `(observed[word_index] & bit_mask)`. | `bitwise_snapshot::decode_outputs_match_bitwise_snapshot`; verified `binary/forced_duplicate/float/k+delta/bp=5/merge=false/order=input/syndrome=0x43`, field `transitions`. |
| `nary_outcomes_reversed` | In the N-ary float DP, replace `for outcome in &column.outcomes {` with `for outcome in column.outcomes.iter().rev() {`. | `bitwise_snapshot::decode_outputs_match_bitwise_snapshot`; verified `nary/three_route_collision/float/unpruned/bp=0/merge=false/order=rotate/syndrome=0x0`, field `log_evidence`. |

## Prepared-shot and outcome guards

Verified with compiling mutations in an isolated workspace copy with a
separate Cargo target directory, restoring the source before the next row.
Each killer exits 101. Frozen fixtures were never regenerated. The first
failure column quotes the test output, including panic messages for readiness
guards; a test that expects a panic instead fails with "did not panic".

Killers run with `cargo test --locked -p pecos-trellis --test prepared NAME`.
Facade and Python guards for the same change live in
`exp/pecos-bp-trellis/tests/MUTANTS.md`.

| Mutant | Exact compiling edit | Killer | First failing line verbatim |
|---|---|---|---|
| `binary_float_ignores_k` | In `exp/pecos-trellis/src/lib.rs within fn process_binary_range(`, replace `params.k,` with `self.config.k,`. | `binary_float_k_override` | ``assertion `left != right` failed: override must change outcome or telemetry`` |
| `binary_float_ignores_delta` | In `exp/pecos-trellis/src/lib.rs within fn process_binary_range(`, replace `params.delta,` with `self.config.delta,`. | `binary_float_delta_override` | ``assertion `left != right` failed: override must change outcome or telemetry`` |
| `binary_float_zero_no_path_drops` | In `exp/pecos-trellis/src/lib.rs within fn decode_attempt_binary(`, replace `dropped_states: progress.dropped_states,` with `dropped_states: 0,`. | `no_path_drops_in_all_four_arms` | `positive drops, nary=false, integer=false` |
| `nary_float_ignores_k` | In `exp/pecos-trellis/src/lib.rs within fn decode_attempt_nary(`, replace `params.k,` with `self.config.k,`. | `nary_float_k_override` | ``assertion `left != right` failed: override must change outcome or telemetry`` |
| `nary_float_ignores_delta` | In `exp/pecos-trellis/src/lib.rs within fn decode_attempt_nary(`, replace `params.delta,` with `self.config.delta,`. | `nary_float_delta_override` | ``assertion `left != right` failed: override must change outcome or telemetry`` |
| `nary_float_zero_no_path_drops` | In `exp/pecos-trellis/src/lib.rs within fn decode_attempt_nary(`, replace `                    dropped_states,\n                    bp_seconds` with `                    dropped_states: 0,\n                    bp_seconds`. | `no_path_drops_in_all_four_arms` | `positive drops, nary=true, integer=false` |
| `binary_maxlog_ignores_k` | In `exp/pecos-trellis/src/lib.rs within fn decode_attempt_binary_maxlog(`, replace `params.k,` with `self.config.k,`. | `binary_maxlog_k_override` | ``assertion `left != right` failed: override must change outcome or telemetry`` |
| `binary_maxlog_ignores_delta` | In `exp/pecos-trellis/src/lib.rs within fn decode_attempt_binary_maxlog(`, replace `params.delta,` with `self.config.delta,`. | `binary_maxlog_delta_override` | ``assertion `left != right` failed: override must change outcome or telemetry`` |
| `binary_maxlog_zero_no_path_drops` | In `exp/pecos-trellis/src/lib.rs within fn decode_attempt_binary_maxlog(`, replace `                    dropped_states,\n                    bp_seconds` with `                    dropped_states: 0,\n                    bp_seconds`. | `no_path_drops_in_all_four_arms` | `positive drops, nary=false, integer=true` |
| `nary_maxlog_ignores_k` | In `exp/pecos-trellis/src/lib.rs within fn decode_attempt_nary_maxlog(`, replace `params.k,` with `self.config.k,`. | `nary_maxlog_k_override` | ``assertion `left != right` failed: override must change outcome or telemetry`` |
| `nary_maxlog_ignores_delta` | In `exp/pecos-trellis/src/lib.rs within fn decode_attempt_nary_maxlog(`, replace `params.delta,` with `self.config.delta,`. | `nary_maxlog_delta_override` | ``assertion `left != right` failed: override must change outcome or telemetry`` |
| `nary_maxlog_zero_no_path_drops` | In `exp/pecos-trellis/src/lib.rs within fn decode_attempt_nary_maxlog(`, replace `                    dropped_states,\n                    bp_seconds` with `                    dropped_states: 0,\n                    bp_seconds`. | `no_path_drops_in_all_four_arms` | `positive drops, nary=true, integer=true` |
| `attempt_skips_validation` | In `exp/pecos-trellis/src/lib.rs within pub fn attempt(`, replace `if let Err(error) = params.validate(self.model.config.metric_mode)` with `if let Err(error) = Ok::<(), DecoderError>(())`. | `parameter_rejection_matrix_precedes_readiness` | `attempt requires a Ready prepare` |
| `accept_zero_k` | In `exp/pecos-trellis/src/lib.rs within impl PruneParams {`, replace `if self.k == 0` with `if false`. | `parameter_rejection_matrix_precedes_readiness` | `attempt requires a Ready prepare` |
| `accept_nan_delta` | In `exp/pecos-trellis/src/lib.rs within impl PruneParams {`, replace `self.delta.is_nan() \|\| self.delta < 0.0` with `self.delta < 0.0`. | `parameter_rejection_matrix_precedes_readiness` | `attempt requires a Ready prepare` |
| `accept_negative_delta` | In `exp/pecos-trellis/src/lib.rs within impl PruneParams {`, replace `self.delta.is_nan() \|\| self.delta < 0.0` with `self.delta.is_nan()`. | `parameter_rejection_matrix_precedes_readiness` | `attempt requires a Ready prepare` |
| `accept_maxlog_infinity` | In `exp/pecos-trellis/src/lib.rs within impl PruneParams {`, replace `metric_mode == MetricMode::MaxLogInt && !self.delta.is_finite()` with `false && metric_mode == MetricMode::MaxLogInt && !self.delta.is_finite()`. | `parameter_rejection_matrix_precedes_readiness` | `attempt requires a Ready prepare` |
| `skip_bp_capability_rule` | In `exp/pecos-trellis/src/lib.rs within pub fn attempt(`, replace `if self.model.config.bp_score_iterations > 0` with `if false && self.model.config.bp_score_iterations > 0`. | `bp_capabilities_and_refresh_count` | `attempt requires a Ready prepare` |
| `attempt_accepts_unprepared` | In `exp/pecos-trellis/src/lib.rs within pub fn attempt(`, replace `.expect("attempt requires a Ready prepare")` with `.unwrap_or((false, 0.0))`. | `unprepared_attempt_panics` | `prepared shot` |
| `keep_ready_after_error` | In `exp/pecos-trellis/src/lib.rs within pub fn prepare(`, replace `self.scratch.prepared = None;` with `// readiness incorrectly retained`. | `dimension_error_invalidates_ready` | `note: test did not panic as expected at exp/pecos-trellis/tests/prepared.rs:341:4` |
| `keep_ready_after_residual` | In `exp/pecos-trellis/src/lib.rs within pub fn prepare(`, replace `self.scratch.prepared = None;` with `// readiness incorrectly retained`. | `residual_invalidates_ready` | `note: test did not panic as expected at exp/pecos-trellis/tests/prepared.rs:350:4` |
| `fresh_worker_copies_ready` | In `exp/pecos-trellis/src/lib.rs within pub fn fresh_worker(`, replace `scratch: TrellisScratch::new(&self.model),` with `scratch: self.scratch.clone(),`. | `fresh_worker_starts_unprepared` | `note: test did not panic as expected at exp/pecos-trellis/tests/prepared.rs:333:4` |
| `attempt_consumes_ready` | In `exp/pecos-trellis/src/lib.rs within pub fn attempt(`, replace `        attempt\n    }` with `        self.scratch.prepared = None;\n        attempt\n    }`. | `repeated_attempts_and_next_shot_match_fresh_decoders` | `attempt requires a Ready prepare` |
| `attempt_keeps_frontier` | In `exp/pecos-trellis/src/lib.rs within fn decode_attempt_binary(`, replace `        progress.reset(self);` with `        if progress.frontier.parent.masses.is_empty() { progress.reset(self); }`. | `repeated_attempts_and_next_shot_match_fresh_decoders` | ``assertion `left == right` failed`` |
| `forced_observables_zero` | In `exp/pecos-trellis/src/lib.rs within pub fn forced_observables(`, replace `ObsMask::from_words(&self.model.forced_logical)` with `ObsMask::from_words(&[])`. | `forced_observables_and_lowest_residual` | ``assertion `left == right` failed`` |
| `lowest_residual_wrong` | In `exp/pecos-trellis/src/lib.rs within pub fn prepare(`, replace `residual.trailing_zeros() as usize` with `0`. | `forced_observables_and_lowest_residual` | ``assertion `left == right` failed`` |
| `bp_refresh_counter_missing` | In `exp/pecos-trellis/src/lib.rs within fn refresh_bp_suffix_values(`, replace `scratch.bp_refreshes += 1;` with `// no count`. | `bp_capabilities_and_refresh_count` | ``assertion `left == right` failed`` |
| `bp_runs_zero` | In `exp/pecos-trellis/src/lib.rs within pub fn attempt(`, replace `result.bp_runs = u32::from(bp_ran);` with `result.bp_runs = u32::from(bp_ran) * 0;`. | `bp_capabilities_and_refresh_count` | ``assertion `left == right` failed`` |

## Review-round guards

Each edit below was compiled and run in an isolated workspace copy with a
separate Cargo target directory, then restored. Every killer exited 101.

| Mutant | Exact compiling edit | Killer | First failing line verbatim |
|---|---|---|---|
| `capability_half_exact_exempt` | In `exp/pecos-trellis/src/lib.rs`, replace `params.k == usize::MAX && params.delta.is_infinite()` with `params.k == usize::MAX \|\| params.delta.is_infinite()`. Restrict the edit to `TrellisDecoder::attempt`. | `bp_capabilities_and_refresh_count` | `expected InvalidConfiguration naming require a BP graph` |
| `residual_runs_bp` | In `exp/pecos-trellis/src/lib.rs`, replace `            if residual != 0 {` with `            if residual != 0 {\n                self.model.refresh_bp_suffix_values(&mut self.scratch, &observed)?;`. Restrict the edit to `TrellisDecoder::prepare`. | `residual_precheck_skips_bp_refresh` | ``assertion `left == right` failed: residual preparation must skip BP`` |
| `residual_reports_highest_bit` | In `TrellisDecoder::prepare`, replace `residual.trailing_zeros() as usize` with `(63 - residual.leading_zeros()) as usize`. | `residual_reports_the_lowest_detector_within_and_across_words` | ``assertion `left == right` failed`` |
| `residual_scans_words_reversed` | In `TrellisDecoder::prepare`, replace `.enumerate()` on the observed/forced/touched zip with `.enumerate().rev()`. | `residual_reports_the_lowest_detector_within_and_across_words` | ``assertion `left == right` failed`` |
