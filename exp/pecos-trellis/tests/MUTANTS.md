# Trellis engine manual mutant checklist

The repository has no mutation runner. Apply each compiling edit separately,
run the named killer test(s), and restore the source before trying the next
row. "Equivalent" rows document mutations that intentionally have no killer.

Every edit below targets `exp/pecos-trellis/src/lib.rs`, and every killer named
below lives in this crate. Killers that lived in the parity-port crate
(`decode_outputs_match_bitwise_snapshot`,
`unpruned_and_pruned_results_match_upstream_golden_fixtures`) are deliberately
NOT cited here: those suites ship with the port, so a reader of this crate
alone could not run them. `score_fold_restore`'s killers were re-derived by
executing the mutant against this crate's suite rather than assumed.

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
