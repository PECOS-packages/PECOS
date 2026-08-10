# BP Trellis manual mutant checklist

The repository has no mutation runner. Apply each compiling edit separately,
run the named killer test(s), and restore the source before trying the next
row. “Equivalent” rows document mutations that intentionally have no killer.

| Mutant | Exact edit (quoted old → new) | Expected killer test(s) or disposition |
|---|---|---|
| `bptrellis_default_bp_off` | In `BpTrellisConfig::default`, replace `"bp_score_iterations: 5"` with `"bp_score_iterations: 0"`. | `bptrellis_defaults_enable_bp_merge_and_deadline_order` (including its `bp_seconds > 0.0` assertions) |
| `bptrellis_default_merge_off` | In `BpTrellisConfig::default`, replace `"merge_indistinguishable: true"` with `"merge_indistinguishable: false"`. | `bptrellis_defaults_enable_bp_merge_and_deadline_order` (including its `processed_columns` assertion for the planted duplicate pair) |
| `bptrellis_deadline_ignored` | In `BpTrellisDecoder::from_sparse_dem`, replace `"TrellisOrdering::Deadline => Some(deadline_column_order(dem)?)"` with `"TrellisOrdering::Deadline => None"`. | `bptrellis_defaults_enable_bp_merge_and_deadline_order`; `bptrellis_matches_hand_mapped_trellis_for_every_ordering` |
| `escalate_on_any_failure` | In `BpTrellisDecoder::decode`, escalate a successful result whenever `result.status != TrellisStatus::Exact` instead of returning every `TrellisDecodeAttempt::Success` immediately. | `successful_base_decode_is_bit_identical_with_a_configured_ladder`; `wrong_prediction_does_not_escalate` |
| `ladder_skips_accumulation` | Delete `"result.transitions += transitions;"` from the successful-rung branch in `BpTrellisDecoder::decode`. | `no_path_escalates_to_k16_and_accumulates_transitions` |
| `ladder_reuses_base_k` | In `BpTrellisDecoder::from_sparse_dem`, replace `"k: rung_k"` with `"k"` when constructing each escalation rung. | `no_path_escalates_to_k16_and_accumulates_transitions` |
| `bptrellis_diversity_ignored` | In `BpTrellisDecoder::from_sparse_dem`, replace `"label_diverse_retention,"` with `"label_diverse_retention: false,"` in the mapped `TrellisConfig`. | `bptrellis_maps_label_diverse_retention_to_the_engine` |
