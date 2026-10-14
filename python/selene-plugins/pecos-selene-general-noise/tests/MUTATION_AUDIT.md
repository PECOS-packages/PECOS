# General-noise mutation audit

This representative manual audit checks that the conformance suite detects
plausible implementation defects rather than merely executing the affected
paths. Each mutation was applied in isolation on 2026-08-24, its focused test was
run, and the source mutation was immediately reverted.

| Injected defect | Test that rejected it |
| --- | --- |
| Swap the `0 -> 1` and `1 -> 0` measurement parameters | `test_single_qubit_bernoulli_channels` and `test_one_to_zero_readout_channel` |
| Ignore a zero global noise scale | `test_noise_suppression_controls[global-scale]` |
| Disable configured one-qubit seepage | `test_seepage_releases_a_leaked_qubit[one-qubit-specific]` |
| Translate `MeasureLeaked` as an ordinary measurement | Rust `measure_leaked_reports_computational_and_leaked_outcomes` |
| Omit all topology-defined local measurement victims | `test_measurement_crosstalk_is_device_neutral[local-group]` |
| Drop runtime idle duration during nanosecond-to-second conversion | Rust `runtime_timestamps_drive_every_idle_family` |
| Reverse the operands of translated RZZ operations | Rust `seeded_randomized_traces_match_direct_general_noise_execution` |
| Keep the original one-qubit gate on the emission branch | `test_one_qubit_emission_replaces_original_gate` |
| Keep the original two-qubit gate on the emission branch | `test_two_qubit_fault_families` |
| Return a reset without recording the leaked state | Rust `measure_leaked_reports_computational_and_leaked_outcomes` |

The one-qubit emission-replacement mutation initially survived both the default
qutrit cases and the extended generated matrix. The focused deterministic test
listed above was added as a result and was rerun against the mutation to confirm
that it fails. No other mutation in this audit survived its pre-existing focused
test.

This is a bounded audit, not a proof of completeness. Native qutrit simulator
verification and a larger automated property/mutation corpus remain useful
follow-up work.
