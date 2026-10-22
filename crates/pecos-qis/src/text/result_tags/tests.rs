//! Inputs and expectations captured from every case in `test_qis_result_tags.py`.
//! Compiled cases contain the actual Selene output from the HUGR fixtures.

use super::super::extract_result_tag_measurements;
use serde::Deserialize;
use std::collections::BTreeMap;
use std::sync::LazyLock;

#[derive(Deserialize)]
struct Case {
    ir: String,
    expected: Option<super::super::ResultTagMeasurements>,
    error: Option<String>,
}

static CASES: LazyLock<BTreeMap<String, Vec<Case>>> = LazyLock::new(|| {
    serde_json::from_str(include_str!("tests/python_cases.json")).expect("valid captured cases")
});

fn check(name: &str, index: usize) {
    let case = &CASES[name][index];
    let actual = extract_result_tag_measurements(&case.ir);
    if let Some(expected) = &case.expected {
        assert_eq!(
            &actual.unwrap_or_else(|error| panic!("{name}[{index}]: {error}")),
            expected,
            "{name}[{index}]"
        );
    } else {
        assert_eq!(
            actual.unwrap_err().to_string(),
            *case.error.as_ref().unwrap(),
            "{name}[{index}]"
        );
    }
}

#[test]
fn test_compiled_backslash_tag() {
    check("test_compiled_backslash_tag", 0);
}

mod test_compiled_fixture {
    #[test]
    fn case_0() {
        super::check("test_compiled_fixture", 0);
    }
    #[test]
    fn case_1() {
        super::check("test_compiled_fixture", 1);
    }
    #[test]
    fn case_2() {
        super::check("test_compiled_fixture", 2);
    }
    #[test]
    fn case_3() {
        super::check("test_compiled_fixture", 3);
    }
    #[test]
    fn case_4() {
        super::check("test_compiled_fixture", 4);
    }
    #[test]
    fn case_5() {
        super::check("test_compiled_fixture", 5);
    }
}

#[test]
fn test_block_layout_does_not_determine_ordinals() {
    check("test_block_layout_does_not_determine_ordinals", 0);
}

#[test]
fn test_label_and_instruction_on_same_line() {
    check("test_label_and_instruction_on_same_line", 0);
}

mod test_call_outside_instruction_opcode {
    #[test]
    fn case_0() {
        super::check("test_call_outside_instruction_opcode", 0);
    }
    #[test]
    fn case_1() {
        super::check("test_call_outside_instruction_opcode", 1);
    }
    #[test]
    fn case_2() {
        super::check("test_call_outside_instruction_opcode", 2);
    }
    #[test]
    fn case_3() {
        super::check("test_call_outside_instruction_opcode", 3);
    }
    #[test]
    fn case_4() {
        super::check("test_call_outside_instruction_opcode", 4);
    }
    #[test]
    fn case_5() {
        super::check("test_call_outside_instruction_opcode", 5);
    }
    #[test]
    fn case_6() {
        super::check("test_call_outside_instruction_opcode", 6);
    }
    #[test]
    fn case_7() {
        super::check("test_call_outside_instruction_opcode", 7);
    }
    #[test]
    fn case_8() {
        super::check("test_call_outside_instruction_opcode", 8);
    }
}

#[test]
fn test_two_prints_on_one_line_cannot_drop_an_occurrence() {
    check("test_two_prints_on_one_line_cannot_drop_an_occurrence", 0);
}

mod test_terminator_outside_instruction_opcode {
    #[test]
    fn case_0() {
        super::check("test_terminator_outside_instruction_opcode", 0);
    }
    #[test]
    fn case_1() {
        super::check("test_terminator_outside_instruction_opcode", 1);
    }
    #[test]
    fn case_2() {
        super::check("test_terminator_outside_instruction_opcode", 2);
    }
}

mod test_call_names_and_strings_are_not_opcodes {
    #[test]
    fn case_0() {
        super::check("test_call_names_and_strings_are_not_opcodes", 0);
    }
    #[test]
    fn case_1() {
        super::check("test_call_names_and_strings_are_not_opcodes", 1);
    }
    #[test]
    fn case_2() {
        super::check("test_call_names_and_strings_are_not_opcodes", 2);
    }
    #[test]
    fn case_3() {
        super::check("test_call_names_and_strings_are_not_opcodes", 3);
    }
}

mod test_terminator_must_be_last {
    #[test]
    fn case_0() {
        super::check("test_terminator_must_be_last", 0);
    }
    #[test]
    fn case_1() {
        super::check("test_terminator_must_be_last", 1);
    }
    #[test]
    fn case_2() {
        super::check("test_terminator_must_be_last", 2);
    }
    #[test]
    fn case_3() {
        super::check("test_terminator_must_be_last", 3);
    }
    #[test]
    fn case_4() {
        super::check("test_terminator_must_be_last", 4);
    }
    #[test]
    fn case_5() {
        super::check("test_terminator_must_be_last", 5);
    }
    #[test]
    fn case_6() {
        super::check("test_terminator_must_be_last", 6);
    }
    #[test]
    fn case_7() {
        super::check("test_terminator_must_be_last", 7);
    }
    #[test]
    fn case_8() {
        super::check("test_terminator_must_be_last", 8);
    }
    #[test]
    fn case_9() {
        super::check("test_terminator_must_be_last", 9);
    }
    #[test]
    fn case_10() {
        super::check("test_terminator_must_be_last", 10);
    }
    #[test]
    fn case_11() {
        super::check("test_terminator_must_be_last", 11);
    }
    #[test]
    fn case_12() {
        super::check("test_terminator_must_be_last", 12);
    }
    #[test]
    fn case_13() {
        super::check("test_terminator_must_be_last", 13);
    }
    #[test]
    fn case_14() {
        super::check("test_terminator_must_be_last", 14);
    }
    #[test]
    fn case_15() {
        super::check("test_terminator_must_be_last", 15);
    }
    #[test]
    fn case_16() {
        super::check("test_terminator_must_be_last", 16);
    }
    #[test]
    fn case_17() {
        super::check("test_terminator_must_be_last", 17);
    }
    #[test]
    fn case_18() {
        super::check("test_terminator_must_be_last", 18);
    }
    #[test]
    fn case_19() {
        super::check("test_terminator_must_be_last", 19);
    }
    #[test]
    fn case_20() {
        super::check("test_terminator_must_be_last", 20);
    }
    #[test]
    fn case_21() {
        super::check("test_terminator_must_be_last", 21);
    }
}

mod test_block_requires_terminator {
    #[test]
    fn case_0() {
        super::check("test_block_requires_terminator", 0);
    }
    #[test]
    fn case_1() {
        super::check("test_block_requires_terminator", 1);
    }
    #[test]
    fn case_2() {
        super::check("test_block_requires_terminator", 2);
    }
    #[test]
    fn case_3() {
        super::check("test_block_requires_terminator", 3);
    }
    #[test]
    fn case_4() {
        super::check("test_block_requires_terminator", 4);
    }
    #[test]
    fn case_5() {
        super::check("test_block_requires_terminator", 5);
    }
}

mod test_measurement_free_control_flow {
    #[test]
    fn case_0() {
        super::check("test_measurement_free_control_flow", 0);
    }
    #[test]
    fn case_1() {
        super::check("test_measurement_free_control_flow", 1);
    }
    #[test]
    fn case_2() {
        super::check("test_measurement_free_control_flow", 2);
    }
    #[test]
    fn case_3() {
        super::check("test_measurement_free_control_flow", 3);
    }
}

mod test_conditional_order_relevant_block_rejected {
    #[test]
    fn case_0() {
        super::check("test_conditional_order_relevant_block_rejected", 0);
    }
    #[test]
    fn case_1() {
        super::check("test_conditional_order_relevant_block_rejected", 1);
    }
}

mod test_loop_order_relevant_block_rejected {
    #[test]
    fn case_0() {
        super::check("test_loop_order_relevant_block_rejected", 0);
    }
    #[test]
    fn case_1() {
        super::check("test_loop_order_relevant_block_rejected", 1);
    }
}

#[test]
fn test_measurement_must_dominate_all_returns() {
    check("test_measurement_must_dominate_all_returns", 0);
}

#[test]
fn test_join_intersects_all_predecessor_dominators() {
    check("test_join_intersects_all_predecessor_dominators", 0);
}

#[test]
fn test_measurement_dominates_several_returns() {
    check("test_measurement_dominates_several_returns", 0);
}

mod test_non_lazy_runtime_measurements_rejected {
    #[test]
    fn case_0() {
        super::check("test_non_lazy_runtime_measurements_rejected", 0);
    }
    #[test]
    fn case_1() {
        super::check("test_non_lazy_runtime_measurements_rejected", 1);
    }
    #[test]
    fn case_2() {
        super::check("test_non_lazy_runtime_measurements_rejected", 2);
    }
    #[test]
    fn case_3() {
        super::check("test_non_lazy_runtime_measurements_rejected", 3);
    }
}

mod test_panic_branches {
    #[test]
    fn case_0() {
        super::check("test_panic_branches", 0);
    }
    #[test]
    fn case_1() {
        super::check("test_panic_branches", 1);
    }
}

mod test_unsupported_programs {
    #[test]
    fn case_0() {
        super::check("test_unsupported_programs", 0);
    }
    #[test]
    fn case_1() {
        super::check("test_unsupported_programs", 1);
    }
    #[test]
    fn case_2() {
        super::check("test_unsupported_programs", 2);
    }
    #[test]
    fn case_3() {
        super::check("test_unsupported_programs", 3);
    }
    #[test]
    fn case_4() {
        super::check("test_unsupported_programs", 4);
    }
    #[test]
    fn case_5() {
        super::check("test_unsupported_programs", 5);
    }
    #[test]
    fn case_6() {
        super::check("test_unsupported_programs", 6);
    }
    #[test]
    fn case_7() {
        super::check("test_unsupported_programs", 7);
    }
    #[test]
    fn case_8() {
        super::check("test_unsupported_programs", 8);
    }
}

#[test]
fn test_defined_callee_counts_at_call_site_and_keeps_ssa_local() {
    check(
        "test_defined_callee_counts_at_call_site_and_keeps_ssa_local",
        0,
    );
}

#[test]
fn test_conditional_transitive_callee_rejected() {
    check("test_conditional_transitive_callee_rejected", 0);
}

mod test_only_exact_read_result_carries_provenance {
    #[test]
    fn case_0() {
        super::check("test_only_exact_read_result_carries_provenance", 0);
    }
    #[test]
    fn case_1() {
        super::check("test_only_exact_read_result_carries_provenance", 1);
    }
    #[test]
    fn case_2() {
        super::check("test_only_exact_read_result_carries_provenance", 2);
    }
    #[test]
    fn case_3() {
        super::check("test_only_exact_read_result_carries_provenance", 3);
    }
    #[test]
    fn case_4() {
        super::check("test_only_exact_read_result_carries_provenance", 4);
    }
    #[test]
    fn case_5() {
        super::check("test_only_exact_read_result_carries_provenance", 5);
    }
    #[test]
    fn case_6() {
        super::check("test_only_exact_read_result_carries_provenance", 6);
    }
}

#[test]
fn test_leaked_measurement_counts_without_scalar_provenance() {
    check(
        "test_leaked_measurement_counts_without_scalar_provenance",
        0,
    );
}

#[test]
fn test_read_in_separate_block() {
    check("test_read_in_separate_block", 0);
}

mod test_all_runtime_print_exports {
    #[test]
    fn case_0() {
        super::check("test_all_runtime_print_exports", 0);
    }
    #[test]
    fn case_1() {
        super::check("test_all_runtime_print_exports", 1);
    }
    #[test]
    fn case_2() {
        super::check("test_all_runtime_print_exports", 2);
    }
    #[test]
    fn case_3() {
        super::check("test_all_runtime_print_exports", 3);
    }
    #[test]
    fn case_4() {
        super::check("test_all_runtime_print_exports", 4);
    }
    #[test]
    fn case_5() {
        super::check("test_all_runtime_print_exports", 5);
    }
    #[test]
    fn case_6() {
        super::check("test_all_runtime_print_exports", 6);
    }
    #[test]
    fn case_7() {
        super::check("test_all_runtime_print_exports", 7);
    }
    #[test]
    fn case_8() {
        super::check("test_all_runtime_print_exports", 8);
    }
    #[test]
    fn case_9() {
        super::check("test_all_runtime_print_exports", 9);
    }
    #[test]
    fn case_10() {
        super::check("test_all_runtime_print_exports", 10);
    }
    #[test]
    fn case_11() {
        super::check("test_all_runtime_print_exports", 11);
    }
    #[test]
    fn case_12() {
        super::check("test_all_runtime_print_exports", 12);
    }
    #[test]
    fn case_13() {
        super::check("test_all_runtime_print_exports", 13);
    }
    #[test]
    fn case_14() {
        super::check("test_all_runtime_print_exports", 14);
    }
    #[test]
    fn case_15() {
        super::check("test_all_runtime_print_exports", 15);
    }
}

#[test]
fn test_llvm_escaped_utf8_tag_and_runtime_length() {
    check("test_llvm_escaped_utf8_tag_and_runtime_length", 0);
}

mod test_invalid_tag_bytes {
    #[test]
    fn case_0() {
        super::check("test_invalid_tag_bytes", 0);
    }
    #[test]
    fn case_1() {
        super::check("test_invalid_tag_bytes", 1);
    }
    #[test]
    fn case_2() {
        super::check("test_invalid_tag_bytes", 2);
    }
}

#[test]
fn test_dead_blocks_and_uncalled_functions_are_ignored() {
    check("test_dead_blocks_and_uncalled_functions_are_ignored", 0);
}

#[test]
fn test_direct_call_with_explicit_function_type_and_pointer_attributes() {
    check(
        "test_direct_call_with_explicit_function_type_and_pointer_attributes",
        0,
    );
}

#[test]
fn test_alias_cannot_hide_measurements() {
    check("test_alias_cannot_hide_measurements", 0);
}

#[test]
fn test_comment_cannot_supply_tag_global_contents() {
    check("test_comment_cannot_supply_tag_global_contents", 0);
}

#[test]
fn test_defined_always_panicking_helper_without_relevant_calls() {
    check(
        "test_defined_always_panicking_helper_without_relevant_calls",
        0,
    );
}

mod test_deep_cfg_uses_iterative_traversals {
    #[test]
    fn case_0() {
        super::check("test_deep_cfg_uses_iterative_traversals", 0);
    }
    #[test]
    fn case_1() {
        super::check("test_deep_cfg_uses_iterative_traversals", 1);
    }
}

mod test_module_cannot_define_runtime_calls {
    #[test]
    fn case_0() {
        super::check("test_module_cannot_define_runtime_calls", 0);
    }
    #[test]
    fn case_1() {
        super::check("test_module_cannot_define_runtime_calls", 1);
    }
    #[test]
    fn case_2() {
        super::check("test_module_cannot_define_runtime_calls", 2);
    }
    #[test]
    fn case_3() {
        super::check("test_module_cannot_define_runtime_calls", 3);
    }
    #[test]
    fn case_4() {
        super::check("test_module_cannot_define_runtime_calls", 4);
    }
    #[test]
    fn case_5() {
        super::check("test_module_cannot_define_runtime_calls", 5);
    }
    #[test]
    fn case_6() {
        super::check("test_module_cannot_define_runtime_calls", 6);
    }
    #[test]
    fn case_7() {
        super::check("test_module_cannot_define_runtime_calls", 7);
    }
    #[test]
    fn case_8() {
        super::check("test_module_cannot_define_runtime_calls", 8);
    }
    #[test]
    fn case_9() {
        super::check("test_module_cannot_define_runtime_calls", 9);
    }
    #[test]
    fn case_10() {
        super::check("test_module_cannot_define_runtime_calls", 10);
    }
    #[test]
    fn case_11() {
        super::check("test_module_cannot_define_runtime_calls", 11);
    }
    #[test]
    fn case_12() {
        super::check("test_module_cannot_define_runtime_calls", 12);
    }
    #[test]
    fn case_13() {
        super::check("test_module_cannot_define_runtime_calls", 13);
    }
    #[test]
    fn case_14() {
        super::check("test_module_cannot_define_runtime_calls", 14);
    }
    #[test]
    fn case_15() {
        super::check("test_module_cannot_define_runtime_calls", 15);
    }
    #[test]
    fn case_16() {
        super::check("test_module_cannot_define_runtime_calls", 16);
    }
    #[test]
    fn case_17() {
        super::check("test_module_cannot_define_runtime_calls", 17);
    }
    #[test]
    fn case_18() {
        super::check("test_module_cannot_define_runtime_calls", 18);
    }
    #[test]
    fn case_19() {
        super::check("test_module_cannot_define_runtime_calls", 19);
    }
    #[test]
    fn case_20() {
        super::check("test_module_cannot_define_runtime_calls", 20);
    }
    #[test]
    fn case_21() {
        super::check("test_module_cannot_define_runtime_calls", 21);
    }
}

mod test_defined_print_cannot_redirect_tag {
    #[test]
    fn case_0() {
        super::check("test_defined_print_cannot_redirect_tag", 0);
    }
    #[test]
    fn case_1() {
        super::check("test_defined_print_cannot_redirect_tag", 1);
    }
}

mod test_malformed_input_is_a_certification_error {
    #[test]
    fn case_0() {
        super::check("test_malformed_input_is_a_certification_error", 0);
    }
    #[test]
    fn case_1() {
        super::check("test_malformed_input_is_a_certification_error", 1);
    }
    #[test]
    fn case_2() {
        super::check("test_malformed_input_is_a_certification_error", 2);
    }
    #[test]
    fn case_3() {
        super::check("test_malformed_input_is_a_certification_error", 3);
    }
    #[test]
    fn case_4() {
        super::check("test_malformed_input_is_a_certification_error", 4);
    }
    #[test]
    fn case_5() {
        super::check("test_malformed_input_is_a_certification_error", 5);
    }
}

mod test_non_llvm_string_escapes_rejected {
    #[test]
    fn case_0() {
        super::check("test_non_llvm_string_escapes_rejected", 0);
    }
    #[test]
    fn case_1() {
        super::check("test_non_llvm_string_escapes_rejected", 1);
    }
    #[test]
    fn case_2() {
        super::check("test_non_llvm_string_escapes_rejected", 2);
    }
}
