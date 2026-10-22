//! Regression cases from Python's `test_helper_symbols.py`.

use super::normalize_pecos_helper_symbols;
use serde::Deserialize;
use std::collections::BTreeMap;
use std::sync::LazyLock;

#[derive(Deserialize)]
struct Case {
    ir: String,
    expected: Option<String>,
    error: Option<String>,
}

static CASES: LazyLock<BTreeMap<String, Vec<Case>>> = LazyLock::new(|| {
    serde_json::from_str(include_str!("python_cases.json")).expect("valid helper cases")
});

fn check(name: &str) {
    for (index, case) in CASES[name].iter().enumerate() {
        let actual = normalize_pecos_helper_symbols(&case.ir);
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
}

#[test]
fn test_rename_every_symbol_occurrence() {
    check("test_rename_every_symbol_occurrence");
}

#[test]
fn test_all_helper_abis() {
    check("test_all_helper_abis");
}

#[test]
fn test_identical_declarations_merge_after_attributes_and_whitespace() {
    check("test_identical_declarations_merge_after_attributes_and_whitespace");
}

#[test]
fn test_conflicting_declarations_report_both_signatures() {
    check("test_conflicting_declarations_report_both_signatures");
}

#[test]
fn test_wrong_abi_reports_expected_and_actual() {
    check("test_wrong_abi_reports_expected_and_actual");
}

#[test]
fn test_helper_definition_is_rejected() {
    check("test_helper_definition_is_rejected");
}

#[test]
fn test_non_helpers_are_untouched() {
    check("test_non_helpers_are_untouched");
}

#[test]
fn test_strings_and_comments_are_untouched() {
    check("test_strings_and_comments_are_untouched");
}

#[test]
fn test_function_body_reference_is_not_a_definition() {
    check("test_function_body_reference_is_not_a_definition");
}

#[test]
fn test_quoted_symbol_with_guppy_scope_is_renamed() {
    check("test_quoted_symbol_with_guppy_scope_is_renamed");
}

#[test]
fn test_declaration_text_inside_multiline_string_is_untouched() {
    check("test_declaration_text_inside_multiline_string_is_untouched");
}

#[test]
fn test_aggregate_declarations_merge_with_spacing_and_return_attributes() {
    check("test_aggregate_declarations_merge_with_spacing_and_return_attributes");
}

#[test]
fn test_declaration_modifiers_are_not_return_types() {
    check("test_declaration_modifiers_are_not_return_types");
}

#[test]
fn test_declarations_with_different_modifiers_merge() {
    check("test_declarations_with_different_modifiers_merge");
}

#[test]
fn test_parameter_names_do_not_change_the_llvm_type() {
    check("test_parameter_names_do_not_change_the_llvm_type");
}

#[test]
fn test_control_characters_inside_string_do_not_split_declarations() {
    check("test_control_characters_inside_string_do_not_split_declarations");
}

#[test]
fn test_multiline_helper_definition_is_rejected() {
    check("test_multiline_helper_definition_is_rejected");
}

#[test]
fn test_multiline_declaration_is_validated() {
    check("test_multiline_declaration_is_validated");
}

#[test]
fn test_two_declarations_on_one_line() {
    check("test_two_declarations_on_one_line");
}

#[test]
fn test_hex_escaped_public_helper_is_validated() {
    check("test_hex_escaped_public_helper_is_validated");
}

#[test]
fn test_hex_escaped_private_helper_is_renamed_everywhere() {
    check("test_hex_escaped_private_helper_is_renamed_everywhere");
}

#[test]
fn test_duplicate_removal_preserves_trailing_comment_and_next_declaration() {
    check("test_duplicate_removal_preserves_trailing_comment_and_next_declaration");
}

#[test]
fn test_duplicate_removal_preserves_internal_comment() {
    check("test_duplicate_removal_preserves_internal_comment");
}

#[test]
fn test_helper_outside_declaration_or_instruction_is_rejected() {
    check("test_helper_outside_declaration_or_instruction_is_rejected");
}

#[test]
fn test_instruction_operands_are_allowed_across_lines() {
    check("test_instruction_operands_are_allowed_across_lines");
}

#[test]
fn test_unrecognized_namer_suffix_is_rejected() {
    check("test_unrecognized_namer_suffix_is_rejected");
}

#[test]
fn test_non_numeric_helper_suffix_is_rejected() {
    check("test_non_numeric_helper_suffix_is_rejected");
}

#[test]
fn test_conflicting_return_pointer_types_report_complete_signatures() {
    check("test_conflicting_return_pointer_types_report_complete_signatures");
}
#[test]
fn calling_conventions_match_the_llvm_boundary() {
    let helper = "pecos_qis_runtime_barrier_qubit_hugr";
    for convention in ["", "ccc", "cc 0", "cc 00", "cc0", "cc00"] {
        let ir = format!("declare {convention} i64 @__hugr__.{helper}.1(i64)");
        assert!(normalize_pecos_helper_symbols(&ir).is_ok(), "{convention}");
    }
    for convention in [
        "fastcc",
        "coldcc",
        "cc 8",
        "cc 1023",
        "cc8",
        "cc1023",
        "aarch64_vector_pcs",
        "aarch64_sve_vector_pcs",
        "aarch64_sme_preservemost_from_x0",
        "aarch64_sme_preservemost_from_x1",
        "aarch64_sme_preservemost_from_x2",
        "amdgpu_cs_chain",
        "amdgpu_cs_chain_preserve",
        "hhvm_ccc",
        "spir_func",
        "ptx_kernel",
        "riscv_vector_cc",
        "riscv_vls_cc",
    ] {
        for name in [helper.to_owned(), format!("__hugr__.{helper}.1")] {
            let ir = format!("declare {convention} i64 @{name}(i64)");
            let error = normalize_pecos_helper_symbols(&ir).unwrap_err().to_string();
            assert!(error.contains("incompatible ABI"), "{convention}: {error}");
        }
        let ir = format!(
            "declare i64 @{helper}(i64)\ndeclare {convention} i64 @__hugr__.{helper}.1(i64)"
        );
        assert!(normalize_pecos_helper_symbols(&ir).is_err(), "{convention}");
    }
}

#[test]
fn malformed_parameter_offsets_count_unicode_characters() {
    let ir = "; é\ndeclare void @unterminated(";
    let error = normalize_pecos_helper_symbols(ir).unwrap_err().to_string();
    assert_eq!(error, "Unterminated function parameter list at offset 30");
}
