// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0

use super::qir_reason;

#[test]
fn malformed_header_constants_are_left_for_llvm() {
    for clause in ["prefix", "prologue", "personality"] {
        for constant in [
            "",
            "i8",
            "i8 c",
            "ptr no_cfi",
            "ptr dso_local_equivalent",
            "ptr getelementptr inbounds",
            "[1 x i8",
            "{ i8 } {",
        ] {
            let ir = format!("define void @f() {clause} {constant}");
            assert_eq!(qir_reason(&ir), None, "malformed IR is left for LLVM: {ir}");
        }
    }
}

#[test]
fn rejects_each_qir_attribute_without_quantum_declarations() {
    for attribute in ["entry_point", "qir_profiles", "required_num_results"] {
        for ir in [
            format!(
                "define void @main() #0 {{ ret void }}\nattributes #0 = {{ \"{attribute}\"=\"1\" }}"
            ),
            format!("define void @main()\n\"{attribute}\"=\"1\"\n{{ ret void }}"),
        ] {
            assert!(qir_reason(&ir).is_some(), "accepted {ir}");
        }
    }
    assert!(qir_reason(r#"attributes #0 = { "entry\5fpoint" }"#).is_some());
    assert!(
        qir_reason(
            r#"define void @main() "entry_point" prefix [11 x i8] c"entry_point" { ret void }"#
        )
        .is_some()
    );
}

#[test]
fn rejects_shared_intrinsic_signatures_without_qir_attributes() {
    for ir in [
        "declare void @__quantum__qis__mz__body(ptr, ptr)",
        "declare void @__quantum__qis__mz__body(%Qubit*, %Result*)",
        "declare void @__quantum__qis__x__body(%Qubit*)",
        "declare ptr @__quantum__qis__m__body(ptr)",
        "declare %Result* @__quantum__qis__m__body(%Qubit*)",
        "declare void @__quantum__qis__rx__body(double, ptr)",
        "declare ptr @__quantum__rt__qubit_allocate()",
        "declare void @__quantum__rt__qubit_release(ptr)",
        "declare %Result* @__quantum__rt__result_get_one()",
        "declare ptr @__quantum__rt__result_allocate()",
        "define void @__quantum__qis__x__body(ptr %q) { ret void }",
        "declare void @\"__quantum__qis__mz__body\"(\n ptr noundef %q,\n ptr nonnull align 8 %r\n)",
        "declare void @\"__quantum__qis__x__b\\6fdy\"(ptr addrspace(1))",
    ] {
        assert!(qir_reason(ir).is_some(), "accepted {ir}");
    }
}

#[test]
fn accepts_qis_and_qir_words_used_as_data() {
    for ir in [
        "declare i32 @__quantum__qis__mz__body(i64)",
        "declare i32 @__quantum__qis__m__body(i64, i64)",
        "declare void @__quantum__qis__rx__body(double, i64)",
        "declare void @__quantum__rt__result_record_output(ptr, ptr)",
        "declare void @__quantum__rt__result_record_output(i8*, i8*)",
        "declare i32 @__quantum__rt__result_get_one(i64)",
        "declare i64 @__quantum__rt__result_allocate()",
        "declare void @__quantum__qis__x__body(i64 \"description\"=\"ptr\")",
        "; declare void @__quantum__qis__mz__body(ptr, ptr)\n; attributes #0 = { \"entry_point\" }",
        "@message = constant [12 x i8] c\"entry_point\\00\"",
        "@message = constant [99 x i8] c\"declare void @__quantum__qis__x__body(ptr)\\00\"",
        "attributes #0 = { \"EntryPoint\" \"note\"=\"entry_point\" \"other\"=\"qir_profiles\" }",
        "define void @\"entry_point\"() section \"entry_point\" gc \"qir_profiles\" { ret void }",
        "define i64 @qmain(i64 %arg) prefix [11 x i8] c\"entry_point\" { ret i64 0 }",
        "define i64 @qmain(i64 %arg) prologue [12 x i8] c\"qir_profiles\" { ret i64 0 }",
        "define i64 @qmain(i64 %arg) partition \"required_num_results\" { ret i64 0 }",
        "define i64 @qmain(i64 %arg) prefix { i64, [11 x i8] } { i64 1, [11 x i8] c\"entry_point\" } { attributes: ret i64 0 }",
        "define i64 @qmain(i64 %arg) prologue <{ [12 x i8] }> <{ [12 x i8] c\"qir_profiles\" }> { declare: ret i64 0 }",
        "define void @main() { %entry_point = add i64 0, 1\n ret void }",
        "!0 = !{!\"entry_point\", !\"qir_profiles\"}",
    ] {
        assert_eq!(qir_reason(ir), None, "rejected {ir}");
    }
}

#[test]
fn accepts_every_checked_in_qis_fixture() {
    let fixtures = [
        (
            "pliron ghz3",
            include_str!("../../pecos-phir-pliron/fixtures/ghz3.ll"),
        ),
        (
            "pliron cz_swap",
            include_str!("../../pecos-phir-pliron/fixtures/cz_swap.ll"),
        ),
        (
            "pliron adaptive_branch",
            include_str!("../../pecos-phir-pliron/fixtures/adaptive_branch.ll"),
        ),
        (
            "pliron branch_measure",
            include_str!("../../pecos-phir-pliron/fixtures/branch_measure.ll"),
        ),
        (
            "cli bell",
            include_str!("../../pecos-cli/tests/test_data/hugr/bell_state.ll"),
        ),
        (
            "pecos bell",
            include_str!("../../pecos/tests/test_data/hugr/bell_state.ll"),
        ),
        ("llvm bell", include_str!("../../../examples/llvm/bell.ll")),
        (
            "llvm qprog",
            include_str!("../../../examples/llvm/qprog.ll"),
        ),
        (
            "bell final",
            include_str!("../../../examples/bell_final.ll"),
        ),
    ];
    for (name, ir) in fixtures {
        assert_eq!(qir_reason(ir), None, "rejected QIS fixture {name}");
    }
}

#[test]
fn rejects_both_checked_in_qir_fixtures() {
    for (name, ir) in [
        (
            "ArithmeticOps.Targeted",
            include_str!(
                "../../../python/quantum-pecos/tests/pecos/integration/ll/ArithmeticOps.Targeted.ll"
            ),
        ),
        (
            "IntegerSupport.TargetedAlt",
            include_str!(
                "../../../python/quantum-pecos/tests/pecos/integration/ll/IntegerSupport.TargetedAlt.ll"
            ),
        ),
    ] {
        assert!(qir_reason(ir).is_some(), "accepted QIR fixture {name}");
    }
}
