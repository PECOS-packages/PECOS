// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0

use super::qir_reason;

#[test]
fn accepts_each_qir_attribute() {
    for key in ["entry_point", "qir_profiles", "required_num_results"] {
        for suffix in ["", "=\"1\""] {
            let ir = format!("attributes #0 = {{ \"{key}\"{suffix} }}");
            assert_eq!(qir_reason(&ir), None, "rejected {ir}");
        }
    }
}

#[test]
fn rejects_pointer_signatures() {
    for ir in [
        "declare void @__quantum__qis__mz__body(ptr, ptr)",
        "declare ptr @__quantum__qis__m__body(ptr)",
        "declare void @__quantum__qis__rx__body(double, ptr)",
        "declare ptr @__quantum__rt__qubit_allocate()",
        "declare void @__quantum__rt__qubit_release(ptr)",
        "declare ptr @__quantum__rt__result_allocate()",
        "declare ptr @__quantum__rt__result_get_one()",
        "declare void @__quantum__qis__s__adj(ptr)",
        "declare void @__quantum__qis__swap__body(ptr, ptr)",
        "declare void @__quantum__qis__x__body(ptr addrspace(1))",
        "declare void @__quantum__qis__x__body(i64 \"description\"=\"(ptr)\", ptr)",
        "declare void @__quantum__qis__x__body(ptr byval(i64))",
        "declare void @\"__quantum__qis__custom gate\"(ptr)",
        "define void @__quantum__qis__x__body(ptr %q) {\n  ret void\n}",
    ] {
        assert!(qir_reason(ir).is_some(), "accepted {ir}");
    }
}

#[test]
fn accepts_integer_signatures_and_data() {
    for ir in [
        "declare i32 @__quantum__qis__mz__body(i64)",
        "declare i32 @__quantum__qis__m__body(i64, i64)",
        "declare void @__quantum__qis__rx__body(double, i64)",
        "declare i64 @__quantum__rt__qubit_allocate()",
        "declare i64 @__quantum__rt__result_allocate()",
        "declare i32 @__quantum__rt__result_get_one(i64)",
        "declare void @__quantum__rt__qubit_release(i64)",
        "declare void @__quantum__rt__result_record_output(ptr, ptr)",
        "declare void @__quantum__rt__custom(ptr)",
        "declare void @__quantum__qis__x__body(i64 \"description\"=\"ptr\")",
        "define void @__quantum__qis__x__body(i64 %ptr) {\n  ret void\n}",
        "attributes #0 = { \"EntryPoint\" \"note\"=\"entry_point\" \"other\"=\"qir_profiles\" \"results\"=\"required_num_results\" }",
        "define void @main() #0 prefix [11 x i8] c\"entry_point\" {\n  ret void\n}\nattributes #0 = { \"EntryPoint\" }",
        "@message = constant [12 x i8] c\"entry_point\\00\"",
        "!0 = !{!\"entry_point\"}",
        "define void @main() {\n  %entry_point = add i64 0, 1\n  ret void\n}",
    ] {
        assert_eq!(qir_reason(ir), None, "rejected {ir}");
    }
}
