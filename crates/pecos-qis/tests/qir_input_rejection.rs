// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0

#![cfg(feature = "selene")]

use pecos_qis::QisHeliosInterface;
use pecos_qis::qis_interface::{ProgramFormat, QisInterface};

const QIR: &str = r#"
define void @main() #0 {
  call void @__quantum__qis__x__body(ptr null)
  call void @__quantum__qis__mz__body(ptr null, ptr inttoptr (i64 1 to ptr))
  ret void
}
declare void @__quantum__qis__x__body(ptr)
declare void @__quantum__qis__mz__body(ptr, ptr)
attributes #0 = { "entry_point" "qir_profiles"="base_profile" "required_num_results"="2" }
"#;

#[test]
fn loader_rejects_qir_text_before_linking() {
    let error = QisHeliosInterface::new()
        .load_program(QIR.as_bytes(), ProgramFormat::LlvmIrText)
        .expect_err("QIR must be rejected before linking its incompatible quantum intrinsics");
    assert!(
        error.to_string().contains("convert QIR to QIS first"),
        "{error}"
    );
}
