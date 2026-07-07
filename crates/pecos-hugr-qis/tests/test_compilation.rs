//! Test HUGR to QIS compilation functionality

use pecos_hugr_qis::prelude::*;

#[test]
fn test_direct_compilation_api() {
    // Test empty HUGR (should fail gracefully)
    let empty_bytes = b"";
    let result = compile_hugr_bytes_to_string(empty_bytes);
    assert!(result.is_err());

    // Test invalid HUGR (should fail gracefully)
    let invalid_bytes = b"not a valid hugr";
    let result = compile_hugr_bytes_to_string(invalid_bytes);
    assert!(result.is_err());
}

#[test]
fn test_compiler_wrapper_api() {
    // Test using the HugrCompiler wrapper
    let compiler = HugrCompiler::new();

    // Test empty HUGR
    let empty_bytes = b"";
    let result = compiler.compile_hugr_bytes_to_string(empty_bytes);
    assert!(result.is_err());

    // Test invalid HUGR
    let invalid_bytes = b"not a valid hugr";
    let result = compiler.compile_hugr_bytes_to_string(invalid_bytes);
    assert!(result.is_err());
}

#[test]
fn test_json_hugr_format() {
    // Test that JSON format is detected and handled
    let json_hugr = br#"{"version": "0.1.0", "nodes": []}"#;
    let result = compile_hugr_bytes_to_string(json_hugr);
    // This should fail because it's not a valid HUGR, but it should
    // at least attempt to parse it as JSON
    assert!(result.is_err());
    if let Err(e) = result {
        let error_msg = e.to_string();
        // Should mention something about HUGR or module loading
        assert!(error_msg.contains("HUGR") || error_msg.contains("Failed"));
    }
}

#[test]
fn test_compile_args() {
    let mut args = CompileArgs::default();
    assert_eq!(args.name, "hugr");
    assert_eq!(args.opt_level, OptimizationLevel::Default);
    assert!(args.entry.is_none());
    assert!(args.target_triple.is_none());
    assert!(args.save_hugr.is_none());

    // Test custom args
    args.opt_level = OptimizationLevel::Aggressive;
    args.target_triple = Some("x86_64-unknown-linux-gnu".to_string());
    args.name = "test".to_string();

    assert_eq!(args.opt_level, OptimizationLevel::Aggressive);
    assert_eq!(
        args.target_triple,
        Some("x86_64-unknown-linux-gnu".to_string())
    );
    assert_eq!(args.name, "test");
}

#[test]
fn test_bitcode_compilation() {
    // Test bitcode compilation with invalid HUGR
    let invalid_bytes = b"not a valid hugr";
    let result = compile_hugr_bytes_to_bitcode(invalid_bytes);
    assert!(result.is_err());
}

#[test]
fn test_check_hugr() {
    // Test check_hugr function
    let invalid_bytes = b"not a valid hugr";
    let result = check_hugr(invalid_bytes);
    assert!(result.is_err());

    let empty_bytes = b"";
    let result = check_hugr(empty_bytes);
    assert!(result.is_err());
}

#[test]
fn test_compiler_config() {
    let config = HugrCompilerConfig {
        name: Some("mymodule".to_string()),
        opt_level: Some(OptimizationLevel::None),
        target_triple: Some("aarch64-apple-darwin".to_string()),
        ..Default::default()
    };

    let compiler = HugrCompiler::with_config(config);

    // Test that config is properly used by attempting compilation
    // (We can't access the private config field directly, but we can test behavior)
    let invalid_bytes = b"not valid";
    let result = compiler.compile_hugr_bytes_to_string(invalid_bytes);
    assert!(result.is_err());
}

#[test]
fn test_target_machine_creation() {
    // Test native target machine
    let result = get_native_target_machine(OptimizationLevel::Default);
    assert!(result.is_ok());

    // Test specific target machine (should work with initialization)
    let result =
        get_target_machine_from_triple("x86_64-unknown-linux-gnu", OptimizationLevel::Default);
    assert!(result.is_ok());
}

#[test]
fn test_optimization_levels() {
    let levels = vec![
        OptimizationLevel::None,
        OptimizationLevel::Less,
        OptimizationLevel::Default,
        OptimizationLevel::Aggressive,
    ];

    for level in levels {
        let result = get_native_target_machine(level);
        assert!(
            result.is_ok(),
            "Failed to create target machine with optimization level {level:?}"
        );
    }
}

#[test]
fn function_local_helper_lowers_to_public_symbol_in_text_and_bitcode() {
    // A Guppy program (captured from `barrier_pair_probe`) that declares and calls a
    // function-local PECOS helper. guppylang qualifies the helper with a `<locals>`
    // segment under the private `__hugr__.*` namespace; BOTH the text IR and the
    // bitcode must expose the public `pecos_qis_runtime_barrier_qubits2_hugr` ABI
    // symbol that pecos-qis-ffi exports, not the private name. (Regression guard for
    // the bitcode path, which a text-only normalization would miss.)
    let hugr = include_bytes!("fixtures/szz_barrier_probe.hugr");

    let text = compile_hugr_bytes_to_string(hugr).expect("text compilation should succeed");
    assert!(
        text.contains("@pecos_qis_runtime_barrier_qubits2_hugr("),
        "text IR is missing the public helper symbol"
    );
    assert!(
        !text.contains("<locals>.pecos_qis_runtime_barrier_qubits2_hugr"),
        "text IR still carries the private helper symbol"
    );

    let bitcode = compile_hugr_bytes_to_bitcode(hugr).expect("bitcode compilation should succeed");
    // LLVM stores symbol names contiguously in the bitcode string table, so a byte
    // scan is reliable. The private form embeds the helper under `<locals>.`; the
    // entry function `barrier_pair_probe` is also function-local but does not match
    // this needle, so it is a clean discriminator.
    let private = b"<locals>.pecos_qis_runtime_barrier_qubits2_hugr";
    let public = b"pecos_qis_runtime_barrier_qubits2_hugr";
    assert!(
        !bitcode.windows(private.len()).any(|w| w == private),
        "bitcode still carries the private helper symbol"
    );
    assert!(
        bitcode.windows(public.len()).any(|w| w == public),
        "bitcode is missing the public helper symbol"
    );
}

#[test]
fn duplicate_helper_declarations_merge_to_one_public_symbol() {
    // A HUGR with two declarations of the same helper -- one module-level and one
    // function-local wrapper -- both normalize to
    // `pecos_qis_runtime_barrier_qubits2_hugr`. A blind rename would let LLVM uniquify
    // the second declaration to `...hugr.1`, which pecos-qis-ffi does not export; the
    // merge must collapse them into one unsuffixed public symbol. (Regression guard
    // for the module-level rename collision.)
    let hugr = include_bytes!("fixtures/szz_barrier_collision.hugr");

    let text = compile_hugr_bytes_to_string(hugr).expect("text compilation should succeed");
    assert!(
        text.contains("@pecos_qis_runtime_barrier_qubits2_hugr("),
        "text IR is missing the public helper symbol"
    );
    // The helper name is never followed by a dot: no `.1` collision suffix and no
    // private `__hugr__...pecos_qis_runtime_barrier_qubits2_hugr.<id>` residue.
    assert!(
        !text.contains("pecos_qis_runtime_barrier_qubits2_hugr."),
        "text IR has a suffixed or private helper symbol"
    );

    let bitcode = compile_hugr_bytes_to_bitcode(hugr).expect("bitcode compilation should succeed");
    let suffixed_or_private = b"pecos_qis_runtime_barrier_qubits2_hugr.";
    let public = b"pecos_qis_runtime_barrier_qubits2_hugr";
    assert!(
        !bitcode
            .windows(suffixed_or_private.len())
            .any(|w| w == suffixed_or_private),
        "bitcode has a suffixed or private helper symbol"
    );
    assert!(
        bitcode.windows(public.len()).any(|w| w == public),
        "bitcode is missing the public helper symbol"
    );
}

#[test]
fn conflicting_helper_signatures_fail_loud() {
    // Two declarations that normalize to `pecos_qis_runtime_barrier_qubit_hugr`, one
    // with the wrong signature (`i64 -> { i64, i64 }` vs the ABI `i64 -> i64`). The
    // wrong declaration must be rejected against the fixed pecos-qis-ffi ABI -- LLVM 21
    // opaque pointers + `module.verify()` do not catch a call that disagrees with the
    // export -- so compilation must fail loud instead of shipping ABI-broken IR.
    let hugr = include_bytes!("fixtures/szz_barrier_wrong_signature.hugr");

    let text = compile_hugr_bytes_to_string(hugr);
    assert!(
        text.is_err(),
        "text compilation should fail on a wrong-ABI helper declaration"
    );
    let msg = text.unwrap_err().to_string();
    assert!(
        msg.contains("pecos-qis-ffi ABI") && msg.contains("pecos_qis_runtime_barrier_qubit_hugr"),
        "unexpected error message: {msg}"
    );

    assert!(
        compile_hugr_bytes_to_bitcode(hugr).is_err(),
        "bitcode compilation should fail on a wrong-ABI helper declaration"
    );
}

#[test]
fn single_wrong_helper_signature_fails_loud() {
    // A SINGLE declaration of a recognized helper with a self-consistent but wrong ABI
    // (`pecos_qis_runtime_barrier_qubits2_hugr` declared `i64 -> i64` instead of the
    // exported `(i64, i64) -> { i64, i64 }`). There is no sibling to compare against,
    // so this is caught only by validating the lone declaration against the fixed
    // pecos-qis-ffi ABI. Both text IR and bitcode compilation must fail loud.
    let hugr = include_bytes!("fixtures/szz_barrier_single_wrong_abi.hugr");

    let text = compile_hugr_bytes_to_string(hugr);
    assert!(
        text.is_err(),
        "text compilation should fail on a lone wrong-ABI helper declaration"
    );
    let msg = text.unwrap_err().to_string();
    assert!(
        msg.contains("pecos-qis-ffi ABI") && msg.contains("pecos_qis_runtime_barrier_qubits2_hugr"),
        "unexpected error message: {msg}"
    );

    assert!(
        compile_hugr_bytes_to_bitcode(hugr).is_err(),
        "bitcode compilation should fail on a lone wrong-ABI helper declaration"
    );
}
