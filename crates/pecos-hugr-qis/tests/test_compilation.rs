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
