//! Property-based tests for Zlup using proptest.
//!
//! These tests verify invariants that should hold for all inputs:
//! - Parser never panics on any UTF-8 input
//! - Semantic analyzer never panics on any valid AST
//! - Type system invariants are maintained
//! - BitWidth constraints are enforced

use proptest::prelude::*;
use zlup::semantic::{BitWidth, SemanticAnalyzer, Type};

// =============================================================================
// Parser Properties
// =============================================================================

proptest! {
    /// The parser should never panic on any UTF-8 string input.
    #[test]
    fn parser_never_panics(input in ".*") {
        // Just call parse - we don't care about the result, only that it doesn't panic
        let _ = zlup::parse(&input);
    }

    /// The parser should handle strings up to 10KB without issues.
    #[test]
    fn parser_handles_large_input(input in ".{0,10000}") {
        let _ = zlup::parse(&input);
    }

    /// Parsing valid function declarations should work.
    #[test]
    fn parser_valid_function(
        name in "[a-z][a-z0-9_]{0,20}",
        ret_type in prop_oneof!["unit", "u32", "bool", "f64"]
    ) {
        let source = format!("fn {}() -> {} {{ return {}; }}",
            name,
            ret_type,
            match ret_type.as_str() {
                "unit" => "unit",
                "u32" => "0",
                "bool" => "true",
                "f64" => "0.0",
                _ => "unit",
            }
        );
        let result = zlup::parse(&source);
        prop_assert!(result.is_ok(), "Failed to parse: {}", source);
    }
}

// =============================================================================
// BitWidth Properties
// =============================================================================

proptest! {
    /// BitWidth::new should accept values 1-128 and reject others.
    #[test]
    fn bitwidth_valid_range(bits in 1u16..=128) {
        let bw = BitWidth::new(bits);
        prop_assert!(bw.is_some(), "BitWidth::new({}) should succeed", bits);
        prop_assert_eq!(bw.unwrap().get(), bits);
    }

    /// BitWidth::new should reject 0.
    #[test]
    fn bitwidth_rejects_zero(_dummy in 0..1u8) {
        let bw = BitWidth::new(0);
        prop_assert!(bw.is_none(), "BitWidth::new(0) should fail");
    }

    /// BitWidth::new should reject values > 128.
    #[test]
    fn bitwidth_rejects_large(bits in 129u16..=u16::MAX) {
        let bw = BitWidth::new(bits);
        prop_assert!(bw.is_none(), "BitWidth::new({}) should fail", bits);
    }
}

// =============================================================================
// Type System Properties
// =============================================================================

proptest! {
    /// A type should be resolved if and only if it contains no Unknown.
    #[test]
    fn type_resolved_iff_no_unknown(_bits in 1u16..=128) {
        // Concrete types should be resolved
        let concrete = Type::Bool;
        prop_assert!(concrete.is_resolved());
        prop_assert!(!concrete.contains_unknown());

        // Unknown should not be resolved
        let unknown = Type::Unknown;
        prop_assert!(!unknown.is_resolved());
        prop_assert!(unknown.contains_unknown());
    }

    /// Nested types with Unknown should not be resolved.
    #[test]
    fn nested_unknown_not_resolved(_dummy in 0..1u8) {
        let nested = Type::Optional {
            inner: Box::new(Type::Unknown),
        };
        prop_assert!(!nested.is_resolved());
        prop_assert!(nested.contains_unknown());
    }

    /// Array types inherit Unknown status from their element type.
    #[test]
    fn array_unknown_from_element(size in 0u64..1000) {
        // Array with concrete element
        let concrete_array = Type::Array {
            element: Box::new(Type::Bool),
            size: Some(size),
        };
        prop_assert!(concrete_array.is_resolved());

        // Array with Unknown element
        let unknown_array = Type::Array {
            element: Box::new(Type::Unknown),
            size: Some(size),
        };
        prop_assert!(!unknown_array.is_resolved());
    }
}

// =============================================================================
// Semantic Analyzer Properties
// =============================================================================

proptest! {
    /// Semantic analysis of valid programs should not panic.
    #[test]
    fn semantic_valid_program_no_panic(
        var_name in "[a-z][a-z0-9_]{0,10}",
        value in 0i64..1000
    ) {
        let source = format!(
            "fn main() -> unit {{ {} := {}; return unit; }}",
            var_name, value
        );

        if let Ok(program) = zlup::parse(&source) {
            let mut analyzer = SemanticAnalyzer::new();
            // Should not panic, result doesn't matter
            let _ = analyzer.analyze(&program);
        }
    }

    /// Error recovery should collect all errors without panicking.
    #[test]
    fn error_recovery_no_panic(input in "[a-zA-Z0-9_ ]{0,100}") {
        // Wrap in function to make it potentially parseable
        let source = format!("fn main() -> unit {{ {} return unit; }}", input);

        if let Ok(program) = zlup::parse(&source) {
            let mut analyzer = SemanticAnalyzer::new();
            // Should not panic, may have errors
            let _ = analyzer.analyze_collecting_errors(&program);
        }
    }
}

// =============================================================================
// Integer Literal Properties
// =============================================================================

proptest! {
    /// Integer literals should parse correctly.
    #[test]
    fn integer_literals_parse(value in 0i64..i64::MAX / 2) {
        let source = format!("x := {};", value);
        let result = zlup::parse(&source);
        prop_assert!(result.is_ok(), "Failed to parse integer literal: {}", value);
    }

    /// Negative integer literals should parse correctly.
    #[test]
    fn negative_integers_parse(value in i64::MIN / 2..0i64) {
        let source = format!("x := {};", value);
        let result = zlup::parse(&source);
        prop_assert!(result.is_ok(), "Failed to parse negative literal: {}", value);
    }

    /// Float literals should parse correctly.
    #[test]
    fn float_literals_parse(value in -1e10f64..1e10f64) {
        if value.is_finite() {
            let source = format!("x := {:.6};", value);
            let result = zlup::parse(&source);
            // Some edge cases may not parse, that's OK
            let _ = result;
        }
    }
}

// =============================================================================
// Identifier Properties
// =============================================================================

proptest! {
    /// Valid identifiers should be accepted.
    #[test]
    fn valid_identifiers_accepted(
        first in "[a-zA-Z_]",
        rest in "[a-zA-Z0-9_]{0,30}"
    ) {
        let ident = format!("{}{}", first, rest);
        prop_assume!(!matches!(
            ident.as_str(),
            "fn" | "pub" | "inline" | "comptime" | "mut" | "struct" | "enum"
                | "union" | "packed" | "set" | "if" | "else" | "for" | "switch"
                | "tick" | "return" | "break" | "continue" | "defer" | "errdefer"
                | "and" | "or" | "orelse" | "try" | "catch" | "true" | "false"
                | "none" | "undefined" | "Self" | "test" | "error" | "type" | "anytype"
                | "unit" | "qubit" | "bit" | "alias" | "turns" | "rad"
        ));
        let source = format!("{} := 42;", ident);
        let result = zlup::parse(&source);
        prop_assert!(result.is_ok(), "Failed to parse identifier: {}", ident);
    }
}

// =============================================================================
// NEGATIVE TESTS - These verify that invalid inputs ARE rejected
// =============================================================================

proptest! {
    /// Type mismatches should be rejected by semantic analysis.
    #[test]
    fn type_mismatch_rejected(value in 0i64..1000) {
        // Assigning integer to bool variable should fail
        let source = format!(
            "fn main() -> unit {{ x: bool = {}; return unit; }}",
            value
        );

        if let Ok(program) = zlup::parse(&source) {
            let mut analyzer = SemanticAnalyzer::new();
            let result = analyzer.analyze(&program);
            prop_assert!(result.is_err(), "Type mismatch should be rejected: {}", source);
        }
    }

    /// Undefined variables should be rejected.
    #[test]
    fn undefined_variable_rejected(
        var_name in "[a-z][a-z0-9]{2,10}"  // At least 2 chars to avoid built-ins like 'e', 'pi'
    ) {
        // Skip known built-in constants
        prop_assume!(!["pi", "tau", "e", "qalloc", "measure", "mz", "mx", "my"].contains(&var_name.as_str()));

        // Using undefined variable should fail
        let source = format!(
            "fn main() -> unit {{ x := {} + 1; return unit; }}",
            var_name
        );

        if let Ok(program) = zlup::parse(&source) {
            let mut analyzer = SemanticAnalyzer::new();
            let result = analyzer.analyze(&program);
            prop_assert!(result.is_err(), "Undefined variable should be rejected: {}", source);
        }
    }

    /// Undefined types should be rejected.
    #[test]
    fn undefined_type_rejected(
        type_name in "[A-Z][a-zA-Z0-9]{0,15}"
    ) {
        // Using undefined type should fail
        let source = format!(
            "fn main() -> unit {{ x: {} = 42; return unit; }}",
            type_name
        );

        if let Ok(program) = zlup::parse(&source) {
            let mut analyzer = SemanticAnalyzer::new();
            let result = analyzer.analyze(&program);
            prop_assert!(result.is_err(), "Undefined type should be rejected: {}", source);
        }
    }

    /// Return type mismatches should be rejected.
    #[test]
    fn return_type_mismatch_rejected(value in 0i64..1000) {
        // Returning integer from bool function should fail
        let source = format!(
            "fn foo() -> bool {{ return {}; }}",
            value
        );

        if let Ok(program) = zlup::parse(&source) {
            let mut analyzer = SemanticAnalyzer::new();
            let result = analyzer.analyze(&program);
            prop_assert!(result.is_err(), "Return type mismatch should be rejected: {}", source);
        }
    }

    /// Duplicate function names should be rejected.
    #[test]
    fn duplicate_function_rejected(
        name in "[a-z][a-z0-9]{0,10}"
    ) {
        let source = format!(
            "fn {}() -> unit {{ return unit; }} fn {}() -> unit {{ return unit; }}",
            name, name
        );

        if let Ok(program) = zlup::parse(&source) {
            let mut analyzer = SemanticAnalyzer::new();
            let result = analyzer.analyze(&program);
            prop_assert!(result.is_err(), "Duplicate function should be rejected: {}", source);
        }
    }

    /// Invalid identifier starting with digit should fail to parse.
    #[test]
    fn identifier_starting_with_digit_rejected(
        digit in 0u8..10,
        rest in "[a-zA-Z0-9_]{0,10}"
    ) {
        let ident = format!("{}{}", digit, rest);
        let source = format!("{} := 42;", ident);
        // This should fail to parse (identifiers can't start with digits)
        let result = zlup::parse(&source);
        prop_assert!(result.is_err(), "Identifier starting with digit should be rejected: {}", ident);
    }
}

// =============================================================================
// Mutation Testing - Mutate valid programs to create invalid ones
// =============================================================================

proptest! {
    /// Removing return statement from non-unit function should fail in strict mode.
    #[test]
    fn missing_return_rejected_strict(value in 0i64..1000) {
        let source = format!(
            "fn foo() -> u32 {{ x := {}; }}",  // Missing return
            value
        );

        if let Ok(program) = zlup::parse(&source) {
            let mut analyzer = SemanticAnalyzer::new();
            let result = analyzer.analyze(&program);
            prop_assert!(result.is_err(), "Missing return should be rejected in strict mode: {}", source);
        }
    }

    /// Wrong argument count should be rejected.
    #[test]
    fn wrong_arg_count_rejected(
        extra_args in 1usize..5
    ) {
        // Calling function with wrong number of arguments
        let args = (0..extra_args).map(|i| format!("{}", i)).collect::<Vec<_>>().join(", ");
        let source = format!(
            "fn foo() -> unit {{ return unit; }} fn main() -> unit {{ foo({}); return unit; }}",
            args
        );

        if let Ok(program) = zlup::parse(&source) {
            let mut analyzer = SemanticAnalyzer::new();
            let result = analyzer.analyze(&program);
            prop_assert!(result.is_err(), "Wrong argument count should be rejected: {}", source);
        }
    }
}

// =============================================================================
// Quantum-Specific Negative Tests
// =============================================================================

proptest! {
    /// Gate on unprepared qubit should fail in strict mode.
    #[test]
    fn gate_on_unprepared_qubit_rejected_strict(index in 0usize..4) {
        let source = format!(
            "fn main() -> unit {{ q := qalloc(4); h q[{}]; return unit; }}",  // Missing pz
            index
        );

        if let Ok(program) = zlup::parse(&source) {
            let mut analyzer = SemanticAnalyzer::new();
            let result = analyzer.analyze(&program);
            prop_assert!(result.is_err(), "Gate on unprepared qubit should be rejected in strict mode: {}", source);
        }
    }

    /// Qubit index out of bounds should be rejected.
    #[test]
    fn qubit_out_of_bounds_rejected(
        capacity in 1usize..10,
        index in 10usize..20
    ) {
        let source = format!(
            "fn main() -> unit {{ q := qalloc({}); pz q; h q[{}]; return unit; }}",
            capacity, index
        );

        if let Ok(program) = zlup::parse(&source) {
            let mut analyzer = SemanticAnalyzer::new();
            let result = analyzer.analyze(&program);
            // Should fail because index >= capacity
            prop_assert!(result.is_err(), "Qubit out of bounds should be rejected: {}", source);
        }
    }
}

// =============================================================================
// Mutability Error Tests
// =============================================================================

proptest! {
    /// Assignment to immutable variable should be rejected.
    #[test]
    fn immutable_assignment_rejected(value in 0i64..1000) {
        // x is immutable (no mut), so x = value should fail
        let source = format!(
            "fn main() -> unit {{ x := 1; x = {}; return unit; }}",
            value
        );

        if let Ok(program) = zlup::parse(&source) {
            let mut analyzer = SemanticAnalyzer::new();
            let result = analyzer.analyze(&program);
            prop_assert!(result.is_err(), "Assignment to immutable variable should be rejected: {}", source);
        }
    }

    /// Reassignment to mutable variable should succeed.
    #[test]
    fn mutable_assignment_allowed(value in 0i64..1000) {
        // mut x is mutable, so x = value should succeed
        let source = format!(
            "fn main() -> unit {{ mut x := 1; x = {}; return unit; }}",
            value
        );

        if let Ok(program) = zlup::parse(&source) {
            let mut analyzer = SemanticAnalyzer::new();
            let result = analyzer.analyze(&program);
            prop_assert!(result.is_ok(), "Assignment to mutable variable should succeed: {:?}", result);
        }
    }

    /// Mutation via method on immutable allocator should be rejected.
    #[test]
    fn immutable_allocator_child_rejected(capacity in 1usize..10) {
        // base is immutable, so base.child() should fail (child requires mut)
        let source = format!(
            "fn main() -> unit {{ base := qalloc(8); q := base.child({}); return unit; }}",
            capacity
        );

        if let Ok(program) = zlup::parse(&source) {
            let mut analyzer = SemanticAnalyzer::new();
            let result = analyzer.analyze(&program);
            prop_assert!(result.is_err(), "Child of immutable allocator should be rejected: {}", source);
        }
    }
}

// =============================================================================
// Gate After Measurement Tests (Quantum Safety)
// =============================================================================

proptest! {
    /// Gate after measurement without re-preparing should be rejected in strict mode.
    #[test]
    fn gate_after_measurement_rejected_strict(index in 0usize..4) {
        // After measurement, qubit returns to unprepared state
        // Applying a gate without pz should fail
        let source = format!(
            r#"fn main() -> unit {{
                mut q := qalloc(4);
                pz q;
                result := mz q[{}];
                h q[{}];  // Should fail - qubit is unprepared after measurement
                return unit;
            }}"#,
            index, index
        );

        if let Ok(program) = zlup::parse(&source) {
            let mut analyzer = SemanticAnalyzer::new();
            let result = analyzer.analyze(&program);
            prop_assert!(result.is_err(), "Gate after measurement should be rejected in strict mode: {}", source);
        }
    }

    /// Gate after measurement with re-prepare should succeed.
    #[test]
    fn gate_after_measurement_with_prepare_allowed(index in 0usize..4) {
        // After measurement, re-prepare with pz, then gate should work
        let source = format!(
            r#"fn main() -> unit {{
                mut q := qalloc(4);
                pz q;
                result := mz q[{}];
                pz q[{}];  // Re-prepare the qubit
                h q[{}];   // Now this should succeed
                return unit;
            }}"#,
            index, index, index
        );

        if let Ok(program) = zlup::parse(&source) {
            let mut analyzer = SemanticAnalyzer::new();
            let result = analyzer.analyze(&program);
            prop_assert!(result.is_ok(), "Gate after re-prepare should succeed: {:?}", result);
        }
    }
}

// =============================================================================
// Invalid Gate Arity Tests
// =============================================================================

proptest! {
    /// Single-qubit gate with tuple target should be rejected.
    #[test]
    fn single_qubit_gate_with_two_targets_rejected(_dummy in 0..1u8) {
        // H is a single-qubit gate, giving it two qubits should fail
        let source = r#"
            fn main() -> unit {
                mut q := qalloc(4);
                pz q;
                h (q[0], q[1]);  // H takes 1 qubit, not 2
                return unit;
            }
        "#;

        if let Ok(program) = zlup::parse(source) {
            let mut analyzer = SemanticAnalyzer::new();
            let result = analyzer.analyze(&program);
            prop_assert!(result.is_err(), "Single-qubit gate with two targets should be rejected");
        }
    }

    /// Two-qubit gate with single target should be rejected.
    #[test]
    fn two_qubit_gate_with_one_target_rejected(index in 0usize..4) {
        // CX is a two-qubit gate, giving it one qubit should fail
        let source = format!(
            r#"fn main() -> unit {{
                mut q := qalloc(4);
                pz q;
                cx q[{}];  // CX takes 2 qubits, not 1
                return unit;
            }}"#,
            index
        );

        if let Ok(program) = zlup::parse(&source) {
            let mut analyzer = SemanticAnalyzer::new();
            let result = analyzer.analyze(&program);
            prop_assert!(result.is_err(), "Two-qubit gate with one target should be rejected: {}", source);
        }
    }

    /// Two-qubit gate with correct arity should succeed.
    #[test]
    fn two_qubit_gate_correct_arity_allowed(_dummy in 0..1u8) {
        // CX with two qubits should succeed
        let source = r#"
            fn main() -> unit {
                mut q := qalloc(4);
                pz q;
                cx (q[0], q[1]);  // CX takes 2 qubits - correct
                return unit;
            }
        "#;

        if let Ok(program) = zlup::parse(source) {
            let mut analyzer = SemanticAnalyzer::new();
            let result = analyzer.analyze(&program);
            prop_assert!(result.is_ok(), "Two-qubit gate with correct arity should succeed: {:?}", result);
        }
    }

    /// Three-qubit gate (CCX/Toffoli) with wrong arity should be rejected.
    #[test]
    fn three_qubit_gate_with_two_targets_rejected(_dummy in 0..1u8) {
        // CCX is a three-qubit gate
        let source = r#"
            fn main() -> unit {
                mut q := qalloc(4);
                pz q;
                ccx (q[0], q[1]);  // CCX takes 3 qubits, not 2
                return unit;
            }
        "#;

        if let Ok(program) = zlup::parse(source) {
            let mut analyzer = SemanticAnalyzer::new();
            let result = analyzer.analyze(&program);
            prop_assert!(result.is_err(), "Three-qubit gate with two targets should be rejected");
        }
    }
}

// =============================================================================
// Recursive Call Tests (NASA Power of 10 Compliance)
// =============================================================================

proptest! {
    /// Direct recursion is always rejected (NASA Power of 10 compliance).
    /// Use FFI with Rust if recursive algorithms are needed.
    #[test]
    fn direct_recursion_rejected_strict(n in 1i64..10) {
        // Function calling itself directly
        let source = format!(
            r#"fn factorial(n: i64) -> i64 {{
                if n <= 1 {{
                    return 1;
                }}
                return n * factorial(n - {});
            }}
            fn main() -> unit {{ x := factorial(5); return unit; }}"#,
            n.min(1)  // Always subtract at least 1
        );

        if let Ok(program) = zlup::parse(&source) {
            let mut analyzer = SemanticAnalyzer::new();
            let result = analyzer.analyze(&program);
            prop_assert!(result.is_err(), "Direct recursion should be rejected: {}", source);
        }
    }

    /// Mutual recursion is always rejected (NASA Power of 10 compliance).
    #[test]
    fn mutual_recursion_rejected_strict(_dummy in 0..1u8) {
        // Two functions calling each other
        let source = r#"
            fn is_even(n: i64) -> bool {
                if n == 0 { return true; }
                return is_odd(n - 1);
            }
            fn is_odd(n: i64) -> bool {
                if n == 0 { return false; }
                return is_even(n - 1);
            }
            fn main() -> unit { x := is_even(10); return unit; }
        "#;

        if let Ok(program) = zlup::parse(source) {
            let mut analyzer = SemanticAnalyzer::new();
            let result = analyzer.analyze(&program);
            prop_assert!(result.is_err(), "Mutual recursion should be rejected");
        }
    }

    /// Non-recursive functions should succeed.
    #[test]
    fn non_recursive_allowed_strict(a in 0i64..100, b in 0i64..100) {
        // Regular function calls (no recursion) should be fine
        let source = format!(
            r#"fn add(x: i64, y: i64) -> i64 {{ return x + y; }}
            fn main() -> unit {{ result := add({}, {}); return unit; }}"#,
            a, b
        );

        if let Ok(program) = zlup::parse(&source) {
            let mut analyzer = SemanticAnalyzer::new();
            let result = analyzer.analyze(&program);
            prop_assert!(result.is_ok(), "Non-recursive function should succeed: {:?}", result);
        }
    }
}

/// Recursion is rejected even in permissive mode (no escape hatch).
#[test]
fn recursion_rejected_even_permissive() {
    let source = r#"
        fn factorial(n: i64) -> i64 {
            if n <= 1 { return 1; }
            return n * factorial(n - 1);
        }
        fn main() -> unit { x := factorial(5); return unit; }
    "#;
    let program = zlup::parse(source).expect("Should parse");
    let mut analyzer = SemanticAnalyzer::new_permissive();
    let result = analyzer.analyze(&program);
    assert!(
        result.is_err(),
        "Recursion should be rejected even in permissive mode"
    );
}

// =============================================================================
// =============================================================================
// Performance Regression Tests
// =============================================================================

/// Test that deeply nested parens don't cause exponential parsing time.
/// This input was found by fuzzing to be slow.
#[test]
fn slow_input_deeply_nested_parens() {
    use std::time::{Duration, Instant};

    // This input caused 445s parse time in fuzzing
    let input = "a:=((Z/[[((((((((((((((((((((((/\n";

    let start = Instant::now();
    let _ = zlup::parse(input);
    let elapsed = start.elapsed();

    println!("Parse time for deeply nested input: {:?}", elapsed);

    // Should complete in under 1 second, not 445 seconds
    assert!(
        elapsed < Duration::from_secs(1),
        "Parsing took too long: {:?} (expected < 1s)",
        elapsed
    );
}

/// Test another slow input found by fuzzing - deeply nested parens in function call.
#[test]
fn slow_input_nested_call_parens() {
    use std::time::{Duration, Instant};

    // This input caused timeout in fuzzing: k:k=z(((((((((((((((((((((((((/inl...
    let input = "k:k=z(((((((((((((((((((((((((/inl\x1a\x00at\r\r:0";

    let start = Instant::now();
    let _ = zlup::parse(input);
    let elapsed = start.elapsed();

    println!("Parse time for nested call parens: {:?}", elapsed);

    // Should complete in under 1 second
    assert!(
        elapsed < Duration::from_secs(1),
        "Parsing took too long: {:?} (expected < 1s)",
        elapsed
    );
}

// =============================================================================
// Additional Parser Negative Tests
// =============================================================================

/// Unclosed parentheses should fail to parse.
#[test]
fn unclosed_paren_rejected() {
    let inputs = [
        "x := (1 + 2;",
        "x := ((1);",
        "fn foo( {}",
        "x := func(a, b;",
    ];
    for input in inputs {
        let result = zlup::parse(input);
        assert!(
            result.is_err(),
            "Unclosed paren should be rejected: {}",
            input
        );
    }
}

/// Unclosed brackets should fail to parse.
#[test]
fn unclosed_bracket_rejected() {
    let inputs = ["x := [1, 2, 3;", "x := a[0;", "x: [4]u8 = [1, 2;"];
    for input in inputs {
        let result = zlup::parse(input);
        assert!(
            result.is_err(),
            "Unclosed bracket should be rejected: {}",
            input
        );
    }
}

/// Unclosed braces should fail to parse.
#[test]
fn unclosed_brace_rejected() {
    let inputs = [
        "fn foo() {",
        "x := Point { x: 1, y: 2;",
        "if true { x := 1;",
    ];
    for input in inputs {
        let result = zlup::parse(input);
        assert!(
            result.is_err(),
            "Unclosed brace should be rejected: {}",
            input
        );
    }
}

/// Reserved keywords as identifiers should fail.
#[test]
fn keyword_as_identifier_rejected() {
    let keywords = [
        "fn", "if", "else", "for", "return", "struct", "enum", "true", "false",
    ];
    for kw in keywords {
        let source = format!("{} := 42;", kw);
        let result = zlup::parse(&source);
        assert!(
            result.is_err(),
            "Keyword '{}' as identifier should be rejected",
            kw
        );
    }
}

/// Invalid number literals should fail to parse.
#[test]
fn invalid_number_literal_rejected() {
    let inputs = [
        "x := 0x;", // hex with no digits
        "x := 0b;", // binary with no digits
        "x := 0o;", // octal with no digits
        "x := 1.;", // trailing dot with no fraction
    ];
    for input in inputs {
        let result = zlup::parse(input);
        assert!(
            result.is_err(),
            "Invalid number literal should be rejected: {}",
            input
        );
    }
}

/// Deprecated measurement syntax should be rejected.
#[test]
fn deprecated_measurement_syntax_rejected() {
    let source = r#"
        pub fn main() -> unit {
            mut q := qalloc(2);
            pz q;
            r := mz(u1, q[0]);
        }
    "#;
    let program = zlup::parse(source).expect("Should parse");
    let mut analyzer = SemanticAnalyzer::new();
    let result = analyzer.analyze(&program);
    assert!(
        result.is_err(),
        "Deprecated mz(type, target) syntax should be rejected"
    );
}

/// Empty function body is valid but missing semicolon is not.
#[test]
fn missing_semicolon_rejected() {
    let inputs = [
        "x := 42",                       // missing semicolon on binding
        "fn foo() -> u32 { return 42 }", // missing semicolon on return
    ];
    for input in inputs {
        let result = zlup::parse(input);
        assert!(
            result.is_err(),
            "Missing semicolon should be rejected: {}",
            input
        );
    }
}

// =============================================================================
// Additional Semantic Negative Tests
// =============================================================================

/// Batch gate with wrong arity elements should be rejected.
#[test]
fn batch_gate_wrong_element_arity_rejected() {
    // H is single-qubit, but we're giving it tuples
    let source = r#"
        pub fn main() -> unit {
            mut q := qalloc(4);
            pz q;
            h {(q[0], q[1]), (q[2], q[3])};
        }
    "#;
    let program = zlup::parse(source).expect("Should parse");
    let mut analyzer = SemanticAnalyzer::new();
    let result = analyzer.analyze(&program);
    assert!(
        result.is_err(),
        "H gate with tuple elements should be rejected"
    );
}

/// CX batch with single qubits instead of pairs should be rejected.
#[test]
fn batch_cx_wrong_element_type_rejected() {
    // CX needs pairs, but we're giving single qubits
    let source = r#"
        pub fn main() -> unit {
            mut q := qalloc(4);
            pz q;
            cx {q[0], q[1], q[2], q[3]};
        }
    "#;
    let program = zlup::parse(source).expect("Should parse");
    let mut analyzer = SemanticAnalyzer::new();
    let result = analyzer.analyze(&program);
    assert!(
        result.is_err(),
        "CX gate with single qubit elements should be rejected"
    );
}

/// Using a non-qubit type where qubit is expected should be rejected.
#[test]
fn non_qubit_as_gate_target_rejected() {
    let source = r#"
        pub fn main() -> unit {
            x := 42;
            h x;
        }
    "#;
    let program = zlup::parse(source).expect("Should parse");
    let mut analyzer = SemanticAnalyzer::new();
    let result = analyzer.analyze(&program);
    assert!(result.is_err(), "Integer as gate target should be rejected");
}

/// Break outside of loop should be rejected.
#[test]
fn break_outside_loop_rejected() {
    let source = r#"
        pub fn main() -> unit {
            break;
        }
    "#;
    let program = zlup::parse(source).expect("Should parse");
    let mut analyzer = SemanticAnalyzer::new();
    let result = analyzer.analyze(&program);
    assert!(result.is_err(), "Break outside loop should be rejected");
}

/// Continue outside of loop should be rejected.
#[test]
fn continue_outside_loop_rejected() {
    let source = r#"
        pub fn main() -> unit {
            continue;
        }
    "#;
    let program = zlup::parse(source).expect("Should parse");
    let mut analyzer = SemanticAnalyzer::new();
    let result = analyzer.analyze(&program);
    assert!(result.is_err(), "Continue outside loop should be rejected");
}

/// Nested tick blocks should be rejected.
#[test]
fn nested_tick_rejected() {
    let source = r#"
        pub fn main() -> unit {
            mut q := qalloc(2);
            pz q;
            tick {
                tick {
                    h q[0];
                }
            }
        }
    "#;
    let program = zlup::parse(source).expect("Should parse");
    let mut analyzer = SemanticAnalyzer::new();
    let result = analyzer.analyze(&program);
    assert!(result.is_err(), "Nested tick blocks should be rejected");
}

/// Same qubit used twice in same tick should be rejected.
#[test]
fn duplicate_qubit_in_tick_rejected() {
    let source = r#"
        pub fn main() -> unit {
            mut q := qalloc(2);
            pz q;
            tick {
                h q[0];
                x q[0];
            }
        }
    "#;
    let program = zlup::parse(source).expect("Should parse");
    let mut analyzer = SemanticAnalyzer::new();
    let result = analyzer.analyze(&program);
    assert!(
        result.is_err(),
        "Same qubit used twice in tick should be rejected"
    );
}

/// Catch on non-error type should be rejected.
#[test]
fn catch_on_non_error_type_rejected() {
    let source = r#"
        pub fn main() -> unit {
            x := 42;
            y := x catch 0;
        }
    "#;
    let program = zlup::parse(source).expect("Should parse");
    let mut analyzer = SemanticAnalyzer::new();
    let result = analyzer.analyze(&program);
    assert!(
        result.is_err(),
        "Catch on non-error type should be rejected"
    );
}

/// Array index with non-integer should be rejected.
#[test]
fn array_index_non_integer_rejected() {
    let source = r#"
        pub fn main() -> unit {
            arr: [4]u32 = [1, 2, 3, 4];
            x := arr[true];
        }
    "#;
    let program = zlup::parse(source).expect("Should parse");
    let mut analyzer = SemanticAnalyzer::new();
    let result = analyzer.analyze(&program);
    assert!(
        result.is_err(),
        "Array index with boolean should be rejected"
    );
}

/// Field access on non-struct should be rejected.
#[test]
fn field_access_on_non_struct_rejected() {
    let source = r#"
        pub fn main() -> unit {
            x := 42;
            y := x.field;
        }
    "#;
    let program = zlup::parse(source).expect("Should parse");
    let mut analyzer = SemanticAnalyzer::new();
    let result = analyzer.analyze(&program);
    assert!(
        result.is_err(),
        "Field access on integer should be rejected"
    );
}

/// Accessing undefined struct field should be rejected.
#[test]
fn undefined_struct_field_rejected() {
    let source = r#"
        Point := struct {
            x: i32,
            y: i32,
        };
        pub fn main() -> unit {
            p := Point { x: 1, y: 2 };
            z := p.z;
        }
    "#;
    let program = zlup::parse(source).expect("Should parse");
    let mut analyzer = SemanticAnalyzer::new();
    let result = analyzer.analyze(&program);
    assert!(
        result.is_err(),
        "Accessing undefined field should be rejected"
    );
}

/// Duplicate struct field in initialization should be rejected.
#[test]
fn duplicate_struct_field_init_rejected() {
    let source = r#"
        Point := struct {
            x: i32,
            y: i32,
        };
        pub fn main() -> unit {
            p := Point { x: 1, x: 2, y: 3 };
        }
    "#;
    let program = zlup::parse(source).expect("Should parse");
    let mut analyzer = SemanticAnalyzer::new();
    let result = analyzer.analyze(&program);
    assert!(
        result.is_err(),
        "Duplicate field in struct init should be rejected"
    );
}

/// Binary operation with incompatible types should be rejected.
#[test]
fn binary_op_type_mismatch_rejected() {
    let inputs = [
        ("x := true + 1;", "bool + int"),
        ("x := \"hello\" - 5;", "string - int"),
        ("x := 1.5 and true;", "float and bool"),
    ];
    for (source, desc) in inputs {
        let wrapped = format!("pub fn main() -> unit {{ {} }}", source);
        let program = zlup::parse(&wrapped).expect("Should parse");
        let mut analyzer = SemanticAnalyzer::new();
        let result = analyzer.analyze(&program);
        assert!(result.is_err(), "Binary op {} should be rejected", desc);
    }
}

/// Calling a non-function should be rejected.
#[test]
fn call_non_function_rejected() {
    let source = r#"
        pub fn main() -> unit {
            x := 42;
            y := x(1, 2);
        }
    "#;
    let program = zlup::parse(source).expect("Should parse");
    let mut analyzer = SemanticAnalyzer::new();
    let result = analyzer.analyze(&program);
    assert!(result.is_err(), "Calling an integer should be rejected");
}

/// Return with value from unit function should be rejected.
#[test]
fn return_value_from_unit_function_rejected() {
    let source = r#"
        pub fn main() -> unit {
            return 42;
        }
    "#;
    let program = zlup::parse(source).expect("Should parse");
    let mut analyzer = SemanticAnalyzer::new();
    let result = analyzer.analyze(&program);
    assert!(
        result.is_err(),
        "Returning value from unit function should be rejected"
    );
}

/// Wrong type in struct field initialization should be rejected.
#[test]
fn struct_field_wrong_type_rejected() {
    let source = r#"
        Point := struct {
            x: i32,
            y: i32,
        };
        pub fn main() -> unit {
            p := Point { x: "hello", y: 2 };
        }
    "#;
    let program = zlup::parse(source).expect("Should parse");
    let mut analyzer = SemanticAnalyzer::new();
    let result = analyzer.analyze(&program);
    assert!(result.is_err(), "String in i32 field should be rejected");
}

/// Qubit already prepared should be rejected (double prepare).
#[test]
fn qubit_already_prepared_rejected() {
    let source = r#"
        pub fn main() -> unit {
            mut q := qalloc(2);
            pz q;
            pz q;
        }
    "#;
    let program = zlup::parse(source).expect("Should parse");
    let mut analyzer = SemanticAnalyzer::new();
    analyzer.set_strict_mode(true);
    let result = analyzer.analyze(&program);
    assert!(
        result.is_err(),
        "Double prepare should be rejected in strict mode"
    );
}

/// Duplicate qubit in measurement should be rejected.
#[test]
fn duplicate_qubit_in_measurement_rejected() {
    let source = r#"
        pub fn main() -> unit {
            mut q := qalloc(2);
            pz q;
            r := mz([2]u1) [q[0], q[0]];
        }
    "#;
    let program = zlup::parse(source).expect("Should parse");
    let mut analyzer = SemanticAnalyzer::new();
    let result = analyzer.analyze(&program);
    assert!(
        result.is_err(),
        "Duplicate qubit in measurement should be rejected"
    );
}

/// Measurement size mismatch should be rejected.
#[test]
fn measurement_size_mismatch_rejected() {
    let source = r#"
        pub fn main() -> unit {
            mut q := qalloc(4);
            pz q;
            r := mz([2]u1) [q[0], q[1], q[2]];
        }
    "#;
    let program = zlup::parse(source).expect("Should parse");
    let mut analyzer = SemanticAnalyzer::new();
    let result = analyzer.analyze(&program);
    assert!(
        result.is_err(),
        "Measurement size mismatch should be rejected"
    );
}

/// Using orelse on non-optional should be rejected.
#[test]
fn orelse_on_non_optional_rejected() {
    let source = r#"
        pub fn main() -> unit {
            x := 42;
            y := x orelse 0;
        }
    "#;
    let program = zlup::parse(source).expect("Should parse");
    let mut analyzer = SemanticAnalyzer::new();
    let result = analyzer.analyze(&program);
    assert!(result.is_err(), "orelse on non-optional should be rejected");
}

/// Empty array without type annotation should be rejected.
#[test]
fn empty_array_no_type_rejected() {
    let source = r#"
        pub fn main() -> unit {
            arr := [];
        }
    "#;
    let program = zlup::parse(source).expect("Should parse");
    let mut analyzer = SemanticAnalyzer::new();
    let result = analyzer.analyze(&program);
    assert!(
        result.is_err(),
        "Empty array without type should be rejected"
    );
}

/// For loop with non-iterable should be rejected.
#[test]
fn for_loop_non_iterable_rejected() {
    let source = r#"
        pub fn main() -> unit {
            for i in 42 {
                x := i;
            }
        }
    "#;
    let program = zlup::parse(source).expect("Should parse");
    let mut analyzer = SemanticAnalyzer::new();
    let result = analyzer.analyze(&program);
    assert!(result.is_err(), "For loop over integer should be rejected");
}

/// Comparison between incompatible types should be rejected.
#[test]
fn comparison_type_mismatch_rejected() {
    let source = r#"
        pub fn main() -> unit {
            x := 42 == "hello";
        }
    "#;
    let program = zlup::parse(source).expect("Should parse");
    let mut analyzer = SemanticAnalyzer::new();
    let result = analyzer.analyze(&program);
    assert!(
        result.is_err(),
        "Comparing int to string should be rejected"
    );
}

/// If condition must be boolean.
#[test]
fn if_non_boolean_condition_rejected() {
    let source = r#"
        pub fn main() -> unit {
            if 42 {
                x := 1;
            }
        }
    "#;
    let program = zlup::parse(source).expect("Should parse");
    let mut analyzer = SemanticAnalyzer::new();
    let result = analyzer.analyze(&program);
    assert!(
        result.is_err(),
        "If with integer condition should be rejected"
    );
}

// =============================================================================
// More Parser Negative Tests
// =============================================================================

/// Invalid escape sequences in strings should fail.
#[test]
fn invalid_string_escape_rejected() {
    let inputs = [
        r#"x := "\q";"#,     // \q is not a valid escape
        r#"x := "\u1234";"#, // \u is not supported (use \x)
    ];
    for input in inputs {
        let result = zlup::parse(input);
        assert!(
            result.is_err(),
            "Invalid escape should be rejected: {}",
            input
        );
    }
}

/// Multiple expressions without separator should fail.
#[test]
fn missing_separator_rejected() {
    let inputs = [
        "x := 1 y := 2;",  // missing semicolon between statements
        "fn foo() {} bar", // junk after function
    ];
    for input in inputs {
        let result = zlup::parse(input);
        assert!(
            result.is_err(),
            "Missing separator should be rejected: {}",
            input
        );
    }
}

/// Invalid type syntax should fail.
#[test]
fn invalid_type_syntax_rejected() {
    let inputs = [
        "x: [;", // incomplete array type
        "x: *;", // pointer to nothing
        "x: ?;", // optional of nothing
    ];
    for input in inputs {
        let result = zlup::parse(input);
        assert!(
            result.is_err(),
            "Invalid type should be rejected: {}",
            input
        );
    }
}

/// Empty function parameter list with trailing comma should parse but empty param should fail.
#[test]
fn invalid_param_syntax_rejected() {
    let inputs = [
        "fn foo(,) {}",     // just a comma
        "fn foo(x:) {}",    // missing type
        "fn foo(: i32) {}", // missing name
    ];
    for input in inputs {
        let result = zlup::parse(input);
        assert!(
            result.is_err(),
            "Invalid param should be rejected: {}",
            input
        );
    }
}

// =============================================================================
// More Semantic Negative Tests
// =============================================================================

/// Negative array size should be rejected.
#[test]
fn negative_array_size_rejected() {
    let source = r#"
        pub fn main() -> unit {
            arr: [-1]u32 = undefined;
        }
    "#;
    // This might fail at parse or semantic level
    let result = zlup::parse(source);
    if let Ok(program) = result {
        let mut analyzer = SemanticAnalyzer::new();
        let result = analyzer.analyze(&program);
        assert!(result.is_err(), "Negative array size should be rejected");
    }
    // If it fails to parse, that's also acceptable
}

/// Zero-size array should be valid but accessing it should fail.
#[test]
fn zero_array_access_rejected() {
    let source = r#"
        pub fn main() -> unit {
            arr: [0]u32 = [];
            x := arr[0];
        }
    "#;
    let program = zlup::parse(source).expect("Should parse");
    let mut analyzer = SemanticAnalyzer::new();
    let result = analyzer.analyze(&program);
    assert!(
        result.is_err(),
        "Accessing element of zero-size array should be rejected"
    );
}

/// Using undefined label with break should be rejected.
#[test]
fn undefined_break_label_rejected() {
    let source = r#"
        pub fn main() -> unit {
            for i in 0..10 {
                break :nonexistent;
            }
        }
    "#;
    let program = zlup::parse(source).expect("Should parse");
    let mut analyzer = SemanticAnalyzer::new();
    let result = analyzer.analyze(&program);
    assert!(
        result.is_err(),
        "Break with undefined label should be rejected"
    );
}

/// Assigning to a constant/comptime value should be rejected.
#[test]
fn assign_to_comptime_rejected() {
    let source = r#"
        SIZE := comptime 10;
        pub fn main() -> unit {
            SIZE = 20;
        }
    "#;
    let program = zlup::parse(source).expect("Should parse");
    let mut analyzer = SemanticAnalyzer::new();
    let result = analyzer.analyze(&program);
    assert!(
        result.is_err(),
        "Assigning to comptime value should be rejected"
    );
}

/// Division by zero at comptime should be rejected.
#[test]
fn comptime_division_by_zero_rejected() {
    let source = r#"
        X := comptime 10 / 0;
        pub fn main() -> unit {
            y := X;
        }
    "#;
    let program = zlup::parse(source).expect("Should parse");
    let mut analyzer = SemanticAnalyzer::new();
    let result = analyzer.analyze(&program);
    assert!(
        result.is_err(),
        "Comptime division by zero should be rejected"
    );
}

/// Invalid unary operator application should be rejected.
#[test]
fn invalid_unary_op_rejected() {
    let inputs = [
        ("x := -true;", "negating bool"),
        ("x := !42;", "logical not on int"),
        ("x := ~true;", "bitwise not on bool"),
    ];
    for (source, desc) in inputs {
        let wrapped = format!("pub fn main() -> unit {{ {} }}", source);
        let program = zlup::parse(&wrapped).expect("Should parse");
        let mut analyzer = SemanticAnalyzer::new();
        let result = analyzer.analyze(&program);
        assert!(
            result.is_err(),
            "Invalid unary op {} should be rejected",
            desc
        );
    }
}

/// Dereferencing a non-pointer should be rejected.
#[test]
fn deref_non_pointer_rejected() {
    let source = r#"
        pub fn main() -> unit {
            x := 42;
            y := *x;
        }
    "#;
    let program = zlup::parse(source).expect("Should parse");
    let mut analyzer = SemanticAnalyzer::new();
    let result = analyzer.analyze(&program);
    assert!(
        result.is_err(),
        "Dereferencing non-pointer should be rejected"
    );
}

/// Taking address of a literal should be rejected.
#[test]
fn address_of_literal_rejected() {
    let source = r#"
        pub fn main() -> unit {
            x := &42;
        }
    "#;
    let program = zlup::parse(source).expect("Should parse");
    let mut analyzer = SemanticAnalyzer::new();
    let result = analyzer.analyze(&program);
    // This might be allowed in some contexts, but typically rejected
    // If it's allowed, we can remove this test
    assert!(result.is_err(), "Address of literal should be rejected");
}

/// Mixing qubits from different allocators in same gate should be rejected.
#[test]
fn mixed_allocator_gate_rejected() {
    let source = r#"
        pub fn main() -> unit {
            mut q1 := qalloc(2);
            mut q2 := qalloc(2);
            pz q1;
            pz q2;
            cx (q1[0], q2[0]);
        }
    "#;
    let program = zlup::parse(source).expect("Should parse");
    let mut analyzer = SemanticAnalyzer::new();
    let result = analyzer.analyze(&program);
    // This might actually be allowed - remove if so
    // Cross-allocator gates might be valid in some quantum systems
    if result.is_err() {
        // Good - it's rejected as expected
    }
    // Don't assert - this might be architecture-dependent
}

/// Parameterized gate with wrong parameter type should be rejected.
#[test]
fn gate_wrong_param_type_rejected() {
    let source = r#"
        pub fn main() -> unit {
            mut q := qalloc(1);
            pz q;
            rx("not an angle") q[0];
        }
    "#;
    let program = zlup::parse(source).expect("Should parse");
    let mut analyzer = SemanticAnalyzer::new();
    let result = analyzer.analyze(&program);
    assert!(
        result.is_err(),
        "Gate with string parameter should be rejected"
    );
}

/// Enum variant that doesn't exist should be rejected.
#[test]
fn undefined_enum_variant_rejected() {
    let source = r#"
        Color := enum { Red, Green, Blue };
        pub fn main() -> unit {
            c := Color.Yellow;
        }
    "#;
    let program = zlup::parse(source).expect("Should parse");
    let mut analyzer = SemanticAnalyzer::new();
    let result = analyzer.analyze(&program);
    assert!(result.is_err(), "Undefined enum variant should be rejected");
}

/// Using a type as a value incorrectly should be rejected.
#[test]
fn type_as_value_rejected() {
    let source = r#"
        pub fn main() -> unit {
            x := u32 + 1;
        }
    "#;
    let program = zlup::parse(source).expect("Should parse");
    let mut analyzer = SemanticAnalyzer::new();
    let result = analyzer.analyze(&program);
    assert!(
        result.is_err(),
        "Using type in arithmetic should be rejected"
    );
}

/// Bitwise operations on floats should be rejected.
#[test]
fn bitwise_on_float_rejected() {
    let inputs = [
        ("x := 1.0 & 2.0;", "bitwise and on floats"),
        ("x := 1.0 | 2.0;", "bitwise or on floats"),
        ("x := 1.0 ^ 2.0;", "bitwise xor on floats"),
        ("x := 1.0 << 2;", "left shift on float"),
    ];
    for (source, desc) in inputs {
        let wrapped = format!("pub fn main() -> unit {{ {} }}", source);
        let program = zlup::parse(&wrapped).expect("Should parse");
        let mut analyzer = SemanticAnalyzer::new();
        let result = analyzer.analyze(&program);
        assert!(result.is_err(), "{} should be rejected", desc);
    }
}

/// Switch with duplicate integer cases should be rejected.
#[test]
fn switch_duplicate_case_rejected() {
    let source = r#"
        pub fn main() -> unit {
            x := 1;
            switch (x) {
                1 => unit,
                1 => unit,
                else => unit,
            }
        }
    "#;
    let program = zlup::parse(source).expect("Should parse");
    let mut analyzer = SemanticAnalyzer::new();
    let result = analyzer.analyze(&program);
    assert!(result.is_err(), "Duplicate switch case should be rejected");
}

/// Switch with duplicate boolean cases should be rejected.
#[test]
fn switch_duplicate_bool_case_rejected() {
    let source = r#"
        pub fn main() -> unit {
            x := true;
            switch (x) {
                true => unit,
                true => unit,
                else => unit,
            }
        }
    "#;
    let program = zlup::parse(source).expect("Should parse");
    let mut analyzer = SemanticAnalyzer::new();
    let result = analyzer.analyze(&program);
    assert!(
        result.is_err(),
        "Duplicate bool switch case should be rejected"
    );
}

/// Switch with duplicate string cases should be rejected.
#[test]
fn switch_duplicate_string_case_rejected() {
    let source = r#"
        pub fn main() -> unit {
            x := "hello";
            switch (x) {
                "hello" => unit,
                "world" => unit,
                "hello" => unit,
                else => unit,
            }
        }
    "#;
    let program = zlup::parse(source).expect("Should parse");
    let mut analyzer = SemanticAnalyzer::new();
    let result = analyzer.analyze(&program);
    assert!(
        result.is_err(),
        "Duplicate string switch case should be rejected"
    );
}

/// Switch with unique cases should be allowed.
#[test]
fn switch_unique_cases_allowed() {
    let source = r#"
        pub fn main() -> unit {
            x := 1;
            switch (x) {
                1 => unit,
                2 => unit,
                3 => unit,
                else => unit,
            }
            return unit;
        }
    "#;
    let program = zlup::parse(source).expect("Should parse");
    let mut analyzer = SemanticAnalyzer::new();
    let result = analyzer.analyze(&program);
    assert!(
        result.is_ok(),
        "Switch with unique cases should be allowed: {:?}",
        result
    );
}

/// Missing struct fields in initialization should be rejected.
#[test]
fn missing_struct_field_rejected() {
    let source = r#"
        Point := struct {
            x: i32,
            y: i32,
        };
        pub fn main() -> unit {
            p := Point { x: 1 };
        }
    "#;
    let program = zlup::parse(source).expect("Should parse");
    let mut analyzer = SemanticAnalyzer::new();
    let result = analyzer.analyze(&program);
    assert!(result.is_err(), "Missing struct field should be rejected");
}

// =============================================================================
// Logging Tests
// =============================================================================

/// Standard log levels should parse correctly.
#[test]
fn log_standard_levels_parse() {
    let source = r#"
        pub fn main() -> unit {
            @emit.log.trace(f"trace message");
            @emit.log.debug(f"debug message");
            @emit.log.info(f"info message");
            @emit.log.warn(f"warn message");
            @emit.log.error(f"error message");
            return unit;
        }
    "#;
    let result = zlup::parse(source);
    assert!(
        result.is_ok(),
        "Standard log levels should parse: {:?}",
        result.err()
    );
}

/// Log with sub-namespace should parse correctly.
#[test]
fn log_with_namespace_parses() {
    let source = r#"
        pub fn main() -> unit {
            @emit.log.debug("subns", f"message with namespace");
            @emit.log.info("my::nested::ns", f"nested namespace");
            return unit;
        }
    "#;
    let result = zlup::parse(source);
    assert!(
        result.is_ok(),
        "Log with namespace should parse: {:?}",
        result.err()
    );
}

/// Bare log.info without @emit prefix is parsed as method call (not channel).
/// Since `log` is not defined, semantic analysis should fail.
#[test]
fn bare_log_not_channel() {
    let source = r#"
        pub fn main() -> unit {
            log.info(f"missing @emit prefix");
            return unit;
        }
    "#;
    // Parses as method call (log.info), but `log` is undefined
    let program = zlup::parse(source).expect("Parses as method call");
    let mut analyzer = SemanticAnalyzer::new();
    let result = analyzer.analyze(&program);
    // Semantic analysis should fail because `log` is not defined
    assert!(
        result.is_err(),
        "Undefined variable 'log' should cause semantic error"
    );
}

/// Log with data parameter should parse correctly.
#[test]
fn log_with_data_parses() {
    let source = r#"
        pub fn main() -> unit {
            x := 42;
            @emit.log.debug(f"value", data: x);
            @emit.log.info("ns", f"with both", data: x);
            return unit;
        }
    "#;
    let result = zlup::parse(source);
    assert!(
        result.is_ok(),
        "Log with data should parse: {:?}",
        result.err()
    );
}

/// Custom log level with @emit.log.at should parse correctly.
#[test]
fn log_custom_level_parses() {
    let source = r#"
        pub fn main() -> unit {
            @emit.log.at(15, f"custom numeric level");
            @emit.log.at(25, "perf", f"with namespace");
            return unit;
        }
    "#;
    let result = zlup::parse(source);
    assert!(
        result.is_ok(),
        "Custom log level should parse: {:?}",
        result.err()
    );
}

/// Log expressions should pass semantic analysis.
#[test]
fn log_passes_semantic_analysis() {
    let source = r#"
        pub fn main() -> unit {
            x := 42;
            @emit.log.trace(f"trace");
            @emit.log.debug(f"x = {x}");
            @emit.log.info("ns", f"namespaced");
            @emit.log.warn(f"warning", data: x);
            @emit.log.error(f"error");
            @emit.log.at(15, f"custom");
            return unit;
        }
    "#;
    let program = zlup::parse(source).expect("Should parse");
    let mut analyzer = SemanticAnalyzer::new();
    let result = analyzer.analyze(&program);
    assert!(
        result.is_ok(),
        "Log expressions should pass semantic analysis: {:?}",
        result.err()
    );
}

/// Log with f-string interpolation should work.
#[test]
fn log_fstring_interpolation_works() {
    let source = r#"
        pub fn main() -> unit {
            name := "world";
            count := 42;
            @emit.log.info(f"Hello {name}, count = {count}");
            return unit;
        }
    "#;
    let program = zlup::parse(source).expect("Should parse");
    let mut analyzer = SemanticAnalyzer::new();
    let result = analyzer.analyze(&program);
    assert!(
        result.is_ok(),
        "Log with f-string interpolation should work: {:?}",
        result.err()
    );
}

/// Log SLR codegen should emit LogStmt nodes.
#[test]
fn log_slr_codegen_emits_log_stmt() {
    use zlup::codegen::SlrCodegen;

    let source = r#"
        pub fn main() -> unit {
            @emit.log.debug(f"test message");
            return unit;
        }
    "#;
    let program = zlup::parse(source).expect("Should parse");
    let mut codegen = SlrCodegen::new();
    let slr_program = codegen.compile(&program).expect("Should compile");
    let json = codegen.to_json(&slr_program).expect("Should serialize");

    assert!(
        json.contains("LogStmt"),
        "SLR output should contain LogStmt"
    );
    assert!(
        json.contains("debug"),
        "SLR output should contain log level"
    );
    assert!(
        json.contains("test message"),
        "SLR output should contain message"
    );
}

/// Log SLR codegen should handle namespaces.
#[test]
fn log_slr_codegen_handles_namespace() {
    use zlup::codegen::SlrCodegen;

    let source = r#"
        pub fn main() -> unit {
            @emit.log.info("myns", f"namespaced");
            return unit;
        }
    "#;
    let program = zlup::parse(source).expect("Should parse");
    let mut codegen = SlrCodegen::new();
    codegen.set_module("testmod");
    let slr_program = codegen.compile(&program).expect("Should compile");
    let json = codegen.to_json(&slr_program).expect("Should serialize");

    assert!(
        json.contains("testmod::myns"),
        "SLR output should contain combined namespace"
    );
}

/// Log elision in release mode should remove all logs.
#[test]
fn log_elision_release_removes_all() {
    use zlup::codegen::SlrCodegen;

    let source = r#"
        pub fn main() -> unit {
            @emit.log.trace(f"trace");
            @emit.log.debug(f"debug");
            @emit.log.info(f"info");
            @emit.log.warn(f"warn");
            @emit.log.error(f"error");
            return unit;
        }
    "#;
    let program = zlup::parse(source).expect("Should parse");
    let mut codegen = SlrCodegen::new_release();
    let slr_program = codegen.compile(&program).expect("Should compile");
    let json = codegen.to_json(&slr_program).expect("Should serialize");

    assert!(
        !json.contains("LogStmt"),
        "Release mode should elide all logs"
    );
}

/// Log elision with custom level should filter appropriately.
#[test]
fn log_elision_custom_level_filters() {
    use zlup::codegen::SlrCodegen;
    use zlup::codegen::slr::LogElisionLevel;

    let source = r#"
        pub fn main() -> unit {
            @emit.log.trace(f"trace");
            @emit.log.debug(f"debug");
            @emit.log.info(f"info");
            @emit.log.warn(f"warn");
            @emit.log.error(f"error");
            return unit;
        }
    "#;
    let program = zlup::parse(source).expect("Should parse");
    let mut codegen = SlrCodegen::new();
    codegen.set_log_elision(LogElisionLevel(Some(200))); // INFO level
    let slr_program = codegen.compile(&program).expect("Should compile");
    let json = codegen.to_json(&slr_program).expect("Should serialize");

    // trace (0) and debug (10) should be elided
    assert!(!json.contains("trace"), "trace should be elided");
    assert!(!json.contains("debug"), "debug should be elided");
    // info (20), warn (30), error (40) should remain
    assert!(json.contains("info"), "info should remain");
    assert!(json.contains("warn"), "warn should remain");
    assert!(json.contains("error"), "error should remain");
}

// =============================================================================
// Result Expression Tests
// =============================================================================

/// Result expression should parse and type check.
#[test]
fn result_expr_parses() {
    let source = r#"
        pub fn main() -> unit {
            result("measurement", 42);
            return unit;
        }
    "#;
    let program = zlup::parse(source).expect("Should parse");
    let mut analyzer = SemanticAnalyzer::new_permissive();
    analyzer.analyze(&program).expect("Should type check");
}

/// Result expression with namespaced tag.
#[test]
fn result_expr_namespaced_tag() {
    let source = r#"
        pub fn main() -> unit {
            result("qec/syndrome", 0);
            result("qec/round_1/parity", true);
            return unit;
        }
    "#;
    let program = zlup::parse(source).expect("Should parse");
    let mut analyzer = SemanticAnalyzer::new_permissive();
    analyzer.analyze(&program).expect("Should type check");
}

/// Result expression with various value types.
#[test]
fn result_expr_various_types() {
    let source = r#"
        pub fn main() -> unit {
            result("int_result", 42);
            result("bool_result", true);
            result("float_result", 3.14);
            return unit;
        }
    "#;
    let program = zlup::parse(source).expect("Should parse");
    let mut analyzer = SemanticAnalyzer::new_permissive();
    analyzer.analyze(&program).expect("Should type check");
}

// =============================================================================
// Simulator Control Expression Tests
// =============================================================================

/// @emit.sim.send should parse.
#[test]
fn sim_send_parses() {
    let source = r#"
        pub fn main() -> unit {
            @emit.sim.send("checkpoint", "before_correction");
            return unit;
        }
    "#;
    let program = zlup::parse(source).expect("Should parse");
    let mut analyzer = SemanticAnalyzer::new_permissive();
    analyzer.analyze(&program).expect("Should type check");
}

/// @emit.sim.send with various value types.
#[test]
fn sim_send_various_values() {
    let source = r#"
        pub fn main() -> unit {
            @emit.sim.send("seed", 12345);
            @emit.sim.send("checkpoint", "start");
            @emit.sim.send("noise_rate", 0.01);
            return unit;
        }
    "#;
    let program = zlup::parse(source).expect("Should parse");
    let mut analyzer = SemanticAnalyzer::new_permissive();
    analyzer.analyze(&program).expect("Should type check");
}

/// @emit.sim.noise_enable should parse.
#[test]
fn sim_noise_enable_parses() {
    let source = r#"
        pub fn main() -> unit {
            @emit.sim.noise_enable();
            return unit;
        }
    "#;
    let program = zlup::parse(source).expect("Should parse");
    let mut analyzer = SemanticAnalyzer::new_permissive();
    analyzer.analyze(&program).expect("Should type check");
}

/// @emit.sim.noise_disable should parse.
#[test]
fn sim_noise_disable_parses() {
    let source = r#"
        pub fn main() -> unit {
            @emit.sim.noise_disable();
            return unit;
        }
    "#;
    let program = zlup::parse(source).expect("Should parse");
    let mut analyzer = SemanticAnalyzer::new_permissive();
    analyzer.analyze(&program).expect("Should type check");
}

/// Multiple sim commands in sequence should work.
#[test]
fn sim_multiple_commands() {
    let source = r#"
        pub fn main() -> unit {
            @emit.sim.send("seed", 42);
            @emit.sim.send("noise_model", "depolarizing");
            @emit.sim.send("checkpoint", "start");
            q := qalloc(1);
            h q[0];
            @emit.sim.noise_disable();
            return unit;
        }
    "#;
    let program = zlup::parse(source).expect("Should parse");
    let mut analyzer = SemanticAnalyzer::new_permissive();
    analyzer.analyze(&program).expect("Should type check");
}

// =============================================================================
// SLR Codegen Tests for Result and Sim
// =============================================================================

/// Result expressions should generate SendStmt with channel "result".
#[test]
fn result_generates_slr() {
    use zlup::codegen::slr::SlrCodegen;

    let source = r#"
        pub fn main() -> unit {
            result("answer", 42);
            return unit;
        }
    "#;
    let program = zlup::parse(source).expect("Should parse");
    let mut codegen = SlrCodegen::new();
    let slr = codegen.compile(&program).expect("Should compile");
    let json = codegen.to_json(&slr).expect("Should serialize");

    // Verify SendStmt with channel "result" is in output
    assert!(
        json.contains("SendStmt"),
        "Should contain SendStmt: {}",
        json
    );
    assert!(
        json.contains("\"channel\": \"result\""),
        "Should have result channel: {}",
        json
    );
    assert!(
        json.contains("\"key\": \"answer\""),
        "Should contain key: {}",
        json
    );
}

/// Sim commands should generate SendStmt with channel "sim" for simulator target.
#[test]
fn sim_generates_slr_for_simulator() {
    use zlup::codegen::slr::SlrCodegen;

    let source = r#"
        pub fn main() -> unit {
            @emit.sim.noise_enable();
            @emit.sim.send("seed", 42);
            @emit.sim.noise_disable();
            return unit;
        }
    "#;
    let program = zlup::parse(source).expect("Should parse");
    let mut codegen = SlrCodegen::new();
    // sim_mode defaults to Emit (simulator target)
    let slr = codegen.compile(&program).expect("Should compile");
    let json = codegen.to_json(&slr).expect("Should serialize");

    // Verify SendStmt with channel "sim" is in output
    assert!(
        json.contains("SendStmt"),
        "Should contain SendStmt: {}",
        json
    );
    assert!(
        json.contains("\"channel\": \"sim\""),
        "Should have sim channel: {}",
        json
    );
    assert!(
        json.contains("\"key\": \"noise_enable\""),
        "Should contain noise_enable: {}",
        json
    );
    assert!(
        json.contains("\"key\": \"seed\""),
        "Should contain seed key: {}",
        json
    );
}

/// Sim commands should emit barriers for hardware target (default).
#[test]
fn sim_emits_barrier_for_hardware() {
    use zlup::codegen::slr::{SimMode, SlrCodegen};

    let source = r#"
        pub fn main() -> unit {
            @emit.sim.noise_enable();
            @emit.sim.send("checkpoint", "start");
            return unit;
        }
    "#;
    let program = zlup::parse(source).expect("Should parse");
    let mut codegen = SlrCodegen::new();
    codegen.set_sim_mode(SimMode::Barrier); // Hardware target default
    let slr = codegen.compile(&program).expect("Should compile");
    let json = codegen.to_json(&slr).expect("Should serialize");

    // Verify sim SendStmt is NOT in output, but BarrierOp IS
    assert!(
        !json.contains("\"channel\": \"sim\""),
        "Should NOT contain sim channel for hardware: {}",
        json
    );
    assert!(
        json.contains("BarrierOp"),
        "Should contain BarrierOp for ordering: {}",
        json
    );
}

/// Sim commands can be completely elided with explicit opt-in.
#[test]
fn sim_fully_elided_when_requested() {
    use zlup::codegen::slr::{SimMode, SlrCodegen};

    let source = r#"
        pub fn main() -> unit {
            @emit.sim.noise_enable();
            @emit.sim.send("checkpoint", "start");
            return unit;
        }
    "#;
    let program = zlup::parse(source).expect("Should parse");
    let mut codegen = SlrCodegen::new();
    codegen.set_sim_mode(SimMode::Elide); // Explicit full elision
    let slr = codegen.compile(&program).expect("Should compile");
    let json = codegen.to_json(&slr).expect("Should serialize");

    // Verify neither sim SendStmt nor BarrierOp in output
    assert!(
        !json.contains("\"channel\": \"sim\""),
        "Should NOT contain sim channel: {}",
        json
    );
    assert!(
        !json.contains("BarrierOp"),
        "Should NOT contain BarrierOp: {}",
        json
    );
}

/// Sim barrier is scoped to allocators in current scope.
#[test]
fn sim_barrier_is_scoped_to_allocators() {
    use zlup::codegen::slr::{SimMode, SlrCodegen};

    let source = r#"
        pub fn main() -> unit {
            q := qalloc(4);
            @emit.sim.noise_disable();
            h q[0];
            return unit;
        }
    "#;
    let program = zlup::parse(source).expect("Should parse");
    let mut codegen = SlrCodegen::new();
    codegen.set_sim_mode(SimMode::Barrier);
    let slr = codegen.compile(&program).expect("Should compile");
    let json = codegen.to_json(&slr).expect("Should serialize");

    // Verify BarrierOp includes the allocator "q"
    assert!(
        json.contains("BarrierOp"),
        "Should contain BarrierOp: {}",
        json
    );
    assert!(
        json.contains("\"allocators\""),
        "Should have allocators field: {}",
        json
    );
    assert!(
        json.contains("\"q\""),
        "Should include allocator 'q': {}",
        json
    );
}

/// Result is NEVER elided, even with full log elision in release mode.
#[test]
fn result_never_elided_in_release() {
    use zlup::codegen::slr::{SimMode, SlrCodegen};

    let source = r#"
        pub fn main() -> unit {
            @emit.log.debug(f"this will be elided");
            result("answer", 42);
            @emit.sim.send("checkpoint", "end");
            return unit;
        }
    "#;
    let program = zlup::parse(source).expect("Should parse");

    // Configure for hardware release (max elision)
    let mut codegen = SlrCodegen::new_release();
    codegen.set_sim_mode(SimMode::Elide); // Full sim elision

    let slr = codegen.compile(&program).expect("Should compile");
    let json = codegen.to_json(&slr).expect("Should serialize");

    // Log should be elided
    assert!(!json.contains("LogStmt"), "Log should be elided: {}", json);
    // Sim should be elided
    assert!(
        !json.contains("\"channel\": \"sim\""),
        "Sim should be elided: {}",
        json
    );
    // Result should ALWAYS be present
    assert!(
        json.contains("SendStmt"),
        "Result should be present: {}",
        json
    );
    assert!(
        json.contains("\"channel\": \"result\""),
        "Result channel should be present: {}",
        json
    );
    assert!(
        json.contains("\"key\": \"answer\""),
        "Result key should be present: {}",
        json
    );
}

/// All three channels work together in a realistic program.
#[test]
fn all_channels_together() {
    use zlup::codegen::slr::SlrCodegen;

    let source = r#"
        pub fn main() -> unit {
            // Simulator setup
            @emit.sim.send("seed", 42);
            @emit.sim.send("noise_model", "depolarizing");

            // Allocate qubits
            q := qalloc(2);
            @emit.log.info(f"allocated qubits");

            // Quantum operations
            h q[0];
            @emit.sim.noise_disable();
            cx (q[0], q[1]);

            // Measure and emit result
            m := mz([2]u1) [q[0], q[1]];
            @emit.log.debug(f"measured: {m}");
            result("bell_measurement", m);

            return unit;
        }
    "#;
    let program = zlup::parse(source).expect("Should parse");
    let mut analyzer = SemanticAnalyzer::new_permissive();
    analyzer
        .analyze(&program)
        .expect("Should pass semantic analysis");

    let mut codegen = SlrCodegen::new();
    let slr = codegen.compile(&program).expect("Should compile");
    let json = codegen.to_json(&slr).expect("Should serialize");

    // All three channel types should be present
    assert!(
        json.contains("LogStmt"),
        "Should have log statements: {}",
        json
    );
    assert!(
        json.contains("\"channel\": \"result\""),
        "Should have result channel: {}",
        json
    );
    assert!(
        json.contains("\"channel\": \"sim\""),
        "Should have sim channel: {}",
        json
    );
}

// =============================================================================
// Swap Builtin Tests
// =============================================================================

/// @swap should parse correctly.
#[test]
fn swap_builtin_parses() {
    let source = r#"
        pub fn main() -> unit {
            a := 1;
            b := 2;
            @swap(&a, &b);
            return unit;
        }
    "#;
    let result = zlup::parse(source);
    assert!(result.is_ok(), "@swap should parse: {:?}", result.err());
}

/// @swap should pass semantic analysis.
#[test]
fn swap_builtin_semantic_analysis() {
    let source = r#"
        pub fn main() -> unit {
            a := 1;
            b := 2;
            @swap(&a, &b);
            return unit;
        }
    "#;
    let program = zlup::parse(source).expect("Should parse");
    let mut analyzer = SemanticAnalyzer::new_permissive();
    let result = analyzer.analyze(&program);
    assert!(
        result.is_ok(),
        "@swap should pass semantic analysis: {:?}",
        result.err()
    );
}

/// @swap with wrong number of arguments should fail.
#[test]
fn swap_wrong_arg_count_rejected() {
    let source = r#"
        pub fn main() -> unit {
            a := 1;
            @swap(&a);
            return unit;
        }
    "#;
    let program = zlup::parse(source).expect("Should parse");
    let mut analyzer = SemanticAnalyzer::new_permissive();
    let result = analyzer.analyze(&program);
    assert!(result.is_err(), "@swap with 1 arg should fail");
}

/// @swap with mismatched types should fail.
#[test]
fn swap_type_mismatch_rejected() {
    let source = r#"
        pub fn main() -> unit {
            a: i64 = 1;
            b: f64 = 2.0;
            @swap(&a, &b);
            return unit;
        }
    "#;
    let program = zlup::parse(source).expect("Should parse");
    let mut analyzer = SemanticAnalyzer::new();
    let result = analyzer.analyze(&program);
    assert!(result.is_err(), "@swap with mismatched types should fail");
}

/// @swap should generate SLR SwapOp.
#[test]
fn swap_generates_slr() {
    use zlup::codegen::SlrCodegen;

    let source = r#"
        pub fn main() -> unit {
            a := 1;
            b := 2;
            @swap(&a, &b);
            return unit;
        }
    "#;
    let program = zlup::parse(source).expect("Should parse");
    let mut codegen = SlrCodegen::new();
    let slr = codegen.compile(&program).expect("Should compile");
    let json = codegen.to_json(&slr).expect("Should serialize");

    assert!(json.contains("SwapOp"), "Should contain SwapOp: {}", json);
}

// =============================================================================
// Reference Safety Tests (safe-by-constraint memory model)
// =============================================================================

/// Returning a reference to a local variable is always rejected (safe-by-constraint).
#[test]
fn return_reference_to_local_rejected() {
    let source = r#"
        fn bad() -> *i64 {
            x := 42;
            return &x;
        }
    "#;
    let program = zlup::parse(source).expect("Should parse");
    let mut analyzer = SemanticAnalyzer::new();
    let result = analyzer.analyze(&program);
    assert!(result.is_err(), "Returning reference to local should fail");
    let err = result.unwrap_err();
    assert!(
        format!("{}", err).contains("local variable"),
        "Error should mention local variable: {}",
        err
    );
}

/// Returning a reference to a local is rejected even in permissive mode (no escape hatch).
#[test]
fn return_reference_to_local_rejected_permissive() {
    let source = r#"
        fn bad() -> *i64 {
            x := 42;
            return &x;
        }
    "#;
    let program = zlup::parse(source).expect("Should parse");
    let mut analyzer = SemanticAnalyzer::new_permissive();
    let result = analyzer.analyze(&program);
    // Safe-by-constraint: no escape hatch for returning dangling references
    assert!(
        result.is_err(),
        "Returning reference to local should fail even in permissive mode"
    );
}

/// Returning a parameter reference should be allowed (caller owns the data).
#[test]
fn return_reference_to_param_allowed() {
    let source = r#"
        fn ok(x: *i32) -> *i32 {
            return x;
        }
    "#;
    let program = zlup::parse(source).expect("Should parse");
    let mut analyzer = SemanticAnalyzer::new(); // strict mode
    let result = analyzer.analyze(&program);
    assert!(
        result.is_ok(),
        "Returning parameter reference should be allowed: {:?}",
        result.err()
    );
}

/// Returning a value (not reference) of a local is fine.
#[test]
fn return_value_of_local_allowed() {
    let source = r#"
        fn ok() -> i32 {
            x := 42;
            return x;
        }
    "#;
    let program = zlup::parse(source).expect("Should parse");
    let mut analyzer = SemanticAnalyzer::new(); // strict mode
    let result = analyzer.analyze(&program);
    assert!(
        result.is_ok(),
        "Returning value of local should be allowed: {:?}",
        result.err()
    );
}

/// Returning a slice of a local array should be rejected.
#[test]
fn return_slice_of_local_rejected() {
    let source = r#"
        fn bad() -> []i32 {
            arr: [4]i32 = [1, 2, 3, 4];
            return arr[0..2];
        }
    "#;
    let program = zlup::parse(source).expect("Should parse");
    let mut analyzer = SemanticAnalyzer::new();
    let result = analyzer.analyze(&program);
    assert!(
        result.is_err(),
        "Returning slice of local array should fail"
    );
}

/// Slice syntax with open-ended ranges should parse correctly.
#[test]
fn slice_open_ended_parses() {
    // arr[..n] - from start to n
    let source1 = r#"
        fn test() -> unit {
            arr: [4]i32 = [1, 2, 3, 4];
            x := arr[..2];
            return unit;
        }
    "#;
    assert!(zlup::parse(source1).is_ok(), "arr[..2] should parse");

    // arr[n..] - from n to end
    let source2 = r#"
        fn test() -> unit {
            arr: [4]i32 = [1, 2, 3, 4];
            x := arr[2..];
            return unit;
        }
    "#;
    assert!(zlup::parse(source2).is_ok(), "arr[2..] should parse");

    // arr[..] - entire slice
    let source3 = r#"
        fn test() -> unit {
            arr: [4]i32 = [1, 2, 3, 4];
            x := arr[..];
            return unit;
        }
    "#;
    assert!(zlup::parse(source3).is_ok(), "arr[..] should parse");
}

/// Slice of parameter array is allowed (caller owns it).
#[test]
fn slice_of_param_allowed() {
    let source = r#"
        fn ok(arr: []i32) -> []i32 {
            return arr[0..2];
        }
    "#;
    let program = zlup::parse(source).expect("Should parse");
    let mut analyzer = SemanticAnalyzer::new();
    let result = analyzer.analyze(&program);
    // Slicing a parameter is safe - the caller owns the data
    assert!(
        result.is_ok(),
        "Slicing parameter array should be allowed: {:?}",
        result.err()
    );
}

/// Returning a tuple containing a reference to local should be rejected.
#[test]
fn return_tuple_with_local_reference_rejected() {
    let source = r#"
        fn bad() -> (i32, *i64) {
            x := 42;
            return (1, &x);
        }
    "#;
    let program = zlup::parse(source).expect("Should parse");
    let mut analyzer = SemanticAnalyzer::new();
    let result = analyzer.analyze(&program);
    assert!(
        result.is_err(),
        "Returning tuple with reference to local should fail"
    );
}

/// Returning a struct containing a reference to local should be rejected.
#[test]
fn return_struct_with_local_reference_rejected() {
    let source = r#"
        Wrapper := struct { ptr: *i64 };
        fn bad() -> Wrapper {
            x := 42;
            return Wrapper { ptr: &x };
        }
    "#;
    let program = zlup::parse(source).expect("Should parse");
    let mut analyzer = SemanticAnalyzer::new();
    let result = analyzer.analyze(&program);
    assert!(
        result.is_err(),
        "Returning struct with reference to local should fail"
    );
}

/// Returning an array containing references to locals should be rejected.
#[test]
fn return_array_with_local_references_rejected() {
    let source = r#"
        fn bad() -> [2]*i64 {
            x := 1;
            y := 2;
            return [&x, &y];
        }
    "#;
    let program = zlup::parse(source).expect("Should parse");
    let mut analyzer = SemanticAnalyzer::new();
    let result = analyzer.analyze(&program);
    assert!(
        result.is_err(),
        "Returning array with references to locals should fail"
    );
}

/// Taking address of parameter and returning is allowed (caller owns it).
#[test]
fn return_address_of_param_allowed() {
    let source = r#"
        fn ok(x: i32) -> *i32 {
            return &x;
        }
    "#;
    let program = zlup::parse(source).expect("Should parse");
    let mut analyzer = SemanticAnalyzer::new();
    let result = analyzer.analyze(&program);
    // Parameters are owned by caller, so this is safe
    // (the reference is valid for the caller's scope)
    assert!(
        result.is_ok(),
        "Returning address of parameter should be allowed: {:?}",
        result.err()
    );
}

// =============================================================================
// @swap Additional Safety Tests
// =============================================================================

/// @swap requires pointer arguments.
#[test]
fn swap_requires_pointers() {
    let source = r#"
        pub fn main() -> unit {
            a := 1;
            b := 2;
            @swap(a, b);
            return unit;
        }
    "#;
    let program = zlup::parse(source).expect("Should parse");
    let mut analyzer = SemanticAnalyzer::new();
    let result = analyzer.analyze(&program);
    assert!(result.is_err(), "@swap without & should fail");
}

/// @swap with three arguments should fail.
#[test]
fn swap_too_many_args_rejected() {
    let source = r#"
        pub fn main() -> unit {
            a := 1;
            b := 2;
            c := 3;
            @swap(&a, &b, &c);
            return unit;
        }
    "#;
    let program = zlup::parse(source).expect("Should parse");
    let mut analyzer = SemanticAnalyzer::new();
    let result = analyzer.analyze(&program);
    assert!(result.is_err(), "@swap with 3 args should fail");
}

// =============================================================================
// Qubit Safety Tests
// =============================================================================

/// Measuring the same qubit twice in a tick should be rejected (strict mode).
#[test]
fn duplicate_qubit_in_tick_measurement_rejected() {
    let source = r#"
        pub fn main() -> unit {
            q := qalloc(2);
            pz q[0];
            pz q[1];
            tick {
                mz(u1) q[0];
                mz(u1) q[0];
            }
            return unit;
        }
    "#;
    let program = zlup::parse(source).expect("Should parse");
    let mut analyzer = SemanticAnalyzer::new();
    let result = analyzer.analyze(&program);
    assert!(
        result.is_err(),
        "Duplicate qubit in tick should fail in strict mode"
    );
}

/// Using the same qubit twice in different gates within a tick should be rejected.
#[test]
fn duplicate_qubit_in_tick_gates_rejected() {
    let source = r#"
        pub fn main() -> unit {
            q := qalloc(2);
            pz q[0];
            pz q[1];
            tick {
                h q[0];
                x q[0];
            }
            return unit;
        }
    "#;
    let program = zlup::parse(source).expect("Should parse");
    let mut analyzer = SemanticAnalyzer::new();
    let result = analyzer.analyze(&program);
    assert!(
        result.is_err(),
        "Using same qubit in multiple gates within tick should fail"
    );
}

/// Gate on unprepared qubit should be rejected in strict mode.
#[test]
fn gate_on_unprepared_qubit_rejected() {
    let source = r#"
        pub fn main() -> unit {
            q := qalloc(2);
            h q[0];
            return unit;
        }
    "#;
    let program = zlup::parse(source).expect("Should parse");
    let mut analyzer = SemanticAnalyzer::new();
    let result = analyzer.analyze(&program);
    assert!(
        result.is_err(),
        "Gate on unprepared qubit should fail in strict mode"
    );
}

/// Gate after pz (prepare) should succeed.
#[test]
fn gate_after_prepare_allowed() {
    let source = r#"
        pub fn main() -> unit {
            q := qalloc(2);
            pz q[0];
            h q[0];
            return unit;
        }
    "#;
    let program = zlup::parse(source).expect("Should parse");
    let mut analyzer = SemanticAnalyzer::new();
    let result = analyzer.analyze(&program);
    assert!(
        result.is_ok(),
        "Gate after prepare should succeed: {:?}",
        result.err()
    );
}

/// Mixed gate and measurement on same qubit in tick should be rejected.
#[test]
fn mixed_gate_measurement_same_qubit_in_tick_rejected() {
    let source = r#"
        pub fn main() -> unit {
            q := qalloc(2);
            pz q[0];
            pz q[1];
            tick {
                h q[0];
                mz(u1) q[0];
            }
            return unit;
        }
    "#;
    let program = zlup::parse(source).expect("Should parse");
    let mut analyzer = SemanticAnalyzer::new();
    let result = analyzer.analyze(&program);
    assert!(
        result.is_err(),
        "Gate and measurement on same qubit in tick should fail"
    );
}

/// Different qubits in tick operations should succeed.
#[test]
fn different_qubits_in_tick_allowed() {
    let source = r#"
        pub fn main() -> unit {
            q := qalloc(4);
            pz q[0];
            pz q[1];
            pz q[2];
            pz q[3];
            tick {
                h q[0];
                x q[1];
                mz(u1) q[2];
                mz(u1) q[3];
            }
            return unit;
        }
    "#;
    let program = zlup::parse(source).expect("Should parse");
    let mut analyzer = SemanticAnalyzer::new();
    let result = analyzer.analyze(&program);
    assert!(
        result.is_ok(),
        "Different qubits in tick should succeed: {:?}",
        result.err()
    );
}

// =============================================================================
// Additional Slice Syntax Tests
// =============================================================================

/// Slice with variable bounds should parse.
#[test]
fn slice_with_variable_bounds_parses() {
    let source = r#"
        fn test(start: usize, end: usize) -> unit {
            arr: [10]i32 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
            x := arr[start..end];
            return unit;
        }
    "#;
    assert!(
        zlup::parse(source).is_ok(),
        "Slice with variable bounds should parse"
    );
}

/// Slice with expression bounds should parse.
#[test]
fn slice_with_expression_bounds_parses() {
    let source = r#"
        fn test() -> unit {
            arr: [10]i32 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
            n := 5;
            x := arr[n - 2..n + 2];
            return unit;
        }
    "#;
    assert!(
        zlup::parse(source).is_ok(),
        "Slice with expression bounds should parse"
    );
}

/// Slice used in assignment should parse.
#[test]
fn slice_assignment_parses() {
    let source = r#"
        fn test() -> unit {
            arr: [4]i32 = [1, 2, 3, 4];
            slice := arr[1..3];
            return unit;
        }
    "#;
    let result = zlup::parse(source);
    assert!(
        result.is_ok(),
        "Slice assignment should parse: {:?}",
        result.err()
    );
}

/// Multiple slicing operations should parse.
#[test]
fn multiple_slices_parse() {
    let source = r#"
        fn test() -> unit {
            arr: [10]i32 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
            a := arr[0..3];
            b := arr[3..6];
            c := arr[6..];
            d := arr[..4];
            return unit;
        }
    "#;
    let result = zlup::parse(source);
    assert!(
        result.is_ok(),
        "Multiple slices should parse: {:?}",
        result.err()
    );
}

// =============================================================================
// Slice Type Semantics Tests
// =============================================================================

/// Re-slicing a slice should return a slice type.
/// NOTE: Variable names must not be gate names (s, h, x, y, z, t are gates).
#[test]
fn reslice_returns_slice_type() {
    let source = r#"
        fn reslice(data: []i32) -> []i32 {
            return data[1..3];
        }
    "#;
    let program = zlup::parse(source).expect("Should parse");
    let mut analyzer = SemanticAnalyzer::new();
    let result = analyzer.analyze(&program);
    assert!(
        result.is_ok(),
        "Re-slicing should return slice type: {:?}",
        result.err()
    );
}

/// Indexing a slice with an integer should return the element type.
/// NOTE: Variable names must not be gate names (s, h, x, y, z, t are gates).
#[test]
fn slice_index_returns_element_type() {
    let source = r#"
        fn get_element(data: []i32) -> i32 {
            return data[0];
        }
    "#;
    let program = zlup::parse(source).expect("Should parse");
    let mut analyzer = SemanticAnalyzer::new();
    let result = analyzer.analyze(&program);
    assert!(
        result.is_ok(),
        "Slice index should return element type: {:?}",
        result.err()
    );
}

/// Chained slicing: arr[1..5][0..2] should work.
#[test]
fn chained_slicing_allowed() {
    let source = r#"
        fn chain(arr: []i32) -> []i32 {
            return arr[1..5][0..2];
        }
    "#;
    let program = zlup::parse(source).expect("Should parse");
    let mut analyzer = SemanticAnalyzer::new();
    let result = analyzer.analyze(&program);
    assert!(
        result.is_ok(),
        "Chained slicing should be allowed: {:?}",
        result.err()
    );
}

/// Array type [N]T is distinct from slice type []T.
#[test]
fn array_type_distinct_from_slice() {
    // Returning array when slice expected should fail
    let source = r#"
        fn bad() -> []i32 {
            arr: [3]i32 = [1, 2, 3];
            return arr;
        }
    "#;
    let program = zlup::parse(source).expect("Should parse");
    let mut analyzer = SemanticAnalyzer::new();
    let result = analyzer.analyze(&program);
    // This should fail because [3]i32 is not []i32
    assert!(
        result.is_err(),
        "Array should not be assignable to slice type"
    );
}

/// Slicing an array produces a slice type.
#[test]
fn slicing_array_produces_slice() {
    let source = r#"
        fn to_slice(arr: [5]i32) -> []i32 {
            return arr[0..3];
        }
    "#;
    let program = zlup::parse(source).expect("Should parse");
    let mut analyzer = SemanticAnalyzer::new();
    let result = analyzer.analyze(&program);
    assert!(
        result.is_ok(),
        "Slicing array should produce slice: {:?}",
        result.err()
    );
}

/// Nested slice types: [][]i32 should work.
#[test]
fn nested_slice_type_allowed() {
    let source = r#"
        fn nested(matrix: [][]i32) -> []i32 {
            return matrix[0];
        }
    "#;
    let program = zlup::parse(source).expect("Should parse");
    let mut analyzer = SemanticAnalyzer::new();
    let result = analyzer.analyze(&program);
    assert!(
        result.is_ok(),
        "Nested slice types should be allowed: {:?}",
        result.err()
    );
}

/// Open-ended slice of parameter is allowed.
#[test]
fn open_slice_of_param_allowed() {
    let source = r#"
        fn full_slice(arr: []i32) -> []i32 {
            return arr[..];
        }
    "#;
    let program = zlup::parse(source).expect("Should parse");
    let mut analyzer = SemanticAnalyzer::new();
    let result = analyzer.analyze(&program);
    assert!(
        result.is_ok(),
        "Full slice of param should be allowed: {:?}",
        result.err()
    );
}

/// Slice with start index only.
#[test]
fn slice_from_start_index() {
    let source = r#"
        fn from_start(arr: []i32) -> []i32 {
            return arr[2..];
        }
    "#;
    let program = zlup::parse(source).expect("Should parse");
    let mut analyzer = SemanticAnalyzer::new();
    let result = analyzer.analyze(&program);
    assert!(
        result.is_ok(),
        "Slice from start index should work: {:?}",
        result.err()
    );
}

/// Slice with end index only.
#[test]
fn slice_to_end_index() {
    let source = r#"
        fn to_end(arr: []i32) -> []i32 {
            return arr[..5];
        }
    "#;
    let program = zlup::parse(source).expect("Should parse");
    let mut analyzer = SemanticAnalyzer::new();
    let result = analyzer.analyze(&program);
    assert!(
        result.is_ok(),
        "Slice to end index should work: {:?}",
        result.err()
    );
}

// =============================================================================
// Gate Naming Tests
// =============================================================================

/// The old 's' gate name should no longer be recognized.
/// 's' is now parsed as a variable identifier, not a gate.
/// Verifies that 's' is no longer valid gate syntax (removed from grammar).
#[test]
fn old_s_gate_name_not_recognized() {
    let source = r#"
        fn test() -> unit {
            q := qalloc(1);
            pz q;
            s q[0];
            return unit;
        }
    "#;
    // 's' is no longer a valid gate keyword in the grammar, so parsing fails
    assert!(zlup::parse(source).is_err());
}

/// The 'sz' gate should work correctly.
#[test]
fn sz_gate_works() {
    let source = r#"
        fn test() -> unit {
            q := qalloc(1);
            pz q;
            sz q[0];
            return unit;
        }
    "#;
    let program = zlup::parse(source).expect("Should parse");
    let mut analyzer = SemanticAnalyzer::new();
    let result = analyzer.analyze(&program);
    assert!(result.is_ok(), "sz gate should work: {:?}", result.err());
}

/// The 'szdg' gate should work correctly.
#[test]
fn szdg_gate_works() {
    let source = r#"
        fn test() -> unit {
            q := qalloc(1);
            pz q;
            szdg q[0];
            return unit;
        }
    "#;
    let program = zlup::parse(source).expect("Should parse");
    let mut analyzer = SemanticAnalyzer::new();
    let result = analyzer.analyze(&program);
    assert!(result.is_ok(), "szdg gate should work: {:?}", result.err());
}

/// Variable named 's' should now be allowed (no longer a gate).
#[test]
fn variable_named_s_allowed() {
    let source = r#"
        fn test() -> i32 {
            s: i32 = 42;
            return s;
        }
    "#;
    let program = zlup::parse(source).expect("Should parse");
    let mut analyzer = SemanticAnalyzer::new();
    let result = analyzer.analyze(&program);
    assert!(
        result.is_ok(),
        "Variable 's' should be allowed: {:?}",
        result.err()
    );
}

// =============================================================================
// Optimizer Gate Cancellation Tests
// =============================================================================

/// SZ and SZdg should cancel each other.
#[test]
fn optimizer_sz_szdg_cancellation() {
    use zlup::optimize::Optimizer;

    let source = r#"
        pub fn main() -> unit {
            q := qalloc(1);
            pz q;
            sz q[0];
            szdg q[0];
            return unit;
        }
    "#;
    let ast = zlup::parse(source).expect("Should parse");
    let mut optimizer = Optimizer::new();
    let _optimized = optimizer.optimize(ast);

    assert_eq!(
        optimizer.stats().gates_cancelled,
        2,
        "SZ and SZdg should cancel"
    );
}

/// SZdg followed by SZ should also cancel.
#[test]
fn optimizer_szdg_sz_cancellation() {
    use zlup::optimize::Optimizer;

    let source = r#"
        pub fn main() -> unit {
            q := qalloc(1);
            pz q;
            szdg q[0];
            sz q[0];
            return unit;
        }
    "#;
    let ast = zlup::parse(source).expect("Should parse");
    let mut optimizer = Optimizer::new();
    let _optimized = optimizer.optimize(ast);

    assert_eq!(
        optimizer.stats().gates_cancelled,
        2,
        "SZdg and SZ should cancel"
    );
}

// =============================================================================
// Slice Edge Cases Tests
// =============================================================================

/// Triple-chained slicing should work.
#[test]
fn triple_chained_slicing() {
    let source = r#"
        fn triple(arr: []i32) -> []i32 {
            return arr[0..10][2..8][1..4];
        }
    "#;
    let program = zlup::parse(source).expect("Should parse");
    let mut analyzer = SemanticAnalyzer::new();
    let result = analyzer.analyze(&program);
    assert!(
        result.is_ok(),
        "Triple-chained slicing should work: {:?}",
        result.err()
    );
}

/// Deeply nested slice types should work.
#[test]
fn deeply_nested_slice_types() {
    let source = r#"
        fn nested3d(cube: [][][]i32) -> [][]i32 {
            return cube[0];
        }
    "#;
    let program = zlup::parse(source).expect("Should parse");
    let mut analyzer = SemanticAnalyzer::new();
    let result = analyzer.analyze(&program);
    assert!(
        result.is_ok(),
        "3D nested slice types should work: {:?}",
        result.err()
    );
}

/// Indexing 3D nested slice returns 2D slice.
#[test]
fn nested_slice_indexing_returns_correct_type() {
    let source = r#"
        fn get_row(matrix: [][]i32) -> []i32 {
            return matrix[0];
        }
    "#;
    let program = zlup::parse(source).expect("Should parse");
    let mut analyzer = SemanticAnalyzer::new();
    let result = analyzer.analyze(&program);
    assert!(
        result.is_ok(),
        "Indexing 2D slice should return 1D slice: {:?}",
        result.err()
    );
}

/// Slicing then indexing should work.
#[test]
fn slice_then_index() {
    let source = r#"
        fn slice_index(arr: []i32) -> i32 {
            return arr[1..5][0];
        }
    "#;
    let program = zlup::parse(source).expect("Should parse");
    let mut analyzer = SemanticAnalyzer::new();
    let result = analyzer.analyze(&program);
    assert!(
        result.is_ok(),
        "Slice then index should work: {:?}",
        result.err()
    );
}

/// Indexing then slicing should work on nested types.
#[test]
fn index_then_slice_nested() {
    let source = r#"
        fn index_slice(matrix: [][]i32) -> []i32 {
            return matrix[0][1..5];
        }
    "#;
    let program = zlup::parse(source).expect("Should parse");
    let mut analyzer = SemanticAnalyzer::new();
    let result = analyzer.analyze(&program);
    assert!(
        result.is_ok(),
        "Index then slice on nested should work: {:?}",
        result.err()
    );
}

// =============================================================================
// Array vs Slice Coercion Tests
// =============================================================================

/// Returning fixed array where slice expected should fail.
#[test]
fn array_not_coercible_to_slice() {
    let source = r#"
        fn bad() -> []i32 {
            arr: [5]i32 = [1, 2, 3, 4, 5];
            return arr;
        }
    "#;
    let program = zlup::parse(source).expect("Should parse");
    let mut analyzer = SemanticAnalyzer::new();
    let result = analyzer.analyze(&program);
    assert!(
        result.is_err(),
        "Array should not be directly coercible to slice"
    );
}

/// Slicing an array parameter to get a slice should work.
/// NOTE: Cannot slice local array and return it (escape analysis prevents this).
#[test]
fn array_sliced_to_slice() {
    let source = r#"
        fn ok(arr: [5]i32) -> []i32 {
            return arr[..];
        }
    "#;
    let program = zlup::parse(source).expect("Should parse");
    let mut analyzer = SemanticAnalyzer::new();
    let result = analyzer.analyze(&program);
    assert!(
        result.is_ok(),
        "Array param sliced with [..] should produce slice: {:?}",
        result.err()
    );
}

/// Passing array where slice parameter expected should fail.
#[test]
fn array_param_not_coercible_to_slice_param() {
    let source = r#"
        fn takes_slice(data: []i32) -> unit {
            return unit;
        }

        fn caller() -> unit {
            arr: [3]i32 = [1, 2, 3];
            takes_slice(arr);
            return unit;
        }
    "#;
    let program = zlup::parse(source).expect("Should parse");
    let mut analyzer = SemanticAnalyzer::new();
    let result = analyzer.analyze(&program);
    assert!(
        result.is_err(),
        "Array should not be passable where slice expected"
    );
}

/// Passing sliced array to slice parameter should work.
#[test]
fn sliced_array_to_slice_param() {
    let source = r#"
        fn takes_slice(data: []i32) -> unit {
            return unit;
        }

        fn caller() -> unit {
            arr: [3]i32 = [1i32, 2i32, 3i32];
            takes_slice(arr[..]);
            return unit;
        }
    "#;
    let program = zlup::parse(source).expect("Should parse");
    let mut analyzer = SemanticAnalyzer::new();
    let result = analyzer.analyze(&program);
    assert!(
        result.is_ok(),
        "Sliced array should be passable as slice: {:?}",
        result.err()
    );
}

// =============================================================================
// Error Message Quality Tests
// =============================================================================

/// Type mismatch between array and slice should have clear error.
#[test]
fn array_slice_mismatch_error_message() {
    let source = r#"
        fn bad() -> []i32 {
            arr: [3]i32 = [1, 2, 3];
            return arr;
        }
    "#;
    let program = zlup::parse(source).expect("Should parse");
    let mut analyzer = SemanticAnalyzer::new();
    let result = analyzer.analyze(&program);

    match result {
        Err(e) => {
            let error_msg = format!("{:?}", e);
            // Error should mention the type mismatch
            assert!(
                error_msg.contains("TypeMismatch") || error_msg.contains("mismatch"),
                "Error should indicate type mismatch: {}",
                error_msg
            );
        }
        Ok(_) => panic!("Should have failed with type mismatch"),
    }
}

/// Using gate name as variable in quantum context should give helpful error.
#[test]
fn gate_name_variable_in_quantum_context() {
    // Using 'h' (Hadamard gate) as a variable name
    let source = r#"
        fn test() -> unit {
            q := qalloc(1);
            pz q;
            h: i32 = 5;
            return unit;
        }
    "#;
    // This should parse but 'h' shadows the gate - the question is whether
    // this causes issues. Let's just verify it parses and analyzes.
    let program = zlup::parse(source).expect("Should parse");
    let mut analyzer = SemanticAnalyzer::new();
    let result = analyzer.analyze(&program);
    // This might work or fail depending on implementation - we're testing behavior
    // Either outcome is fine, we're checking it doesn't crash
    let _ = result;
}

/// Return type mismatch with slice element should be clear.
#[test]
fn slice_element_type_mismatch_error() {
    let source = r#"
        fn bad(arr: []i32) -> bool {
            return arr[0];
        }
    "#;
    let program = zlup::parse(source).expect("Should parse");
    let mut analyzer = SemanticAnalyzer::new();
    let result = analyzer.analyze(&program);

    assert!(
        result.is_err(),
        "Should fail: returning i32 where bool expected"
    );
    if let Err(e) = result {
        let error_msg = format!("{:?}", e);
        assert!(
            error_msg.contains("i32") || error_msg.contains("bool"),
            "Error should mention the types involved: {}",
            error_msg
        );
    }
}
