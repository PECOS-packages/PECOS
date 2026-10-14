//! End-to-end tests for Guppy to Zlup compilation.
//!
//! These tests verify the full pipeline from Guppy source to valid Zlup output.

use guppy_zlup::{lint_and_compile, lint_source};

/// Test helper to compile Guppy source and verify Zlup output.
fn compile_and_check(source: &str, expected_patterns: &[&str]) {
    let result = lint_and_compile(source, None);
    match result {
        Ok(zlup) => {
            for pattern in expected_patterns {
                assert!(
                    zlup.contains(pattern),
                    "Expected pattern '{}' not found in output:\n{}",
                    pattern,
                    zlup
                );
            }
        }
        Err(e) => panic!("Compilation failed: {:?}", e),
    }
}

/// Test helper to verify linting catches errors.
fn lint_should_error(source: &str, expected_rule: &str) {
    let result = lint_source(source, None);
    assert!(
        result.has_errors,
        "Expected lint errors but got none for rule {}",
        expected_rule
    );
    assert!(
        result
            .diagnostics
            .iter()
            .any(|d| d.rule_id == expected_rule),
        "Expected rule {} but got: {:?}",
        expected_rule,
        result
            .diagnostics
            .iter()
            .map(|d| &d.rule_id)
            .collect::<Vec<_>>()
    );
}

// =============================================================================
// Basic Function Tests
// =============================================================================

#[test]
fn test_e2e_simple_function() {
    let source = r#"
def add(a: int, b: int) -> int:
    return a + b
"#;
    compile_and_check(
        source,
        &["fn add(a: i64, b: i64)", "-> i64", "return a + b"],
    );
}

#[test]
fn test_e2e_void_function() {
    let source = r#"
def noop() -> None:
    pass
"#;
    compile_and_check(source, &["fn noop()", "-> unit", "return;"]);
}

#[test]
fn test_e2e_variable_binding() {
    let source = r#"
def compute() -> int:
    x: int = 10
    y: int = 20
    return x + y
"#;
    compile_and_check(
        source,
        &["mut x: i64 = 10", "mut y: i64 = 20", "return x + y"],
    );
}

// =============================================================================
// Control Flow Tests
// =============================================================================

#[test]
fn test_e2e_if_else() {
    let source = r#"
def abs_val(x: int) -> int:
    if x < 0:
        return -x
    else:
        return x
"#;
    compile_and_check(source, &["if (x < 0)", "return -x", "else", "return x"]);
}

#[test]
fn test_e2e_for_loop() {
    let source = r#"
def sum_range() -> int:
    total: int = 0
    for i in range(10):
        total = total + i
    return total
"#;
    // total = total + i (assignment, not binding) since total is already declared
    compile_and_check(source, &["for i in 0..10", "total = total + i"]);
}

#[test]
fn test_e2e_nested_for() {
    let source = r#"
def nested() -> int:
    sum: int = 0
    for i in range(3):
        for j in range(3):
            sum = sum + 1
    return sum
"#;
    compile_and_check(source, &["for i in 0..3", "for j in 0..3"]);
}

// =============================================================================
// Quantum Operation Tests
// =============================================================================

#[test]
fn test_e2e_single_qubit_gate() {
    let source = r#"
def apply_h() -> None:
    q = qubit[1]
    h(q[0])
"#;
    compile_and_check(source, &["qalloc(1)", "h q[0]"]);
}

#[test]
fn test_e2e_two_qubit_gate() {
    let source = r#"
def apply_cx() -> None:
    q = qubit[2]
    cx(q[0], q[1])
"#;
    compile_and_check(source, &["qalloc(2)", "cx (q[0], q[1])"]);
}

#[test]
fn test_e2e_multiple_gates() {
    let source = r#"
def bell_prep() -> None:
    q = qubit[2]
    h(q[0])
    cx(q[0], q[1])
"#;
    compile_and_check(source, &["h q[0]", "cx (q[0], q[1])"]);
}

#[test]
fn test_e2e_measure_single() {
    let source = r#"
def measure_one() -> None:
    q = qubit[1]
    h(q[0])
    m = measure(q[0])
"#;
    compile_and_check(source, &["h q[0]", "mz(u1) q[0]"]);
}

#[test]
fn test_e2e_measure_register() {
    let source = r#"
def measure_all() -> None:
    q = qubit[4]
    for i in range(4):
        h(q[i])
    m = measure(q)
"#;
    compile_and_check(source, &["qalloc(4)", "for i in 0..4", "mz([4]u1) q"]);
}

// =============================================================================
// Lint Rule Tests
// =============================================================================

#[test]
fn test_e2e_lint_unbounded_loop() {
    let source = r#"
def bad_loop() -> None:
    while True:
        pass
"#;
    lint_should_error(source, "ZLUP001");
}

#[test]
fn test_e2e_lint_recursion() {
    let source = r#"
def factorial(n: int) -> int:
    if n <= 1:
        return 1
    return n * factorial(n - 1)
"#;
    lint_should_error(source, "ZLUP002");
}

#[test]
fn test_e2e_lint_dynamic_alloc_in_loop() {
    let source = r#"
def bad_alloc() -> None:
    for i in range(10):
        items = []
        items.append(i)
"#;
    lint_should_error(source, "ZLUP003");
}

#[test]
fn test_e2e_lint_eval() {
    let source = r#"
def bad_eval() -> None:
    eval("print('hello')")
"#;
    lint_should_error(source, "ZLUP004");
}

// =============================================================================
// Type Tests
// =============================================================================

#[test]
fn test_e2e_bool_type() {
    let source = r#"
def check(x: int) -> bool:
    return x > 0
"#;
    compile_and_check(source, &["-> bool", "return x > 0"]);
}

#[test]
fn test_e2e_float_type() {
    let source = r#"
def half(x: float) -> float:
    return x / 2.0
"#;
    compile_and_check(source, &["x: f64", "-> f64"]);
}

#[test]
fn test_e2e_parameterized_gate() {
    // Note: Guppy convention is gate(qubit, angle)
    let source = r#"
def main() -> None:
    q = qubit[1]
    rz(q[0], 3.14)
"#;
    compile_and_check(source, &["qalloc(1)", "rz(3.14 / 2 turns) q[0]"]);
}

#[test]
fn test_e2e_parameterized_gate_preserves_guppy_half_turns() {
    let guppy_source = r#"
def main() -> None:
    q = qubit[1]
    rz(q[0], 3.14)
"#;
    let zlup_source = lint_and_compile(guppy_source, None).expect("Guppy compilation failed");
    let program = zlup::parse(&zlup_source).expect("generated Zlup failed to parse");
    let slr = zlup::codegen::SlrCodegen::new()
        .compile(&program)
        .expect("generated Zlup failed to lower");
    let zlup::codegen::slr::SlrStatement::Gate(gate) = &slr.body[0] else {
        panic!("expected an RZ gate");
    };
    let zlup::codegen::slr::SlrExpression::Literal(literal) = &gate.params[0] else {
        panic!("expected a literal angle");
    };
    let zlup::codegen::slr::SlrLiteralValue::Angle(turns) = &literal.value else {
        panic!("expected an angle in turns");
    };
    assert_eq!(*turns, 1.57);
}

#[test]
fn test_e2e_symbolic_parameterized_gate_is_typed_as_angle() {
    let source = r#"
def main(theta: float) -> None:
    q = qubit[1]
    rz(q[0], theta)
"#;

    compile_and_check(
        source,
        &[
            "fn main(theta: f64)",
            "__zlup_angle_theta: a64 = theta / 2 turns",
            "rz(__zlup_angle_theta) q[0]",
        ],
    );
}

#[test]
fn test_e2e_symbolic_angle_propagates_through_gate_arithmetic() {
    let source = r#"
def main(theta: float) -> None:
    q = qubit[2]
    rz(q[0], theta)
    rz(q[1], theta * 2)
"#;

    compile_and_check(
        source,
        &[
            "fn main(theta: f64)",
            "__zlup_angle_theta: a64 = theta / 2 turns",
            "rz(__zlup_angle_theta) q[0]",
            "rz(__zlup_angle_theta * 2) q[1]",
        ],
    );
}

#[test]
fn test_e2e_local_angle_propagates_through_gate_arithmetic() {
    let source = r#"
def main() -> None:
    q = qubit[2]
    x = 0.5
    rz(q[0], x)
    rz(q[1], x * 2)
"#;

    compile_and_check(
        source,
        &[
            "x := 0.5",
            "__zlup_angle_x: a64 = x / 2 turns",
            "rz(__zlup_angle_x) q[0]",
            "rz(__zlup_angle_x * 2) q[1]",
        ],
    );
}

#[test]
fn test_e2e_angle_type_propagates_to_derived_local() {
    let source = r#"
def main() -> None:
    q = qubit[1]
    x = 0.5
    rz(q[0], x)
    y = x + 1.0
"#;

    compile_and_check(
        source,
        &[
            "x := 0.5",
            "__zlup_angle_x: a64 = x / 2 turns",
            "y := x + 1",
            "__zlup_angle_y: a64 = y / 2 turns",
        ],
    );
}

#[test]
fn test_e2e_local_constant_angle_is_converted_at_its_binding() {
    let source = r#"
def main() -> None:
    q = qubit[1]
    x = 0.5
    rz(q[0], x)
"#;

    let zlup = lint_and_compile(source, None).expect("Guppy compilation failed");
    assert!(
        zlup.contains("__zlup_angle_x: a64 = x / 2 turns"),
        "generated source:\n{zlup}"
    );
    assert!(
        zlup.contains("rz(__zlup_angle_x) q[0]"),
        "generated source:\n{zlup}"
    );

    let program = zlup::parse(&zlup).expect("generated Zlup failed to parse");
    let slr = zlup::codegen::SlrCodegen::new()
        .compile(&program)
        .expect("comptime local angle should lower to SLR");
    let zlup::codegen::slr::SlrStatement::Gate(gate) = &slr.body[0] else {
        panic!("expected RZ gate");
    };
    let zlup::codegen::slr::SlrExpression::Literal(literal) = &gate.params[0] else {
        panic!("expected resolved angle");
    };
    assert!(matches!(
        literal.value,
        zlup::codegen::slr::SlrLiteralValue::Angle(0.25)
    ));
}

#[test]
fn test_e2e_angle_parameter_keeps_non_angle_uses() {
    let source = r#"
def scale(x: float) -> float:
    return x * 2.0
def main(theta: float) -> None:
    q = qubit[1]
    rz(q[0], theta)
    rz(q[0], scale(theta))
    if theta > 0.5:
        h(q[0])
"#;

    compile_and_check(
        source,
        &[
            "fn scale(x: f64)",
            "fn main(theta: f64)",
            "__zlup_angle_theta: a64 = theta / 2 turns",
            "rz(__zlup_angle_theta) q[0]",
            "rz(scale(theta) / 2 turns) q[0]",
            "if (theta > 0.5)",
        ],
    );
}

#[test]
fn test_e2e_angle_parameter_keeps_scalar_return_type() {
    let source = r#"
def echo(theta: float) -> float:
    q = qubit[1]
    rz(q[0], theta)
    return theta
"#;

    compile_and_check(
        source,
        &[
            "fn echo(theta: f64)",
            "-> f64",
            "__zlup_angle_theta: a64 = theta / 2 turns",
            "return theta",
        ],
    );
}

#[test]
fn test_e2e_integer_angle_keeps_integer_index() {
    let source = r#"
def main() -> None:
    q = qubit[4]
    i = 2
    rz(q[0], i)
    h(q[i])
"#;

    let zlup = lint_and_compile(source, None).expect("Guppy compilation failed");
    for expected in [
        "i := 2",
        "__zlup_angle_i: a64 = i / 2 turns",
        "rz(__zlup_angle_i) q[0]",
        "h q[i]",
    ] {
        assert!(zlup.contains(expected), "missing {expected:?}:\n{zlup}");
    }
}

#[test]
fn test_e2e_angle_parameter_used_only_in_arithmetic() {
    let source = r#"
def main(theta: float) -> None:
    q = qubit[1]
    rz(q[0], theta * 2)
"#;

    compile_and_check(
        source,
        &[
            "fn main(theta: f64)",
            "__zlup_angle_theta: a64 = theta / 2 turns",
            "rz(__zlup_angle_theta * 2) q[0]",
        ],
    );
}

#[test]
fn test_e2e_parameter_never_used_as_angle_stays_scalar() {
    let source = r#"
def scale(theta: float) -> float:
    return theta * 2.0
"#;

    let zlup = lint_and_compile(source, None).expect("Guppy compilation failed");
    assert!(
        zlup.contains("fn scale(theta: f64)"),
        "generated source:\n{zlup}"
    );
    assert!(
        !zlup.contains("__zlup_angle_theta"),
        "generated source:\n{zlup}"
    );
}

#[test]
fn test_e2e_integer_parameter_keeps_index_type() {
    let source = r#"
def main(i: int) -> None:
    q = qubit[4]
    rz(q[0], i)
    h(q[i])
"#;

    compile_and_check(
        source,
        &[
            "fn main(i: i64)",
            "__zlup_angle_i: a64 = i / 2 turns",
            "rz(__zlup_angle_i) q[0]",
            "h q[i]",
        ],
    );
}

#[test]
fn test_e2e_angle_value_flowing_through_function_call() {
    let source = r#"
def scale(x: float) -> float:
    return x * 2.0
def main(theta: float) -> None:
    q = qubit[1]
    phi = scale(theta)
    rz(q[0], phi)
"#;

    compile_and_check(
        source,
        &[
            "fn main(theta: f64)",
            "phi := scale(theta)",
            "__zlup_angle_phi: a64 = phi / 2 turns",
            "rz(__zlup_angle_phi) q[0]",
        ],
    );
}

#[test]
fn test_e2e_derived_angles_convert_once_per_origin() {
    let source = r#"
def main(theta: float) -> None:
    q = qubit[3]
    rz(q[0], theta)
    phi = theta * 2
    rz(q[1], phi)
    psi = phi + theta
    rz(q[2], psi)
"#;

    let zlup = lint_and_compile(source, None).expect("Guppy compilation failed");
    for name in ["theta", "phi", "psi"] {
        let declaration = format!("__zlup_angle_{name}: a64");
        assert_eq!(
            zlup.matches(&declaration).count(),
            1,
            "angle origin {name} was not converted exactly once:\n{zlup}"
        );
        assert!(
            zlup.contains(&format!("rz(__zlup_angle_{name})")),
            "gate did not use converted {name}:\n{zlup}"
        );
    }
    assert!(
        zlup.contains("phi := theta * 2"),
        "generated source:\n{zlup}"
    );
    assert!(
        zlup.contains("psi := phi + theta"),
        "generated source:\n{zlup}"
    );
}

#[test]
fn test_e2e_loop_variable_angle_has_no_synthesized_binding() {
    let source = r#"
def main() -> None:
    q = qubit[4]
    for i in range(4):
        rz(q[i], i)
"#;

    let zlup = lint_and_compile(source, None).expect("Guppy compilation failed");
    assert!(
        zlup.contains("for i in 0..4") && zlup.contains("rz(i / 2 turns) q[i]"),
        "generated source:\n{zlup}"
    );
    assert!(
        !zlup.contains("__zlup_angle_i"),
        "loop captures must not reference an unbound angle alias:\n{zlup}"
    );
}

#[test]
fn test_e2e_measurement_result_angle_has_no_synthesized_binding() {
    let source = r#"
def main() -> None:
    q = qubit[2]
    m = measure(q[0])
    rz(q[1], m)
"#;

    let zlup = lint_and_compile(source, None).expect("Guppy compilation failed");
    assert!(
        zlup.contains("rz(m / 2 turns) q[1]"),
        "generated source:\n{zlup}"
    );
    assert!(
        !zlup.contains("__zlup_angle_m"),
        "measurement results must not reference an unbound angle alias:\n{zlup}"
    );
}

#[test]
fn test_e2e_mixed_parameter_and_loop_angle_origins() {
    let source = r#"
def main(theta: float) -> None:
    q = qubit[4]
    for i in range(4):
        rz(q[i], theta + i)
"#;

    let zlup = lint_and_compile(source, None).expect("Guppy compilation failed");
    assert!(
        zlup.contains("__zlup_angle_theta: a64 = theta / 2 turns"),
        "generated source:\n{zlup}"
    );
    assert!(
        zlup.contains("rz(__zlup_angle_theta + i / 2 turns) q[i]"),
        "generated source:\n{zlup}"
    );
    assert!(
        !zlup.contains("__zlup_angle_i"),
        "loop captures must not reference an unbound angle alias:\n{zlup}"
    );
}

#[test]
fn test_e2e_mutable_angle_binding_is_refreshed_after_assignment() {
    let source = r#"
def main() -> None:
    q = qubit[2]
    theta: float = 0.5
    rz(q[0], theta)
    theta = 0.25
    rz(q[1], theta)
"#;

    let zlup = lint_and_compile(source, None).expect("Guppy compilation failed");
    for expected in [
        "mut theta: f64 = 0.5",
        "mut __zlup_angle_theta: a64 = theta / 2 turns",
        "theta = 0.25",
        "__zlup_angle_theta = theta / 2 turns",
        "rz(__zlup_angle_theta) q[1]",
    ] {
        assert!(zlup.contains(expected), "missing {expected:?}:\n{zlup}");
    }
    assert!(
        zlup.find("theta = 0.25") < zlup.find("__zlup_angle_theta = theta / 2 turns"),
        "converted angle must refresh after its source assignment:\n{zlup}"
    );
}

#[test]
fn test_e2e_synthesized_angle_bindings_avoid_user_names_per_function() {
    let source = r#"
def first(theta: float, __zlup_angle_theta: float) -> None:
    q = qubit[1]
    rz(q[0], theta)

def second(theta: float, __zlup_angle_theta: float) -> None:
    q = qubit[1]
    rz(q[0], theta)
"#;

    let zlup = lint_and_compile(source, None).expect("Guppy compilation failed");
    assert_eq!(
        zlup.matches("__zlup_angle_theta_2: a64 = theta / 2 turns")
            .count(),
        2,
        "fresh-name selection must be isolated per function:\n{zlup}"
    );
    assert_eq!(
        zlup.matches("rz(__zlup_angle_theta_2) q[0]").count(),
        2,
        "gates must use the collision-free aliases:\n{zlup}"
    );
}

#[test]
fn test_e2e_three_qubit_gate() {
    let source = r#"
def toffoli_test() -> None:
    q = qubit[3]
    ccx(q[0], q[1], q[2])
"#;
    compile_and_check(source, &["qalloc(3)", "ccx"]);
}

#[test]
fn test_e2e_multiple_different_gates() {
    let source = r#"
def gate_sequence() -> None:
    q = qubit[2]
    h(q[0])
    x(q[1])
    cx(q[0], q[1])
    z(q[0])
"#;
    compile_and_check(source, &["h q[0]", "x q[1]", "cx (q[0], q[1])", "z q[0]"]);
}

// =============================================================================
// Expression Tests
// =============================================================================

#[test]
fn test_e2e_arithmetic() {
    let source = r#"
def math(a: int, b: int) -> int:
    return (a + b) * (a - b)
"#;
    compile_and_check(source, &["a + b", "a - b"]);
}

#[test]
fn test_e2e_comparison() {
    let source = r#"
def compare(a: int, b: int) -> bool:
    return a <= b
"#;
    compile_and_check(source, &["a <= b"]);
}

#[test]
fn test_e2e_unary() {
    let source = r#"
def negate(x: int) -> int:
    return -x
"#;
    compile_and_check(source, &["return -x"]);
}

#[test]
fn test_e2e_boolean_and() {
    let source = r#"
def both(a: bool, b: bool) -> bool:
    return a and b
"#;
    compile_and_check(source, &["a and b"]);
}

#[test]
fn test_e2e_boolean_or() {
    let source = r#"
def either(a: bool, b: bool) -> bool:
    return a or b
"#;
    compile_and_check(source, &["a or b"]);
}

#[test]
fn test_e2e_chained_boolean() {
    let source = r#"
def all_three(a: bool, b: bool, c: bool) -> bool:
    return a and b and c
"#;
    // Chained: (a and b) and c
    compile_and_check(source, &["and"]);
}

// =============================================================================
// Control Flow Tests (while, break, continue)
// =============================================================================

#[test]
fn test_e2e_while_loop_with_break() {
    // While loops are transformed to bounded for loops with break condition
    let source = r#"
def count_up(limit: int) -> int:
    i: int = 0
    while i < limit:
        i = i + 1
    return i
"#;
    // Should transform to for loop with break
    compile_and_check(source, &["for", "break"]);
}

#[test]
fn test_e2e_break_statement() {
    let source = r#"
def find_first() -> int:
    result: int = 0
    for i in range(10):
        if i > 5:
            result = i
            break
    return result
"#;
    compile_and_check(source, &["break"]);
}

#[test]
fn test_e2e_continue_statement() {
    let source = r#"
def skip_evens() -> int:
    total: int = 0
    for i in range(10):
        if i % 2 == 0:
            continue
        total = total + i
    return total
"#;
    compile_and_check(source, &["continue"]);
}

// =============================================================================
// Edge Case Tests
// =============================================================================

#[test]
fn test_e2e_augmented_assignment_add() {
    let source = r#"
def increment() -> int:
    x: int = 5
    x += 3
    return x
"#;
    compile_and_check(source, &["mut x: i64 = 5", "x = x + 3"]);
}

#[test]
fn test_e2e_augmented_assignment_sub() {
    let source = r#"
def decrement() -> int:
    x: int = 10
    x -= 4
    return x
"#;
    compile_and_check(source, &["x = x - 4"]);
}

#[test]
fn test_e2e_augmented_assignment_mul() {
    let source = r#"
def double() -> int:
    x: int = 5
    x *= 2
    return x
"#;
    compile_and_check(source, &["x = x * 2"]);
}

#[test]
fn test_e2e_nested_for_loops() {
    let source = r#"
def matrix_sum() -> int:
    total: int = 0
    for i in range(3):
        for j in range(4):
            total += 1
    return total
"#;
    compile_and_check(
        source,
        &["for i in 0..3", "for j in 0..4", "total = total + 1"],
    );
}

#[test]
fn test_e2e_deeply_nested_loops() {
    let source = r#"
def cube_sum() -> int:
    total: int = 0
    for i in range(2):
        for j in range(2):
            for k in range(2):
                total += 1
    return total
"#;
    compile_and_check(source, &["for i in 0..2", "for j in 0..2", "for k in 0..2"]);
}

#[test]
fn test_e2e_nested_loop_with_outer_var() {
    let source = r#"
def product_sum() -> int:
    total: int = 0
    for i in range(5):
        for j in range(3):
            total = total + i * j
    return total
"#;
    compile_and_check(source, &["total = total + i * j"]);
}

#[test]
fn test_e2e_break_in_nested_loop() {
    let source = r#"
def find_product() -> int:
    result: int = 0
    for i in range(5):
        for j in range(5):
            if i * j > 6:
                result = i * j
                break
    return result
"#;
    compile_and_check(source, &["if (i * j > 6)", "break"]);
}

#[test]
fn test_e2e_multiple_variables() {
    let source = r#"
def swap_like() -> int:
    a: int = 1
    b: int = 2
    c: int = 3
    a = b
    b = c
    c = a
    return a + b + c
"#;
    compile_and_check(
        source,
        &["mut a: i64 = 1", "mut b: i64 = 2", "mut c: i64 = 3"],
    );
}

#[test]
fn test_e2e_complex_arithmetic() {
    let source = r#"
def calculate(x: int, y: int) -> int:
    return (x + y) * (x - y) + x * y
"#;
    compile_and_check(source, &["x + y", "x - y", "x * y"]);
}

#[test]
fn test_e2e_chained_comparison() {
    let source = r#"
def in_range(x: int, lo: int, hi: int) -> bool:
    return x >= lo and x <= hi
"#;
    compile_and_check(source, &["x >= lo", "x <= hi", "and"]);
}

#[test]
fn test_e2e_boolean_operators() {
    let source = r#"
def complex_bool(a: bool, b: bool, c: bool) -> bool:
    return (a and b) or (a and c) or (b and c)
"#;
    compile_and_check(source, &["and", "or"]);
}

#[test]
fn test_e2e_if_elif_else() {
    let source = r#"
def classify(x: int) -> int:
    if x < 0:
        return -1
    elif x == 0:
        return 0
    else:
        return 1
"#;
    compile_and_check(source, &["if (x < 0)", "return -1", "else", "return 1"]);
}

#[test]
fn test_e2e_nested_if() {
    let source = r#"
def nested_cond(a: int, b: int) -> int:
    if a > 0:
        if b > 0:
            return 1
        else:
            return 2
    else:
        return 3
"#;
    compile_and_check(
        source,
        &[
            "if (a > 0)",
            "if (b > 0)",
            "return 1",
            "return 2",
            "return 3",
        ],
    );
}

#[test]
fn test_e2e_while_with_multiple_conditions() {
    let source = r#"
def bounded_count(limit: int) -> int:
    i: int = 0
    count: int = 0
    while i < limit and count < 50:
        count += 1
        i += 1
    return count
"#;
    compile_and_check(source, &["for", "break", "and"]);
}

#[test]
fn test_e2e_multiple_quantum_registers() {
    let source = r#"
def multi_reg() -> None:
    a = qubit[2]
    b = qubit[3]
    h(a[0])
    h(b[0])
    cx(a[0], b[0])
"#;
    compile_and_check(
        source,
        &[
            "qalloc(2)",
            "qalloc(3)",
            "h a[0]",
            "h b[0]",
            "cx (a[0], b[0])",
        ],
    );
}

#[test]
fn test_e2e_loop_with_quantum_ops() {
    let source = r#"
def apply_h_to_all() -> None:
    q = qubit[4]
    for i in range(4):
        h(q[i])
"#;
    compile_and_check(source, &["qalloc(4)", "for i in 0..4", "h q[i]"]);
}

#[test]
fn test_e2e_modulo_operator() {
    let source = r#"
def is_even(x: int) -> bool:
    return x % 2 == 0
"#;
    compile_and_check(source, &["x % 2 == 0"]);
}

#[test]
fn test_e2e_integer_division() {
    let source = r#"
def half(x: int) -> int:
    return x // 2
"#;
    compile_and_check(source, &["x / 2"]);
}

// =============================================================================
// Analysis Integration Tests
// =============================================================================

#[test]
fn test_e2e_analysis_after_compile() {
    use zlup::analysis::{AllocatorAnalysis, analyze_parallelism};

    let source = r#"
def bell() -> None:
    q = qubit[2]
    h(q[0])
    cx(q[0], q[1])
"#;

    // Compile Guppy to Zlup
    let zlup_source = lint_and_compile(source, None).expect("compile failed");

    // Parse Zlup and run analysis
    let program = zlup::parse(&zlup_source).expect("parse failed");

    let allocator_analysis = AllocatorAnalysis::analyze(&program);
    assert!(
        allocator_analysis.allocators.contains_key("q"),
        "Should detect allocator q"
    );

    let summaries = analyze_parallelism(&program);
    assert!(!summaries.is_empty(), "Should have function summaries");

    let bell_summary = summaries.iter().find(|s| s.function_name == "bell");
    assert!(bell_summary.is_some(), "Should have bell function summary");
    assert!(
        bell_summary.unwrap().quantum_ops >= 2,
        "Should have at least 2 quantum ops"
    );
}

#[test]
fn test_e2e_analysis_disjoint_registers() {
    use zlup::analysis::analyze_parallelism;

    let source = r#"
def parallel_prep() -> None:
    q1 = qubit[2]
    q2 = qubit[2]
    h(q1[0])
    h(q2[0])
"#;

    let zlup_source = lint_and_compile(source, None).expect("compile failed");
    let program = zlup::parse(&zlup_source).expect("parse failed");

    let summaries = analyze_parallelism(&program);
    let summary = &summaries[0];

    // Two independent H gates on different allocators should enable parallelism
    assert!(
        summary.max_parallelism >= 2,
        "Disjoint allocators should enable parallelism"
    );
}

#[test]
fn test_e2e_analysis_classical_quantum_independence() {
    use zlup::analysis::analyze_parallelism;

    let source = r#"
def mixed_ops() -> None:
    q = qubit[2]
    x = 1 + 2
    h(q[0])
    y = x * 3
"#;

    let zlup_source = lint_and_compile(source, None).expect("compile failed");
    let program = zlup::parse(&zlup_source).expect("parse failed");

    let summaries = analyze_parallelism(&program);
    let summary = &summaries[0];

    assert!(summary.classical_ops > 0, "Should have classical ops");
    assert!(summary.quantum_ops > 0, "Should have quantum ops");
}

#[test]
fn test_e2e_lint_error_blocks_analysis() {
    // Source with lint error (unbounded loop)
    let source = r#"
def bad_loop() -> None:
    while True:
        pass
"#;

    let result = lint_source(source, None);
    assert!(result.has_errors, "Should detect lint error");

    // lint_and_compile should fail
    let compile_result = lint_and_compile(source, None);
    assert!(
        compile_result.is_err(),
        "Should fail to compile with lint errors"
    );
}
