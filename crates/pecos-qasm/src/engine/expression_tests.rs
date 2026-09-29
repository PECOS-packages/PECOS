use super::*;
use crate::parser::expressions::parse_integer_to_bitvec;

fn run(source: &str) -> Result<QASMEngine, PecosError> {
    let program = QASMProgram::from_str(&format!("OPENQASM 2.0; {source}"))?;
    let mut engine = QASMEngine::new(program);
    engine.allow_complex_conditionals(true);
    engine.process_program_impl()?;
    Ok(engine)
}

fn integer(value: &str) -> Expression {
    Expression::Integer(parse_integer_to_bitvec(value).unwrap())
}

#[test]
fn plain_store_sign_extends_to_100_bits() {
    let engine = run("creg w[100]; w = -1;").unwrap();
    assert_eq!(
        engine.classical_registers["w"],
        BitVec::<u8, Lsb0>::repeat(true, 100)
    );
}

#[test]
fn conditional_store_sign_extends_to_100_bits() {
    let engine = run("creg w[100]; if (1 == 1) w = -1;").unwrap();
    assert_eq!(
        engine.classical_registers["w"],
        BitVec::<u8, Lsb0>::repeat(true, 100)
    );
}

#[test]
fn wide_condition_uses_every_bit() {
    for bit in [70, 99] {
        let engine = run(&format!(
            "creg w[100]; creg c[1]; w[{bit}] = 1; if (w) c = 1;"
        ))
        .unwrap();
        assert!(engine.classical_registers["c"][0]);
    }
}

#[test]
fn runtime_division_by_zero_errors() {
    assert!(matches!(
        run("creg a[4]; creg c[8]; a = 7; c = a / 0;"),
        Err(PecosError::RuntimeDivisionByZero)
    ));
}

#[test]
fn literal_division_by_zero_errors() {
    assert!(matches!(
        run("creg c[8]; c = 7 / 0;"),
        Err(PecosError::RuntimeDivisionByZero)
    ));
}

#[test]
fn classical_declaration_width_bounds() {
    for width in [0, 65536, 65537] {
        assert!(matches!(
            QASMProgram::from_str(&format!("OPENQASM 2.0; creg unused[{width}];")),
            Err(PecosError::CompileInvalidRegisterSize(_))
        ));
    }
    for width in [1, 65535] {
        let engine = run(&format!("creg c[{width}]; c = -1;")).unwrap();
        assert_eq!(engine.classical_registers["c"].len(), width);
        assert!(engine.classical_registers["c"].all());
    }
    // The bound belongs to classical declarations, not the shared identifier parser.
    assert!(QASMProgram::from_str("OPENQASM 2.0; qreg q[65536];").is_ok());
}

#[test]
fn wide_signed_fold_keeps_complete_pattern() {
    let engine =
        run("creg w[65]; w = (-(18446744073709551616 - 18446744073709551617)) << 64;").unwrap();
    let mut expected = BitVec::<u8, Lsb0>::repeat(false, 65);
    expected.set(64, true);
    assert_eq!(engine.classical_registers["w"], expected);
}

#[test]
fn boolean_complement_and_indexed_truthiness() {
    let engine =
        run("creg a[1]; creg b[8]; creg c[2]; a = 1; b = ~a[0]; c[0] = 2; if (a) c[1] = 2;")
            .unwrap();
    assert!(engine.classical_registers["b"].not_any());
    assert!(engine.classical_registers["c"].all());
}

#[test]
fn shared_semantic_anchors() {
    for (expression, expected) in [
        ("8 / 2", 4),
        ("-7 / 2", 253),
        ("(5+3)*2", 16),
        ("~1 | 0", 254),
        ("a / -2", 0),
        ("-(1-3)", 2),
        ("(1-3) < 0", 1),
        ("(1 == 1) > -1", 0),
        ("8 >> 64", 8),
        ("-8 >> 2", 254),
        ("8 >> -(-1)", 4),
        ("1 << 64", 0),
        ("1 << 65536", 1),
        ("1 << 18446744073709551617", 1),
    ] {
        let engine = run(&format!("creg a[4]; creg w[8]; a = 15; w = {expression};")).unwrap();
        assert_eq!(
            engine.classical_registers["w"].load_le::<u8>(),
            expected,
            "{expression}"
        );
    }
    for width in [4, 64, 100] {
        let engine = run(&format!("creg c[{width}]; creg eq[1]; creg gt[1]; creg lt[1]; c = -1; if (c == -1) eq = 1; if (c > -1) gt = 1; if (c < -1) lt = 1;")).unwrap();
        assert!(engine.classical_registers["c"].all());
        assert_eq!(engine.classical_registers["eq"][0], width >= 64);
        assert!(!engine.classical_registers["gt"][0]);
        assert_eq!(engine.classical_registers["lt"][0], width < 64);
    }
    let engine = run("creg w[100]; w = 1 << 80;").unwrap();
    assert!(engine.classical_registers["w"].not_any());
    for expression in ["8 << -1", "1 << 9223372036854775808"] {
        assert!(
            run(&format!("creg w[100]; w = {expression};"))
                .unwrap_err()
                .to_string()
                .contains("Negative shift amount")
        );
    }
}

#[test]
fn rng_setters_evaluate_arguments_without_folding() {
    for name in ["RNGseed", "RNGindex", "RNGbound"] {
        for arg in [
            integer("1"),
            Expression::Variable("a".to_string()),
            Expression::BitId("a".to_string(), 0),
            Expression::BitId("a".to_string(), 1),
            Expression::BinaryOp {
                op: "==".to_string(),
                left: Box::new(integer("1")),
                right: Box::new(integer("1")),
            },
        ] {
            let mut engine = QASMEngine::default();
            engine
                .classical_registers
                .insert("a".to_string(), bitvec![u8, Lsb0; 1, 0]);
            let expected = u64::from(!matches!(arg, Expression::BitId(_, 1)));
            let ExprValue::Unsigned(value) = engine.evaluate_rng_models(name, &[arg], 100).unwrap()
            else {
                panic!("Void result must be unsigned");
            };
            assert_eq!(value.size(), 100);
            assert!(value.is_zero());
            match name {
                "RNGseed" => {
                    let mut reference = QASMEngine::default();
                    reference.rng_model.set_seed(expected);
                    assert_eq!(engine.rng_model.rng_gen, reference.rng_model.rng_gen);
                }
                "RNGindex" => assert_eq!(engine.rng_model.count, expected),
                "RNGbound" => assert_eq!(u64::from(engine.rng_model.curr_bound), expected),
                _ => unreachable!(),
            }
        }
        for expression in ["a", "a[0]", "1 == 1", "-(-1)"] {
            run(&format!("creg a[1]; a = 1; {name}({expression});")).unwrap();
        }
        for expression in ["-1", "1 - 3", "-18446744073709551616"] {
            let error = run(&format!("{name}({expression});"))
                .unwrap_err()
                .to_string();
            assert!(error.contains(name), "{error}");
            assert!(error.contains("non-negative"), "{error}");
            let expected = if expression == "1 - 3" {
                "-2"
            } else {
                expression
            };
            assert!(error.contains(expected), "{error}");
        }
        let error = run(&format!(
            "{name}(-340282366920938463463374607431768211456);"
        ))
        .unwrap_err()
        .to_string();
        assert!(
            error.contains(name) && error.contains("-0x100000000000000000000000000000000"),
            "{error}"
        );
        assert!(
            run(&format!("{name}(1 << 9223372036854775808);"))
                .unwrap_err()
                .to_string()
                .contains("Negative shift amount")
        );
    }
}

#[test]
fn rng_setter_scalar_boundaries() {
    for (name, limit, above) in [
        ("RNGseed", "18446744073709551615", "18446744073709551616"),
        ("RNGindex", "18446744073709551615", "18446744073709551616"),
        ("RNGbound", "4294967295", "4294967296"),
    ] {
        let mut engine = QASMEngine::default();
        // set_index advances iteratively. Put the generator at the requested index
        // to test this boundary without generating 2^64-1 random numbers.
        if name == "RNGindex" {
            engine.rng_model.count = u64::MAX;
        }
        engine
            .evaluate_rng_models(name, &[integer(limit)], 1)
            .unwrap();
        let error = engine
            .evaluate_rng_models(name, &[integer(above)], 1)
            .unwrap_err()
            .to_string();
        assert!(error.contains(name));
        assert!(error.contains("does not fit"));
        // A wide register with only a low bit is accepted by value, not width.
        let mut engine = QASMEngine::default();
        let mut bits = BitVec::repeat(false, 100);
        bits.set(0, true);
        engine.classical_registers.insert("wide".to_string(), bits);
        engine
            .evaluate_rng_models(name, &[Expression::Variable("wide".to_string())], 1)
            .unwrap();
        engine
            .classical_registers
            .get_mut("wide")
            .unwrap()
            .set(99, true);
        let error = engine
            .evaluate_rng_models(name, &[Expression::Variable("wide".to_string())], 1)
            .unwrap_err()
            .to_string();
        assert!(error.contains(name) && error.contains("does not fit"));
    }
    let mut engine = QASMEngine::default();
    engine.rng_model.count = 4_294_967_296;
    engine
        .evaluate_rng_models("RNGindex", &[integer("4294967296")], 1)
        .unwrap();
    assert_eq!(engine.rng_model.count, 4_294_967_296);
}

#[test]
fn rng_results_zero_extend_scalar_at_target_width() {
    for width in [1, 32, 33, 100] {
        let mut engine = QASMEngine::default();
        let mut reference = QASMEngine::default();
        let expected = u64::from(reference.rng_model.rng_num());
        let ExprValue::Unsigned(value) = engine.evaluate_rng_models("RNGnum", &[], width).unwrap()
        else {
            panic!("Unsigned RNG result");
        };
        assert_eq!(usize::from(value.size()), width);
        let mask = if width == 1 { 1 } else { u64::MAX };
        assert_eq!(value.to_u64(), Some(expected & mask));
    }
    for width in [33, 100] {
        let engine = run(&format!("creg w[{width}]; w = RNGnum();")).unwrap();
        assert!(engine.classical_registers["w"][32..].not_any());
    }
}

#[cfg(feature = "wasm")]
mod wasm {
    use super::*;
    use crate::foreign_objects::ForeignObject;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[derive(Clone, Debug)]
    struct Probe {
        calls: Arc<AtomicUsize>,
        returns: Vec<i64>,
        expected_args: Vec<i64>,
    }

    impl ForeignObject for Probe {
        fn clone_box(&self) -> Box<dyn ForeignObject> {
            Box::new(self.clone())
        }
        fn init(&mut self) -> Result<(), PecosError> {
            Ok(())
        }
        fn new_instance(&mut self) -> Result<(), PecosError> {
            Ok(())
        }
        fn get_funcs(&self) -> Vec<String> {
            vec!["probe".to_string()]
        }
        fn exec(&mut self, _: &str, args: &[i64]) -> Result<Vec<i64>, PecosError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            assert_eq!(args, self.expected_args);
            Ok(self.returns.clone())
        }
        fn as_any(&self) -> &dyn Any {
            self
        }
        fn as_any_mut(&mut self) -> &mut dyn Any {
            self
        }
    }

    #[test]
    fn wasm_width_rejected_before_foreign_call() {
        for width in [65, 100] {
            for bit in [None, Some(0), Some(64), Some(width - 1), Some(70)]
                .into_iter()
                .filter(|bit| bit.is_none_or(|bit| bit < width))
            {
                let mut engine = QASMEngine::default();
                let calls = Arc::new(AtomicUsize::new(0));
                engine.foreign_object = Some(Box::new(Probe {
                    calls: calls.clone(),
                    returns: vec![1],
                    expected_args: vec![],
                }));
                let mut bits = BitVec::repeat(false, width);
                if let Some(bit) = bit {
                    bits.set(bit, true);
                }
                engine.classical_registers.insert("wide".to_string(), bits);
                let error = engine
                    .evaluate_wasm_expr(
                        "probe",
                        &[integer("1"), Expression::Variable("wide".to_string())],
                        8,
                    )
                    .unwrap_err();
                assert!(matches!(error, PecosError::Input(_)));
                let error = error.to_string();
                assert!(
                    error.contains("probe")
                        && error.contains("argument 2")
                        && error.contains(&width.to_string()),
                    "{error}"
                );
                assert_eq!(calls.load(Ordering::SeqCst), 0);
            }
        }
    }

    #[test]
    fn wasm_arguments_and_results_preserve_scalar_patterns() {
        for (width, pattern, expected) in [(4, 15, 15), (64, 1, 1), (64, u64::MAX, -1)] {
            for returns in [vec![], vec![17], vec![-1]] {
                let mut engine = QASMEngine::default();
                let bits =
                    pecos_core::bitvec::from_expr_value(&ExprValue::unsigned(pattern), width)
                        .unwrap();
                engine.classical_registers.insert("a".to_string(), bits);
                let calls = Arc::new(AtomicUsize::new(0));
                engine.foreign_object = Some(Box::new(Probe {
                    calls: calls.clone(),
                    returns: returns.clone(),
                    expected_args: vec![expected],
                }));
                let expr = Expression::FunctionCall {
                    name: "probe".to_string(),
                    args: vec![Expression::Variable("a".to_string())],
                };
                let ExprValue::Unsigned(value) = engine
                    .evaluate_expression_bitvec_with_width(&expr, 100)
                    .unwrap()
                else {
                    panic!("Unsigned WASM result");
                };
                assert_eq!(value.size(), 100);
                assert_eq!(
                    value.to_u64(),
                    Some(returns.first().copied().unwrap_or(0).cast_unsigned())
                );
                assert!((64..100).all(|bit| !value.get_bit(bit)));
                assert_eq!(calls.load(Ordering::SeqCst), 1);
            }
        }
    }
}
