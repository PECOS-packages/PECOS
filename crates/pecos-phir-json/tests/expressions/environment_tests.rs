// No need to import PecosError for these tests
use pecos_phir_json::v0_1::ast::{ArgItem, Expression};
use pecos_phir_json::v0_1::environment::{DataType, Environment};
use pecos_phir_json::v0_1::expression::ExpressionEvaluator;

#[test]
fn test_variable_environment() {
    // Create a variable environment
    let mut env = Environment::new();

    // Add variables of different types
    env.add_variable("i8_var", DataType::I8, 8).unwrap();
    env.add_variable("u8_var", DataType::U8, 8).unwrap();
    env.add_variable("i32_var", DataType::I32, 32).unwrap();
    env.add_variable("qubits", DataType::Qubits, 4).unwrap();

    // Set values
    env.set("i8_var", 100).unwrap();
    env.set("u8_var", 200).unwrap();
    env.set("i32_var", 12345).unwrap();

    // Verify values
    assert_eq!(env.get("i8_var").map(|v| v.as_i64()), Some(100));
    assert_eq!(env.get("u8_var").map(|v| v.as_u64()), Some(200));
    assert_eq!(env.get("i32_var").map(|v| v.as_i64()), Some(12345));

    // Test type constraints
    env.set("i8_var", 130).unwrap(); // Should wrap around due to i8 constraints
    assert_eq!(
        env.get("i8_var").map(|v| v.as_u64()),
        Some(0xFFFF_FFFF_FFFF_FF82)
    ); // -126 as u64

    env.set("u8_var", 300).unwrap(); // Should be masked to 44 (300 % 256)
    assert_eq!(env.get("u8_var").map(|v| v.as_u64()), Some(44));

    // Test bit operations
    env.add_variable("bits", DataType::U8, 8).unwrap();
    env.set("bits", 0).unwrap();

    env.set_bit("bits", 0, 1).unwrap(); // Set bit 0
    env.set_bit("bits", 2, 1).unwrap(); // Set bit 2
    env.set_bit("bits", 4, 1).unwrap(); // Set bit 4

    assert_eq!(env.get("bits").map(|v| v.as_u64()), Some(0b0001_0101)); // Binary 21

    // Test getting individual bits
    assert!(env.get_bit("bits", 0).unwrap().0);
    assert!(!env.get_bit("bits", 1).unwrap().0);
    assert!(env.get_bit("bits", 2).unwrap().0);

    // Test reset_values
    env.reset_values();
    assert_eq!(env.get("i8_var").map(|v| v.as_u64()), Some(0));
    assert_eq!(env.get("u8_var").map(|v| v.as_u64()), Some(0));
    assert_eq!(env.get("i32_var").map(|v| v.as_u64()), Some(0));
    assert_eq!(env.get("bits").map(|v| v.as_u64()), Some(0));

    // Make sure variables still exist after reset
    assert!(env.has_variable("i8_var"));
    assert!(env.has_variable("u8_var"));
    assert!(env.has_variable("i32_var"));
    assert!(env.has_variable("bits"));
}

#[test]
fn test_expression_evaluation() {
    // Create an environment with test variables
    let mut env = Environment::new();
    env.add_variable("a", DataType::I32, 32).unwrap();
    env.add_variable("b", DataType::I32, 32).unwrap();
    env.add_variable("c", DataType::I32, 32).unwrap();

    env.set("a", 10).unwrap();
    env.set("b", 5).unwrap();
    env.set("c", 2).unwrap();

    let mut evaluator = ExpressionEvaluator::new(&env);

    // Test basic expression types
    let expr_int = Expression::Integer(42);
    assert_eq!(evaluator.eval_expr(&expr_int).unwrap(), 42);

    let expr_var = Expression::Variable("a".to_string());
    assert_eq!(evaluator.eval_expr(&expr_var).unwrap(), 10);

    // Test arithmetic operations
    let expr_add = Expression::Operation {
        cop: "+".to_string(),
        args: vec![
            ArgItem::Simple("a".to_string()),
            ArgItem::Simple("b".to_string()),
        ],
    };
    assert_eq!(evaluator.eval_expr(&expr_add).unwrap(), 15);

    let expr_sub = Expression::Operation {
        cop: "-".to_string(),
        args: vec![
            ArgItem::Simple("a".to_string()),
            ArgItem::Simple("b".to_string()),
        ],
    };
    assert_eq!(evaluator.eval_expr(&expr_sub).unwrap(), 5);

    let expr_mul = Expression::Operation {
        cop: "*".to_string(),
        args: vec![
            ArgItem::Simple("a".to_string()),
            ArgItem::Simple("b".to_string()),
        ],
    };
    assert_eq!(evaluator.eval_expr(&expr_mul).unwrap(), 50);

    let expr_div = Expression::Operation {
        cop: "/".to_string(),
        args: vec![
            ArgItem::Simple("a".to_string()),
            ArgItem::Simple("b".to_string()),
        ],
    };
    assert_eq!(evaluator.eval_expr(&expr_div).unwrap(), 2);

    // Test bit operations
    let bitwise_and = Expression::Operation {
        cop: "&".to_string(),
        args: vec![
            ArgItem::Simple("a".to_string()),
            ArgItem::Simple("b".to_string()),
        ],
    };
    assert_eq!(evaluator.eval_expr(&bitwise_and).unwrap(), 0); // 10 & 5 = 0

    let bitwise_or = Expression::Operation {
        cop: "|".to_string(),
        args: vec![
            ArgItem::Simple("a".to_string()),
            ArgItem::Simple("b".to_string()),
        ],
    };
    assert_eq!(evaluator.eval_expr(&bitwise_or).unwrap(), 15); // 10 | 5 = 15

    let xor_expr = Expression::Operation {
        cop: "^".to_string(),
        args: vec![
            ArgItem::Simple("a".to_string()),
            ArgItem::Simple("b".to_string()),
        ],
    };
    assert_eq!(evaluator.eval_expr(&xor_expr).unwrap(), 15); // 10 ^ 5 = 15

    // Test nested expressions
    let nested_expr = Expression::Operation {
        cop: "*".to_string(),
        args: vec![
            ArgItem::Expression(Box::new(Expression::Operation {
                cop: "+".to_string(),
                args: vec![
                    ArgItem::Simple("a".to_string()),
                    ArgItem::Simple("b".to_string()),
                ],
            })),
            ArgItem::Simple("c".to_string()),
        ],
    };
    assert_eq!(evaluator.eval_expr(&nested_expr).unwrap(), 30); // (10 + 5) * 2 = 30

    // Test complex nested expression
    let complex_expr = Expression::Operation {
        cop: "-".to_string(),
        args: vec![
            ArgItem::Expression(Box::new(Expression::Operation {
                cop: "*".to_string(),
                args: vec![
                    ArgItem::Simple("a".to_string()),
                    ArgItem::Simple("c".to_string()),
                ],
            })),
            ArgItem::Expression(Box::new(Expression::Operation {
                cop: "/".to_string(),
                args: vec![ArgItem::Simple("b".to_string()), ArgItem::Integer(1)],
            })),
        ],
    };
    assert_eq!(evaluator.eval_expr(&complex_expr).unwrap(), 15); // (10 * 2) - (5 / 1) = 20 - 5 = 15
}
