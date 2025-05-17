use pecos_qasm::{Operation, parser::QASMParser};

#[test]
fn test_complex_classical_operations() {
    let qasm = r#"
        OPENQASM 2.0;
        include "hqslib1.inc";

        qreg q[1];
        creg c[4];
        creg a[2];
        creg b[3];
        creg d[1];

        c = 2;
        c = a;
        if (b != 2) c[1] = b[1] & a[1] | a[0];
        c = b & a;
        b = a + b;
        b[1] = b[0] + ~b[2];
        c = a - (b**c);
        d = a << 1;
        d = c >> 2;
        c[0] = 1;
        b = a * c / b;
        d[0] = a[0] ^ 1;
        if(c>=2) h q[0];
        if(d == 1) rx((0.5+0.5)*pi) q[0];
    "#;

    let program = QASMParser::parse_str(qasm).expect("Failed to parse complex classical operations");
    
    // Count different types of operations
    let mut classical_assignments = 0;
    let mut conditionals = 0;
    let mut gates = 0;
    
    for op in &program.operations {
        match op {
            Operation::ClassicalAssignment { .. } => classical_assignments += 1,
            Operation::If { .. } => conditionals += 1,
            Operation::Gate { .. } => gates += 1,
            _ => {}
        }
    }
    
    // Based on the debug output, we have 11 assignments, 3 conditionals
    assert_eq!(classical_assignments, 11, "Should have 11 classical assignments");
    assert_eq!(conditionals, 3, "Should have 3 conditional statements (one contains an assignment)");
    
    // The gates are inside the conditionals, not at the top level
    assert_eq!(gates, 0, "Gates are inside conditionals, not at top level");
    
    // Check some specific operations
    let mut found_power_op = false;
    let mut found_bitwise_ops = false;
    let mut found_arithmetic_ops = false;
    let mut found_shift_ops = false;
    
    for op in &program.operations {
        let expr_str = format!("{:?}", op);
        
        // Check for various operations in the debug string
        if expr_str.contains("**") {
            found_power_op = true;
        }
        if expr_str.contains("&") || expr_str.contains("|") || expr_str.contains("^") || expr_str.contains("~") {
            found_bitwise_ops = true;
        }
        if expr_str.contains("+") || expr_str.contains("-") || expr_str.contains("*") || expr_str.contains("/") {
            found_arithmetic_ops = true;
        }
        if expr_str.contains("<<") || expr_str.contains(">>") {
            found_shift_ops = true;
        }
    }
    
    assert!(found_arithmetic_ops, "Should have arithmetic operations");
    assert!(found_bitwise_ops, "Should have bitwise operations");
    assert!(found_shift_ops, "Should have shift operations");
    assert!(found_power_op, "Should have power operation");
}

#[test]
fn test_conditional_quantum_gates() {
    let qasm = r#"
        OPENQASM 2.0;
        include "hqslib1.inc";

        qreg q[1];
        creg c[4];
        creg d[1];

        c = 3;
        d = 1;
        
        if(c>=2) h q[0];
        if(d == 1) rx((0.5+0.5)*pi) q[0];
    "#;

    let program = QASMParser::parse_str(qasm).expect("Failed to parse conditional gates");
    
    // Find the conditional operations
    let mut h_conditional = false;
    let mut rx_conditional = false;
    
    for op in &program.operations {
        if let Operation::If { condition, operation } = op {
            let cond_str = format!("{:?}", condition);
            
            if cond_str.contains(">=") {
                // This should be the H gate conditional
                if let Operation::Gate { name, .. } = &**operation {
                    if name == "h" {
                        h_conditional = true;
                    }
                }
            }
            
            if cond_str.contains("==") {
                // This should be the RX gate conditional  
                if let Operation::Gate { name, .. } = &**operation {
                    if name == "rx" {
                        rx_conditional = true;
                    }
                }
            }
        }
    }
    
    assert!(h_conditional, "Should have conditional h gate");
    assert!(rx_conditional, "Should have conditional rx gate");
}

#[test]
fn test_register_size_arithmetic() {
    let qasm = r#"
        OPENQASM 2.0;
        include "hqslib1.inc";

        creg a[2];
        creg b[3];
        creg c[4];
        
        c = b & a;  // bitwise AND between different sized registers
        b = a + b;  // addition with different sized registers
    "#;

    let program = QASMParser::parse_str(qasm).expect("Failed to parse register arithmetic");
    
    // Check that operations with different register sizes are parsed
    let mut found_bitwise_and = false;
    let mut found_addition = false;
    
    for op in &program.operations {
        let op_str = format!("{:?}", op);
        
        if op_str.contains("&") {
            found_bitwise_and = true;
        }
        if op_str.contains("+") {
            found_addition = true;
        }
    }
    
    assert!(found_bitwise_and, "Should have bitwise AND operation");
    assert!(found_addition, "Should have addition operation");
}

#[test]
fn test_complex_expression_parsing() {
    let qasm = r#"
        OPENQASM 2.0;
        include "hqslib1.inc";

        creg a[2];
        creg b[3];
        creg c[4];
        
        c = a - (b**c);  // subtraction with power in parentheses
        b[1] = b[0] + ~b[2];  // indexed assignment with bitwise NOT
        c[1] = b[1] & a[1] | a[0];  // complex bitwise expression
        b = a * c / b;  // chained arithmetic
    "#;

    let program = QASMParser::parse_str(qasm).expect("Failed to parse complex expressions");
    
    // Find specific patterns in the expressions
    let mut found_power_in_parens = false;
    let mut found_indexed_assignment = false;
    let mut found_complex_bitwise = false;
    let mut found_chained_arithmetic = false;
    
    for op in &program.operations {
        let op_str = format!("{:?}", op);
            
        // Check for power operation in subtraction
        if op_str.contains("-") && op_str.contains("**") {
            found_power_in_parens = true;
        }
        
        // Check for indexed assignment
        if let Operation::ClassicalAssignment { is_indexed, target, .. } = op {
            if *is_indexed && target == "b" {
                found_indexed_assignment = true;
            }
        }
        
        // Check for complex bitwise expression (AND and OR)
        if op_str.contains("&") && op_str.contains("|") {
            found_complex_bitwise = true;
        }
        
        // Check for chained arithmetic (multiply and divide)
        if op_str.contains("*") && op_str.contains("/") {
            found_chained_arithmetic = true;
        }
    }
    
    assert!(found_power_in_parens, "Should parse power operation in parentheses");
    assert!(found_indexed_assignment, "Should parse indexed assignment");
    assert!(found_complex_bitwise, "Should parse complex bitwise expression");
    assert!(found_chained_arithmetic, "Should parse chained arithmetic");
}