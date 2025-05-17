use pecos_qasm::QASMParser;
use pest::Parser;
use pest::iterators::Pair;

fn debug_pairs(pair: &Pair<pecos_qasm::parser::Rule>, depth: usize) {
    let indent = "  ".repeat(depth);
    println!(
        "{}Rule: {:?}, Text: '{}'",
        indent,
        pair.as_rule(),
        pair.as_str()
    );

    let pairs = pair.clone().into_inner();
    for inner_pair in pairs {
        debug_pairs(&inner_pair, depth + 1);
    }
}

#[test]
fn test_pest_expr_parsing() {
    let expr = "b ^ a";

    // Parse using the expr rule directly to see what's happening
    match pecos_qasm::parser::QASMParser::parse(pecos_qasm::parser::Rule::expr, expr) {
        Ok(mut pairs) => {
            println!("Successfully parsed expression");
            let pair = pairs.next().unwrap();
            debug_pairs(&pair, 0);
        }
        Err(e) => {
            println!("Failed to parse expression:");
            println!("{e}");
        }
    }
}

#[test]
fn test_binary_operators() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        creg a[2];
        creg b[2];
        creg c[2];

        b = 2;
        a = 1;
        c = b + a;  // Addition instead of XOR as a test
    "#;

    let program = match QASMParser::parse_str(qasm) {
        Ok(prog) => prog,
        Err(e) => {
            panic!("Failed to parse: {e:?}");
        }
    };

    // Just check that parsing succeeded
    assert_eq!(program.classical_registers.len(), 3);
    assert_eq!(program.operations.len(), 3);
}
