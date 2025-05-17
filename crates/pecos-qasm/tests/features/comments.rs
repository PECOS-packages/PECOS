use pecos_qasm::{Operation, QASMParser};

#[test]
fn test_international_comments() {
    // Test that the parser correctly handles various international characters in comments
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";

        // 🚀 Quantum computing! 🎉
        qreg q[3];
        creg c[3];

        h q[0];

        // 日本語のコメント：量子もつれ (Japanese: Quantum entanglement)
        cx q[0], q[1];

        // Comentario en español: Superposición cuántica (Spanish: Quantum superposition)
        h q[1];
        cx q[1], q[2];
        h q[2];

        // हिंदी में टिप्पणी: क्वांटम गेट (Hindi: Quantum gate)
        // 한국어 주석: 양자 측정 (Korean: Quantum measurement)
        measure q[0] -> c[0];
        measure q[1] -> c[1];
        measure q[2] -> c[2];

        // Mixed emojis and text: 🌟✨ Quantum magic! ✨🌟
        // Mathematical symbols: ∀x∈ℂ, |ψ⟩ = α|0⟩ + β|1⟩
        // Special characters: ñ § € £ ¥ © ® ™ • ° ± ≠ ≤ ≥ ∞
    "#;

    let result = QASMParser::parse_str(qasm);

    match result {
        Ok(program) => {
            // Verify the program parsed correctly despite the international comments
            println!("Successfully parsed QASM with international comments");

            // Count operations to ensure comments didn't interfere with parsing
            let gate_count = program
                .operations
                .iter()
                .filter(|op| matches!(op, Operation::Gate { .. }))
                .count();

            let measure_count = program
                .operations
                .iter()
                .filter(|op| matches!(op, Operation::Measure { .. }))
                .count();

            // We expect: 3 H gates, 2 CX gates, 3 measure operations
            assert_eq!(gate_count, 5, "Expected 5 gates (3 H + 2 CX)");
            assert_eq!(measure_count, 3, "Expected 3 measure operations");

            // Verify the registers were created correctly
            assert_eq!(
                program.quantum_registers.len(),
                1,
                "Expected 1 quantum register"
            );
            assert_eq!(
                program.classical_registers.len(),
                1,
                "Expected 1 classical register"
            );
            assert_eq!(program.total_qubits, 3, "Expected 3 qubits total");
        }
        Err(e) => {
            panic!("Failed to parse QASM with international comments: {e}");
        }
    }
}

#[test]
fn test_inline_comments_with_emojis() {
    // Test inline comments with special characters
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";

        qreg q[2];

        h q[0]; // 🔮 Creating superposition
        cx q[0], q[1]; // 🔗 Entangling qubits | 量子もつれ

        // Test multiple comment styles on same line
        h q[1]; // English // Español: Hadamard
    "#;

    let result = QASMParser::parse_str(qasm);

    match result {
        Ok(program) => {
            println!("Successfully parsed QASM with inline emoji comments");

            // Verify the operations
            let operations: Vec<String> = program
                .operations
                .iter()
                .filter_map(|op| match op {
                    Operation::Gate { name, .. } => Some(name.clone()),
                    _ => None,
                })
                .collect();

            assert_eq!(
                operations,
                vec!["H", "CX", "H"],
                "Expected H, CX, H sequence"
            );
        }
        Err(e) => {
            panic!("Failed to parse QASM with inline emoji comments: {e}");
        }
    }
}

#[test]
fn test_edge_case_comments() {
    // Test edge cases with special Unicode characters
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";

        // Zero-width characters: \u{200B}‌‍
        // Right-to-left override: ‏مرحبا‎
        // Combining characters: é = e + ́  (combining acute)
        // Mathematical symbols: ∮∂Ω⊗∇²ψ = 0
        // Box drawing: ┌─┬─┐│ │ │├─┼─┤└─┴─┘
        // Miscellaneous symbols: ♠♣♥♦☀☁☂☃★☆☎☏✓✗

        qreg q[1];
        h q[0]; // Final gate 🏁
    "#;

    let result = QASMParser::parse_str(qasm);

    match result {
        Ok(program) => {
            println!("Successfully parsed QASM with edge case Unicode comments");
            assert_eq!(program.operations.len(), 1, "Expected 1 operation");
        }
        Err(e) => {
            panic!("Failed to parse QASM with edge case comments: {e}");
        }
    }
}
