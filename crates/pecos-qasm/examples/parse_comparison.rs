use pecos_qasm::{QASMParser, ParseConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let qasm = r#"
        OPENQASM 2.0;
        qreg q[2];
        h q[0];
    "#;

    // Method 1: Simple default parsing
    let _program1 = QASMParser::parse_str(qasm)?;

    // Method 2: Config struct for custom configuration
    let mut config = ParseConfig::default();
    config.search_paths.push("/custom/path".into());
    config.expand_gates = false;
    let _program2 = QASMParser::parse_with_config(qasm, config)?;

    // Method 3: Existing convenience methods
    let _program3 = QASMParser::parse_str_raw(qasm)?;

    println!("All parsing methods work successfully");
    Ok(())
}