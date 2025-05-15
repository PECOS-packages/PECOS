use pecos_qasm::{QASMParser, ParseConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let qasm = r#"
        OPENQASM 2.0;
        qreg q[2];
        h q[0];
    "#;
    
    // Method 1: Simple parsing with defaults
    let program1 = QASMParser::parse_str(qasm)?;
    
    // Method 2: Parse from file
    // let program2 = QASMParser::parse_file("quantum.qasm")?;
    
    // Method 3: Custom configuration
    let mut config = ParseConfig::default();
    config.search_paths.push("/custom/path".into());
    config.expand_gates = false;
    let program3 = QASMParser::parse_with_config(qasm, config)?;
    
    // Method 4: Quick convenience method (for compatibility)
    let program4 = QASMParser::parse_str(qasm)?;
    
    println!("All parsing methods worked successfully");
    Ok(())
}