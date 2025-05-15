use pecos_qasm::parser::QASMParser;
use std::env;
use std::fs;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    
    if args.len() < 2 {
        eprintln!("Usage: {} <qasm_file> [--preprocess-only]", args[0]);
        eprintln!("\nThis tool shows QASM code after preprocessing and expansion.");
        eprintln!("Options:");
        eprintln!("  --preprocess-only   Show only phase 1 (include resolution)");
        eprintln!("  (default)          Show phases 1 & 2 (include resolution + gate expansion)");
        return Ok(());
    }
    
    let filename = &args[1];
    let preprocess_only = args.len() > 2 && args[2] == "--preprocess-only";
    
    // Read the file
    let qasm = fs::read_to_string(filename)?;
    
    if preprocess_only {
        // Show just phase 1 - preprocessed QASM
        println!("=== Phase 1: Preprocessed QASM (includes resolved) ===");
        let preprocessed = QASMParser::preprocess(&qasm)?;
        println!("{}", preprocessed);
    } else {
        // Show phase 1
        println!("=== Phase 1: Preprocessed QASM (includes resolved) ===");
        let preprocessed = QASMParser::preprocess(&qasm)?;
        println!("{}", preprocessed);
        
        // Show phases 1 & 2 - fully expanded QASM
        println!("\n=== Phase 2: Expanded QASM (all gates to native operations) ===");
        let expanded = QASMParser::preprocess_and_expand(&qasm)?;
        println!("{}", expanded);
    }
    
    Ok(())
}