/// Demonstration of the simplified include system
/// No distinction between virtual/filesystem/system includes

// This would be the new simplified API:

use pecos_qasm::{ParseConfig, QASMParser};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Simple QASM that uses includes
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";     // System include
        include "custom.inc";     // User include
        include "gates/more.inc"; // Filesystem include
        
        qreg q[2];
        h q[0];
        custom_gate q[0],q[1];
        more_gate q[1];
    "#;
    
    // Simple configuration - no distinction between include types
    let mut config = ParseConfig::default();
    
    // Add includes directly (what used to be "virtual")
    config.includes.push((
        "custom.inc".to_string(),
        "gate custom_gate a,b { cx a,b; }".to_string()
    ));
    
    // Add filesystem search paths
    config.search_paths.push("/path/to/includes".into());
    config.search_paths.push("./local/includes".into());
    
    // Everything is treated uniformly - the parser doesn't care
    // where the content came from
    let program = QASMParser::parse_with_config(qasm, config)?;
    
    println!("Parsed successfully!");
    println!("Gates loaded: {:?}", program.gate_definitions.keys().collect::<Vec<_>>());
    
    Ok(())
}

/* Benefits of this approach:

1. Simpler API - just includes and paths
2. No artificial distinctions - content is content
3. User overrides work naturally (later adds override earlier)
4. Implementation is cleaner - single include resolution path
5. Users don't need to understand "virtual" vs "filesystem"

The system automatically:
- Pre-loads system includes (lowest priority) 
- Adds user includes (override system)
- Searches filesystem when needed
- Caches everything in memory
*/