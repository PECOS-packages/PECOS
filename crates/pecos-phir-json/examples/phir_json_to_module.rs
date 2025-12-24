use pecos_phir::ModuleRonExt;
use pecos_phir_json::phir_json_to_module;
use std::env;
use std::fs;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: {} <input.phir.json> [output.ron]", args[0]);
        eprintln!("\nThis tool converts PHIR-JSON files to PHIR Module format.");
        eprintln!("If an output file is specified, it will serialize the module to PHIR-RON.");
        eprintln!("If no output file is specified, prints module info to stdout.");
        std::process::exit(1);
    }

    let input_path = &args[1];
    let output_path = args.get(2);

    // Step 1: Read the .phir.json file
    println!("Reading PHIR-JSON from: {input_path}");
    let json_content = fs::read_to_string(input_path)?;

    // Step 2: Convert directly to PHIR Module
    println!("Converting PHIR-JSON to PHIR Module...");
    let module = phir_json_to_module(&json_content)?;
    println!("Successfully created PHIR Module: {}", module.name);
    println!("  - {} blocks in main region", module.body.blocks.len());
    if let Some(main_block) = module.body.blocks.first() {
        println!(
            "  - {} operations in main block",
            main_block.operations.len()
        );
    }

    if let Some(output) = output_path {
        // Step 3 (optional): Serialize the module to PHIR-RON for debugging
        println!("\nSerializing PHIR Module to PHIR-RON...");
        let ron_text = module.to_ron()?;

        // Write RON to file
        fs::write(output, &ron_text)?;
        println!("Wrote PHIR-RON to: {output}");
        println!("  - {} characters written", ron_text.len());
    } else {
        // Print module structure to stdout
        println!("\nModule structure:");
        println!("  Name: {}", module.name);
        println!("  Attributes: {:?}", module.attributes);
        println!("  Region kind: {:?}", module.body.kind);

        // Print operations
        if let Some(main_block) = module.body.blocks.first() {
            println!("\nOperations in main block:");
            for (i, op) in main_block.operations.iter().enumerate() {
                println!("  {}: {:?}", i, op.operation);
            }
        }
    }

    println!("\nConversion path: PHIR-JSON → PHIR Module");
    if output_path.is_some() {
        println!("Debug path: PHIR Module → PHIR-RON (for inspection)");
    }

    Ok(())
}
