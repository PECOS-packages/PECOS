//! Test to ensure Windows stub generation is consistent with runtime.rs exports

use std::fs;
use std::path::Path;

// Include the stub generator module
#[allow(dead_code)]
#[path = "../src/platform/windows_stub_gen.rs"]
mod windows_stub_gen;

#[test]
fn test_exports_match_runtime() {
    // This test runs on all platforms to ensure consistency
    let runtime_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/runtime.rs");
    let content = fs::read_to_string(runtime_path).expect("Failed to read runtime.rs");

    let runtime_exports = extract_runtime_exports(&content);

    // Get the list of exports from the actual EXPORTED_FUNCTIONS constant
    let stub_exports: Vec<&str> = windows_stub_gen::EXPORTED_FUNCTIONS
        .iter()
        .map(|f| f.name)
        .filter(|&name| name != "main") // main is special, not in runtime.rs
        .collect();

    // Check for missing exports
    let mut missing = Vec::new();
    for export in &runtime_exports {
        if !stub_exports.contains(&export.as_str()) {
            missing.push(export.clone());
        }
    }

    assert!(
        missing.is_empty(),
        "\nWindows stub generator is missing the following exports from runtime.rs:\n  {}\n\
             Please update EXPORTED_FUNCTIONS in src/platform/windows_stub_gen.rs",
        missing.join("\n  ")
    );

    // Check for extra exports
    let mut extra = Vec::new();
    for &export in &stub_exports {
        if !runtime_exports.contains(&export.to_string()) {
            extra.push(export.to_string());
        }
    }

    assert!(
        extra.is_empty(),
        "\nWindows stub generator has the following exports not found in runtime.rs:\n  {}\n\
             Please update EXPORTED_FUNCTIONS in src/platform/windows_stub_gen.rs",
        extra.join("\n  ")
    );

    println!("Windows stub exports are consistent with runtime.rs");
}

fn extract_runtime_exports(content: &str) -> Vec<String> {
    let mut exports = Vec::new();

    // Simple line-based parsing for #[unsafe(no_mangle)] functions
    let lines: Vec<&str> = content.lines().collect();
    for i in 0..lines.len() {
        if lines[i].contains("#[unsafe(no_mangle)]") {
            // Look at the next few lines for the function name
            for j in 1..=3 {
                if i + j < lines.len()
                    && let Some(func_name) = extract_function_name(lines[i + j])
                {
                    exports.push(func_name);
                    break;
                }
            }
        }
    }

    exports
}

fn extract_function_name(line: &str) -> Option<String> {
    // Match: pub unsafe extern "C" fn function_name(
    if line.contains("pub") && line.contains("extern") && line.contains("fn") {
        let parts: Vec<&str> = line.split_whitespace().collect();
        for i in 0..parts.len() {
            if parts[i] == "fn" && i + 1 < parts.len() {
                // Extract function name (remove parentheses)
                let name = parts[i + 1].split('(').next()?;
                return Some(name.to_string());
            }
        }
    }
    None
}

#[test]
fn test_def_and_stub_generation() {
    // Test that the generation functions work correctly
    let def_content = windows_stub_gen::generate_def_file();
    assert!(def_content.contains("EXPORTS"));
    assert!(def_content.contains("qir_runtime_reset"));
    assert!(def_content.contains("__quantum__qis__h__body"));
    assert!(def_content.contains("main @1 NONAME"));

    let c_content = windows_stub_gen::generate_c_stub();
    assert!(c_content.contains("BinaryCommands"));
    assert!(c_content.contains("_DllMainCRTStartup"));
    assert!(c_content.contains("__quantum__qis__h__body(int qubit)"));
    assert!(!c_content.contains("main()")); // main should not be in stub
}
