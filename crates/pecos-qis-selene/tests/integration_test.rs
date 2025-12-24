use pecos_qis_core::{ProgramFormat, QisInterface};
use pecos_qis_selene::QisHeliosInterface;

#[test]
fn test_simple_bell_state() {
    // Use the Helios library path set by build.rs
    let helios_lib = env!("HELIOS_LIB_PATH");
    unsafe {
        std::env::set_var("HELIOS_LIB_PATH", helios_lib);
    }

    // Read the test LLVM IR
    let ll_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/test_data/simple_bell.ll"
    );
    let ll_contents = std::fs::read_to_string(ll_path).expect("Failed to read test LLVM IR file");

    // Create interface and load program
    let mut interface = QisHeliosInterface::new();
    interface
        .load_program(ll_contents.as_bytes(), ProgramFormat::LlvmIrText)
        .expect("Failed to load program");

    // Collect operations
    let operations = interface
        .collect_operations()
        .expect("Failed to collect operations");

    // Verify operations were collected
    println!("Collected {} operations", operations.operations.len());
    println!("Operations: {:#?}", operations.operations);

    // Should have:
    // - 2 AllocateQubit operations
    // - 1 H gate
    // - 1 CX gate
    // - 2 Measure operations
    // - 2 ReleaseQubit operations
    assert!(
        operations.operations.len() >= 6,
        "Expected at least 6 operations, got {}",
        operations.operations.len()
    );
}
