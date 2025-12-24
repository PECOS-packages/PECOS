use pecos_phir::{PhirConfig, compile_hugr_bytes_via_phir};

#[test]
fn test_compile_hugr_bytes() {
    // Simple HUGR JSON test case
    let hugr_json = r#"{
        "version": "v1",
        "modules": [{
            "parent": 0,
            "nodes": []
        }]
    }"#;

    let config = PhirConfig::default();
    let result = compile_hugr_bytes_via_phir(hugr_json.as_bytes(), &config);

    // The function should now work without the "Binary HUGR format not yet supported" error
    match result {
        Ok(mlir_text) => {
            println!("Successfully compiled HUGR to MLIR: {mlir_text}");
            assert!(!mlir_text.is_empty());
        }
        Err(e) => {
            // Should not get the binary format error anymore
            assert!(
                !e.to_string()
                    .contains("Binary HUGR format not yet supported")
            );
        }
    }
}
