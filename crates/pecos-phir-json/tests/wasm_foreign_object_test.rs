#[cfg(all(test, feature = "wasm"))]
mod tests {
    use pecos_phir_json::v0_1::foreign_objects::ForeignObject;
    use pecos_phir_json::v0_1::wasm_foreign_object::WasmtimeForeignObject;
    use std::path::Path;
    // Box is imported automatically, no need to explicitly import it

    #[test]
    fn test_wasm_foreign_object_from_wat() {
        // Use the CARGO_MANIFEST_DIR environment variable to get the absolute path
        let test_wat_path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("assets")
            .join("add.wat");

        // Create WebAssembly foreign object
        let mut foreign_object = WasmtimeForeignObject::new(&test_wat_path).unwrap();

        // Initialize
        foreign_object.init().unwrap();

        // Get available functions
        let funcs = foreign_object.get_funcs();
        assert!(funcs.contains(&"add".to_string()));

        // Execute add function
        let result = foreign_object.exec("add", &[3, 4]).unwrap();
        assert_eq!(result[0], 7);
    }
}
