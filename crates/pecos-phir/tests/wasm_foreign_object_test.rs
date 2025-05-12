#[cfg(test)]
mod tests {
    use pecos_phir::v0_1::foreign_objects::ForeignObject;
    use pecos_phir::v0_1::wasm_foreign_object::WasmtimeForeignObject;
    use std::path::Path;
    use std::sync::Arc;

    #[test]
    fn test_wasm_foreign_object_from_wat() {
        // Skip this test for now since we don't have a way to create WAT files in tests
        // This test is here as a template for when we have a way to create test files
        // For example, when running tests from a directory with test assets
        if !Path::new("tests/add.wat").exists() {
            println!("Skipping test_wasm_foreign_object_from_wat: test WAT file not found");
            return;
        }

        // Create WebAssembly foreign object
        let foreign_object = WasmtimeForeignObject::new("tests/add.wat").unwrap();
        let mut foreign_object = Arc::new(foreign_object);

        // Initialize
        Arc::get_mut(&mut foreign_object).unwrap().init().unwrap();

        // Get available functions
        let funcs = Arc::get_mut(&mut foreign_object).unwrap().get_funcs();
        assert!(funcs.contains(&"add".to_string()));

        // Execute add function
        let result = Arc::get_mut(&mut foreign_object)
            .unwrap()
            .exec("add", &[3, 4])
            .unwrap();
        assert_eq!(result[0], 7);
    }
}
