#![cfg(unix)]

/// Discriminates on Linux; macOS binds the fixture's import at load either way.
#[test]
fn rejects_unresolved_import_at_load() {
    let directory = tempfile::tempdir().unwrap();
    let source = directory.path().join("missing_import.c");
    let library = directory.path().join(if cfg!(target_os = "macos") {
        "missing_import.dylib"
    } else {
        "missing_import.so"
    });
    std::fs::write(
        &source,
        "extern void pecos_test_missing_import(void);\n\
         void pecos_test_export(void) { pecos_test_missing_import(); }\n",
    )
    .unwrap();
    let mut compiler = std::process::Command::new("cc");
    compiler.args(["-shared", "-fPIC"]);
    #[cfg(target_os = "linux")]
    compiler.arg("-Wl,-z,lazy");
    #[cfg(target_os = "macos")]
    compiler.arg("-Wl,-undefined,dynamic_lookup");
    let output = compiler
        .arg(&source)
        .arg("-o")
        .arg(&library)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "C fixture compilation failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let Err(error) = pecos_foreign::discovery::load_plugin(&library) else {
        panic!("loaded a plugin with an unresolved import");
    };
    assert!(
        matches!(
            error,
            pecos_foreign::discovery::PluginError::LoadFailed(_, _)
        ),
        "{error:?}"
    );
    assert!(
        error.to_string().contains("pecos_test_missing_import"),
        "unexpected loader error: {error}"
    );
}
