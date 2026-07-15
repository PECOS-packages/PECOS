use pecos_wasm::{ForeignObject, WasmForeignObject};
use std::path::PathBuf;

fn fixture_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("wat")
        .join(name)
}

#[test]
fn finite_start_function_is_not_interrupted_during_instantiation() {
    let wat_path = fixture_path("finite_start_loop.wat");
    let mut wasm = WasmForeignObject::with_timeout(wat_path, 1.0)
        .expect("a finite start function should complete within its timeout");

    // init() creates a second instance, so this also covers re-instantiation.
    wasm.init()
        .expect("re-instantiation should re-arm the epoch deadline");
}
