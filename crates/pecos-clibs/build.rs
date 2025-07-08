use std::env;
use std::path::PathBuf;

fn main() {
    // Get the root directory of the workspace
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let manifest_path = PathBuf::from(&manifest_dir);
    let workspace_root = manifest_path
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf();
    
    // Build the C libraries from clibs/
    let clibs_root = workspace_root.join("clibs");
    
    // Build rng_pcg library
    let rng_pcg_path = clibs_root.join("rng_pcg");
    cc::Build::new()
        .file(rng_pcg_path.join("rng_pcg.c"))
        .include(&rng_pcg_path)
        .compile("rng_pcg");
    
    // To add a new C library:
    // 1. Create a new directory in clibs/ with your library files
    // 2. Add a similar cc::Build block here
    // 3. Add FFI bindings in src/lib.rs
    
    // Tell Cargo to rerun build script if C source changes
    println!("cargo:rerun-if-changed={}", rng_pcg_path.join("rng_pcg.c").display());
    println!("cargo:rerun-if-changed={}", rng_pcg_path.join("rng_pcg.h").display());
}