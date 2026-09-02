use std::{env, fs, path::PathBuf};

fn main() {
    println!("cargo:rerun-if-env-changed=FRONTIER_DEM_PATH");
    println!("cargo:rerun-if-changed=model.dem");

    let manifest_dir = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let source = env::var_os("FRONTIER_DEM_PATH")
        .map_or_else(|| manifest_dir.join("model.dem"), PathBuf::from);
    println!("cargo:rerun-if-changed={}", source.display());
    let output = PathBuf::from(env::var_os("OUT_DIR").unwrap()).join("model.dem");

    let dem = fs::read_to_string(&source)
        .unwrap_or_else(|error| panic!("failed to read DEM {}: {error}", source.display()));
    if dem.lines().any(|line| {
        let line = line.trim_start();
        line.starts_with("repeat") || line.starts_with("shift_detectors")
    }) {
        panic!(
            "{} is not flattened; flatten it first (for example with stim.DetectorErrorModel.flattened())",
            source.display()
        );
    }
    fs::write(output, dem).expect("failed to stage embedded DEM");
}
