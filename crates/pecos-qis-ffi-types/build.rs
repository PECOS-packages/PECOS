use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use std::{env, fs, process::Command};

fn add_field(hash: &mut Sha256, name: &str, value: &[u8]) {
    hash.update((name.len() as u64).to_le_bytes());
    hash.update(name.as_bytes());
    hash.update((value.len() as u64).to_le_bytes());
    hash.update(value);
}

fn rust_sources(dir: &Path, files: &mut Vec<PathBuf>) {
    for entry in fs::read_dir(dir).expect("read shared QIS types sources") {
        let path = entry.expect("read source entry").path();
        if path.is_dir() {
            rust_sources(&path, files);
        } else if path.extension().is_some_and(|ext| ext == "rs") {
            files.push(path);
        }
    }
}

fn main() {
    let root = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").expect("Cargo manifest directory"));
    let mut hash = Sha256::new();
    add_field(&mut hash, "format", b"pecos-qis-rust-build-v1");
    let mut files = vec![root.join("Cargo.toml"), root.join("build.rs")];
    rust_sources(&root.join("src"), &mut files);
    files.sort();
    // Watch the directory too, so adding a new shared source changes the stamp.
    println!("cargo:rerun-if-changed=src");
    for path in files {
        let relative = path.strip_prefix(&root).unwrap();
        println!("cargo:rerun-if-changed={}", relative.display());
        let name = relative
            .to_str()
            .expect("UTF-8 source path")
            .replace('\\', "/");
        add_field(
            &mut hash,
            &name,
            &fs::read(&path).expect("read shared source"),
        );
    }
    println!("cargo:rerun-if-env-changed=RUSTC");
    let compiler = Command::new(env::var_os("RUSTC").expect("Cargo Rust compiler"))
        .arg("-vV")
        .output()
        .expect("query Rust compiler identity");
    assert!(compiler.status.success(), "rustc -vV failed");
    add_field(&mut hash, "rustc", &compiler.stdout);

    // Cargo exposes only the filename stem for a custom JSON target, so it
    // cannot identify that specification reliably. Support built-in targets;
    // do not silently stamp an unknown target without its layout definition.
    let targets = Command::new(env::var_os("RUSTC").expect("Cargo Rust compiler"))
        .args(["--print", "target-list"])
        .output()
        .expect("query built-in Rust targets");
    assert!(targets.status.success(), "rustc --print target-list failed");
    let target = env::var("TARGET").expect("Cargo target");
    assert!(
        std::str::from_utf8(&targets.stdout)
            .expect("UTF-8 Rust target list")
            .lines()
            .any(|built_in| built_in == target),
        "Custom Rust target {target:?} is unsupported by the QIS native build fingerprint; use a built-in Rust target"
    );

    for key in [
        "TARGET",
        "PROFILE",
        "DEBUG",
        "OPT_LEVEL",
        "CARGO_ENCODED_RUSTFLAGS",
    ] {
        println!("cargo:rerun-if-env-changed={key}");
        let value = env::var(key).unwrap_or_else(|_| panic!("Cargo did not provide {key}"));
        add_field(&mut hash, key, value.as_bytes());
    }
    let mut cfg = env::vars()
        .filter(|(key, _)| key.starts_with("CARGO_CFG_") || key.starts_with("CARGO_FEATURE_"))
        .collect::<Vec<_>>();
    cfg.sort();
    for (key, value) in cfg {
        println!("cargo:rerun-if-env-changed={key}");
        add_field(&mut hash, &key, value.as_bytes());
    }
    let digest = hash.finalize();
    let words = digest
        .as_chunks::<8>()
        .0
        .iter()
        .map(|chunk| {
            let hex = format!("{:016x}", u64::from_be_bytes(*chunk));
            format!(
                "0x{}_{}_{}_{}",
                &hex[..4],
                &hex[4..8],
                &hex[8..12],
                &hex[12..]
            )
        })
        .collect::<Vec<_>>()
        .join(", ");
    let output = PathBuf::from(env::var_os("OUT_DIR").expect("Cargo output directory"));
    fs::write(
        output.join("abi_fingerprint.rs"),
        format!("pub const QIS_FFI_ABI_FINGERPRINT: [u64; 4] = [{words}];\n"),
    )
    .expect("write shared QIS build fingerprint");
}
