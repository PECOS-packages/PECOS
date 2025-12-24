/// Build script for pecos-rslib.
///
/// Note: When building via maturin (the recommended approach), most of this
/// configuration is handled automatically. This build.rs primarily provides
/// compatibility for direct `cargo build` usage on macOS.
///
/// See: <https://pyo3.rs/v0.23.4/building-and-distribution>
fn main() {
    // For macOS, add required linker args for Python extension modules.
    // This is only needed for manual `cargo build` - maturin handles this automatically.
    #[cfg(target_os = "macos")]
    {
        // Link against the system C++ library from dyld shared cache
        // Prioritize /usr/lib to prevent opportunistic linking to Homebrew's libunwind
        println!("cargo:rustc-link-search=native=/usr/lib");
        println!("cargo:rustc-link-lib=c++");
        println!("cargo:rustc-link-arg=-Wl,-search_paths_first");
    }
}
