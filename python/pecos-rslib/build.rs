/// This build script helps with `PyO3` configuration.
fn main() {
    // Ensure rebuild when build.rs itself changes
    println!("cargo:rerun-if-changed=build.rs");
    // Ensure rebuild when any source files change
    println!("cargo:rerun-if-changed=src");
    // Ensure rebuild when config files change
    println!("cargo:rerun-if-changed=Cargo.toml");
    println!("cargo:rerun-if-changed=pyproject.toml");

    // For macOS, add required linker args for Python extension modules
    #[cfg(target_os = "macos")]
    {
        pyo3_build_config::add_extension_module_link_args();

        // Link against the system C++ library from dyld shared cache
        // Prioritize /usr/lib to prevent opportunistic linking to Homebrew's libunwind
        println!("cargo:rustc-link-search=native=/usr/lib");
        println!("cargo:rustc-link-lib=c++");
        println!("cargo:rustc-link-arg=-Wl,-search_paths_first");
    }
}
