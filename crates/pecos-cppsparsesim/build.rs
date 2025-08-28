fn main() {
    // Build C++ source files
    let mut build = cc::Build::new();
    build
        .cpp(true)
        .file("src/sparsesim.cpp")
        .file("src/cxx_shim.cpp")
        .include("src");

    // Use C++14 or newer to avoid issues with older cross-compilers
    // that don't fully support C++11 type traits like is_trivially_move_constructible
    let target = std::env::var("TARGET").unwrap_or_default();

    // For cross-compilation (especially aarch64), we need at least C++14
    // to ensure type traits are available
    if target.contains("aarch64") || target.contains("arm") {
        // Try C++17 first, fall back to C++14
        if build.is_flag_supported("-std=c++17").unwrap_or(false) {
            build.std("c++17");
        } else {
            build.std("c++14");
        }
    } else {
        build.std("c++14");
    }

    build.compile("sparsesim");

    // Generate cxx bridge code with same C++ standard
    let mut bridge = cxx_build::bridge("src/lib.rs");
    bridge.file("src/cxx_shim.cpp");

    // Match the same C++ standard for cxx bridge
    if target.contains("aarch64") || target.contains("arm") {
        if bridge.is_flag_supported("-std=c++17").unwrap_or(false) {
            bridge.std("c++17");
        } else {
            bridge.std("c++14");
        }
    } else {
        bridge.std("c++14");
    }

    bridge.compile("cppsparsesim-bridge");

    // Tell cargo to rerun if source files change
    println!("cargo:rerun-if-changed=src/lib.rs");
    println!("cargo:rerun-if-changed=src/sparsesim.cpp");
    println!("cargo:rerun-if-changed=src/sparsesim.h");
    println!("cargo:rerun-if-changed=src/cxx_shim.cpp");
    println!("cargo:rerun-if-changed=src/cxx_shim.h");
}
