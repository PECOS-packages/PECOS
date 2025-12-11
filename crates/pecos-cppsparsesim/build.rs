use std::env;

/// Get the PECOS build profile from environment
/// Returns "dev", "release", or "native"
fn get_pecos_profile() -> String {
    env::var("PECOS_PROFILE").unwrap_or_else(|_| {
        // Fall back to detecting from OPT_LEVEL if PECOS_PROFILE not set
        let opt_level = env::var("OPT_LEVEL").unwrap_or_else(|_| "0".to_string());
        if opt_level == "0" {
            "dev".to_string()
        } else {
            "release".to_string()
        }
    })
}

/// Apply PECOS profile optimization flags to a `cc::Build`
fn apply_profile_flags(build: &mut cc::Build, target: &str) {
    let profile = get_pecos_profile();

    if target.contains("windows") {
        // MSVC optimization flags
        match profile.as_str() {
            "native" => {
                build.opt_level(2); // /O2
                build.flag_if_supported("/arch:AVX2"); // Common native optimization for modern CPUs
            }
            "release" => {
                build.opt_level(2); // /O2
            }
            _ => {
                // Dev: use default (no optimization)
            }
        }
    } else {
        // GCC/Clang optimization flags
        match profile.as_str() {
            "native" => {
                build.flag_if_supported("-O3");
                build.flag_if_supported("-march=native");
            }
            "release" => {
                build.flag_if_supported("-O3");
            }
            _ => {
                // Dev: no optimization for fastest compile
                build.flag_if_supported("-O0");
            }
        }
    }
}

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
    let target = env::var("TARGET").unwrap_or_default();

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

    // On Windows, embed debug info in .obj files (no PDB) for parallel build reliability
    if target.contains("windows") {
        build.flag("/Z7");
    }

    // Apply PECOS profile optimization flags
    apply_profile_flags(&mut build, &target);

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

    // On macOS, use the -stdlib=libc++ flag to ensure proper C++ standard library linkage
    if target.contains("darwin") {
        bridge.flag("-stdlib=libc++");
        // Note: Linker-specific flags are passed via cargo:rustc-link-arg below, not here
    }

    // On Windows, embed debug info in .obj files (no PDB) for parallel build reliability
    if target.contains("windows") {
        bridge.flag("/Z7");
    }

    // Apply PECOS profile optimization flags to bridge
    apply_profile_flags(&mut bridge, &target);

    bridge.compile("cppsparsesim-bridge");

    // On macOS, link against the system C++ library from dyld shared cache
    if target.contains("darwin") {
        println!("cargo:rustc-link-search=native=/usr/lib");
        println!("cargo:rustc-link-lib=c++");
        println!("cargo:rustc-link-arg=-Wl,-search_paths_first");
    }

    // Tell cargo to rerun if source files change
    println!("cargo:rerun-if-changed=src/lib.rs");
    println!("cargo:rerun-if-changed=src/sparsesim.cpp");
    println!("cargo:rerun-if-changed=src/sparsesim.h");
    println!("cargo:rerun-if-changed=src/cxx_shim.cpp");
    println!("cargo:rerun-if-changed=src/cxx_shim.h");
    println!("cargo:rerun-if-env-changed=PECOS_PROFILE");
}
