//! Build script for LDPC decoder integration

use log::info;
use pecos_build::{Error, Manifest, Result, ensure_dep_ready, report_cache_config};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

/// Get the build profile from Cargo's environment
/// Returns "debug", "release", or "native"
///
/// Note: Cargo's PROFILE env var only reports "debug" or "release" even for custom profiles
/// (due to backward compatibility - see RFC 2678). Custom profiles inherit from these base
/// profiles, so PROFILE reflects the parent. To detect custom profiles like "native", we
/// check the `OUT_DIR` path which contains the actual profile directory name.
///
/// Profile behavior:
/// - "debug" -> no C++ optimization, fast compile
/// - "release" -> full optimization (-O3)
/// - "native" -> full optimization + CPU-specific (-O3 -march=native)
fn get_build_profile() -> String {
    // First check OUT_DIR for custom profile name (e.g., target/native/build/...)
    // Custom profiles get their own directory under target/
    if let Ok(out_dir) = env::var("OUT_DIR") {
        // OUT_DIR looks like: .../target/<profile>/build/<crate>-<hash>/out
        // We want to extract <profile>
        let parts: Vec<&str> = out_dir.split(std::path::MAIN_SEPARATOR).collect();
        if let Some(target_idx) = parts.iter().position(|&p| p == "target")
            && let Some(profile_name) = parts.get(target_idx + 1)
        {
            return match *profile_name {
                "native" => "native",
                "release" => "release",
                "debug" => "debug",
                _ => {
                    // Unknown profile, fall back to PROFILE env var
                    if env::var("PROFILE").as_deref() == Ok("release") {
                        "release"
                    } else {
                        "debug"
                    }
                }
            }
            .to_string();
        }
    }

    // Fallback to PROFILE env var (will be "debug" or "release")
    match env::var("PROFILE").as_deref() {
        Ok("release") => "release".to_string(),
        _ => "debug".to_string(),
    }
}

/// Main build function for LDPC
pub fn build() -> Result<()> {
    // Tell Cargo when to rerun this build script
    println!("cargo:rerun-if-changed=build_ldpc.rs");
    println!("cargo:rerun-if-changed=src/bridge.rs");
    println!("cargo:rerun-if-changed=src/bridge.cpp");
    println!("cargo:rerun-if-changed=include/ldpc_ffi.h");

    // Also rerun if the user forces a rebuild
    println!("cargo:rerun-if-env-changed=FORCE_REBUILD");

    let out_dir = PathBuf::from(env::var("OUT_DIR")?);

    // Always emit link directives - these are cached by Cargo
    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=ldpc-bridge");

    // Get LDPC source (downloads to ~/.pecos/cache/, extracts to ~/.pecos/deps/)
    let manifest = Manifest::find_and_load_validated()?;
    let ldpc_dir = ensure_dep_ready("ldpc", &manifest)?;

    // Build using cxx
    build_cxx_bridge(&ldpc_dir)?;

    Ok(())
}

fn fix_header_guard_conflict(src_cpp_dir: &Path) -> Result<()> {
    // Fix the header guard conflict in union_find.hpp
    // Both union_find.hpp and lsd.hpp use the same header guard UF2_H
    // This causes compilation errors when both headers are included
    // Issue exists in commit 31cf9f33872f32579af1efbe1e84552d42b03ea8
    let union_find_path = src_cpp_dir.join("union_find.hpp");

    if union_find_path.exists() {
        let content = fs::read_to_string(&union_find_path)?;

        // Only apply patch if not already applied
        if content.contains("#ifndef UF2_H") {
            let fixed_content = content
                .replace("#ifndef UF2_H", "#ifndef UNION_FIND_H")
                .replace("#define UF2_H", "#define UNION_FIND_H");
            fs::write(&union_find_path, fixed_content)?;
            info!("Fixed header guard conflict in union_find.hpp");
        }
    }

    Ok(())
}

/// Replace a pinned MBP fragment, accepting exactly one complete source version.
fn patch_mbp_fragment(
    content: &mut String,
    path: &Path,
    upstream: &str,
    patched: &str,
    count: usize,
) -> Result<()> {
    // Some patches retain the upstream anchor (includes and decision guards).
    let old_count = content.replace(patched, "").matches(upstream).count();
    let new_count = content.matches(patched).count();
    if old_count == 0 && new_count == count {
        return Ok(());
    }
    if old_count != count || new_count != 0 {
        return Err(Error::Config(format!(
            "{} does not contain the expected MBP fragment {count} time(s): {upstream:?}; \
             the pinned ldpc source changed or is partially patched, so update the MBP fixes in build_ldpc.rs",
            path.display()
        )));
    }
    *content = content.replace(upstream, patched);
    Ok(())
}

fn read_mbp_header(path: &Path) -> Result<String> {
    fs::read_to_string(path).map_err(|error| {
        Error::Config(format!(
            "Cannot read {}: {error}; check the path and permissions",
            path.display()
        ))
    })
}

fn write_mbp_header(path: &Path, content: &str) -> Result<()> {
    // Shared cache: publish atomically so other checkouts never read a partial header.
    let temporary = path.with_extension(format!("hpp.pecos-{}.tmp", std::process::id()));
    fs::write(&temporary, content).map_err(|error| {
        Error::Config(format!(
            "Cannot write {}: {error}; check the path and permissions",
            temporary.display()
        ))
    })?;
    fs::rename(&temporary, path).map_err(|error| {
        // A failed publication must not leave a stale process-private header.
        let _ = fs::remove_file(&temporary);
        Error::Config(format!(
            "Cannot rename {} to {}: {error}; check the path and permissions",
            temporary.display(),
            path.display()
        ))
    })?;
    Ok(())
}

fn fix_mbp(src_cpp_dir: &Path) -> Result<()> {
    let path = src_cpp_dir.join("mbp.hpp");
    let original = read_mbp_header(&path)?;
    let mut content = original.clone();
    fix_mbp_iterate_methods(&mut content, &path)?;
    fix_mbp_numeric_guards(&mut content, &path)?;
    if content != original {
        // Publish every MBP edit together. Concurrent first builds must never
        // replace a fully patched header with a loops-only intermediate version.
        write_mbp_header(&path, &content)?;
        info!("Fixed MBP message passing and numeric guards");
    }
    Ok(())
}

fn fix_mbp_iterate_methods(content: &mut String, path: &Path) -> Result<()> {
    // Upgrade both the pinned pointer syntax and the old PECOS patch. The latter
    // used the correct iterator API but copied entries, losing every message write
    // and comparing addresses of locals instead of excluding the current edge.
    if content.contains("iterate_column_ptr(") || content.contains("iterate_row_ptr(") {
        for (upstream, patched, count) in [
            ("pcm->iterate_column_ptr(", "pcm->iterate_column(", 5),
            ("pcm->iterate_row_ptr(", "pcm->iterate_row(", 2),
            ("if (g != e)", "if (&g != &e)", 2),
        ] {
            patch_mbp_fragment(content, path, upstream, patched, count)?;
        }
        // Exact access counts also detect changes to the pinned pointer version.
        for (upstream, patched, count, existing) in [
            ("e->pauli", "e.pauli", 3, 1),
            ("e->qubit_to_stab_msgs", "e.qubit_to_stab_msgs", 4, 0),
            ("e->stab_to_qubit_msgs", "e.stab_to_qubit_msgs", 8, 0),
            ("e->row_index", "e.row_index", 2, 0),
            ("g->pauli", "g.pauli", 4, 0),
            ("g->qubit_to_stab_msgs", "g.qubit_to_stab_msgs", 4, 0),
            ("g->col_index", "g.col_index", 1, 0),
        ] {
            if content.matches(upstream).count() != count
                || content.matches(patched).count() != existing
            {
                return Err(Error::Config(format!(
                    "{} has unexpected MBP pointer accesses for {upstream}; \
                     update fix_mbp_iterate_methods in build_ldpc.rs for the pinned ldpc source",
                    path.display()
                )));
            }
            *content = content.replace(upstream, patched);
        }
    }
    for (upstream, patched, count) in [
        (
            "for (auto e: pcm->iterate_column(i))",
            "for (auto& e: pcm->iterate_column(i))",
            1,
        ),
        (
            "for (auto e: pcm->iterate_column(qubit))",
            "for (auto& e: pcm->iterate_column(qubit))",
            4,
        ),
        (
            "for (auto g: pcm->iterate_row(stab_index))",
            "for (auto& g: pcm->iterate_row(stab_index))",
            2,
        ),
    ] {
        patch_mbp_fragment(content, path, upstream, patched, count)?;
    }
    if content.matches("if (&g != &e)").count() != 2
        || content.contains("for (auto e:")
        || content.contains("for (auto g:")
    {
        return Err(Error::Config(format!(
            "{} has unexpected MBP edge iteration or exclusion; \
             update fix_mbp_iterate_methods in build_ldpc.rs for the pinned ldpc source",
            path.display()
        )));
    }
    Ok(())
}

fn fix_mbp_numeric_guards(content: &mut String, path: &Path) -> Result<()> {
    // Upgrade either the original no-op guard or the previous zero-only fix.
    const ORIGINAL: &str = "if (demoninator <= 0) {\n                                    stab_update_product = 1e-12;\n                                }";
    const ZERO_ONLY: &str = "if (demoninator <= 0) {\n                                    demoninator = 1e-12;\n                                }";
    const FLOOR: &str = "demoninator = std::max(demoninator, 1e-12);";
    let previous = if content.contains(ORIGINAL) {
        ORIGINAL
    } else {
        ZERO_ONLY
    };
    patch_mbp_fragment(content, path, previous, FLOOR, 2)?;
    for (upstream, patched, count) in [
        (
            "#include <iostream>\n",
            "#include <algorithm>\n#include <iostream>\n",
            1,
        ),
        (
            "else if (stab_update_product <= -1) stab_update_product = 1 + 1e-8;",
            "else if (stab_update_product <= -1) stab_update_product = -(1 - 1e-8);",
            1,
        ),
        (
            "#include <utility>\n",
            "#include <stdexcept>\n#include <string>\n#include <utility>\n",
            1,
        ),
        (
            "                    if (log_prob_ratios[w][qubit] < min) {",
            "                    // PECOS FIX: reject NaN before the hard decision.\n                    if (std::isnan(log_prob_ratios[w][qubit])) {\n                        throw std::runtime_error(\n                            \"MBP log-probability ratio for qubit \" + std::to_string(qubit) +\n                            \" Pauli \" + std::to_string(w + 1) + \" is NaN\");\n                    }\n                    if (log_prob_ratios[w][qubit] < min) {",
            1,
        ),
    ] {
        patch_mbp_fragment(content, path, upstream, patched, count)?;
    }
    Ok(())
}

fn fix_msvc_compatibility(src_cpp_dir: &Path) -> Result<()> {
    // Fix MSVC compatibility issues in lsd.hpp
    // MSVC doesn't recognize 'or' keyword without #include <iso646.h>
    // Also need to fix some C++17 syntax issues
    let lsd_path = src_cpp_dir.join("lsd.hpp");

    if lsd_path.exists() {
        let content = fs::read_to_string(&lsd_path)?;

        // Only apply patch if not already applied
        if !content.contains("#include <iso646.h>") {
            // Add iso646.h include for MSVC compatibility at the top
            let lines: Vec<&str> = content.lines().collect();
            let mut new_lines = Vec::new();

            // Find the first #include and add our fix before it
            let mut added_includes = false;
            for line in &lines {
                if !added_includes && line.starts_with("#include") {
                    new_lines.push("#ifdef _MSC_VER".to_string());
                    new_lines.push("#include <iso646.h>".to_string());
                    new_lines.push("#endif".to_string());
                    added_includes = true;
                }

                // Replace 'or' with '||' for better MSVC compatibility
                let fixed_line = line.replace(" or ", " || ");
                new_lines.push(fixed_line);
            }

            let fixed_content = new_lines.join("\n");
            fs::write(&lsd_path, fixed_content)?;
            info!("Fixed MSVC compatibility issues in lsd.hpp");
        }
    }

    Ok(())
}

fn fix_uninitialized_pointers(src_cpp_dir: &Path) -> Result<()> {
    // Fix uninitialized pointer bug in osd.hpp
    // The LuDecomposition pointer is not initialized, but the destructor always tries to delete it
    // When osd_method == OSD_OFF, osd_setup() returns early without initializing LuDecomposition
    // This causes heap corruption when the destructor tries to delete uninitialized memory
    // Issue exists in commit 31cf9f33872f32579af1efbe1e84552d42b03ea8
    let osd_path = src_cpp_dir.join("osd.hpp");

    if osd_path.exists() {
        let mut content = fs::read_to_string(&osd_path)?;
        let mut modified = false;

        // Fix 1: Initialize LuDecomposition pointer
        if content.contains("RowReduce<ldpc::bp::BpEntry>* LuDecomposition;")
            && !content.contains("LuDecomposition = nullptr")
        {
            content = content.replace(
                "ldpc::gf2sparse_linalg::RowReduce<ldpc::bp::BpEntry>* LuDecomposition;",
                "ldpc::gf2sparse_linalg::RowReduce<ldpc::bp::BpEntry>* LuDecomposition = nullptr;",
            );

            content = content.replace(
                "~OsdDecoder(){\n            delete this->LuDecomposition;\n        };",
                "~OsdDecoder(){\n            if (this->LuDecomposition) delete this->LuDecomposition;\n        };",
            );
            modified = true;
            info!("Fixed uninitialized pointer bug in osd.hpp");
        }

        // Fix 2: Fix buffer overflow in COMBINATION_SWEEP when osd_order > k
        // The code loops i,j from 0 to osd_order but accesses candidate[i] and candidate[j]
        // where candidate has size k. When osd_order > k, this is out of bounds.
        if content.contains("for(int i = 0; i<this->osd_order;i++){")
            && !content.contains("// PECOS FIX: clamp to k")
        {
            content = content.replace(
                "            if(this->osd_method == COMBINATION_SWEEP){\n                for(int i=0; i<k; i++) {\n                    std::vector<uint8_t> osd_candidate;\n                    osd_candidate.resize(k,0);\n                    osd_candidate[i]=1; \n                    this->osd_candidate_strings.push_back(osd_candidate);\n                }\n\n                for(int i = 0; i<this->osd_order;i++){\n                    for(int j = 0; j<this->osd_order; j++){",
                "            if(this->osd_method == COMBINATION_SWEEP){\n                for(int i=0; i<k; i++) {\n                    std::vector<uint8_t> osd_candidate;\n                    osd_candidate.resize(k,0);\n                    osd_candidate[i]=1; \n                    this->osd_candidate_strings.push_back(osd_candidate);\n                }\n\n                // PECOS FIX: clamp to k to prevent buffer overflow when osd_order > k\n                int max_order = (this->osd_order < k) ? this->osd_order : k;\n                for(int i = 0; i<max_order;i++){\n                    for(int j = 0; j<max_order; j++){",
            );
            modified = true;
            info!("Fixed buffer overflow in OSD COMBINATION_SWEEP");
        }

        if modified {
            fs::write(&osd_path, content)?;
        }
    }

    Ok(())
}

fn fix_osd_dense_combination_sweep(src_cpp_dir: &Path) -> Result<()> {
    // osd_dense.hpp (used by the LSD decoder for each cluster) has the same COMBINATION_SWEEP
    // overflow as osd.hpp: candidates have size k (the cluster's non-pivot columns), but the
    // pair loops run to osd_order and write candidate[i] and candidate[j]. A cluster with
    // k < lsd_order writes past the vector. A pair with i < k <= j repeats the single-bit
    // candidate e_i; with both indices >= k, the first k entries are all zero and reproduce
    // the LSD-0 solution. Neither can win because selection uses strict <, so clamping to k
    // preserves results and removes the out-of-bounds writes.
    const UPSTREAM: &str = "                for (auto i = 0; i < this->osd_order; i++) {\n                    for (auto j = 0; j < this->osd_order; j++) {";
    const PATCHED: &str = "                // PECOS FIX: clamp to k to prevent buffer overflow when osd_order > k\n                int max_order = (this->osd_order < k) ? this->osd_order : k;\n                for (auto i = 0; i < max_order; i++) {\n                    for (auto j = 0; j < max_order; j++) {";
    let path = src_cpp_dir.join("osd_dense.hpp");
    let content = fs::read_to_string(&path).map_err(|error| {
        Error::Config(format!(
            "Cannot read {}: {error}; check the path and permissions",
            path.display()
        ))
    })?;
    if content.contains(PATCHED) {
        return Ok(());
    }
    if content.matches(UPSTREAM).count() != 1 {
        return Err(Error::Config(format!(
            "{} does not contain the expected COMBINATION_SWEEP loop exactly once; \
             the pinned ldpc source changed, so update fix_osd_dense_combination_sweep in build_ldpc.rs",
            path.display()
        )));
    }
    // The source lives in the shared ~/.pecos/deps cache: write a process-private file and
    // rename it over the original so a concurrent build never reads a partial file.
    let temporary = path.with_extension(format!("hpp.pecos-{}.tmp", std::process::id()));
    fs::write(&temporary, content.replace(UPSTREAM, PATCHED)).map_err(|error| {
        Error::Config(format!(
            "Cannot write {}: {error}; check the path and permissions",
            temporary.display()
        ))
    })?;
    fs::rename(&temporary, &path).map_err(|error| {
        Error::Config(format!(
            "Cannot rename {} to {}: {error}; check the path and permissions",
            temporary.display(),
            path.display()
        ))
    })?;
    info!("Fixed buffer overflow in dense OSD COMBINATION_SWEEP");
    Ok(())
}

fn bound_product_sum_messages(src_cpp_dir: &Path) -> Result<()> {
    // Interior priors can round tanh products to +/-1, making the inverse infinite
    // and subsequent BP arithmetic NaN. Bound both schedules before the inverse.
    const PARALLEL: &str = "                                e.check_to_bit_msg =\n                                        message_sign * std::log((1 + e.check_to_bit_msg) / (1 - e.check_to_bit_msg));";
    const SERIAL: &str = "                                e.check_to_bit_msg = pow(-1, syndrome[check_index]) *\n                                                     std::log((1 + e.check_to_bit_msg) / (1 - e.check_to_bit_msg));";
    const BOUND: &str = "                                // PECOS FIX: bound product-sum inverse; only x == +/-1 changes.\n                                // Every other representable tanh product is bit-identical.\n                                const double bound = std::nextafter(1.0, 0.0);\n                                if (e.check_to_bit_msg > bound) e.check_to_bit_msg = bound;\n                                if (e.check_to_bit_msg < -bound) e.check_to_bit_msg = -bound;\n";
    let path = src_cpp_dir.join("bp.hpp");
    let mut content = fs::read_to_string(&path).map_err(|error| {
        Error::Config(format!(
            "Cannot read {}: {error}; check the path and permissions",
            path.display()
        ))
    })?;
    let mut modified = false;
    for upstream in [PARALLEL, SERIAL] {
        let patched = format!("{BOUND}{upstream}");
        if content.matches(&patched).count() == 1 && content.matches(upstream).count() == 1 {
            continue;
        }
        if content.matches(upstream).count() != 1 || content.contains(&patched) {
            return Err(Error::Config(format!(
                "{} does not contain the expected product-sum inverse exactly once; the pinned ldpc source changed, so update bound_product_sum_messages in build_ldpc.rs",
                path.display()
            )));
        }
        content = content.replace(upstream, &patched);
        modified = true;
    }
    if modified {
        // Shared cache: publish atomically so other checkouts never read a partial header.
        let temporary = path.with_extension(format!("hpp.pecos-{}.tmp", std::process::id()));
        fs::write(&temporary, content).map_err(|error| {
            Error::Config(format!(
                "Cannot write {}: {error}; check the path and permissions",
                temporary.display()
            ))
        })?;
        fs::rename(&temporary, &path).map_err(|error| {
            Error::Config(format!(
                "Cannot rename {} to {}: {error}; check the path and permissions",
                temporary.display(),
                path.display()
            ))
        })?;
    }
    Ok(())
}

fn build_cxx_bridge(ldpc_dir: &Path) -> Result<()> {
    let src_cpp_dir = ldpc_dir.join("src_cpp");
    let include_dir = ldpc_dir.join("include");

    // Fix header guard conflict between union_find.hpp and lsd.hpp
    fix_header_guard_conflict(&src_cpp_dir)?;

    // Publish all MBP fixes as one atomic cache update.
    fix_mbp(&src_cpp_dir)?;

    // Fix MSVC compatibility issues
    fix_msvc_compatibility(&src_cpp_dir)?;

    // Fix uninitialized pointers that cause heap corruption on Windows
    fix_uninitialized_pointers(&src_cpp_dir)?;

    // Fix out-of-bounds writes in the LSD decoder's dense OSD
    fix_osd_dense_combination_sweep(&src_cpp_dir)?;
    bound_product_sum_messages(&src_cpp_dir)?;

    // Build the cxx bridge first to generate headers
    let mut build = cxx_build::bridge("src/bridge.rs");

    let target = env::var("TARGET").unwrap_or_default();

    // On macOS, explicitly use system clang to ensure SDK paths are correct.
    // The PECOS LLVM clang may be in PATH but doesn't have SDK headers configured,
    // causing "math.h file not found" errors during compilation.
    if target.contains("darwin") && env::var("CXX").is_err() && env::var("CC").is_err() {
        build.compiler("/usr/bin/clang++");
    }

    build
        .file("src/bridge.cpp")
        .include(&src_cpp_dir)
        .include(&include_dir)
        .include(include_dir.join("robin_map"))
        .include(include_dir.join("rapidcsv"))
        .include("include")
        .warnings(false); // Disable cxx's default warning flags so we can set our own

    // Use C++17 when available, fall back to C++14 for older compilers
    // This helps with cross-compilation where older toolchains may not fully support C++17
    if target.contains("aarch64") || target.contains("arm") {
        // For ARM targets, check what's supported
        if build.is_flag_supported("-std=c++17").unwrap_or(false) {
            build.std("c++17");
        } else {
            build.std("c++14");
        }
    } else {
        // For other targets, use C++17
        build.std("c++17");
    }

    // Report ccache/sccache configuration
    report_cache_config();

    // Use build profile for optimization settings
    let profile = get_build_profile();
    match profile.as_str() {
        "native" => {
            // Native profile: release optimizations + CPU-specific optimizations
            build.flag_if_supported("-O3");
            // Only use -march=native if not cross-compiling
            if env::var("CARGO_CFG_TARGET_ARCH").ok() == env::var("HOST_ARCH").ok() {
                build.flag_if_supported("-march=native");
            }
        }
        "release" => {
            // Release profile: full optimization
            build.flag_if_supported("-O3");
        }
        _ => {
            // Dev profile: no optimization for faster compilation
            build.flag_if_supported("-O0");
            build.flag_if_supported("-g"); // Include debug symbols
        }
    }

    // Suppress warnings from external code
    if cfg!(not(target_env = "msvc")) {
        // For GCC/Clang
        build
            .flag("-w") // Suppress all warnings
            .flag_if_supported("-fopenmp"); // Enable OpenMP if available

        // On macOS, use the -stdlib=libc++ flag to ensure proper C++ standard library linkage
        if target.contains("darwin") {
            build.flag("-stdlib=libc++");
            // Prevent opportunistic linking to Homebrew's libunwind (Xcode 15+ issue)
            build.flag("-L/usr/lib");
            build.flag("-Wl,-search_paths_first");
        }
    } else {
        // For MSVC
        build
            .flag("/W0") // Warning level 0 (no warnings)
            // NOTE: OpenMP disabled on Windows to avoid CRT issues - can cause heap corruption
            // when combined with dynamic CRT (/MD) in multi-threaded scenarios
            // .flag_if_supported("/openmp") // Enable OpenMP if available
            .flag_if_supported("/permissive-") // Enable standards-compliant C++ parsing
            .flag_if_supported("/Zc:__cplusplus"); // Report correct __cplusplus macro value

        // CRITICAL: Use the same CRT as Rust/cxx to avoid heap corruption on Windows
        // Dependencies like cxx are always built with release CRT (/MD) even in debug builds
        // We must match this to avoid heap corruption when memory crosses DLL boundaries
        // Always use /MD (release CRT) to match dependencies
        build.flag("/MD"); // Dynamic CRT, release version (matches cxx and other deps)
    }

    build.compile("ldpc-bridge");

    // On macOS, link against the system C++ library from dyld shared cache
    if target.contains("darwin") {
        println!("cargo:rustc-link-search=native=/usr/lib");
        println!("cargo:rustc-link-lib=c++");
        println!("cargo:rustc-link-arg=-Wl,-search_paths_first");
    }

    Ok(())
}
