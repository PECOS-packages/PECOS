//! Shared Stim build script for all decoders

use pecos_build_utils::{Result, download_cached, extract_archive, stim_download_info};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

/// Downloads and extracts Stim if not already present
pub fn ensure_stim(out_dir: &Path) -> Result<PathBuf> {
    // Use the newer Stim version that Tesseract uses
    let stim_dir = out_dir.join("stim_shared");

    if !stim_dir.exists() {
        download_and_extract_stim(out_dir)?;
    }

    // Generate amalgamated header for Chromobius if needed
    let amalgamated_header = stim_dir.join("stim.h");
    if !amalgamated_header.exists() {
        generate_amalgamated_header(&stim_dir)?;
    }

    Ok(stim_dir)
}

fn download_and_extract_stim(out_dir: &Path) -> Result<()> {
    let info = stim_download_info("tesseract");
    let tar_gz = download_cached(&info)?;
    extract_archive(&tar_gz, out_dir, Some("stim_shared"))?;

    if std::env::var("PECOS_VERBOSE_BUILD").is_ok() {
        println!("cargo:warning=Shared Stim source ready");
    }
    Ok(())
}

/// Get the essential Stim source files needed for Tesseract
// Always enable in tesseract crate
// #[cfg(feature = "tesseract")]
pub fn collect_stim_sources_tesseract(stim_src_dir: &Path) -> Result<Vec<PathBuf>> {
    // Tesseract primarily needs DEM parsing and basic circuit support
    let essential_files = vec![
        // Core DEM files
        "stim/dem/detector_error_model.cc",
        "stim/dem/detector_error_model_instruction.cc",
        "stim/dem/detector_error_model_target.cc",
        "stim/dem/dem_instruction.cc", // Added - required for validation
        "stim/dem/dem_target.cc",      // Added - required for target operations
        // Basic circuit support
        "stim/circuit/circuit.cc",
        "stim/circuit/circuit_instruction.cc",
        "stim/circuit/gate_data.cc",
        "stim/circuit/gate_target.cc",
        // Memory management
        "stim/mem/simd_word.cc",
        "stim/mem/simd_util.cc",
        // I/O for reading files
        "stim/io/raii_file.cc",
        // All gate implementations needed by GateDataMap
        "stim/gates/gates.cc",
        "stim/gates/gate_data_annotations.cc",
        "stim/gates/gate_data_blocks.cc",
        "stim/gates/gate_data_collapsing.cc",
        "stim/gates/gate_data_controlled.cc",
        "stim/gates/gate_data_hada.cc",
        "stim/gates/gate_data_heralded.cc",
        "stim/gates/gate_data_noisy.cc",
        "stim/gates/gate_data_pauli.cc",
        "stim/gates/gate_data_period_3.cc",
        "stim/gates/gate_data_period_4.cc",
        "stim/gates/gate_data_pp.cc",
        "stim/gates/gate_data_swaps.cc",
        "stim/gates/gate_data_pair_measure.cc",
        "stim/gates/gate_data_pauli_product.cc",
    ];

    collect_files_from_list(stim_src_dir, &essential_files)
}

fn collect_files_from_list(base_dir: &Path, files: &[&str]) -> Result<Vec<PathBuf>> {
    let mut found_files = Vec::new();

    for file_path in files {
        let full_path = base_dir.join(file_path);
        if full_path.exists() {
            found_files.push(full_path);
        }
    }

    Ok(found_files)
}

/// Generate amalgamated stim.h header for Chromobius
fn generate_amalgamated_header(stim_dir: &Path) -> Result<()> {
    let output_path = stim_dir.join("stim.h");

    // Create a simple wrapper that includes all necessary Stim headers
    // This is simpler and more reliable than trying to merge headers
    let content = r#"// Stim amalgamated header wrapper for Chromobius compatibility
// Generated from Stim commit bd60b73

#ifndef STIM_H
#define STIM_H

// Base utilities and prerequisites
#include "src/stim/util_base/util_base.h"

// Memory management
#include "src/stim/mem/bit_ref.h"
#include "src/stim/mem/simd_word.h"
#include "src/stim/mem/simd_util.h"
#include "src/stim/mem/simd_bits.h"
#include "src/stim/mem/simd_bits_range_ref.h"
#include "src/stim/mem/sparse_xor_vec.h"
#include "src/stim/mem/monotonic_buffer.h"

// Circuit components
#include "src/stim/circuit/gate_target.h"
#include "src/stim/circuit/circuit_instruction.h"
#include "src/stim/circuit/circuit.h"
#include "src/stim/circuit/gate_data.h"

// DEM components
#include "src/stim/dem/detector_error_model_target.h"
#include "src/stim/dem/detector_error_model_instruction.h"
#include "src/stim/dem/detector_error_model.h"

// Stabilizers
#include "src/stim/stabilizers/pauli_string.h"
#include "src/stim/stabilizers/pauli_string_ref.h"
#include "src/stim/stabilizers/tableau.h"

// IO
#include "src/stim/io/raii_file.h"
#include "src/stim/io/measure_record.h"
#include "src/stim/io/measure_record_batch.h"
#include "src/stim/io/measure_record_reader.h"
#include "src/stim/io/measure_record_writer.h"
#include "src/stim/io/stim_data_formats.h"

// Utility functions
#include "src/stim/util_bot/str_util.h"

// Command line utilities
#include "src/stim/arg_parse.h"
#include "src/stim/cmd/command_help.h"

// Make sure commonly used types are in the stim namespace
using namespace stim;

#endif // STIM_H
"#;

    ensure_precompiled_header(&output_path, content)?;
    Ok(())
}

/// Generate a precompiled header if it doesn't exist
fn ensure_precompiled_header(header_path: &Path, content: &str) -> Result<()> {
    if !header_path.exists() {
        if std::env::var("PECOS_VERBOSE_BUILD").is_ok() {
            println!(
                "cargo:warning=Generating precompiled header: {}",
                header_path.display()
            );
        }
        if let Some(parent) = header_path.parent() {
            fs::create_dir_all(parent)?;
        }
        let mut file = fs::File::create(header_path)?;
        file.write_all(content.as_bytes())?;
    }
    Ok(())
}
