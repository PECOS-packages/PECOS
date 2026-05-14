//! Stim build support for Chromobius decoder

use log::info;
use pecos_build::Result;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

/// Get the essential Stim source files needed for Chromobius
pub fn collect_stim_sources(stim_src_dir: &Path) -> Vec<PathBuf> {
    // Chromobius needs more comprehensive Stim functionality
    let essential_files = vec![
        // Core DEM files
        "stim/dem/detector_error_model.cc",
        "stim/dem/detector_error_model_instruction.cc",
        "stim/dem/detector_error_model_target.cc",
        "stim/dem/dem_instruction.cc",
        "stim/dem/dem_target.cc",
        // Utilities (parse_int64 needed by dem_instruction.cc)
        "stim/util_bot/arg_parse.cc",
        // Circuit support
        "stim/circuit/circuit.cc",
        "stim/circuit/circuit_instruction.cc",
        "stim/circuit/gate_data.cc",
        "stim/circuit/gate_target.cc",
        "stim/circuit/gate_decomposition.cc",
        // Memory management. simd_word.cc and sparse_xor_vec.cc are upstream
        // Stim "translation-unit anchor" files for header-only types: their
        // contents are just `#include "<header>.h"`, so they compile to empty
        // .o files with no symbols. Including them adds nothing at link time
        // and triggers macOS BSD ranlib's `has no symbols` warnings.
        "stim/mem/bit_ref.cc",
        "stim/mem/simd_util.cc",
        // Stabilizer operations (needed for Chromobius)
        "stim/stabilizers/pauli_string.cc",
        "stim/stabilizers/flex_pauli_string.cc",
        "stim/stabilizers/tableau.cc",
        // I/O
        "stim/io/raii_file.cc",
        "stim/io/measure_record_batch.cc",
        "stim/io/measure_record_reader.cc",
        "stim/io/measure_record_writer.cc",
        // Gate implementations (all required by GateDataMap)
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

fn collect_files_from_list(base_dir: &Path, files: &[&str]) -> Vec<PathBuf> {
    let mut found_files = Vec::new();

    for file_path in files {
        let full_path = base_dir.join(file_path);
        if full_path.exists() {
            found_files.push(full_path);
        } else {
            info!("Stim source file not found: {}", full_path.display());
        }
    }

    info!("Found {} Stim source files", found_files.len());

    found_files
}

/// Generate amalgamated stim.h header for Chromobius
pub fn generate_amalgamated_header(stim_dir: &Path) -> Result<()> {
    let output_path = stim_dir.join("stim.h");

    if output_path.exists() {
        return Ok(());
    }

    let content = r#"// Stim amalgamated header wrapper for Chromobius compatibility
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

    info!("Generating amalgamated header: {}", output_path.display());
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut file = fs::File::create(output_path)?;
    file.write_all(content.as_bytes())?;

    Ok(())
}
