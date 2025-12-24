//! Stim build support for `PyMatching` decoder

use log::info;
use std::path::{Path, PathBuf};

/// Get the essential Stim source files needed for `PyMatching`
pub fn collect_stim_sources(stim_src_dir: &Path) -> Vec<PathBuf> {
    // PyMatching needs comprehensive Stim functionality for DEM operations
    let essential_files = vec![
        // Core DEM files
        "stim/dem/detector_error_model.cc",
        "stim/dem/detector_error_model_instruction.cc",
        "stim/dem/detector_error_model_target.cc",
        "stim/dem/dem_instruction.cc",
        "stim/dem/dem_target.cc",
        // Circuit support
        "stim/circuit/circuit.cc",
        "stim/circuit/circuit_instruction.cc",
        "stim/circuit/gate_data.cc",
        "stim/circuit/gate_target.cc",
        "stim/circuit/gate_decomposition.cc",
        // Memory management
        "stim/mem/bit_ref.cc",
        "stim/mem/simd_word.cc",
        "stim/mem/simd_util.cc",
        "stim/mem/sparse_xor_vec.cc",
        // Stabilizer operations (needed for MWPM)
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
