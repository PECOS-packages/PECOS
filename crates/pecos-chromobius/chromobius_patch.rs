//! Utilities for patching Chromobius to work with newer Stim versions

use pecos_build::Result;
use std::fs;
use std::path::Path;

/// Apply compatibility patches to Chromobius source
pub fn patch_chromobius_for_newer_stim(chromobius_dir: &Path) -> Result<()> {
    // Check if patches have already been applied
    let patch_marker = chromobius_dir.join(".patches_applied");
    if patch_marker.exists() {
        // Silently skip if already patched
        return Ok(());
    }

    if std::env::var("PECOS_VERBOSE_BUILD").is_ok() {
        println!("cargo:warning=Applying Chromobius compatibility patches...");
    }

    // Based on our analysis, the main potential incompatibilities are:
    // 1. DEM instruction iteration API changes
    // 2. Method name changes on DetectorErrorModel
    // 3. Changes in how coordinates are stored/accessed
    // 4. Changes in the iter_flatten_error_instructions callback signature

    // Apply patches to specific files that might need updates
    let files_to_check = vec![
        "src/chromobius/decode/decoder.cc",
        "src/chromobius/graph/collect_atomic_errors.cc",
        "src/chromobius/graph/collect_nodes.cc",
        "src/chromobius/graph/collect_composite_errors.cc",
    ];

    let mut any_patched = false;
    for file_path in files_to_check {
        let full_path = chromobius_dir.join(file_path);
        if full_path.exists() {
            // Check if we need to patch this file
            if needs_dem_api_patch(&full_path)? {
                apply_dem_api_patch(&full_path)?;
                any_patched = true;
            }
        }
    }

    if any_patched {
        // Mark patches as applied
        fs::write(patch_marker, "1")?;
        if std::env::var("PECOS_VERBOSE_BUILD").is_ok() {
            println!("cargo:warning=Chromobius patches applied successfully");
        }
    } else if std::env::var("PECOS_VERBOSE_BUILD").is_ok() {
        println!("cargo:warning=No Chromobius patches needed");
    }
    Ok(())
}

/// Check if a file needs DEM API patches
fn needs_dem_api_patch(file_path: &Path) -> Result<bool> {
    let content = fs::read_to_string(file_path)?;

    // Check for patterns that might indicate old API usage
    // Don't patch if already patched
    if content.contains("// CHROMOBIUS_PATCHED") {
        return Ok(false);
    }

    // Check for potentially problematic API usage
    let needs_patch = content.contains("iter_flatten_error_instructions")
        || content.contains("repeat_block_body(")
        || content.contains("repeat_block_rep_count(")
        || content.contains(".instructions");

    Ok(needs_patch)
}

/// Apply DEM API compatibility patches
fn apply_dem_api_patch(file_path: &Path) -> Result<()> {
    let mut content = fs::read_to_string(file_path)?;

    // Add patch marker
    content = format!("// CHROMOBIUS_PATCHED: Compatibility patches for newer Stim\n{content}");

    // Patch 1: Fix append_detector_instruction calls
    // The newer Stim added a third parameter (tag) to append_detector_instruction
    // Old: append_detector_instruction({}, target)
    // New: append_detector_instruction({}, target, "")

    // Fix the specific pattern we found in decoder.cc
    content = content.replace(
        "result.mobius_dem.append_detector_instruction(\n            {}, stim::DemTarget::relative_detector_id(result.node_colors.size() * 2 - 1));",
        "result.mobius_dem.append_detector_instruction(\n            {}, stim::DemTarget::relative_detector_id(result.node_colors.size() * 2 - 1), \"\");"
    );

    // Fix the patterns in collect_nodes.cc
    content = content.replace(
        "out_mobius_dem->append_detector_instruction(*coord_buffer, d0);",
        "out_mobius_dem->append_detector_instruction(*coord_buffer, d0, \"\");",
    );

    content = content.replace(
        "out_mobius_dem->append_detector_instruction(*coord_buffer, d1);",
        "out_mobius_dem->append_detector_instruction(*coord_buffer, d1, \"\");",
    );

    // Patch 2: Fix append_error_instruction calls
    // The newer Stim also added a third parameter (tag) to append_error_instruction
    // Old: append_error_instruction(probability, targets)
    // New: append_error_instruction(probability, targets, "")

    // Fix the pattern in collect_composite_errors.cc
    content = content.replace(
        "out_mobius_dem->append_error_instruction(p, composite_error_buffer);",
        "out_mobius_dem->append_error_instruction(p, composite_error_buffer, \"\");",
    );

    fs::write(file_path, content)?;

    if std::env::var("PECOS_VERBOSE_BUILD").is_ok() {
        println!(
            "cargo:warning=Patched {} for append_detector_instruction API change",
            file_path.display()
        );
    }
    Ok(())
}
