//! Example: Converting unified API results to different formats
//!
//! The unified simulation API returns `ShotVec`, which can be converted
//! to other formats as needed for compatibility or specific use cases.

use pecos_engines::{
    shot_results::{Data, Shot, ShotVec},
    shots_to_columnar,
};
use std::collections::BTreeMap;

fn main() {
    // Note: In real usage, you would use actual engine builders from the crates:
    // - pecos_qasm::unified_engine_builder::qasm_engine()
    // - pecos_qis_sim::engine_builder::qis_engine()
    // - pecos_selene_engine::selene_executable()
    //
    // This example focuses on the result format conversions rather than
    // the engine implementation details.

    // For this example, we'll create a sample ShotVec directly
    let mut shot_vec = ShotVec::new();

    // Add some sample shots with different registers
    for i in 0..10 {
        let mut shot = Shot::default();

        // Add measurement results for two registers
        let mut data = BTreeMap::new();
        data.insert("q0".to_string(), Data::U32(i % 2));
        data.insert("q1".to_string(), Data::U32((i / 2) % 2));
        data.insert("phase".to_string(), Data::F64(f64::from(i) * 0.1));

        shot.data = data;
        shot_vec.shots.push(shot);
    }

    // Convert to ShotMap (for display, analysis, etc.)
    match shot_vec.try_as_shot_map() {
        Ok(shot_map) => {
            println!("ShotMap format: {shot_map:?}");
            // Use shot_map.display() for pretty printing
            // Use shot_map.iter() for analysis
        }
        Err(e) => {
            println!("Cannot convert to ShotMap: {e}");
            // This happens when shots have different register structures
        }
    }

    // Convert to columnar format (BTreeMap<String, Vec<i64>>)
    // This format provides columnar data access for analysis
    let columnar = shots_to_columnar(&shot_vec);
    println!("Columnar format: {columnar:?}");
    // Each register name maps to a vector of values across all shots

    // Direct access to shots
    for shot in shot_vec.shots.iter().take(5) {
        println!("Shot: {shot:?}");
        // Access individual shot data
    }

    // Example: Working with individual register data
    println!("\n=== Direct Register Access ===");
    if let Ok(shot_map) = shot_vec.try_as_shot_map() {
        // Access specific register data
        if let Some(q0_values) = shot_map.get("q0") {
            println!("q0 register has {} values", q0_values.len());
        }

        // Iterate over all registers
        for (register_name, values) in shot_map.iter() {
            println!("Register '{}': {} values", register_name, values.len());
        }
    }
}
