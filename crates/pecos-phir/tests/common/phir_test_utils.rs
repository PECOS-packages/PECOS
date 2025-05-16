#![allow(dead_code)]

use pecos_core::errors::PecosError;
use pecos_engines::{MonteCarloEngine, NoiseModel, PassThroughNoiseModel, ShotResults};
use pecos_phir::v0_1::ast::PHIRProgram;
use pecos_phir::v0_1::engine::PHIREngine;

/// Run a PHIR simulation and get the results using JSON string
///
/// # Arguments
///
/// * `json` - PHIR program as a JSON string
/// * `shots` - Number of shots to run
/// * `workers` - Number of workers to use
/// * `seed` - Optional seed for reproducibility
/// * `noise_model` - Optional noise model to use (defaults to `PassThroughNoiseModel`)
/// * `wasm_path` - Optional path to a WebAssembly file (.wat or .wasm) for foreign function integration
///
/// # Returns
///
/// * `ShotResults` - The results of the simulation
///
/// # Examples
///
/// Basic usage without WebAssembly:
///
/// ```no_run
/// let results = run_phir_simulation_from_json(
///     phir_json,
///     1,                      // Just one shot
///     1,                      // Single worker
///     Some(42),               // Seed for reproducibility
///     None::<PassThroughNoiseModel>,  // No noise model (pass-through)
///     None::<&std::path::Path>,       // No WebAssembly file
/// )?;
/// ```
///
/// Using with WebAssembly:
///
/// ```no_run
/// let wasm_path = std::path::Path::new("path/to/file.wat");
/// let results = run_phir_simulation_from_json(
///     phir_json,
///     1,                      // Just one shot
///     1,                      // Single worker
///     Some(42),               // Seed for reproducibility
///     None::<PassThroughNoiseModel>,  // No noise model (pass-through)
///     Some(&wasm_path),       // WebAssembly file for foreign function calls
/// )?;
/// ```
pub fn run_phir_simulation_from_json<T: NoiseModel + 'static, P: AsRef<std::path::Path> + Clone>(
    json: &str,
    shots: usize,
    workers: usize,
    seed: Option<u64>,
    noise_model: Option<T>,
    wasm_path: Option<P>,
) -> Result<ShotResults, PecosError> {
    // Parse JSON into PHIRProgram
    let program: PHIRProgram = serde_json::from_str(json)
        .map_err(|e| PecosError::Input(format!("Failed to parse PHIR program: {e}")))?;

    // Create a PHIR engine from the program (clone it to keep the original)
    #[allow(unused_mut)]
    let mut engine = PHIREngine::from_program(program.clone())?;

    // If WebAssembly path is provided, set up the WebAssembly foreign object
    #[cfg(not(feature = "wasm"))]
    if let Some(_wasm_file_path) = wasm_path {
        return Err(PecosError::Input(
            "WebAssembly support requires the 'wasm' feature to be enabled".to_string(),
        ));
    }

    #[cfg(feature = "wasm")]
    if let Some(wasm_file_path) = wasm_path {
        // Box is sufficient since we don't need shared ownership
        use pecos_phir::v0_1::foreign_objects::ForeignObject;
        use pecos_phir::v0_1::wasm_foreign_object::WasmtimeForeignObject;

        // Create and initialize the WebAssembly foreign object
        let mut foreign_object = WasmtimeForeignObject::new(wasm_file_path.as_ref())?;
        foreign_object.init()?;
        let foreign_object: Box<dyn ForeignObject> = Box::new(foreign_object);

        // Set the foreign object in the engine (only once!)
        engine.set_foreign_object(foreign_object);
    }

    // Use the provided noise model or default to PassThroughNoiseModel
    let noise_model_box: Box<dyn NoiseModel> = match noise_model {
        Some(model) => Box::new(model),
        None => Box::new(PassThroughNoiseModel),
    };

    // Debug: Print the engine state before running
    println!("Debug - Starting simulation with engine: {engine:?}");

    // Run the Monte Carlo engine
    let results = MonteCarloEngine::run_with_noise_model(
        Box::new(engine),
        noise_model_box,
        shots,
        workers,
        seed,
    )
    .map_err(|e| {
        PecosError::with_context(e, "Failed to run Monte Carlo engine with noise model")
    })?;

    // Debug: Print register information from results
    println!("Debug - Results received: {results:?}");
    println!("Debug - Registers (u32): {:?}", results.register_shots);
    println!("Debug - Registers (u64): {:?}", results.register_shots_u64);
    println!("Debug - Registers (i64): {:?}", results.register_shots_i64);

    Ok(results)
}

/// Assert that a register has an expected value in a `ShotResults`
///
/// # Arguments
///
/// * `results` - The simulation results
/// * `register_name` - The name of the register to check
/// * `expected_value` - The expected value of the register
///
/// # Panics
///
/// * If the register does not exist
/// * If the register value does not match the expected value
pub fn assert_register_value(results: &ShotResults, register_name: &str, expected_value: i64) {
    // Special case for "output" and "result" - this is for backward compatibility with tests
    // after refactoring removed special case handling of these names
    if register_name == "output" && !results.register_shots_i64.contains_key("output") {
        // Check if "result" exists instead, since our refactoring no longer does automatic mapping
        if let Some(values) = results.register_shots_i64.get("result") {
            assert!(
                !values.is_empty(),
                "Register 'result' (checked as fallback for '{register_name}') found but has no values"
            );
            assert_eq!(
                values[0], expected_value,
                "Register 'result' (checked as fallback for '{}') has i64 value {} but expected {}",
                register_name, values[0], expected_value
            );
            println!("NOTICE: Test looked for 'output' but found 'result' with correct value");
            return;
        }
    }

    // First check in i64 registers which is most accurate for our expected values
    if let Some(values) = results.register_shots_i64.get(register_name) {
        assert!(
            !values.is_empty(),
            "Register '{register_name}' found but has no values"
        );
        assert_eq!(
            values[0], expected_value,
            "Register '{}' has i64 value {} but expected {}",
            register_name, values[0], expected_value
        );
        return;
    }

    // Then check in the u32 registers
    if let Some(values) = results.register_shots.get(register_name) {
        assert!(
            !values.is_empty(),
            "Register '{register_name}' found but has no values"
        );
        // Convert to i64 for comparison
        let value_i64 = i64::from(values[0]);
        assert_eq!(
            value_i64, expected_value,
            "Register '{}' has u32 value {} but expected {} as i64",
            register_name, values[0], expected_value
        );
        return;
    }

    // Finally check in u64 registers
    if let Some(values) = results.register_shots_u64.get(register_name) {
        assert!(
            !values.is_empty(),
            "Register '{register_name}' found but has no values"
        );
        // For large u64 values outside the i64 range, this could fail
        if let Ok(value_i64) = i64::try_from(values[0]) {
            assert_eq!(
                value_i64, expected_value,
                "Register '{}' has u64 value {} but expected {} as i64",
                register_name, values[0], expected_value
            );
            return;
        }
        panic!(
            "Register '{}' has u64 value {} which is too large to convert to i64 for comparison",
            register_name, values[0]
        );
    }

    // Fall back to checking "result" if "output" was requested but not found
    if register_name == "output" {
        println!("NOTICE: 'output' register not found, falling back to check 'result'");
        return assert_register_value(results, "result", expected_value);
    }

    panic!(
        "Register '{}' not found in any register types. Available registers: {:?}",
        register_name,
        results
            .register_shots
            .keys()
            .chain(results.register_shots_u64.keys())
            .chain(results.register_shots_i64.keys())
            .collect::<std::collections::HashSet<_>>()
    );
}
