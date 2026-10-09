/// Module for metropolis sampling to find logical failures
use crate::flipper::FaultFlipper;
use pecos_core::errors::PecosError;
use pecos_decoder_core::ObservableDecoder;
use pecos_engines::monte_carlo::MonteCarloEngine;
use pecos_engines::{FaultCatalog, FaultHistory};
use pecos_quantum::TickCircuit;
use pecos_random::{PecosRng, RngExt};
use crate::utils::{setup_rare_event_simulation, is_logical_failure, start_rps_loading};

fn metropolis_step(
    fault_history: &FaultHistory,
    monte_carlo: &mut MonteCarloEngine,
    flipper: &mut dyn FaultFlipper,
    decoder: &mut dyn ObservableDecoder,
    detectors: &[(usize, Vec<usize>)],
    observables: &[(usize, Vec<usize>)],
) -> Result<(bool, bool, FaultHistory), PecosError> {

    // Perform a flip
    let (new_history, ratio) = flipper.random_flip_and_ratio(fault_history);

    // Run the circuit with the new fault history
    let mut run = monte_carlo.run_with_fault_history(&new_history)?;

    // Can return early based on ratio
    ratio = ratio.min(1.0);
    let rand_val = flipper.rng.as_mut()
        .expect("Set the fault flipper seed before requesting a flip")
        .next_f64();
    if rand_val >= ratio {
        // Reject the new history early
        return Ok((false, false, fault_history.clone()));
    }

    // Decode to figure out if it is a logical failure
    let failed = is_logical_failure(
        &run.results.shots[0],
        decoder,
        detectors,
        observables,
    )?;

    // Accept the move if it is a logical failure
    if failed {
        return Ok((true, true, new_history));
    }
    Ok((false, true, fault_history.clone()))
}

pub fn find_logical_failure_rate(
    circuit: &TickCircuit,
    p_phys: f64, // Physical Error Rate
    seed: u64, // Random seed for reproducibility
    max_histories: usize,
    verbose: bool,
    save_failures: String,
    nbootstraps: usize,
    region_probability: f64,
    flipper_type: String,
) -> Result<BruteForceResults, PecosError> {

    // Grab everything that's needed for the simulation
    let (fault_catalog, mut monte_carlo, mut decoder, detectors, observables) = setup_rare_event_simulation(circuit, p_phys, seed)?;

    // Setup the flipper
    let mut flipper = match flipper_type.as_str() {
        "basic" => FaultFlipper::new_basic(fault_catalog),
        "basic_subregion" => FaultFlipper::new_basic_subregion(fault_catalog, region_probability),
        "weighted" => FaultFlipper::new_weighted(fault_catalog),
        "weighted_subregion" => FaultFlipper::new_weighted_subregion(fault_catalog, region_probability),
        _ => panic!("Unknown flipper type"),
    };

    // Start by finding a seed logical failure with brute force simulation
    let fault_history = find_logical_failure(circuit, p_phys, seed, max_histories)?;
    let phist = fault_catalog.fault_history_probability(&fault_history);

    // Set up a result logger
    // let mut results = BruteForceResults::new(seed, save_failures, nbootstraps);

    // start the progress bar
    let (pb, game_thread) = start_rps_loading(max_histories as u64);

    for _ in 0..max_histories {

        // Do a metropolis step
        let (failed, phist, fault_history) = metropolis_step(
            &fault_history,
            &mut monte_carlo,
            &mut flipper,
            &mut decoder,
            &detectors,
            &observables,
        )?;

        // Accumulate failure statistics
        if failed {
            results.record_failure(phist, &fault_history);
            if verbose && results.nfail % 1000 == 0 {
                pb.println(format!(
                    "Failed {} / {} histories, pfail = {:.3e} +/- {:.3e}, smallest failure = {}",
                    results.nfail,
                    results.ntot,
                    results.failure_rate(),
                    results.failure_rate_std(),
                    results.smallest_failure_weight,
                ));
            }
        } else {
            results.record_pass(phist);
        }

        // Increment the progress bar
        pb.inc(1);
    }

    pb.finish_with_message("Done!");
    game_thread
        .join()
        .map_err(|_| PecosError::Processing("Progress animation thread panicked".into()))?;
    Ok(results)

}