/// Module for metropolis sampling to find logical failures
use crate::brute_force::{BruteForceResults, find_logical_failure};
use crate::flipper::FaultFlipper;
use pecos_core::errors::PecosError;
use pecos_decoder_core::ObservableDecoder;
use pecos_engines::monte_carlo::MonteCarloEngine;
use pecos_engines::FaultHistory;
use pecos_quantum::TickCircuit;
use pecos_random::PecosRng;
use crate::utils::{setup_rare_event_simulation, is_logical_failure, start_rps_loading};

pub fn metropolis_step(
    fault_history: &FaultHistory,
    monte_carlo: &mut MonteCarloEngine,
    flipper: &mut FaultFlipper,
    rng: &mut PecosRng,
    decoder: &mut dyn ObservableDecoder,
    detectors: &[(usize, Vec<usize>)],
    observables: &[(usize, Vec<usize>)],
) -> Result<(bool, bool, FaultHistory), PecosError> {

    // Perform a flip
    let (new_history, ratio) = flipper.random_flip_and_ratio(fault_history);

    // Can return early based on ratio
    if rng.next_f64() >= ratio.min(1.0) {
        // Reject the new history early
        return Ok((false, false, fault_history.clone()));
    }

    // Run the circuit with the new fault history
    let run = monte_carlo.run_with_fault_history(&new_history)?;

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
