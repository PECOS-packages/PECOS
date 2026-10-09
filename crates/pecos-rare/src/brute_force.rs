/// Module for brute force monte carlo simulations to find logical failures
use crate::utils::{setup_rare_event_simulation, is_logical_failure, start_rps_loading};
use pecos_core::errors::PecosError;
use pecos_decoder_core::ObservableDecoder;
use pecos_engines::monte_carlo::MonteCarloEngine;
use pecos_engines::{FaultCatalog, FaultHistory};
use pecos_quantum::TickCircuit;
use pecos_random::{PecosRng, RngExt};

// Simple structure to hold the results of the brute force calculation
pub struct BruteForceResult {
    seed: u64,
    nfail: usize,
    ntot: usize,
    pfails: Vec<f64>,
    ptots: Vec<f64>,
    results: Vec<u8>,
    rng: PecosRng,
    save_failures: String,
    failures: Vec<FaultHistory>,
    smallest_failure_weight: usize,
    nbootstraps: usize,
}

impl BruteForceResult {
    pub fn new(seed: u64, save_failures: String, nbootstraps: usize) -> Self {
        Self {
            seed,
            nfail: 0,
            ntot: 0,
            pfails: Vec::new(),
            ptots: Vec::new(),
            results: Vec::new(),
            rng: PecosRng::seed_from_u64(seed),
            save_failures,
            failures: Vec::new(),
            smallest_failure_weight: usize::MAX,
            nbootstraps: nbootstraps,
        }
    }

    pub fn record_failure(&mut self, phist: f64, history: &FaultHistory) {
        self.nfail += 1;
        self.ntot += 1;
        self.pfails.push(phist);
        self.ptots.push(phist);
        self.results.push(1);
        if self.save_failures.to_lowercase() == "true" {
            self.failures.push(history.clone());
        }
        else if self.save_failures.to_lowercase() != "last" {
            self.failures.clear();
            self.failures.push(history.clone());
        }
        else if self.save_failures.to_lowercase() != "false" {
            panic!("Invalid value for save_failures: {}", self.save_failures);
        }
        let failure_weight = history.weight();
        if failure_weight < self.smallest_failure_weight {
            self.smallest_failure_weight = failure_weight;
        }
    }

    pub fn record_pass(&mut self, phist: f64) {
        self.ntot += 1;
        self.ptots.push(phist);
        self.results.push(0);
    }

    // Returns the average failure rate
    pub fn failure_rate_average(&self) -> f64 {
        if self.ntot == 0 {
            0.0
        } else {
            self.nfail as f64 / self.ntot as f64
        }
    }

    // Returns the average failure rate
    pub fn failure_rate(&self) -> f64 {
        self.failure_rate_average()
    }

    fn bootstrap_failure_rates(&self) -> Vec<f64> {
        let mut rng = self.rng.clone();
        let mut resampled_rates = Vec::with_capacity(self.nbootstraps);
        for _ in 0..self.nbootstraps {
            let mut resampled_results = Vec::with_capacity(self.results.len());
            for _ in 0..self.results.len() {
                // Generate a random index using the PecosRng
                let idx = rng.random_range(0..self.results.len());
                resampled_results.push(self.results[idx]);
            }
            let resampled_rate = resampled_results.iter().copied().map(f64::from).sum::<f64>()
                / resampled_results.len() as f64;
            resampled_rates.push(resampled_rate);
        }
        resampled_rates
    }

    /// Returns the standard deviation of the failure rate
    pub fn failure_rate_std(&self) -> f64 {
        // Collect the bootstrapped rates
        let rates = self.bootstrap_failure_rates();
        let mean = rates.iter().sum::<f64>() / rates.len() as f64;
        let variance = rates.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / rates.len() as f64;
        variance.sqrt()
    }

    // Returns the confidence interval of the failure rate
    pub fn failure_rate_ci(&self, percentage: f64) -> (f64, f64) {
        let mut rates = self.bootstrap_failure_rates();
        rates.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let lower_idx = ((1.0 - percentage) / 2.0 * rates.len() as f64).floor() as usize;
        let upper_idx = ((1.0 + percentage) / 2.0 * rates.len() as f64).ceil() as usize;
        (rates[lower_idx], rates[upper_idx])
    }

    // Returns the 99% confidence interval of the failure rate
    pub fn failure_rate_ci99(&self) -> (f64, f64) {
        self.failure_rate_ci(0.99)
    }

    // Returns the 95% confidence interval of the failure rate
    pub fn failure_rate_ci95(&self) -> (f64, f64) {
        self.failure_rate_ci(0.95)
    }

    // Write the results to a json file
    pub fn write_json(&self, path: &str, level: usize) -> Result<(), PecosError> {
        use std::fs::File;
        use serde_json::json;

        let data = match level {
            0 => {
                json!({
                    "seed": self.seed,
                    "number_of_failures": self.nfail,
                    "number_of_samples": self.ntot,
                    "failure_rate": self.failure_rate(),
                })
            }
            1 => {
                json!({
                    "seed": self.seed,
                    "number_of_failures": self.nfail,
                    "number_of_samples": self.ntot,
                    "failure_rate_std": self.failure_rate_std(),
                    "failure_rate_ci95": self.failure_rate_ci95(),
                    "failure_rate_ci99": self.failure_rate_ci99(),
                    "failure_rate": self.failure_rate(),
                    "smallest_failure_weight": self.smallest_failure_weight,
                })
            }
            2 => {
                json!({
                    "seed": self.seed,
                    "number_of_failures": self.nfail,
                    "number_of_samples": self.ntot,
                    "failure_rate_std": self.failure_rate_std(),
                    "failure_rate_ci95": self.failure_rate_ci95(),
                    "failure_rate_ci99": self.failure_rate_ci99(),
                    "failure_rate": self.failure_rate(),
                    "smallest_failure_weight": self.smallest_failure_weight,
                    "failure_probabilities": self.pfails,
                    "all_probabilities": self.ptots,
                    "results": self.results,
                })
            }
            _ => return Err(PecosError::Input("JSON level must be 0, 1, or 2".into())),
        };
        let file = File::create(path)?;
        serde_json::to_writer(file, &data)
            .map_err(|error| PecosError::with_context(error, "Unable to write JSON to file"))
    }
}


// Runs a single monte carlo simulation, and returns whether it was a logical failure,
// the probability of the sampled history, and the fault history itself.
fn mc_trial(
    monte_carlo: &mut MonteCarloEngine,
    fault_catalog: &FaultCatalog,
    decoder: &mut dyn ObservableDecoder,
    detectors: &[(usize, Vec<usize>)],
    observables: &[(usize, Vec<usize>)],
) -> Result<(bool, f64, FaultHistory), PecosError> {

    // Run a single simulation and collect fault information
    let mut run = monte_carlo.run_with_fault_tracking(1)?;
    let fault_history = run.fault_histories.pop().ok_or_else(|| {
        PecosError::Processing("Simulation returned no fault history".into())
    })?;
    let phist = fault_catalog.fault_history_probability(&fault_history);

    // Check if it failed
    let failed = is_logical_failure(
            &run.results.shots[0],
            decoder,
            detectors,
            observables,
    )?;

    Ok((failed, phist, fault_history))
}

pub fn find_logical_failure_rate(
    circuit: &TickCircuit,
    p_phys: f64, // Physical Error Rate
    seed: u64, // Random seed for reproducibility
    max_histories: usize,
    verbose: bool,
    save_failures: String,
    nbootstraps: usize,
    check_frequency: usize,
    error_threshold: f64,
) -> Result<BruteForceResult, PecosError> {

    // Grab everything that's needed for the simulation
    let (fault_catalog, mut monte_carlo, mut decoder, detectors, observables) = setup_rare_event_simulation(circuit, p_phys, seed)?;

    // Set up a result logger
    let mut results = BruteForceResult::new(seed, save_failures, nbootstraps);

    // start the progress bar
    let (pb, game_thread) = start_rps_loading(max_histories as u64);

    for _ in 0..max_histories {

        // Try a simulation
        let (failed, phist, fault_history) = mc_trial(&mut monte_carlo, &fault_catalog, decoder.as_mut(), &detectors, &observables)?;

        // Accumulate failure statistics
        if failed {
            results.record_failure(phist, &fault_history);
            if verbose && results.nfail % check_frequency == 0 {

                let avg = results.failure_rate();
                let std = results.failure_rate_std();
                let error = std / avg;
                if error < error_threshold {
                    pb.println(format!(
                        "Failed {} / {} histories, pfail = {:.3e} +/- {:.3e}, Percent error {:.3}% below threshold {:.3}%",
                        results.nfail,
                        results.ntot,
                        avg,
                        std,
                        error*100.0,
                        error_threshold*100.0,
                    ));
                    break
                }
                else {
                    pb.println(format!(
                        "Failed {} / {} histories, pfail = {:.3e} +/- {:.3e}, Percent error {:.3}% above threshold {:.3}%",
                        results.nfail,
                        results.ntot,
                        avg,
                        std,
                        error*100.0,
                        error_threshold*100.0,
                    ));
                }
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

/// Finds a single logical failure
pub fn find_logical_failure(
    circuit: &TickCircuit,
    p_phys: f64, // Physical Error Rate
    seed: u64, // Random seed for reproducibility
    max_histories: usize,
) -> Result<FaultHistory, PecosError> {

    // Grab everything that's needed for the simulation
    let (fault_catalog, mut monte_carlo, mut decoder, detectors, observables) = setup_rare_event_simulation(circuit, p_phys, seed)?;

    // Loop until we hit a failure
    for _ in 0..max_histories {

        // Try a simulation
        let (failed, _phist, fault_history) = mc_trial(&mut monte_carlo, &fault_catalog, decoder.as_mut(), &detectors, &observables)?;

        if failed {
            return Ok(fault_history);
        }
    }
    Err(PecosError::LogicalFailureNotFound)
}

pub fn find_logical_failures(
    circuit: &TickCircuit,
    p_phys: f64, // Physical Error Rate
    seed: u64, // Random seed for reproducibility
    max_histories: usize,
    nfail: usize
) -> Result<Vec<FaultHistory>, PecosError> {
    
    // Loop until we find the requested number of failures
    let mut failures = Vec::new();
    let (fault_catalog, mut monte_carlo, mut decoder, detectors, observables) = setup_rare_event_simulation(circuit, p_phys, seed)?;

    for _ in 0..max_histories {

        // Perform a trial
        let (failed, _phist, fault_history) = mc_trial(&mut monte_carlo, &fault_catalog, decoder.as_mut(), &detectors, &observables)?;
        
        // Save the result, if it was a logical failure
        if failed {
            failures.push(fault_history);
        }

        // If we have enough failures, then return them
        if failures.len() >= nfail {
            return Ok(failures);
        }
    }
    Err(PecosError::LogicalFailureNotFound)
}