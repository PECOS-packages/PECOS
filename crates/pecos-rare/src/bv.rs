// Brayvi-Vargo style "splitting" method for computing failure probabilities
use crate::utils::{setup_rare_event_simulation, is_logical_failure};
use crate::metropolis::metropolis_step;
use pecos_core::errors::PecosError;
use pecos_decoder_core::ObservableDecoder;
use pecos_engines::monte_carlo::MonteCarloEngine;
use pecos_engines::{FaultCatalog, FaultHistory};
use pecos_quantum::TickCircuit;
use pecos_rare::FaultFlipper;

pub struct BVResults {
    seed: u64,
    initial_results: Option<BruteForceResults>,
    metropolis_results: Vec<MetropolisResults>,
    ratio_estimates: Vec<f64>,
}

impl BVResults {
    pub fn new(seed: u64) -> Self {
        Self {
            seed,
            initial_results: None,
            metropolis_results: Vec::new(),
            ratio_estimates: Vec::new(),
            probabilities: Vec::new(),
        }
    }

    // Saves the results from the initial brute force result
    pub fn cache_initial_failure(&mut self, brute_force_result: &BruteForceResults) {
        self.initial_results = Some(brute_force_result.clone());
    }

    pub fn cache_metropolis_results(&mut self, metropolis_results: &MetropolisResults) {
        self.metropolis_results.push(metropolis_results.clone());
    }

    pub fn cache_ratio_estimate(&mut self, ratio_estimate: f64) {
        self.ratio_estimates.push(ratio_estimate);
    }

    pub fn probability(&self, index: usize) -> f64 {
        let mut pfail = self.initial_results.as_ref()?.failure_rate();
        if index == 0 {
            return pfail
        }
    
        if self.ratio_estimates.is_empty() {
            panic!("No ratio estimates available to compute probability.");
        }

        for (i, &ratio) in self.ratio_estimates.iter().enumerate() {
            pfail *= ratio;
            if i + 1 == index {
                return pfail;
            }
        }

        panic!("Index out of bounds for probability computation.");
    }

    pub fn all_probabilities(&self) -> Vec<f64> {
        let mut probabilities = Vec::new();
        probabilities.push(self.probability(0));
        for &ratio in &self.ratio_estimates {
            probabilities.push(probabilities.last().unwrap() * ratio);
        }
        probabilities
    }

    pub fn write_json(&self, file_path: &str, level: usize) -> std::io::Result<()> {
        use std::fs::File;
        use serde_json::json;

        let data = match level {
            0 => {
                json!({
                    "seed": self.seed,
                    "probabilities": self.all_probabilities(),
                    "ratio_estimates": self.ratio_estimates,
                })
            }
            _ => return Err(PecosError::Input("JSON level must be 0".into())),
        };
        let file = File::create(file_path)?;
        serde_json::to_writer(file, &data)
            .map_err(|error| PecosError::with_context(error, "Unable to write JSON to file"))
    }
}

pub struct MetropolisResults {
    accepted: Vec<bool>,
    ran_decoder: Vec<bool>,
    fault_history_probabilities: Vec<f64>,
    fault_history_probabilities_plus: Vec<f64>,
    fault_history_probabilities_minus: Vec<f64>,
    fault_ratios_plus: Vec<f64>,
    fault_ratios_minus: Vec<f64>,
    bennett_ratios_plus: Vec<f64>,
    bennett_ratios_minus: Vec<f64>,
    convergence_threshold: f64,
}

impl MetropolisResults {

    pub fn new(convergence_threshold: f64) -> Self {
        Self {
            convergence_threshold,
            accepted: Vec::new(),
            ran_decoder: Vec::new(),
            fault_history_probabilities: Vec::new(),
            fault_history_probabilities_plus: Vec::new(),
            fault_history_probabilities_minus: Vec::new(),
            fault_ratios_plus: Vec::new(),
            fault_ratios_minus: Vec::new(),
            bennett_ratios_plus: Vec::new(),
            bennett_ratios_minus: Vec::new(),
        }
    }

    pub fn nsample(&self) -> usize {
        self.fault_history_probabilities.len()
    }

    pub fn record_step(
        &mut self,
        accepted: bool,
        ran_decoder: bool,
        phist: f64,
        phist_plus: Option<f64>,
        phist_minus: Option<f64>,
    ) -> None {
        self.accepted.push(accepted);
        self.ran_decoder.push(ran_decoder);

        self.fault_history_probabilities.push(phist);

        if let Some(phist_minus) = phist_minus {
            self.fault_history_probabilities_minus.push(phist_minus);
            self.fault_ratios_minus.push(phist / phist_minus); 
            self.bennett_ratios_minus.push(self.bennett_function(phist / phist_minus));
        }

        if let Some(phist_plus) = phist_plus {
            self.fault_history_probabilities_plus.push(phist_plus);
            self.fault_ratios_plus.push(phist / phist_plus);
            self.bennett_ratios_plus.push(self.bennett_function(phist / phist_plus));
        }
    }

    pub fn average_fault_history_probability(&self) -> f64 {
        let sum: f64 = self.fault_history_probabilities.iter().sum();
        sum / self.fault_history_probabilities.len() as f64
    }

    pub fn average_fault_history_probability_plus(&self) -> f64 {
        let sum: f64 = self.fault_history_probabilities_plus.iter().sum();
        sum / self.fault_history_probabilities_plus.len() as f64
    }

    pub fn average_fault_history_probability_minus(&self) -> f64 {
        let sum: f64 = self.fault_history_probabilities_minus.iter().sum();
        sum / self.fault_history_probabilities_minus.len() as f64
    }

    pub fn bennett_function(&self, val: f64) -> f64 {
        1.0 / (1.0 + val)
    }

    pub fn average_bennett_function_plus(&self) -> f64 {
        self.bennett_ratios_plus.iter().sum() / self.bennett_ratios_plus.len() as f64
    }

    pub fn average_bennett_function_minus(&self) -> f64 {
        self.bennett_ratios_minus.iter().sum() / self.bennett_ratios_minus.len() as f64
    }

    pub fn variance_bennett_function_plus(&self) -> f64 {
        let average = self.average_bennett_function_plus();
        let variance: f64 = self.bennett_ratios_plus.iter().map(|&x| (x - mean).powi(2)).sum();
        variance / (self.bennett_ratios_plus.len()-1) as f64
    }

    pub fn variance_bennett_function_minus(&self) -> f64 {
        let average = self.average_bennett_function_minus();
        let variance: f64 = self.bennett_ratios_minus.iter().map(|&x| (x - mean).powi(2)).sum();
        variance / (self.bennett_ratios_minus.len()-1) as f64
    }

    pub fn average_and_variance_bennett_function_plus(&self) -> Result<(f64, f64), PecosError> {
        let average = self.average_bennett_function_plus();
        let variance: f64 = self.bennett_ratios_plus.iter().map(|&x| (x - mean).powi(2)).sum();
        Ok((average, variance / (self.bennett_ratios_plus.len()-1) as f64))
    }

    pub fn average_and_variance_bennett_function_minus(&self) -> Result<(f64, f64), PecosError> {
        let average = self.average_bennett_function_minus();
        let variance: f64 = self.bennett_ratios_minus.iter().map(|&x| (x - mean).powi(2)).sum();
        Ok((average, variance / (self.bennett_ratios_minus.len()-1) as f64))
    }

    pub fn statistical_error_plus(&self) -> f64 {
        let (average, variance) = self.average_and_variance_bennett_function_plus().unwrap();
        let std = variance.powf(0.5);
        (std / average) / (self.nsample() as f64).sqrt()
    }

    pub fn statistical_error_minus(&self) -> f64 {
        let (average, variance) = self.average_and_variance_bennett_function_minus().unwrap();
        let std = variance.powf(0.5);
        (std / average) / (self.nsample() as f64).sqrt()
    }

    pub fn statistical_error(&self) -> f64 {
        if self.bennett_ratios_plus.is_empty() && self.bennett_ratios_minus.is_empty() {
            panic!("No samples available to calculate statistical error");
        }
        else if self.bennett_ratios_plus.is_empty() {
            return self.statistical_error_minus();
        }
        else if self.bennett_ratios_minus.is_empty() {
            return self.statistical_error_plus();
        }
        // If both plus and minus have samples, calculate the maximum of their statistical errors
        self.statistical_error_plus().max(self.statistical_error_minus())
    }

    // Returns average over first half of samples
    pub fn subset_average_bennett_function_plus(&self) -> f64 {
        self.bennett_ratios_plus.iter().take(self.bennett_ratios_plus.len() / 2).sum::<f64>() / (self.bennett_ratios_plus.len() / 2) as f64
    }

    // Returns average over first half of samples
    pub fn subset_average_bennett_function_minus(&self) -> f64 {
        self.bennett_ratios_minus.iter().take(self.bennett_ratios_minus.len() / 2).sum::<f64>() / (self.bennett_ratios_minus.len() / 2) as f64
    }

    pub fn mixing_discrepancy_plus(&self) -> f64 {
        let subset_average_plus = self.subset_average_bennett_function_plus();
        let average_plus = self.average_bennett_function_plus();
        (average_plus - subset_average_plus).abs() / average_plus.abs()
    }

    pub fn mixing_discrepancy_minus(&self) -> f64 {
        let subset_average_minus = self.subset_average_bennett_function_minus();
        let average_minus = self.average_bennett_function_minus();
        (average_minus - subset_average_minus).abs() / average_minus.abs()
    }

    pub fn mixing_discrepancy(&self) -> f64 {
        if self.bennett_ratios_plus.is_empty() && self.bennett_ratios_minus.is_empty() {
            panic!("No samples available to calculate mixing discrepancy");
        }
        else if self.bennett_ratios_plus.is_empty() {
            return self.mixing_discrepancy_minus();
        }
        else if self.bennett_ratios_minus.is_empty() {
            return self.mixing_discrepancy_plus();
        }
        // If both plus and minus have samples, calculate the maximum of their discrepancies
        self.mixing_discrepancy_plus().max(self.mixing_discrepancy_minus())
    }

    pub fn convergence_criteria(&self) -> (f64, f64) {
        let mixing_discrepancy = self.mixing_discrepancy();
        let statistical_error = self.statistical_error();
        (mixing_discrepancy, statistical_error)
    }

    pub fn is_converged(&self) -> bool {
        let (mixing_discrepancy, statistical_error) = self.convergence_criteria();
        mixing_discrepancy + statistical_error < self.convergence_threshold
    }

    pub fn ratio_denominator_weighted_estimation_sum(&self, ratio_estimate: f64) -> f64 {
        let mut sum = 0.;
        for idx in 0..self.nsample() {
            sum += self.bennett_function(self.fault_history_probabilities[idx] / self.fault_history_probabilities_minus[idx] / ratio_estimate)
        }
        sum / (self.nsample() as f64)
    }

    pub fn ratio_numerator_weighted_estimation_sum(&self, ratio_estimate: f64) -> f64 {
        let mut sum = 0.;
        for idx in 0..self.nsample() {
            sum += self.bennett_function(self.fault_history_probabilities[idx] / self.fault_history_probabilities_plus[idx] * ratio_estimate)
        }
        sum / (self.nsample() as f64)
    }
}

pub fn find_logical_failure_rate(
    circuit: &TickCircuit,
    prob_sequence: Vec<f64>, // Sequence of physical error probabilities
    seed: u64, // Random seed for reproducibility
    max_brute_force_histories: usize,
    nbootstraps_brute_force: usize,
    check_frequency_brute_force: usize,
    error_threshold: f64,
    flipper_type: &str,
    region_probability: f64,
    nmetropolis: usize,
    burn_in: usize,
    max_ratio_iterations: usize,
) -> Result<BVResults, PecosError> {

    // Initialize the BVResult structure
    let results = BVResults::new(seed);

    // Grab everything that's needed for the simulation
    let p_phys = prob_sequence[0];
    let (fault_catalog, mut monte_carlo, mut decoder, detectors, observables) = setup_rare_event_simulation(circuit, p_phys, seed)?;

    // Find the initial failure probability
    let level_threshold = error_threshold / (prob_sequence.len() as f64).powf(0.5);
    let brute_force_results = find_logical_failure_rate(
        &circuit,
        p_phys,
        seed,
        max_brute_force_histories,
        true,
        "last".into(),
        nbootstraps_brute_force,
        check_frequency_brute_force,
        level_threshold,
    )?;

    // Cache the initial failure results
    results.cache_initial_failure(&brute_force_results);

    // Extract the failing fault history from the results
    let fault_history = brute_force_results.failures[0];

    // Loop over the remaining probabilities in the sequence
    for (i, &p_phys) in prob_sequence.iter().enumerate() {

        // Set up everything at this probability level
        let p_phys = prob_sequence[i];
        let (fault_catalog, mut monte_carlo, mut decoder, detectors, observables) = setup_rare_event_simulation(circuit, p_phys, seed)?;
        let phist = fault_catalog.fault_history_probability(&fault_history);

        // Grab information from the previous level
        let use_minus = i > 0;
        if use_minus {
            let p_phys_minus = prob_sequence[i - 1];
            let fault_catalog_minus = setup_rare_event_simulation(circuit, p_phys_minus, seed)[0];
            let phist_minus = fault_catalog_minus.fault_history_probability(&fault_history);
        }

        // Grab information from the next level
        let use_plus = i < prob_sequence.len() - 1;
        if use_plus {
            let p_phys_plus = prob_sequence[i + 1];
            let fault_catalog_plus = setup_rare_event_simulation(circuit, p_phys_plus, seed)[0];
            let phist_plus = fault_catalog_plus.fault_history_probability(&fault_history);
        }

        // Set up the flipper
        let mut flipper = match flipper_type.as_str() {
            "basic" => FaultFlipper::new_basic(fault_catalog),
            "basic_subregion" => FaultFlipper::new_basic_subregion(fault_catalog, region_probability),
            "weighted" => FaultFlipper::new_weighted(fault_catalog),
            "weighted_subregion" => FaultFlipper::new_weighted_subregion(fault_catalog, region_probability),
            _ => panic!("Unknown flipper type"),
        };

        // Set up structure to collect results
        let mut met_results = MetropolisResults::new(level_threshold);

        // Take a bunch of metropolis steps
        for step in 0..nmetropolis {

            // Perform a single metropolis step here
            let (accepted, ran_decoder, fault_history) = metropolis_step(
                &fault_history,
                &mut monte_carlo,
                &mut flipper,
                &mut decoder,
                &detectors,
                &observables,
            );

            // Compute fault histories
            if accepted {
                let phist = fault_catalog.fault_history_probability(&fault_history);
                if use_minus {
                    let phist_minus = fault_catalog_minus.fault_history_probability(&fault_history);
                }
                else {
                    let phist_minus = None;
                }
                if use_plus {
                    let phist_plus = Some(fault_catalog_plus.fault_history_probability(&fault_history));
                }
                else {
                    let phist_plus = None;
                }
            }

            // Cache the results
            if step > burn_in {
                met_results.record_step(accepted, ran_decoder, phist, phist_plus, phist_minus);
            }

            // Perform an occasional check for convergence
            if step > burn_in && step - burn_in % check_interval == 0 {
                let converged = met_results.is_converged();
                if converged {
                    break;
                }
            }
        }
        // Save the metropolis results for this level
        results.cache_metropolis_results(&met_results);

        // Self consistently estimate the probability ratio
        if i > 1 {
            let ratio_estimate = 1.0;
            
            for estimate_iter in 0..max_ratio_iterations {

                let new_ratio_estimate = (
                    ratio_estimate * 
                    metropolis_results[i - 1].ratio_numerator_weighted_estimation_sum(ratio_estimate) /
                    metropolis_results[i].ratio_denominator_weighted_estimation_sum(ratio_estimate)
                );

                // Check for ratio convergence
                if (new_ratio_estimate - ratio_estimate).abs()/ratio_estimate.abs() < level_threshold {
                    ratio_estimate = new_ratio_estimate;
                    break;
                }
                ratio_estimate = new_ratio_estimate;

                // Issue a warning if the ratio estimate did not converge within the maximum number of iterations
                if estimate_iter == max_ratio_iterations - 1 {
                    eprintln!("Warning: ratio estimate did not converge within {} iterations", max_ratio_iterations);
                }
            }
            // Cache the results
            results.cache_ratio_estimate(ratio_estimate);
            print(results.probabilities(i));
        }
    }
    results
}

pub fn gen_sequence(p_start: f64, p_end: f64, num_points: usize, distance: usize) -> Vec<f64> {
    let mut sequence = Vec::new();

    if num_points <= 1 {
        panic!("Number of points must be greater than 1");
    }

    sequence.push(p_start);
    for _ in 1..num_points-1 {
        let p_prev = sequence.last().copied().unwrap();
        let wi = ((distance as f64) / 2).max(p_prev * num_points as f64);
        let p_next =  p_prev / (2.0).powf( 1.0 / wi.powf(0.5) );
        sequence.push(p_next);
    }
    sequence.push(p_end);

    sequence
}