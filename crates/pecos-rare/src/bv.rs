// Brayvi-Vargo style "splitting" method for computing failure probabilities
use crate::brute_force::{self, BruteForceResults};
use crate::flipper::FaultFlipper;
use crate::utils::{setup_rare_event_simulation, start_rps_loading};
use crate::metropolis::metropolis_step;
use pecos_core::errors::PecosError;
use pecos_quantum::TickCircuit;
use pecos_random::PecosRng;

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
        let mut pfail = self.initial_results.as_ref()
            .expect("Initial failure results have not been cached")
            .failure_rate();
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

    pub fn write_json(
        &self,
        file_path: &str,
        level: usize,
        physical_error_rates: Option<&[f64]>,
    ) -> Result<(), PecosError> {
        use std::fs::File;
        use serde_json::json;

        let initial_results = self.initial_results.as_ref().ok_or_else(|| {
            PecosError::Input("Initial failure results have not been cached".into())
        })?;
        if let Some(rates) = physical_error_rates {
            if rates.len() != self.ratio_estimates.len() + 1 || !valid_sequence(rates) {
                return Err(PecosError::Input(
                    "Physical error rates must match the logical probabilities and form a valid probability sequence".into(),
                ));
            }
        }

        let metropolis_results = self.metropolis_results.iter()
            .map(|results| results.json_data(level))
            .collect::<Result<Vec<_>, _>>()?;
        let mut data = json!({
            "seed": self.seed,
            "probabilities": self.all_probabilities(),
            "ratio_estimates": self.ratio_estimates,
            "initial_results": initial_results.json_data(level)?,
            "metropolis_results": metropolis_results,
        });
        if let Some(rates) = physical_error_rates {
            data["physical_error_rates"] = json!(rates);
        }
        let file = File::create(file_path)?;
        serde_json::to_writer_pretty(file, &data)
            .map_err(|error| PecosError::with_context(error, "Unable to write JSON to file"))
    }
}

#[derive(Clone)]
pub struct MetropolisResults {
    accepted_steps: usize,
    decoder_calls: usize,
    fault_history_probabilities: Vec<f64>,
    fault_history_probabilities_plus: Vec<f64>,
    fault_history_probabilities_minus: Vec<f64>,
    fault_ratios_plus: Vec<f64>,
    fault_ratios_minus: Vec<f64>,
    bennett_ratios_plus: Vec<f64>,
    bennett_ratios_minus: Vec<f64>,
    convergence_threshold: f64,
    minimum_accepted_flips: usize,
    convergence_history: Vec<(usize, f64, f64)>,
    burn_in_steps: usize,
    burn_in_accepted_steps: usize,
    burn_in_decoder_calls: usize,
}

impl MetropolisResults {

    pub fn new(convergence_threshold: f64, minimum_accepted_flips: usize) -> Self {
        Self {
            convergence_threshold,
            minimum_accepted_flips,
            accepted_steps: 0,
            decoder_calls: 0,
            fault_history_probabilities: Vec::new(),
            fault_history_probabilities_plus: Vec::new(),
            fault_history_probabilities_minus: Vec::new(),
            fault_ratios_plus: Vec::new(),
            fault_ratios_minus: Vec::new(),
            bennett_ratios_plus: Vec::new(),
            bennett_ratios_minus: Vec::new(),
            convergence_history: Vec::new(),
            burn_in_steps: 0,
            burn_in_accepted_steps: 0,
            burn_in_decoder_calls: 0,
        }
    }

    pub(crate) fn json_data(&self, level: usize) -> Result<serde_json::Value, PecosError> {
        use serde_json::json;

        if level > 2 {
            return Err(PecosError::Input("JSON level must be 0, 1, or 2".into()));
        }
        let convergence_history: Vec<_> = self.convergence_history.iter()
            .map(|&(number_of_samples, mixing_discrepancy, statistical_error)| {
                json!({
                    "number_of_samples": number_of_samples,
                    "number_of_steps": number_of_samples + self.burn_in_steps,
                    "mixing_discrepancy": mixing_discrepancy,
                    "statistical_error": statistical_error,
                    "converged": mixing_discrepancy + statistical_error < self.convergence_threshold,
                })
            })
            .collect();
        let mut data = json!({
            "number_of_steps": self.nsample() + self.burn_in_steps,
            "number_of_samples": self.nsample(),
            "number_of_accepted_steps": self.accepted_steps + self.burn_in_accepted_steps,
            "number_of_decoder_calls": self.decoder_calls + self.burn_in_decoder_calls,
            "burn_in_steps": self.burn_in_steps,
            "convergence_threshold": self.convergence_threshold,
            "convergence_history": convergence_history,
        });
        if level == 2 {
            for (name, values) in [
                ("accepted", json!(self.accepted_steps)),
                ("ran_decoder", json!(self.decoder_calls)),
                ("fault_history_probabilities", json!(self.fault_history_probabilities)),
                ("fault_history_probabilities_plus", json!(self.fault_history_probabilities_plus)),
                ("fault_history_probabilities_minus", json!(self.fault_history_probabilities_minus)),
            ] {
                data[name] = values;
            }
        }
        Ok(data)
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
    ) {
        if accepted {
            self.accepted_steps += 1;
        }
        if ran_decoder {
            self.decoder_calls += 1;
        }

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
        self.bennett_ratios_plus.iter().sum::<f64>() / self.bennett_ratios_plus.len() as f64
    }

    pub fn average_bennett_function_minus(&self) -> f64 {
        self.bennett_ratios_minus.iter().sum::<f64>() / self.bennett_ratios_minus.len() as f64
    }

    pub fn variance_bennett_function_plus(&self) -> f64 {
        let average = self.average_bennett_function_plus();
        let variance: f64 = self.bennett_ratios_plus.iter().map(|&value| (value - average).powi(2)).sum();
        variance / (self.bennett_ratios_plus.len()-1) as f64
    }

    pub fn variance_bennett_function_minus(&self) -> f64 {
        let average = self.average_bennett_function_minus();
        let variance: f64 = self.bennett_ratios_minus.iter().map(|&value| (value - average).powi(2)).sum();
        variance / (self.bennett_ratios_minus.len()-1) as f64
    }

    pub fn average_and_variance_bennett_function_plus(&self) -> Result<(f64, f64), PecosError> {
        let average = self.average_bennett_function_plus();
        let variance: f64 = self.bennett_ratios_plus.iter().map(|&value| (value - average).powi(2)).sum();
        Ok((average, variance / (self.bennett_ratios_plus.len()-1) as f64))
    }

    pub fn average_and_variance_bennett_function_minus(&self) -> Result<(f64, f64), PecosError> {
        let average = self.average_bennett_function_minus();
        let variance: f64 = self.bennett_ratios_minus.iter().map(|&value| (value - average).powi(2)).sum();
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

    pub fn convergence_criteria(&mut self) -> (f64, f64) {
        let mixing_discrepancy = self.mixing_discrepancy();
        let statistical_error = self.statistical_error();
        self.convergence_history.push((self.nsample(), mixing_discrepancy, statistical_error));
        (mixing_discrepancy, statistical_error)
    }

    pub fn is_converged(&mut self) -> bool {
        if self.nsample() < 2 {
            return false;
        }
        let (mixing_discrepancy, statistical_error) = self.convergence_criteria();
        if self.accepted_steps < self.minimum_accepted_flips {
            return false;
        }
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

fn valid_sequence(prob_sequence: &[f64]) -> bool {
    !prob_sequence.is_empty()
        && prob_sequence.iter().all(|probability| probability.is_finite() && *probability > 0.0 && *probability < 1.0)
        && prob_sequence.windows(2).all(|pair| pair[0] >= pair[1])
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
    minimum_accepted_flips: usize,
) -> Result<BVResults, PecosError> {

    // Check that the probability sequence is valid
    if !valid_sequence(&prob_sequence) {
        return Err(PecosError::Input("Physical probabilities must be non-increasing and strictly between 0 and 1".into()));
    }

    // Check the various sampling parameters for validity
    if max_brute_force_histories == 0 || nbootstraps_brute_force == 0
        || check_frequency_brute_force == 0 || max_ratio_iterations == 0
        || nmetropolis.saturating_sub(burn_in) < 2
        || !error_threshold.is_finite() || error_threshold <= 0.0
        || !region_probability.is_finite() || !(0.0..=1.0).contains(&region_probability)
    {
        return Err(PecosError::Input("Invalid sampling budget, convergence threshold, or region probability".into()));
    }

    // Check the flipper value for validity
    if !matches!(flipper_type, "basic" | "basic_subregion" | "weighted" | "weighted_subregion") {
        return Err(PecosError::Input(format!("Unknown flipper type: {flipper_type}")));
    }

    // Initialize the BVResult structure
    let mut results = BVResults::new(seed);

    // Grab the decoder & info at lowest error rate
    let (_, _, mut decoder, detectors, observables) = setup_rare_event_simulation(circuit, prob_sequence[prob_sequence.len() - 1], seed)?;

    // Find the initial failure probability
    let p_phys = prob_sequence[0];
    let level_threshold = error_threshold / (prob_sequence.len() as f64).powf(0.5);
    let brute_force_results = brute_force::find_logical_failure_rate(
        circuit,
        p_phys,
        seed,
        max_brute_force_histories,
        true,
        "last".into(),
        nbootstraps_brute_force,
        check_frequency_brute_force,
        level_threshold,
    )?;
    println!("Initial failure probability: {}", brute_force_results.failure_rate());
    println!("-----------------------------------------------------------------");

    // Cache the initial failure results
    results.cache_initial_failure(&brute_force_results);

    // Extract the failing fault history from the results
    let mut fault_history = brute_force_results.failures.last().cloned()
        .ok_or(PecosError::LogicalFailureNotFound)?;

    let mut rng = PecosRng::seed_from_u64(seed.wrapping_add(prob_sequence.len() as u64));

    // Loop over the remaining probabilities in the sequence
    for (level, &p_phys) in prob_sequence.iter().enumerate() {

        println!("-----------------------------------------------------------------");
        println!("Starting simulation for p_phys = {}, level {} / {}", p_phys, level, prob_sequence.len());

        // Set up everything at this probability level
        let level_seed = seed.wrapping_add(level as u64);
        let (fault_catalog, mut monte_carlo, _, _, _) = setup_rare_event_simulation(circuit, p_phys, level_seed)?;

        // Grab information from the previous level
        let fault_catalog_minus = if level > 0 {
            Some(setup_rare_event_simulation(circuit, prob_sequence[level - 1], level_seed)?.0)
        } else {
            None
        };

        // Grab information from the next level
        let fault_catalog_plus = if level < prob_sequence.len() - 1 {
            Some(setup_rare_event_simulation(circuit, prob_sequence[level + 1], level_seed)?.0)
        } else {
            None
        };

        // Set up the flipper
        let mut flipper = match flipper_type {
            "basic" => FaultFlipper::new_basic(fault_catalog.clone()),
            "basic_subregion" => FaultFlipper::new_basic_subregion(fault_catalog.clone(), region_probability),
            "weighted" => FaultFlipper::new_weighted(fault_catalog.clone()),
            "weighted_subregion" => FaultFlipper::new_weighted_subregion(fault_catalog.clone(), region_probability),
            _ => unreachable!("Flipper type was validated above"),
        };
        flipper.set_seed(level_seed);

        // Set up structure to collect results
        let mut met_results = MetropolisResults::new(level_threshold, minimum_accepted_flips);
        let (pb, game_thread) = start_rps_loading(nmetropolis as u64);

        // Take a bunch of metropolis steps
        for step in 0..nmetropolis {

            // Perform a single metropolis step here
            let (accepted, ran_decoder, next_history) = match metropolis_step(
                &fault_history,
                &mut monte_carlo,
                &mut flipper,
                &mut rng,
                decoder.as_mut(),
                &detectors,
                &observables,
            ) {
                Ok(outcome) => outcome,
                Err(error) => {
                    pb.abandon_with_message("Simulation failed");
                    let _ = game_thread.join();
                    return Err(error);
                }
            };
            fault_history = next_history;

            // Compute fault history probabilities and cache the results
            if step >= burn_in {
                let phist = fault_catalog.fault_history_probability(&fault_history);
                let phist_plus = fault_catalog_plus.as_ref()
                    .map(|catalog| catalog.fault_history_probability(&fault_history));
                let phist_minus = fault_catalog_minus.as_ref()
                    .map(|catalog| catalog.fault_history_probability(&fault_history));
                met_results.record_step(accepted, ran_decoder, phist, phist_plus, phist_minus);
            } else {
                met_results.burn_in_steps += 1;
                met_results.burn_in_accepted_steps += usize::from(accepted);
                met_results.burn_in_decoder_calls += usize::from(ran_decoder);
            }
            pb.inc(1);

            // Perform an occasional check for convergence
            if met_results.nsample() >= 2 && met_results.nsample() % check_frequency_brute_force == 0 {
                let converged = met_results.is_converged();
                let &(_, mixing_discrepancy, statistical_error) = met_results.convergence_history.last()
                    .expect("Convergence check must cache its criteria");
                pb.println(format!(
                    "Level {}/{} p={:.3e}, {}/{} accepted flips, {} samples: mixing discrepancy={:.3e}, statistical error={:.3e} (target < {:.3e})",
                    level + 1,
                    prob_sequence.len(),
                    p_phys,
                    met_results.accepted_steps,
                    minimum_accepted_flips,
                    met_results.nsample(),
                    mixing_discrepancy,
                    statistical_error,
                    met_results.convergence_threshold,
                ));
                if converged {
                    break;
                }
            }
        }
        pb.finish_and_clear();
        game_thread.join()
            .map_err(|_| PecosError::Processing("Progress animation thread panicked".into()))?;

        // Save the metropolis results for this level
        results.cache_metropolis_results(&met_results);

        // Self consistently estimate the probability ratio
        if level > 0 {
            let mut ratio_estimate = 1.0;
            
            for estimate_iter in 0..max_ratio_iterations {

                let new_ratio_estimate =
                    ratio_estimate * 
                    results.metropolis_results[level - 1].ratio_numerator_weighted_estimation_sum(ratio_estimate) /
                    results.metropolis_results[level].ratio_denominator_weighted_estimation_sum(ratio_estimate);

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
        }

        println!("Finished simulation for p_phys = {}, level {} / {}", p_phys, level, prob_sequence.len());
        println!("Error Probability: {:?}", results.probability(level));
        println!("Required {} samples", met_results.nsample());
        println!("-----------------------------------------------------------------");
    }
    Ok(results)
}

pub fn gen_sequence(p_start: f64, p_end: f64, num_points: usize) -> Vec<f64> {
    // Generic log spaced sequence
    if num_points <= 1 {
        panic!("Number of points must be greater than 1");
    }

    let step = (p_end.log10() - p_start.log10()) / (num_points - 1) as f64;
    (0..num_points)
        .map(|i| 10.0_f64.powf(p_start.log10() + i as f64 * step))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use pecos_engines::FaultHistory;

    #[test]
    fn sequence_stops_at_the_target_without_reversing_direction() {
        let sequence = gen_sequence(0.1, 0.00001, 100);
        assert_eq!(sequence[0], 0.1);
        assert!((*sequence.last().unwrap() - 0.00001).abs() < 1e-15);
        assert!(sequence.windows(2).all(|pair| pair[0] > pair[1]));
        assert!(sequence.len() <= 100);
    }

    #[test]
    fn bv_chain_moves_beyond_the_initial_history_neighborhood() {
        let mut circuit = TickCircuit::new();
        circuit.tick().pz(&[0, 1, 2, 3]);
        let measurements = circuit.tick().mz(&[0, 1, 2, 3]);
        circuit.detector(&[measurements[1]]).unwrap();
        circuit.observable_labeled("L0", &[measurements[0]]).unwrap();
        circuit.set_meta(
            "detectors",
            pecos_quantum::Attribute::String(
                serde_json::json!([{"id": 0, "meas_ids": [1]}]).to_string(),
            ),
        );
        circuit.set_meta(
            "num_measurements",
            pecos_quantum::Attribute::String(circuit.num_measurements().to_string()),
        );

        let results = find_logical_failure_rate(
            &circuit, vec![0.25, 0.125], 0, 128, 16, 32, 1e-6,
            "basic", 1.0, 128, 4, 1_000,
        ).unwrap();

        for chain in &results.metropolis_results {
            assert_eq!(chain.burn_in_steps, 4);
            assert!(chain.burn_in_accepted_steps <= chain.burn_in_decoder_calls);
            assert!(chain.burn_in_decoder_calls <= chain.burn_in_steps);
            assert!(!chain.convergence_history.is_empty());
            assert_eq!(chain.convergence_history.len(), chain.nsample() / 32);
            assert_eq!(chain.convergence_history[0].0, 32);
        }
        let probabilities = &results.metropolis_results[0].fault_history_probabilities;
        let smallest = probabilities.iter().copied().fold(f64::INFINITY, f64::min);
        let largest = probabilities.iter().copied().fold(0.0, f64::max);
        let single_flip_ratio: f64 = (1.0 - 0.25) / 0.25;
        assert!(
            largest / smallest > single_flip_ratio.powi(2) * (1.0 + 1e-12),
            "The chain must explore beyond one-site changes of a fixed seed history",
        );

        let converged_results = find_logical_failure_rate(
            &circuit, vec![0.25, 0.25], 0, 128, 16, 32, 10.0,
            "basic", 1.0, 128, 4, 1_000,
        ).unwrap();
        for chain in &converged_results.metropolis_results {
            assert_eq!(chain.nsample(), 32);
            assert_eq!(chain.burn_in_steps, 4);
            assert_eq!(chain.convergence_history, vec![(32, 0.0, 0.0)]);
        }
    }

    #[test]
    fn bennett_estimate_recovers_a_constant_probability_ratio() {
        let mut lower = MetropolisResults::new(0.05);
        let mut upper = MetropolisResults::new(0.05);
        for accepted in [true, false] {
            lower.record_step(accepted, true, 0.1, Some(0.2), None);
            upper.record_step(accepted, true, 0.2, None, Some(0.1));
        }
        let ratio = lower.ratio_numerator_weighted_estimation_sum(1.0)
            / upper.ratio_denominator_weighted_estimation_sum(1.0);
        assert!((ratio - 2.0).abs() < 1e-12);
        assert_eq!(lower.nsample(), 2);
        assert!(lower.is_converged());
        assert!(upper.is_converged());
    }

    #[test]
    fn probabilities_include_the_first_adjacent_level_ratio() {
        let mut initial = BruteForceResults::new(0, "false".into(), 100);
        initial.record_failure(&FaultHistory::Depolarizing(Vec::new()));
        for _ in 0..3 {
            initial.record_pass();
        }
        let mut results = BVResults::new(0);
        results.cache_initial_failure(&initial);
        results.cache_ratio_estimate(0.5);
        results.cache_ratio_estimate(0.2);
        assert_eq!(results.all_probabilities(), vec![0.25, 0.125, 0.025]);
        assert_eq!(results.probability(2), 0.025);
    }

    #[test]
    fn json_levels_include_initial_results_and_cached_metropolis_data() {
        use serde_json::{Value, json};
        use std::fs::{self, File};

        let mut initial = BruteForceResults::new(7, "false".into(), 200);
        initial.record_failure(&FaultHistory::Depolarizing(Vec::new()));
        for _ in 0..3 {
            initial.record_pass();
        }
        let mut results = BVResults::new(7);
        results.cache_initial_failure(&initial);
        results.cache_ratio_estimate(0.5);
        for (probability, plus, minus) in [(0.2, Some(0.1), None), (0.1, None, Some(0.2))] {
            let mut chain = MetropolisResults::new(0.05);
            chain.burn_in_steps = 2;
            chain.burn_in_accepted_steps = 1;
            chain.burn_in_decoder_calls = 2;
            chain.record_step(true, true, probability, plus, minus);
            chain.record_step(false, false, probability, plus, minus);
            assert!(chain.is_converged());
            chain.record_step(true, true, 2.0 * probability, plus, minus);
            let expected = (chain.mixing_discrepancy(), chain.statistical_error());
            assert!(!chain.is_converged());
            assert_eq!(chain.convergence_history, vec![(2, 0.0, 0.0), (3, expected.0, expected.1)]);
            results.cache_metropolis_results(&chain);
        }

        let directory = std::env::temp_dir();
        let bv_path = directory.join(format!("pecos-bv-json-{}.json", std::process::id()));
        let initial_path = directory.join(format!("pecos-bv-initial-json-{}.json", std::process::id()));
        let rates = [0.2, 0.1];
        for level in 0..=2 {
            results.write_json(bv_path.to_str().unwrap(), level, Some(&rates)).unwrap();
            initial.write_json(initial_path.to_str().unwrap(), level).unwrap();
            let data: Value = serde_json::from_reader(File::open(&bv_path).unwrap()).unwrap();
            let initial_data: Value = serde_json::from_reader(File::open(&initial_path).unwrap()).unwrap();
            assert_eq!(data["initial_results"], initial_data);
            assert_eq!(data["initial_results"].get("failure_rate_std").is_some(), level >= 1);
            assert_eq!(data["initial_results"].get("results").is_some(), level == 2);
            assert_eq!(data["seed"], 7);
            assert_eq!(data["physical_error_rates"], json!(rates));
            assert_eq!(data["probabilities"], json!([0.25, 0.125]));
            assert_eq!(data["ratio_estimates"], json!([0.5]));
            for (index, chain) in data["metropolis_results"].as_array().unwrap().iter().enumerate() {
                assert_eq!(chain["number_of_steps"], 5);
                assert_eq!(chain["number_of_samples"], 3);
                assert_eq!(chain["number_of_accepted_steps"], 3);
                assert_eq!(chain["number_of_decoder_calls"], 4);
                assert_eq!(chain["burn_in_steps"], 2);
                assert_eq!(chain["convergence_threshold"], 0.05);
                let history = chain["convergence_history"].as_array().unwrap();
                assert_eq!(history.len(), 2);
                for (check, &(samples, discrepancy, error)) in history.iter()
                    .zip(&results.metropolis_results[index].convergence_history)
                {
                    assert_eq!(check["number_of_samples"], samples);
                    assert_eq!(check["number_of_steps"], samples + 2);
                    assert!((check["mixing_discrepancy"].as_f64().unwrap() - discrepancy).abs() < 1e-12);
                    assert!((check["statistical_error"].as_f64().unwrap() - error).abs() < 1e-12);
                }
                assert_eq!(history[0]["converged"], true);
                assert_eq!(history[1]["converged"], false);
                for field in [
                    "accepted", "ran_decoder", "fault_history_probabilities",
                    "fault_history_probabilities_plus", "fault_history_probabilities_minus",
                    "fault_ratios_plus", "fault_ratios_minus",
                    "bennett_ratios_plus", "bennett_ratios_minus",
                ] {
                    assert_eq!(chain.get(field).is_some(), level == 2);
                }
            }
            if level == 2 {
                let chain = &data["metropolis_results"][0];
                assert_eq!(chain["accepted"], json!([true, false, true]));
                assert_eq!(chain["ran_decoder"], json!([true, false, true]));
                assert_eq!(chain["fault_history_probabilities"], json!([0.2, 0.2, 0.4]));
                assert_eq!(chain["fault_history_probabilities_plus"], json!([0.1, 0.1, 0.1]));
                assert_eq!(chain["fault_history_probabilities_minus"], json!([]));
                assert_eq!(chain["fault_ratios_plus"], json!([2.0, 2.0, 4.0]));
            }
        }
        results.write_json(bv_path.to_str().unwrap(), 0, None).unwrap();
        let without_rates = fs::read(&bv_path).unwrap();
        let data: Value = serde_json::from_slice(&without_rates).unwrap();
        assert!(data.get("physical_error_rates").is_none());
        assert!(results.write_json(bv_path.to_str().unwrap(), 3, None).is_err());
        assert!(results.write_json(bv_path.to_str().unwrap(), 0, Some(&[0.2])).is_err());
        assert!(results.write_json(bv_path.to_str().unwrap(), 0, Some(&[0.2, f64::NAN])).is_err());
        assert!(BVResults::new(0).write_json(bv_path.to_str().unwrap(), 0, None).is_err());
        assert_eq!(fs::read(&bv_path).unwrap(), without_rates);
        fs::remove_file(bv_path).unwrap();
        fs::remove_file(initial_path).unwrap();
    }
}