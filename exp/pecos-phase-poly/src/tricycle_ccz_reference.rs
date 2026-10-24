// Copyright 2026 The PECOS Developers
// Licensed under the Apache License, Version 2.0.

//! Static Clifford reference for arXiv:2510.07269, Table II and the appendix
//! "Details for simulations". Three logical-plus blocks undergo one Z-check
//! extraction round without repair, a correlated-Z surrogate, and X readout.
//! Table II's LER corresponds to the particular CX schedule recorded with the
//! paper's run. Other valid colorings admit weight-two undetected logical faults
//! through ancilla-to-data Z propagation and give about 15 times higher LER for
//! C27 at p = 0.001. Acceptance does not depend on the schedule. Cup-product
//! supports and the edge-coloring constructor are shared with `tricycle_ccz_noisy`.
//! Exact and sampled fault-count strata retain their unconditional PMF weights.
//! Omitted mass is reported separately from statistical uncertainty.

use std::collections::BTreeSet;

use pecos_core::{PauliString, gate_type::GateType};
use pecos_qec::BivariateTricycleCode;
use pecos_qec::fault_tolerance::fault_sampler::{
    FaultAlternative, FaultCatalog, FaultChannel, FaultKind, FaultLocation, FaultStratumCounts,
    StochasticNoiseParams, StratifiedEstimate,
};
use pecos_quantum::TickCircuit;
use pecos_random::PecosRng;

use crate::distillation::validate_probability;
use crate::tricycle_ccz_noisy::{ExtractionSchedule, surrogate_supports};

#[cfg(test)]
mod tests;

/// Build the ideal circuit with every `H_X` row and all nine logical X observables.
///
/// # Errors
/// Returns size, scheduling, or annotation errors.
pub fn paper_circuit(
    code: &BivariateTricycleCode,
    schedule: &ExtractionSchedule,
) -> Result<TickCircuit, String> {
    schedule.validate(code)?;
    let n = code.num_qubits();
    let total = n
        .checked_add(code.hz().num_checks())
        .and_then(|v| v.checked_mul(3))
        .ok_or("three-block circuit size overflows usize")?;
    let mut circuit = TickCircuit::new();
    circuit.tick().px(&(0..3 * n).collect::<Vec<_>>());
    circuit.tick_at(0).pz(&(3 * n..total).collect::<Vec<_>>());
    for pairs in schedule.three_blocks() {
        circuit.tick().cx(&pairs);
    }
    circuit.tick().mz(&(3 * n..total).collect::<Vec<_>>());
    let measurements = circuit.tick().mx(&(0..3 * n).collect::<Vec<_>>());
    for block in 0..3 {
        for row in code.hx().rows() {
            let records = row
                .iter()
                .enumerate()
                .filter_map(|(q, &b)| (b != 0).then_some(measurements[block * n + q]))
                .collect::<Vec<_>>();
            circuit.detector(&records).map_err(|e| e.to_string())?;
        }
        for row in code.logical_x().rows() {
            let records = row
                .iter()
                .enumerate()
                .filter_map(|(q, &b)| (b != 0).then_some(measurements[block * n + q]))
                .collect::<Vec<_>>();
            circuit.observable(&records).map_err(|e| e.to_string())?;
        }
    }
    Ok(circuit)
}

/// Build paper-noise mechanisms, including one explicit surrogate per data qubit.
/// Ancilla preparation and all measurements flip with probability 2p/3.
/// Data preparation is ideal. CX faults have fifteen equiprobable alternatives.
///
/// # Errors
/// Returns invalid probability, circuit, or catalog errors.
pub fn paper_catalog(
    code: &BivariateTricycleCode,
    schedule: &ExtractionSchedule,
    p: f64,
) -> Result<FaultCatalog, String> {
    validate_probability(p)?;
    let circuit = paper_circuit(code, schedule)?;
    let mut catalog = FaultCatalog::from_circuit(&circuit).map_err(|e| e.to_string())?;
    catalog.with_noise(&StochasticNoiseParams {
        p1: 0.0,
        p2: p,
        p_meas: 2.0 * p / 3.0,
        p_prep: 2.0 * p / 3.0,
    });
    for location in &mut catalog.locations {
        if location.gate_type == GateType::PX {
            location.channel_probability = 0.0;
            location.no_fault_probability = 1.0;
            for alternative in &mut location.faults {
                alternative.absolute_probability = 0.0;
            }
        }
    }
    let measurements = catalog
        .locations
        .iter()
        .filter(|l| l.gate_type == GateType::MX)
        .map(|l| (l.qubits[0], l.faults[0].clone()))
        .collect::<std::collections::BTreeMap<_, _>>();
    let measurement_tick = catalog
        .locations
        .iter()
        .find(|l| l.gate_type == GateType::MX)
        .ok_or("paper circuit has no data measurements")?
        .tick;
    for support in surrogate_supports(code) {
        let mut affected_measurements = BTreeSet::new();
        let mut affected_detectors = BTreeSet::new();
        let mut affected_observables = BTreeSet::new();
        for &q in &support {
            let record = &measurements[&q];
            xor_into(&mut affected_measurements, &record.affected_measurements);
            xor_into(&mut affected_detectors, &record.affected_detectors);
            xor_into(&mut affected_observables, &record.affected_observables);
        }
        catalog.locations.push(FaultLocation {
            // Not a circuit gate: the Z acts just before the final MX gates, so it
            // is filed under their tick, and `gate_index` 0 names the first of them.
            tick: measurement_tick,
            gate_index: 0,
            gate_type: GateType::Z,
            qubits: support.clone(),
            channel: FaultChannel::Explicit,
            channel_probability: p,
            no_fault_probability: 1.0 - p,
            num_alternatives: 1,
            faults: vec![FaultAlternative {
                kind: FaultKind::Pauli,
                pauli: Some(PauliString::zs(&support)),
                affected_measurements: affected_measurements.into_iter().collect(),
                affected_detectors: affected_detectors.into_iter().collect(),
                affected_observables: affected_observables.into_iter().collect(),
                affected_tracked_paulis: Vec::new(),
                conditional_probability: 1.0,
                absolute_probability: p,
            }],
        });
    }
    Ok(catalog)
}

fn xor_into(target: &mut BTreeSet<usize>, values: &[usize]) {
    for &value in values {
        if !target.remove(&value) {
            target.insert(value);
        }
    }
}

/// Unconditional contribution of one fault-count stratum.
///
/// Contributions are linear diagnostics and can underflow to zero for extreme
/// catalogs; the combined [`StratifiedEstimate`] works in log space throughout.
#[derive(Clone, Debug)]
pub struct StratumEstimate {
    /// Number of firing mechanisms.
    pub k: usize,
    /// True for exhaustive enumeration.
    pub exact: bool,
    /// Represented exact configurations or sampled shots.
    pub configurations: u64,
    /// Probability of this fault count.
    pub mass: f64,
    /// Contribution to P(success).
    pub success_probability: f64,
    /// Contribution to P(success and logical failure).
    pub failure_probability: f64,
}

/// Combined result, with fault-count contributions for diagnosing disagreements.
#[derive(Clone, Debug)]
pub struct ReferenceEstimate {
    /// Probabilities, standard errors, omitted mass, and tail bound.
    pub estimate: StratifiedEstimate,
    /// Individual stratum contributions; divide failures by total success for LER contributions.
    pub strata: Vec<StratumEstimate>,
}

/// Enumerate through `k_exact`, then sample through `k_max` with independent streams.
/// Uses the covariance-aware ratio variance of [`StratifiedEstimate`], with
/// exactly zero variance for enumerated strata. Exact strata come from
/// `FaultCatalog::exact_fault_stratum`.
/// Batches bound temporary storage.
/// Zero samples explicitly omit the remaining strata. As in `StratifiedEstimate`,
/// zero observed failures do not establish a zero true failure probability.
///
/// # Errors
/// Rejects invalid stratum limits or catalogs unsupported by fault-count sampling.
pub fn estimate(
    catalog: &FaultCatalog,
    k_exact: usize,
    k_max: usize,
    samples_per_stratum: u32,
    seed: u64,
) -> Result<ReferenceEstimate, String> {
    if k_exact > k_max {
        return Err("k_exact must not exceed k_max".into());
    }
    let pmf = catalog.fault_count_pmf(k_max).map_err(|e| e.to_string())?;
    let mut rng = PecosRng::seed_from_u64(seed);
    let mut strata = Vec::new();
    let mut sampled = Vec::new();
    let mut exact = Vec::new();
    for (k, &mass) in pmf.masses().iter().enumerate() {
        let mut stratum = StratumEstimate {
            k,
            exact: k <= k_exact,
            configurations: 0,
            mass,
            success_probability: 0.0,
            failure_probability: 0.0,
        };
        if stratum.exact {
            let contribution = catalog.exact_fault_stratum(k).map_err(|e| e.to_string())?;
            stratum.success_probability = contribution.survival_probability();
            stratum.failure_probability = contribution.failure_probability();
            stratum.configurations = configuration_count(catalog, k)?;
            exact.push(contribution);
        } else if samples_per_stratum > 0 {
            let (mut survived, mut failed) = (0_u32, 0_u32);
            let mut remaining = samples_per_stratum;
            while remaining > 0 {
                let batch = remaining.min(4096);
                for configuration in catalog
                    .sample_fault_configurations(k, batch as usize, rng.next_u64())
                    .map_err(|e| e.to_string())?
                {
                    if configuration.affected_detectors.is_empty() {
                        survived += 1;
                        failed += u32::from(!configuration.affected_observables.is_empty());
                    }
                }
                remaining -= batch;
            }
            let n = f64::from(samples_per_stratum);
            let a = f64::from(failed) / n;
            let b = f64::from(survived) / n;
            sampled.push(FaultStratumCounts {
                k,
                attempted: samples_per_stratum as usize,
                survived: survived as usize,
                failed: failed as usize,
            });
            stratum.configurations = u64::from(samples_per_stratum);
            stratum.success_probability = mass * b;
            stratum.failure_probability = mass * a;
        }
        strata.push(stratum);
    }
    Ok(ReferenceEstimate {
        estimate: StratifiedEstimate::from_counts_and_exact(&pmf, &sampled, &exact)
            .map_err(|e| e.to_string())?,
        strata,
    })
}

fn configuration_count(catalog: &FaultCatalog, k: usize) -> Result<u64, String> {
    let mut counts = vec![0_u64; k + 1];
    counts[0] = 1;
    for location in &catalog.locations {
        let alternatives = u64::try_from(
            location
                .faults
                .iter()
                .filter(|a| a.absolute_probability > 0.0)
                .count(),
        )
        .map_err(|e| e.to_string())?;
        for j in (1..=k).rev() {
            counts[j] = alternatives
                .checked_mul(counts[j - 1])
                .and_then(|v| counts[j].checked_add(v))
                .ok_or_else(|| format!("{k}-fault configuration count overflows u64"))?;
        }
    }
    Ok(counts[k])
}
