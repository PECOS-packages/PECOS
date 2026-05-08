// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0

//! EEG types: generator kinds and accumulators.

use crate::Bm;
use std::collections::BTreeMap;

/// The four types of Elementary Error Generators.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum EegType {
    /// Hamiltonian: H_P[rho] = -i[P, rho]. Coherent rotations.
    H,
    /// Stochastic: S_P[rho] = P rho P - rho. Pauli errors.
    S,
    /// Correlation: C_{P,Q}. Two-Pauli correlations.
    C,
    /// Active: A_{P,Q}. Phase-dependent interference.
    A,
}

/// A single Elementary Error Generator with coefficient.
#[derive(Clone, Debug)]
pub struct Eeg {
    pub eeg_type: EegType,
    pub label_p: Bm,
    pub label_q: Bm,
    pub coeff: f64,
}

/// Accumulator for Hamiltonian EEGs (H_P type).
///
/// Stores the sum of coefficients for each Pauli label. This is the
/// first-order BCH result: G_c = Sigma epsilon_P H_P.
#[derive(Clone, Debug, Default)]
pub struct HamiltonianAccumulator {
    generators: BTreeMap<Bm, f64>,
}

impl HamiltonianAccumulator {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add H_P with coefficient epsilon.
    /// Accumulates (first-order BCH = sum).
    pub fn add(&mut self, label: Bm, coeff: f64) {
        *self.generators.entry(label).or_insert(0.0) += coeff;
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.generators.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.generators.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = (&Bm, &f64)> {
        self.generators.iter()
    }

    pub fn prune(&mut self, threshold: f64) {
        self.generators.retain(|_, c| c.abs() > threshold);
    }
}

/// Accumulator for Stochastic EEGs (S_P type).
///
/// S-type generators commute, so first-order BCH is exact.
#[derive(Clone, Debug, Default)]
pub struct StochasticAccumulator {
    generators: BTreeMap<Bm, f64>,
}

impl StochasticAccumulator {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add(&mut self, label: Bm, coeff: f64) {
        *self.generators.entry(label).or_insert(0.0) += coeff;
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.generators.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.generators.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = (&Bm, &f64)> {
        self.generators.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hamiltonian_accumulator() {
        let mut acc = HamiltonianAccumulator::new();
        acc.add(Bm::z(0), 0.05);
        acc.add(Bm::z(0), 0.03);
        acc.add(Bm::x(1), 0.02);

        assert_eq!(acc.len(), 2);
        let z0 = acc.generators.get(&Bm::z(0)).unwrap();
        assert!((z0 - 0.08).abs() < 1e-10);
    }
}
