// Copyright 2026 The PECOS Developers
// Licensed under the Apache License, Version 2.0.

use pecos_random::PecosRng;

/// One joint distribution on consecutive `(X, Z)` bits for each support qubit.
#[derive(Clone, Debug)]
pub struct NoiseChannel {
    /// Physical support, in symbol allocation order.
    pub qubits: Vec<usize>,
    /// First noise symbol; qubit `j` uses `first_symbol + 2*j` and the next bit.
    pub first_symbol: usize,
    /// Alternatives `(probability, component bits)`, retaining their correlations.
    pub alternatives: Vec<(f64, Vec<bool>)>,
}

impl NoiseChannel {
    pub(super) fn sample(&self, rng: &mut PecosRng, bits: &mut [bool]) {
        // Scale by the sum to accommodate the floating-point sum of a normalized
        // input distribution without discarding any small positive alternatives.
        let total: f64 = self.alternatives.iter().map(|(p, _)| p).sum();
        let draw = rng.next_f64();
        let mut cumulative = 0.0;
        let index = self
            .alternatives
            .iter()
            .position(|(p, _)| {
                cumulative += p;
                draw < cumulative / total
            })
            .expect("program validation guarantees a nonempty distribution with a finite positive total");
        let values = &self.alternatives[index].1;
        bits[self.first_symbol..self.first_symbol + values.len()].copy_from_slice(values);
    }
}
