// Copyright 2026 The PECOS Developers
// Licensed under the Apache License, Version 2.0.

use pecos_core::{ChannelExpr, PauliBitmaskSmall};
use pecos_quantum::channel::pauli_string_to_bitmask;
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

pub(super) fn alternatives(channel: &ChannelExpr) -> Result<Vec<(f64, PauliBitmaskSmall)>, String> {
    match channel {
        ChannelExpr::Unitary(unitary) => {
            let width = channel.qubits().into_iter().max().map_or(0, |q| q + 1);
            let pauli = unitary
                .clone()
                .try_to_pauli_string()
                .ok_or_else(|| "unitary is not an exact symbolic Pauli".to_string())?;
            let bits = pauli_string_to_bitmask(width, &pauli).map_err(|e| e.to_string())?;
            Ok(vec![(1.0, bits)])
        }
        ChannelExpr::MixedUnitary(ops) => {
            let mut out = Vec::with_capacity(ops.len());
            let mut total = 0.0;
            for (weight, unitary) in ops {
                if !weight.is_finite() || *weight < 0.0 {
                    return Err("channel probabilities must be finite and nonnegative".into());
                }
                // Convert a deterministic operator, not the weighted mixture:
                // PauliChannel's default cleanup would erase probabilities <=1e-12.
                let converted = alternatives(&ChannelExpr::Unitary(unitary.clone()))?;
                out.push((*weight, converted[0].1.clone()));
                total += weight;
            }
            if out.is_empty() || (total - 1.0).abs() > 1e-12 {
                return Err("channel probabilities must sum to one".into());
            }
            Ok(out)
        }
        ChannelExpr::Tensor(parts) | ChannelExpr::Compose(parts) => {
            let mut out = vec![(1.0, PauliBitmaskSmall::identity())];
            for part in parts {
                let next = alternatives(part)?;
                out = out
                    .iter()
                    .flat_map(|(a, p)| next.iter().map(move |(b, q)| (a * b, p.multiply(q))))
                    .collect();
            }
            Ok(out)
        }
        _ => Err("channel is not a mixture of Pauli unitaries".into()),
    }
}
