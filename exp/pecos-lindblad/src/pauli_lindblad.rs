// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under
// the License.

//! Sparse Pauli-Lindblad noise model (arXiv:2201.09866 generator form).

use rand::{Rng, RngExt};

use crate::basis::{Pauli1, PauliString};

/// Sparse Pauli-Lindblad generator:
/// `N(rho) = exp( sum_k lambda_k * (P_k rho P_k^dag - rho) )`.
/// `rates[i]` is the integrated rate `lambda_k` (dimensionless) for
/// `supports[i]`. All rates are non-negative for forward simulation.
#[derive(Clone, Debug, Default)]
pub struct PauliLindbladModel {
    pub supports: Vec<PauliString>,
    pub rates: Vec<f64>,
}

impl PauliLindbladModel {
    pub fn new(supports: Vec<PauliString>, rates: Vec<f64>) -> Self {
        assert_eq!(supports.len(), rates.len(), "supports/rates length mismatch");
        for &r in &rates {
            assert!(r >= 0.0, "negative PL rate: {}", r);
        }
        Self { supports, rates }
    }

    /// Look up the rate for a given Pauli support. Returns 0 if not present.
    pub fn rate(&self, p: &PauliString) -> f64 {
        self.supports.iter().zip(&self.rates).find(|(s, _)| *s == p).map(|(_, r)| *r).unwrap_or(0.0)
    }

    /// Sample an error realization over integrated duration `t_scale`:
    /// each Pauli term independently fires with probability
    /// `p_k = (1 - exp(-2 * lambda_k * t_scale)) / 2`. Returns the
    /// product Pauli string (may be identity).
    pub fn sample(&self, t_scale: f64, rng: &mut impl Rng) -> PauliString {
        assert!(!self.supports.is_empty(), "cannot sample empty model");
        let n = self.supports[0].num_qubits();
        let mut acc = PauliString(vec![Pauli1::I; n]);
        for (support, &lambda) in self.supports.iter().zip(&self.rates) {
            assert_eq!(support.num_qubits(), n, "ragged supports");
            let p_flip = 0.5 * (1.0 - (-2.0 * lambda * t_scale).exp());
            if rng.random_range(0.0..1.0) < p_flip {
                acc = acc.multiply(support);
            }
        }
        acc
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sample_zero_rates_is_identity() {
        use rand::rngs::StdRng;
        use rand::SeedableRng;
        let supports = vec![
            PauliString::single(Pauli1::X),
            PauliString::single(Pauli1::Y),
            PauliString::single(Pauli1::Z),
        ];
        let model = PauliLindbladModel::new(supports, vec![0.0; 3]);
        let mut rng = StdRng::seed_from_u64(42);
        for _ in 0..100 {
            let s = model.sample(1.0, &mut rng);
            assert_eq!(s, PauliString::single(Pauli1::I));
        }
    }
}
