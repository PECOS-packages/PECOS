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
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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

    /// Sum of all rates. To leading order this is the total probability of
    /// *any* Pauli error firing during the gate.
    pub fn total_rate(&self) -> f64 {
        self.rates.iter().sum()
    }

    /// Sum of rates restricted to a given Pauli weight (number of
    /// non-identity factors).
    pub fn rate_at_weight(&self, weight: usize) -> f64 {
        self.supports
            .iter()
            .zip(&self.rates)
            .filter(|(s, _)| s.weight() == weight)
            .map(|(_, r)| *r)
            .sum()
    }

    /// Largest single rate in the model.
    pub fn max_rate(&self) -> f64 {
        self.rates.iter().copied().fold(0.0, f64::max)
    }

    /// Per-Pauli residual `self - other`. Returns a vector of
    /// `(pauli, self_rate, other_rate, residual)` for every Pauli in the
    /// union of the two models' supports.
    pub fn diff(&self, other: &Self) -> Vec<(PauliString, f64, f64, f64)> {
        use std::collections::HashSet;
        let mut all: HashSet<PauliString> = HashSet::new();
        for p in self.supports.iter().chain(other.supports.iter()) {
            all.insert(p.clone());
        }
        let mut out: Vec<_> = all
            .into_iter()
            .map(|p| {
                let a = self.rate(&p);
                let b = other.rate(&p);
                (p, a, b, a - b)
            })
            .collect();
        out.sort_by(|(_, _, _, x), (_, _, _, y)| {
            y.abs().partial_cmp(&x.abs()).unwrap_or(std::cmp::Ordering::Equal)
        });
        out
    }

    /// `L2` norm of the rate-residual vector between `self` and `other`.
    pub fn residual_l2(&self, other: &Self) -> f64 {
        self.diff(other).iter().map(|(_, _, _, r)| r * r).sum::<f64>().sqrt()
    }

    /// Pauli with the largest absolute residual against `other`. Returns
    /// `None` if both models are empty.
    pub fn max_residual(&self, other: &Self) -> Option<(PauliString, f64)> {
        self.diff(other).into_iter().next().map(|(p, _, _, r)| (p, r))
    }

    /// Leading-order composition of two independent noise sources: rates
    /// add per Pauli. Exact for small rates where `(1 - (1-p_A)(1-p_B)) ≈
    /// p_A + p_B`. For larger rates, prefer [`synthesize_superop`] on the
    /// combined physical Lindbladian directly (all-orders).
    ///
    /// Use case: combine a predicted physical-model PL with an
    /// experimentally-observed residual-noise PL to get an effective
    /// model for circuit-level noise.
    pub fn compose_independent(&self, other: &Self) -> Self {
        use std::collections::HashMap;
        let mut combined: HashMap<PauliString, f64> = HashMap::new();
        for (p, r) in self.supports.iter().zip(&self.rates) {
            combined.insert(p.clone(), *r);
        }
        for (p, r) in other.supports.iter().zip(&other.rates) {
            *combined.entry(p.clone()).or_insert(0.0) += *r;
        }
        let mut entries: Vec<_> = combined.into_iter().collect();
        entries.sort_by(|(a, _), (b, _)| {
            a.0.iter().map(|p| *p as u8).cmp(b.0.iter().map(|p| *p as u8))
        });
        let (supports, rates): (Vec<_>, Vec<_>) = entries.into_iter().unzip();
        Self::new(supports, rates)
    }

    /// Aggregate absolute residual by Pauli weight. Returns `weight -> sum
    /// of |residual|`. Useful for diagnosing which weight class of physics
    /// is missing from the model (e.g. weight-2 residual large =>
    /// correlated two-qubit noise missing).
    pub fn residual_by_weight(&self, other: &Self) -> Vec<(usize, f64)> {
        use std::collections::BTreeMap;
        let mut agg: BTreeMap<usize, f64> = BTreeMap::new();
        for (p, _, _, r) in self.diff(other) {
            *agg.entry(p.weight()).or_insert(0.0) += r.abs();
        }
        agg.into_iter().collect()
    }

    /// Heuristic diagnostic: given a predicted model (`self`) and a
    /// measured model (`other`), suggest physical sources likely missing
    /// from the prediction. Returns human-readable strings ordered by
    /// residual magnitude. Thresholds are intentionally coarse; use as a
    /// starting point, not a final verdict.
    pub fn diagnose_gap(&self, other: &Self, tol: f64) -> Vec<String> {
        let mut msgs = Vec::new();
        let by_weight = self.residual_by_weight(other);

        for (weight, total_abs) in &by_weight {
            if *total_abs < tol {
                continue;
            }
            match weight {
                1 => msgs.push(format!(
                    "weight-1 residual {:.3e}: suggests missing incoherent single-qubit noise \
                     (T_1, T_phi mis-characterized, or extra dephasing/relaxation channels)",
                    total_abs
                )),
                2 => msgs.push(format!(
                    "weight-2 residual {:.3e}: suggests correlated 2-qubit noise not in model \
                     (coherent ZZ crosstalk, leakage-induced correlations, or gate mis-calibration)",
                    total_abs
                )),
                w if *w >= 3 => msgs.push(format!(
                    "weight-{} residual {:.3e}: high-weight residual; suggests multi-qubit \
                     crosstalk or higher-order Magnus corrections needed",
                    w, total_abs
                )),
                _ => {}
            }
        }
        // Highlight the single worst Pauli as a concrete pointer.
        if let Some((p, r)) = self.max_residual(other) {
            if r.abs() >= tol {
                msgs.push(format!(
                    "largest per-Pauli residual: |lambda_{{{}}}^pred - lambda_{{{}}}^meas| = {:.3e}",
                    p, p, r
                ));
            }
        }
        msgs
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
    fn summary_helpers() {
        let supports = vec![
            PauliString::from_str("IX").unwrap(),
            PauliString::from_str("IZ").unwrap(),
            PauliString::from_str("XX").unwrap(),
        ];
        let rates = vec![0.001, 0.003, 0.002];
        let model = PauliLindbladModel::new(supports, rates);
        assert!((model.total_rate() - 0.006).abs() < 1e-12);
        assert!((model.rate_at_weight(1) - 0.004).abs() < 1e-12); // IX + IZ
        assert!((model.rate_at_weight(2) - 0.002).abs() < 1e-12); // XX
        assert!((model.max_rate() - 0.003).abs() < 1e-12);
    }

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
