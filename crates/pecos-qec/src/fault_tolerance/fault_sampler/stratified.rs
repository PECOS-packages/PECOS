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

//! Exact fault-count masses, conditional sampling, and stratified estimates.
//!
//! Fixed-fault-count conditional sampling follows the idea in Clifft
//! (arXiv:2604.27058); no Clifft code is used here. Active locations are grouped
//! by the bits of their firing probability. One suffix table stores
//! `ln E_g(r) = logsumexp_j(ln C(N_g,j) + j ln o_g + ln E_{g+1}(r-j))`,
//! where `o_g = p_g/(1-p_g)`. Both masses and conditional draws use this table.
//!
//! All stored probabilities must be finite and non-negative. A location is
//! active precisely when some alternative has positive absolute probability.
//! Active locations require p < 1, alternatives summing to p, and a no-fault
//! probability of 1-p (relative tolerance 1e-12 for the two equalities).
//! Inactive locations must have a stored no-fault probability exactly one;
//! hence their factors in the enumeration's no-fault product are neutral.

use super::{FaultCatalog, FaultConfiguration, build_fault_configuration};
use pecos_random::{PecosRng, RngExt};
use rand::distr::Open01;
use std::collections::{BTreeMap, BTreeSet};

/// Invalid input to fault-count sampling or estimation.
#[derive(Clone, Debug, PartialEq, thiserror::Error)]
pub enum FaultCountError {
    /// A catalog probability or normalization is inconsistent.
    #[error("inconsistent catalog at location {location}: {reason}")]
    InconsistentCatalog {
        /// Index of the inconsistent location.
        location: usize,
        /// Probability field or normalization that failed validation.
        reason: &'static str,
    },
    /// Log odds require a firing probability strictly below one.
    #[error("active location {location} has channel_probability {probability} >= 1")]
    ActiveProbabilityAtLeastOne {
        /// Index of the active location.
        location: usize,
        /// Unsupported firing probability.
        probability: f64,
    },
    /// The requested stratum exceeds the active location count.
    #[error("fault count {requested} exceeds the {active} active locations")]
    FaultCountTooLarge {
        /// Requested fault count.
        requested: usize,
        /// Number of active locations.
        active: usize,
    },
    /// The requested output batch cannot be allocated.
    #[error("cannot allocate storage for {requested} fault configurations")]
    ConfigurationAllocationFailed {
        /// Number of configurations requested.
        requested: usize,
    },
    /// Counts must satisfy `0 <= failed <= survived <= attempted`, with attempts positive.
    #[error(
        "invalid counts for stratum {k}: require failed <= survived <= attempted and attempted > 0"
    )]
    InvalidCounts {
        /// Stratum with invalid counts.
        k: usize,
    },
    /// Counts refer to a stratum absent from the supplied PMF.
    #[error("stratum {k} is outside the supplied PMF")]
    StratumOutsidePmf {
        /// Missing stratum.
        k: usize,
    },
    /// A stratum was supplied more than once.
    #[error("duplicate counts for stratum {k}")]
    DuplicateStratum {
        /// Repeated stratum.
        k: usize,
    },
}

/// `P(K=k)` and its logarithm for `k=0..=max_k`, plus a Chernoff tail bound.
///
/// The tail bound is exact mathematics evaluated in floating point: when `K`
/// is nearly deterministic (every `p` close to one, `max_k + 1` just above the
/// mean) rounding can place it slightly below the true tail (relative gap near 1e-11
/// observed in that regime).
/// Masses may underflow to zero; log masses remain finite. Values are computed
/// from validated catalog probabilities, to floating-point accuracy.
#[derive(Clone, Debug)]
pub struct FaultCountPmf {
    masses: Vec<f64>,
    log_masses: Vec<f64>,
    tail_bound: f64,
}

impl FaultCountPmf {
    /// Masses indexed by fault count, including zero.
    #[must_use]
    pub fn masses(&self) -> &[f64] {
        &self.masses
    }

    /// Natural logarithms of the masses, including underflowed masses.
    #[must_use]
    pub fn log_masses(&self) -> &[f64] {
        &self.log_masses
    }

    /// Upper bound on `P(K > max_k)`, computed without subtracting masses from one.
    #[must_use]
    pub fn tail_bound(&self) -> f64 {
        self.tail_bound
    }
}

/// Attempted, survived, and survived-and-failed shots in one stratum.
#[derive(Clone, Copy, Debug)]
pub struct FaultStratumCounts {
    /// Number of firing locations in each shot.
    pub k: usize,
    /// Attempted shots `N_k` (must be positive).
    pub attempted: usize,
    /// Shots surviving postselection `S_k`.
    pub survived: usize,
    /// Surviving shots that failed `F_k`.
    pub failed: usize,
}

/// Probability estimates over sampled strata and plug-in standard errors.
///
/// With `a_k=F_k/N_k`, `b_k=S_k/N_k`, this reports `A=sum P_k a_k`,
/// `B=sum P_k b_k`, and `R=A/B`. Variances of A and B sum independent
/// `P_k^2 a_k(1-a_k)/N_k` terms (replace a with b for B). The delta-method
/// ratio variance is `sum P_k^2 v_k/N_k / B^2`, where
/// `v_k=[F_k(1-R)^2+(S_k-F_k)R^2]/N_k-(a_k-R b_k)^2`.
///
/// V1 limitation: a stratum whose observed failure count is zero by chance
/// contributes nothing to the estimated variance of A, so SE(A) can be
/// understated when strata are undersampled. Its contribution to `v_k` for
/// Var(R) is `R^2 b_k(1-b_k)`, not zero. If no sampled stratum has a failure,
/// R is zero and every standard error of A and R is exactly zero; that reflects
/// the sample, not certainty.
#[derive(Clone, Debug)]
pub struct StratifiedEstimate {
    /// Estimated survived-and-failed probability A.
    pub failure_probability: f64,
    /// Estimated survival probability B.
    pub survival_probability: f64,
    /// Estimated conditional failure probability R; None when B is zero.
    pub failure_given_survival: Option<f64>,
    /// Plug-in standard error of A.
    pub failure_standard_error: f64,
    /// Plug-in standard error of B.
    pub survival_standard_error: f64,
    /// Delta-method standard error of R; None when B is zero.
    pub ratio_standard_error: Option<f64>,
    /// Sum of PMF masses for unsampled strata within the supplied PMF.
    pub unsampled_mass: f64,
    /// Bound on mass beyond the PMF, separate from all standard errors.
    pub tail_bound: f64,
}

impl StratifiedEstimate {
    /// Combine counts with a PMF, without sampling or renormalizing omitted strata.
    ///
    /// # Errors
    /// Returns an error for invalid counts, duplicate strata, or strata outside
    /// the supplied PMF. An empty counts slice is valid and omits every stratum.
    pub fn from_counts(
        pmf: &FaultCountPmf,
        counts: &[FaultStratumCounts],
    ) -> Result<Self, FaultCountError> {
        let mut seen = BTreeSet::new();
        let (mut failure, mut survival, mut scale) = (0.0, 0.0, 0.0_f64);
        for count in counts {
            if count.attempted == 0
                || count.failed > count.survived
                || count.survived > count.attempted
            {
                return Err(FaultCountError::InvalidCounts { k: count.k });
            }
            let Some(&p) = pmf.masses.get(count.k) else {
                return Err(FaultCountError::StratumOutsidePmf { k: count.k });
            };
            if !seen.insert(count.k) {
                return Err(FaultCountError::DuplicateStratum { k: count.k });
            }
            let n = count_f64(count.attempted);
            let a = count_f64(count.failed) / n;
            let b = count_f64(count.survived) / n;
            failure += p * a;
            survival += p * b;
            scale = scale.max(p);
        }
        // Variances are accumulated with the largest sampled mass factored out,
        // so squaring masses below ~1e-154 does not underflow the standard errors.
        let weight = |k: usize| {
            if scale > 0.0 {
                pmf.masses[k] / scale
            } else {
                0.0
            }
        };
        let (mut var_a, mut var_b) = (0.0, 0.0);
        for count in counts {
            let n = count_f64(count.attempted);
            let a = count_f64(count.failed) / n;
            let b = count_f64(count.survived) / n;
            let w = weight(count.k);
            var_a += w * w * a * (1.0 - a) / n;
            var_b += w * w * b * (1.0 - b) / n;
        }
        let ratio = (survival > 0.0).then(|| failure / survival);
        let ratio_se = ratio.map(|r| {
            let variance: f64 = counts
                .iter()
                .map(|count| {
                    let n = count_f64(count.attempted);
                    let a = count_f64(count.failed) / n;
                    let b = count_f64(count.survived) / n;
                    // Algebraically the stated v_k, expressed as centered squares
                    // to avoid a negative variance from cancellation.
                    let mean = a - r * b;
                    let v = a * (1.0 - r - mean).powi(2)
                        + (b - a) * (-r - mean).powi(2)
                        + (1.0 - b) * mean.powi(2);
                    weight(count.k).powi(2) * v / n
                })
                .sum();
            scale * variance.sqrt() / survival
        });
        Ok(Self {
            failure_probability: failure,
            survival_probability: survival,
            failure_given_survival: ratio,
            failure_standard_error: scale * var_a.sqrt(),
            survival_standard_error: scale * var_b.sqrt(),
            ratio_standard_error: ratio_se,
            unsampled_mass: pmf
                .masses
                .iter()
                .enumerate()
                .filter(|(k, _)| !seen.contains(k))
                .map(|(_, p)| p)
                .sum(),
            tail_bound: pmf.tail_bound,
        })
    }
}

struct Group {
    locations: Vec<usize>,
    log_weights: Vec<f64>,
}

struct FaultCountTable {
    groups: Vec<Group>,
    suffix: Vec<Vec<f64>>,
    log_no_fault: f64,
    mean: f64,
}

#[allow(clippy::cast_precision_loss)] // counts as f64 for statistics; exact below 2^53
fn count_f64(count: usize) -> f64 {
    count as f64
}

fn relative_eq(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-12 * a.abs().max(b.abs())
}

fn valid_probability(p: f64) -> bool {
    p.is_finite() && p >= 0.0
}

fn log_add(a: f64, b: f64) -> f64 {
    if a == f64::NEG_INFINITY {
        return b;
    }
    if b == f64::NEG_INFINITY {
        return a;
    }
    let hi = a.max(b);
    hi + (a.min(b) - hi).exp().ln_1p()
}

impl FaultCountTable {
    fn new(catalog: &FaultCatalog, max_k: usize) -> Result<Self, FaultCountError> {
        let mut grouped: BTreeMap<u64, Vec<usize>> = BTreeMap::new();
        let mut log_no_fault = 0.0;
        let mut mean = 0.0;
        let mut active = 0;
        for (i, loc) in catalog.locations.iter().enumerate() {
            let inconsistent = |reason| FaultCountError::InconsistentCatalog {
                location: i,
                reason,
            };
            if !valid_probability(loc.channel_probability)
                || !valid_probability(loc.no_fault_probability)
                || loc.faults.iter().any(|alt| {
                    !valid_probability(alt.absolute_probability)
                        || !valid_probability(alt.conditional_probability)
                })
            {
                return Err(inconsistent(
                    "every probability must be finite and non-negative",
                ));
            }
            if loc.faults.iter().any(|alt| alt.absolute_probability > 0.0) {
                let p = loc.channel_probability;
                if p >= 1.0 {
                    return Err(FaultCountError::ActiveProbabilityAtLeastOne {
                        location: i,
                        probability: p,
                    });
                }
                let total: f64 = loc.faults.iter().map(|alt| alt.absolute_probability).sum();
                if !total.is_finite() || !relative_eq(total, p) {
                    return Err(inconsistent(
                        "alternative absolute probabilities must sum to channel_probability",
                    ));
                }
                if !relative_eq(loc.no_fault_probability, 1.0 - p) {
                    return Err(inconsistent(
                        "no_fault_probability must equal 1 - channel_probability",
                    ));
                }
                grouped.entry(p.to_bits()).or_default().push(i);
                active += 1;
            } else if loc.no_fault_probability.to_bits() != 1.0_f64.to_bits() {
                return Err(inconsistent(
                    "inactive location must have no_fault_probability == 1",
                ));
            }
        }
        if max_k > active {
            return Err(FaultCountError::FaultCountTooLarge {
                requested: max_k,
                active,
            });
        }
        let groups: Vec<Group> = grouped
            .into_iter()
            .map(|(bits, locations)| {
                let p = f64::from_bits(bits);
                let log_odds = p.ln() - (-p).ln_1p();
                let n = locations.len();
                log_no_fault += count_f64(n) * (-p).ln_1p();
                mean += count_f64(n) * p;
                let mut log_weights = vec![0.0];
                for j in 1..=n.min(max_k) {
                    let numerator = count_f64(n - j + 1);
                    let denominator = count_f64(j);
                    log_weights
                        .push(log_weights[j - 1] + numerator.ln() - denominator.ln() + log_odds);
                }
                Group {
                    locations,
                    log_weights,
                }
            })
            .collect();
        let mut suffix = vec![vec![f64::NEG_INFINITY; max_k + 1]; groups.len() + 1];
        suffix[groups.len()][0] = 0.0;
        for g in (0..groups.len()).rev() {
            for r in 0..=max_k {
                for j in 0..=groups[g].locations.len().min(r) {
                    suffix[g][r] = log_add(
                        suffix[g][r],
                        groups[g].log_weights[j] + suffix[g + 1][r - j],
                    );
                }
            }
        }
        Ok(Self {
            groups,
            suffix,
            log_no_fault,
            mean,
        })
    }

    fn pmf(&self) -> FaultCountPmf {
        let log_masses: Vec<f64> = self.suffix[0]
            .iter()
            .map(|e| self.log_no_fault + e)
            .collect();
        let a = count_f64(log_masses.len());
        let tail_bound = if self.mean == 0.0 {
            0.0
        } else if a <= self.mean {
            1.0
        } else {
            // Round upward so a positive bound never becomes zero on underflow;
            // a probability bound above one is replaced by the trivial bound.
            (-self.mean + a * (1.0 + self.mean.ln() - a.ln()))
                .exp()
                .next_up()
                .min(1.0)
        };
        FaultCountPmf {
            masses: log_masses.iter().map(|p| p.exp()).collect(),
            log_masses,
            tail_bound,
        }
    }

    fn sample(&self, catalog: &FaultCatalog, k: usize, rng: &mut PecosRng) -> FaultConfiguration {
        let mut remaining = k;
        let mut locations = Vec::with_capacity(k);
        for (g, group) in self.groups.iter().enumerate() {
            let j = draw_log_weights(
                (0..=group.locations.len().min(remaining))
                    .map(|j| (j, group.log_weights[j] + self.suffix[g + 1][remaining - j])),
                rng,
            );
            locations.extend(
                rand::seq::index::sample(rng, group.locations.len(), j)
                    .iter()
                    .map(|i| group.locations[i]),
            );
            remaining -= j;
        }
        locations.sort_unstable();
        let alternatives = locations
            .iter()
            .map(|&i| {
                draw_log_weights(
                    catalog.locations[i]
                        .faults
                        .iter()
                        .enumerate()
                        .filter(|(_, alt)| alt.absolute_probability > 0.0)
                        .map(|(a, alt)| (a, alt.absolute_probability.ln())),
                    rng,
                )
            })
            .collect();
        build_fault_configuration(catalog, locations, alternatives)
    }
}

// Exponential races draw a categorical variable directly from log weights.
// Open01 excludes both endpoints, so logarithms are finite. Every internal
// caller supplies a nonempty range with at least one finite log weight.
fn draw_log_weights(weights: impl Iterator<Item = (usize, f64)>, rng: &mut PecosRng) -> usize {
    weights
        .fold((0, f64::NEG_INFINITY), |best, (index, log_weight)| {
            let uniform: f64 = rng.sample(Open01);
            let score = log_weight - (-uniform.ln()).ln();
            if score > best.1 { (index, score) } else { best }
        })
        .0
}

impl FaultCatalog {
    /// Compute `P(K=k)` for `0..=max_k` and a Chernoff bound on `P(K>max_k)`.
    ///
    /// For `a=max_k+1 > mu=sum p_i`, the bound is
    /// `exp(-mu) (e mu/a)^a`; otherwise it is one (zero when mu is zero).
    /// Active probabilities are grouped by exact f64 bits. The suffix table
    /// costs `O(G max_k^2)` time and `O(G max_k)` space in the worst case.
    ///
    /// # Errors
    /// Rejects non-finite/negative or inconsistent catalog probabilities,
    /// active probabilities at least one, and `max_k > n_active`.
    pub fn fault_count_pmf(&self, max_k: usize) -> Result<FaultCountPmf, FaultCountError> {
        Ok(FaultCountTable::new(self, max_k)?.pmf())
    }

    /// Draw configurations from the exact conditional law given `K=k`.
    ///
    /// One log-domain suffix table and one `PecosRng::seed_from_u64(seed)`
    /// stream serve the entire batch. Group counts have weights
    /// `C(N_g,j) o_g^j E_{g+1}(r-j)`; subsets within groups are uniform and
    /// alternatives have weights `absolute_probability / channel_probability`.
    /// Selected location indices are ascending. Effects and stored probabilities
    /// are constructed by the same helper as exhaustive enumeration.
    ///
    /// # Errors
    /// Rejects non-finite/negative or inconsistent catalog probabilities,
    /// active probabilities at least one, and `k > n_active`, even for zero shots.
    /// Also returns an error if storage for the requested batch cannot be allocated.
    pub fn sample_fault_configurations(
        &self,
        k: usize,
        num_shots: usize,
        seed: u64,
    ) -> Result<Vec<FaultConfiguration>, FaultCountError> {
        let table = FaultCountTable::new(self, k)?;
        let mut rng = PecosRng::seed_from_u64(seed);
        let mut configurations = Vec::new();
        if configurations.try_reserve_exact(num_shots).is_err() {
            return Err(FaultCountError::ConfigurationAllocationFailed {
                requested: num_shots,
            });
        }
        for _ in 0..num_shots {
            configurations.push(table.sample(self, k, &mut rng));
        }
        Ok(configurations)
    }
}

#[cfg(test)]
mod tests;
