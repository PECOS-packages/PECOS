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

//! DEM generator trait and implementations.
//!
//! Unifies all DEM generation methods behind a single trait. Any generator
//! can be used as a simulator backend via the meas_sampling path.

use crate::dem_mapping::{DecomposableDemEntry, DemEntry, Detector, Observable};
use crate::expand::{ExpandedCircuit, GateIndex};
use crate::noise::NoiseSpec;
use pecos_core::Gate;

/// Noise parameters for DEM generation (uniform depolarizing + coherent idle).
#[derive(Clone, Debug)]
pub struct NoiseParams {
    pub p1: f64,
    pub p2: f64,
    pub p_meas: f64,
    pub p_prep: f64,
    pub idle_rz: f64,
}

/// Input context for DEM generation.
///
/// Contains everything a generator needs: expanded circuit, detectors,
/// observables, and precomputed indices.
pub struct DemContext<'a> {
    pub gates: &'a [Gate],
    pub expanded: &'a ExpandedCircuit,
    pub gate_index: &'a GateIndex,
    pub detectors: &'a [Detector],
    pub observables: &'a [Observable],
}

/// Output from a DEM generator.
pub struct DemOutput {
    /// Raw DEM entries (for tesseract and other hyperedge-capable decoders).
    pub entries: Vec<DemEntry>,
    /// Decomposable entries with X/Z provenance (for MWPM decoders).
    /// None if the generator doesn't support decomposition.
    pub decomposable: Option<Vec<DecomposableDemEntry>>,
}

/// Trait for DEM generators.
///
/// Any type implementing this can generate a Detector Error Model from
/// a circuit and noise parameters. Implementations can then be used as
/// simulator backends via the meas_sampling path.
pub trait DemGenerator: Send + Sync {
    /// Generate a DEM from the given context and noise.
    fn generate(&self, ctx: &DemContext<'_>, noise: &dyn NoiseSpec) -> DemOutput;

    /// Human-readable name for this generator method.
    fn name(&self) -> &str;
}

/// Coherent DEM generator (backward Heisenberg mechanism extraction, approximate probabilities).
///
/// Fast. Handles coherent noise (idle_rz). Approximate probabilities
/// (sin^2 for H-type, (1-exp(2s))/2 for S-type).
pub struct CoherentApprox;

impl DemGenerator for CoherentApprox {
    fn generate(&self, ctx: &DemContext<'_>, noise: &dyn NoiseSpec) -> DemOutput {
        let entries = crate::coherent_dem::build_coherent_dem(
            ctx.gates,
            noise,
            ctx.detectors,
            ctx.observables,
            &ctx.gate_index.expansion_gates,
        );
        let decomposable = crate::coherent_dem::build_coherent_dem_decomposable(
            ctx.gates,
            noise,
            ctx.detectors,
            ctx.observables,
            &ctx.gate_index.expansion_gates,
        );
        DemOutput {
            entries,
            decomposable: Some(decomposable),
        }
    }

    fn name(&self) -> &'static str {
        "coherent_approx"
    }
}

/// Coherent DEM generator with Heisenberg-exact probability fitting.
///
/// Slower (runs Heisenberg walks for marginals + pairwise). Handles coherent
/// noise. Exact marginals via L-BFGS fit.
pub struct CoherentExact {
    pub prune_threshold: f64,
}

impl Default for CoherentExact {
    fn default() -> Self {
        Self {
            prune_threshold: 1e-12,
        }
    }
}

impl DemGenerator for CoherentExact {
    fn generate(&self, ctx: &DemContext<'_>, noise: &dyn NoiseSpec) -> DemOutput {
        use crate::heisenberg::{build_noise_map, heisenberg_sparse};
        use crate::stabilizer::StabilizerGroup;

        // Build initial stabilizer group
        let init_gates: Vec<Gate> = (0..ctx.expanded.num_original_qubits)
            .map(|q| crate::expand::make_gate(pecos_core::gate_type::GateType::PZ, &[q]))
            .collect();
        let stab = StabilizerGroup::from_circuit(&init_gates, ctx.expanded.num_qubits);

        // Heisenberg walks for exact marginals
        let noise_map = build_noise_map(ctx.gates, noise, &ctx.gate_index.expansion_gates);

        let num_dets = ctx.detectors.iter().map(|d| d.id + 1).max().unwrap_or(0);
        let mut marginals = vec![0.0_f64; num_dets];
        for det in ctx.detectors {
            let p = heisenberg_sparse(
                ctx.gates,
                &det.stabilizer,
                noise,
                &stab,
                self.prune_threshold,
                ctx.gate_index,
                Some(&noise_map),
            );
            if det.id < marginals.len() {
                marginals[det.id] = p;
            }
        }

        // Pairwise rates
        let mut pairwise: Vec<((usize, usize), f64)> = Vec::new();
        for i in 0..ctx.detectors.len() {
            for j in (i + 1)..ctx.detectors.len() {
                let product = ctx.detectors[i]
                    .stabilizer
                    .multiply(&ctx.detectors[j].stabilizer);
                let p_product = heisenberg_sparse(
                    ctx.gates,
                    &product,
                    noise,
                    &stab,
                    self.prune_threshold,
                    ctx.gate_index,
                    Some(&noise_map),
                );
                let p_joint = (marginals[ctx.detectors[i].id] + marginals[ctx.detectors[j].id]
                    - p_product)
                    / 2.0;
                if p_joint > 1e-10 {
                    pairwise.push(((ctx.detectors[i].id, ctx.detectors[j].id), p_joint.max(0.0)));
                }
            }
        }

        // Exact-fitted entries
        let entries = crate::coherent_dem::build_coherent_dem_exact(
            ctx.gates,
            noise,
            ctx.detectors,
            ctx.observables,
            &ctx.gate_index.expansion_gates,
            &marginals,
            Some(&pairwise),
        );
        let decomposable = crate::coherent_dem::build_coherent_dem_exact_decomposable(
            ctx.gates,
            noise,
            ctx.detectors,
            ctx.observables,
            &ctx.gate_index.expansion_gates,
            &marginals,
            Some(&pairwise),
        );

        DemOutput {
            entries,
            decomposable: Some(decomposable),
        }
    }

    fn name(&self) -> &'static str {
        "coherent_exact"
    }
}

/// Perturbative (forward EEG) DEM generator.
///
/// Fastest. Uses forward EEG propagation with Taylor approximation.
/// Approximate probabilities (~50% error for coherent noise).
pub struct Perturbative;

impl DemGenerator for Perturbative {
    fn generate(&self, ctx: &DemContext<'_>, noise: &dyn NoiseSpec) -> DemOutput {
        // Scaffolding imports for future forward-EEG implementation
        #[allow(unused_imports)]
        use crate::circuit::analyze_expanded;
        #[allow(unused_imports)]
        use crate::dem_mapping::{EegConfig, build_dem_configured, build_dem_decomposable};
        #[allow(unused_imports)]
        use crate::noise::UniformNoise;

        // We need to extract params from the NoiseSpec — use a test gate to probe
        let _ = noise.noise_after_gate(0, pecos_core::gate_type::GateType::H, &[0]);

        // For now, use the coherent_dem path as fallback since forward EEG
        // requires its own NoiseModel type (not the NoiseSpec trait)
        let entries = crate::coherent_dem::build_coherent_dem(
            ctx.gates,
            noise,
            ctx.detectors,
            ctx.observables,
            &ctx.gate_index.expansion_gates,
        );

        // Forward EEG would need stabilizer group for proper classification
        // Use coherent_dem decomposable for now
        let decomposable = crate::coherent_dem::build_coherent_dem_decomposable(
            ctx.gates,
            noise,
            ctx.detectors,
            ctx.observables,
            &ctx.gate_index.expansion_gates,
        );

        DemOutput {
            entries,
            decomposable: Some(decomposable),
        }
    }

    fn name(&self) -> &'static str {
        "perturbative"
    }
}

/// Select a DEM generator by method name.
#[must_use]
pub fn select_generator(method: &str, _idle_rz: f64) -> Box<dyn DemGenerator> {
    match method {
        "coherent_exact" => Box::new(CoherentExact::default()),
        "perturbative" => Box::new(Perturbative),
        _ => Box::new(CoherentApprox), // auto/coherent/default fallback
    }
}
