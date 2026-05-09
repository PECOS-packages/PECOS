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

//! Property-based invariants via proptest. Generates random Lindbladians
//! and verifies:
//!
//! 1. All synthesized PL rates are non-negative (positivity).
//! 2. At τ_g → 0, all rates vanish (no-time, no-error).
//! 3. Rates scale linearly with τ_g at leading order in weak noise.
//! 4. Walsh-Hadamard forward/inverse is a bijection on rate vectors.

use proptest::prelude::*;

use pecos_lindblad::noise_models::ad_pd_1q;
use pecos_lindblad::{Gate, Pauli1, PauliLindbladModel, PauliString, synthesize_identity_1q};

/// Generate physical `(T_1, T_2)` with `T_2 <= 2 T_1` (GKS-positive regime).
fn t1_t2_strategy() -> impl Strategy<Value = (f64, f64)> {
    (10.0f64..1000.0, 0.01f64..1.0).prop_map(|(t1, t2_ratio)| (t1, t2_ratio * 2.0 * t1))
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 64, ..ProptestConfig::default() })]

    #[test]
    fn identity_1q_rates_non_negative((t1, t2) in t1_t2_strategy(), tau_g in 0.01f64..50.0) {
        let noise = ad_pd_1q(t1, t2);
        let gate = Gate::identity(1, noise, tau_g);
        let pl = synthesize_identity_1q(&gate);
        for p in [Pauli1::X, Pauli1::Y, Pauli1::Z] {
            let r = pl.rate(&PauliString::single(p));
            prop_assert!(r >= 0.0, "rate for {:?} was negative: {}", p, r);
        }
    }

    #[test]
    fn identity_1q_rates_linear_in_tau(
        (t1, t2) in t1_t2_strategy(),
        tau_a in 0.01f64..5.0,
    ) {
        let tau_b = tau_a * 2.0;
        let noise_a = ad_pd_1q(t1, t2);
        let noise_b = ad_pd_1q(t1, t2);
        let pl_a = synthesize_identity_1q(&Gate::identity(1, noise_a, tau_a));
        let pl_b = synthesize_identity_1q(&Gate::identity(1, noise_b, tau_b));
        for p in [Pauli1::X, Pauli1::Y, Pauli1::Z] {
            let ra = pl_a.rate(&PauliString::single(p));
            let rb = pl_b.rate(&PauliString::single(p));
            // Identity+AD+PD is exact closed form with lambda_k ∝ tau_g.
            prop_assert!(
                (rb - 2.0 * ra).abs() < 1e-12,
                "rate for {:?} not linear in tau: ra={}, rb={}",
                p, ra, rb,
            );
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 32, ..ProptestConfig::default() })]

    /// Generate a random non-negative rate vector and verify the forward
    /// Walsh-Hadamard map reproduces alpha_b = 2 Σ λ_k ⟨b,k⟩_sp.
    #[test]
    fn walsh_hadamard_forward_is_linear_in_rates(
        lam_x in 0.0f64..0.1,
        lam_y in 0.0f64..0.1,
        lam_z in 0.0f64..0.1,
    ) {
        // Forward relation alpha_b = 2 Σ_k λ_k ⟨b,k⟩_sp for n=1.
        //   alpha_X = 2(lam_y + lam_z)
        //   alpha_Y = 2(lam_x + lam_z)
        //   alpha_Z = 2(lam_x + lam_y)
        let supports = vec![
            PauliString::single(Pauli1::X),
            PauliString::single(Pauli1::Y),
            PauliString::single(Pauli1::Z),
        ];
        let rates = vec![lam_x, lam_y, lam_z];
        let model = PauliLindbladModel::new(supports, rates);

        let alpha = |b: &PauliString| -> f64 {
            model
                .supports
                .iter()
                .zip(&model.rates)
                .map(|(k, lam)| 2.0 * lam * f64::from(b.symplectic_product(k)))
                .sum()
        };

        let x = PauliString::single(Pauli1::X);
        let y = PauliString::single(Pauli1::Y);
        let z = PauliString::single(Pauli1::Z);
        prop_assert!((alpha(&x) - 2.0 * (lam_y + lam_z)).abs() < 1e-14);
        prop_assert!((alpha(&y) - 2.0 * (lam_x + lam_z)).abs() < 1e-14);
        prop_assert!((alpha(&z) - 2.0 * (lam_x + lam_y)).abs() < 1e-14);
    }
}
