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

use pecos_neo::extensible::DecompositionNoiseStrategy;
use pecos_neo::noise::CorrelatedNoiseChannel;
use pecos_neo::noise::composite::prelude::*;
use pecos_neo::noise::composite::{BatchCompositeChannel, CompositeNoiseModelBuilder, Prob};
use pecos_neo::noise::validation::ValidationError;

fn rejects(error: Option<ValidationError>, parameter: &str) {
    let error = error.expect("invalid input must be rejected");
    assert!(error.to_string().contains(parameter), "{error}");
}

#[test]
fn composite_probability_rejects_invalid_input() {
    for p in [f64::NAN, -0.1, 1.1] {
        rejects(CompositeNoiseModelBuilder::new().with_p1(p).err(), "p1");
    }
}

#[test]
fn leakage_scale_rejects_invalid_input() {
    for p in [f64::NAN, -0.1, 1.1] {
        rejects(
            CompositeNoiseModelBuilder::new()
                .with_leakage_scale(p)
                .err(),
            "leakage_scale",
        );
    }
}

#[test]
fn primitive_probability_rejects_invalid_input() {
    for p in [f64::NAN, -0.1, 1.1] {
        rejects(Prob::new(p, nothing()).err(), "probability");
    }
}

#[test]
fn correlated_probability_rejects_invalid_input() {
    for p in [f64::NAN, -0.1, 1.1] {
        rejects(CorrelatedNoiseChannel::new(0.1, p).err(), "correlation");
        rejects(
            CorrelatedNoiseChannel::new(p, 0.1).err(),
            "base_error_probability",
        );
    }
}

#[test]
fn blended_weight_rejects_invalid_input() {
    for p in [f64::NAN, -0.1, 1.1] {
        rejects(
            DecompositionNoiseStrategy::blended(p).err(),
            "high_level_weight",
        );
    }
}

#[test]
fn emission_probability_rejects_invalid_input() {
    for p in [f64::NAN, -0.1, 1.1] {
        rejects(sample_emission_with_prob(p).err(), "prob");
        rejects(
            pecos_neo::noise::composite::IndependentEmissionWithPartnerDepolarize::new(p).err(),
            "emission_prob",
        );
        rejects(amplitude_damping_biased(p).err(), "x_probability");
    }
}

#[test]
fn batch_probability_rejects_invalid_input() {
    for p in [f64::NAN, -0.1, 1.1] {
        rejects(
            BatchCompositeChannel::new("test", p, nothing()).err(),
            "probability",
        );
    }
}
