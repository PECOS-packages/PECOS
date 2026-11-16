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

use pecos_neo::noise::composite::batch::{GeometricSampler, geometric};

#[test]
fn sampler_invalid_probabilities_return_errors() {
    for probability in [-0.1, 1.1, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(GeometricSampler::new(probability).is_err());
    }
}

#[test]
fn geometric_invalid_probabilities_return_errors() {
    for probability in [-0.1, 1.1, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(geometric(probability).is_err());
    }
}

#[test]
fn checked_sampler_invalid_probabilities_return_errors() {
    for probability in [-0.1, 1.1, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(GeometricSampler::new_checked(probability).is_err());
    }
}
