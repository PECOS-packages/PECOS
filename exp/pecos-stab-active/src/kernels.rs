// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file
// except in compliance with the License. You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the
// License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
// express or implied. See the License for the specific language governing permissions and
// limitations under the License.

//! Dense amplitude kernels shared by shot executors.

use num_complex::Complex64;
use pecos_core::Angle64;
use pecos_stab_tn::stab_mps::coordinate_tableau::{CoordinateGate, MeasurementCase};
use pecos_stab_tn::stab_mps::measure::EXPECTATION_ENDPOINT_TOLERANCE;

pub(crate) fn rotate(
    amplitudes: &mut [Complex64],
    theta: Angle64,
    flip: usize,
    sign: usize,
    phase: Complex64,
) {
    // Convert the fixed-point magnitude before restoring the sign: subtracting
    // two floating-point turns would erase tiny negative rotations near zero.
    let radians = if theta > Angle64::HALF_TURN {
        -(-theta).to_radians()
    } else {
        theta.to_radians()
    };
    let (sine, cosine) = (radians / 2.0).sin_cos();
    let coefficient = Complex64::new(0.0, -sine) * phase;
    if flip == 0 {
        for (index, amplitude) in amplitudes.iter_mut().enumerate() {
            *amplitude *= cosine + coefficient * parity(index & sign);
        }
    } else {
        for index in 0..amplitudes.len() {
            let partner = index ^ flip;
            if index < partner {
                let a = amplitudes[index];
                let b = amplitudes[partner];
                amplitudes[index] = cosine * a + coefficient * parity(partner & sign) * b;
                amplitudes[partner] = cosine * b + coefficient * parity(index & sign) * a;
            }
        }
    }
    normalize(amplitudes);
}

pub(crate) fn normalize(amplitudes: &mut [Complex64]) {
    let norm = amplitudes
        .iter()
        .map(Complex64::norm_sqr)
        .sum::<f64>()
        .sqrt();
    assert!(norm > 0.0, "cannot normalize a zero state");
    for amplitude in amplitudes {
        *amplitude /= norm;
    }
}

pub(crate) fn active_expectation(
    amplitudes: &[Complex64],
    flip: usize,
    sign: usize,
    phase: Complex64,
) -> Complex64 {
    amplitudes
        .iter()
        .enumerate()
        .map(|(index, a)| amplitudes[index ^ flip].conj() * phase * parity(index & sign) * a)
        .sum()
}

/// Return the outcome-one probability used by measurement and the dense oracle.
///
/// Active expectations within [`EXPECTATION_ENDPOINT_TOLERANCE`] of either
/// endpoint are treated as that exact eigenvalue before computing probability
/// or drawing from the RNG. This matches the shared measurement policy and
/// prevents a forced projector from amplifying a cancellation residue.
pub(crate) fn measurement_probability(
    amplitudes: &[Complex64],
    case: MeasurementCase,
    flip: usize,
    sign: usize,
    phase: Complex64,
) -> f64 {
    match case {
        MeasurementCase::Random => 0.5,
        MeasurementCase::Deterministic => f64::from(phase.re < 0.0),
        MeasurementCase::Active => {
            let expectation = active_expectation(amplitudes, flip, sign, phase).re;
            let expectation = if 1.0 - expectation.abs() <= EXPECTATION_ENDPOINT_TOLERANCE {
                expectation.signum()
            } else {
                expectation
            };
            ((1.0 - expectation) / 2.0).clamp(0.0, 1.0)
        }
    }
}

pub(crate) fn coordinate_gate(amplitudes: &mut [Complex64], gate: CoordinateGate) {
    match gate {
        CoordinateGate::Sdg(bit) => {
            for (index, amplitude) in amplitudes.iter_mut().enumerate() {
                if index & (1 << bit) != 0 {
                    *amplitude *= Complex64::new(0.0, -1.0);
                }
            }
        }
        CoordinateGate::H(bit) => {
            for index in 0..amplitudes.len() {
                if index & (1 << bit) == 0 {
                    let partner = index ^ (1 << bit);
                    let a = amplitudes[index];
                    let b = amplitudes[partner];
                    amplitudes[index] = (a + b) * std::f64::consts::FRAC_1_SQRT_2;
                    amplitudes[partner] = (a - b) * std::f64::consts::FRAC_1_SQRT_2;
                }
            }
        }
        CoordinateGate::Cx(control, target) => {
            for index in 0..amplitudes.len() {
                if index & (1 << control) != 0 && index & (1 << target) == 0 {
                    amplitudes.swap(index, index ^ (1 << target));
                }
            }
        }
    }
}

pub(crate) fn project(amplitudes: &mut Vec<Complex64>, pivot_bit: usize, value: bool) {
    let bit_mask = 1 << pivot_bit;
    let mut destination = 0;
    for source in 0..amplitudes.len() {
        if (source & bit_mask != 0) == value {
            // Ascending sources keep every write at or below the unread entries.
            amplitudes[destination] = amplitudes[source];
            destination += 1;
        }
    }
    amplitudes.truncate(amplitudes.len() / 2);
}

pub(crate) fn double(amplitudes: &mut Vec<Complex64>) {
    amplitudes.resize(amplitudes.len() * 2, Complex64::new(0.0, 0.0));
}

pub(crate) fn mask(bits: &[usize]) -> usize {
    bits.iter().fold(0, |value, &bit| value | (1 << bit))
}
fn parity(value: usize) -> f64 {
    if value.count_ones().is_multiple_of(2) {
        1.0
    } else {
        -1.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pecos_random::PecosRng;

    #[test]
    fn projection_matches_filter_bit_for_bit() {
        let mut rng = PecosRng::seed_from_u64(0x5a);
        for width in 1..=10 {
            for _ in 0..16 {
                // Arbitrary bit patterns preserve even NaN payloads without arithmetic.
                let input: Vec<_> = (0..1 << width)
                    .map(|_| {
                        Complex64::new(
                            f64::from_bits(rng.next_u64()),
                            f64::from_bits(rng.next_u64()),
                        )
                    })
                    .collect();
                for pivot_bit in 0..width {
                    for value in [false, true] {
                        let expected: Vec<_> = input
                            .iter()
                            .enumerate()
                            .filter(|(index, _)| (index & (1 << pivot_bit) != 0) == value)
                            .map(|(_, &a)| a)
                            .collect();
                        let mut actual = input.clone();
                        let pointer = actual.as_ptr();
                        let capacity = actual.capacity();
                        project(&mut actual, pivot_bit, value);
                        assert_eq!(actual.as_ptr(), pointer);
                        assert_eq!(actual.capacity(), capacity);
                        assert_eq!(actual.len(), expected.len());
                        for (index, (a, b)) in actual.iter().zip(&expected).enumerate() {
                            assert_eq!(
                                (a.re.to_bits(), a.im.to_bits()),
                                (b.re.to_bits(), b.im.to_bits()),
                                "width={width}, pivot={pivot_bit}, value={value}, index={index}"
                            );
                        }
                    }
                }
            }
        }
    }
}
