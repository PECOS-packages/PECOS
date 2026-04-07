// Copyright 2024 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
// in compliance with the License.You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under
// the License.

//! Name resolution for PHIR quantum operations.
//!
//! Translates PHIR gate names to simulator-recognized names, matching
//! the Python `sim_name_resolver` in `pecos/reps/pyphir/name_resolver.py`.

use std::f64::consts::{FRAC_PI_2, PI, TAU};

const ATOL: f64 = 1e-12;

fn isclose(a: f64, b: f64) -> bool {
    (a - b).abs() <= ATOL
}

/// Resolve the simulator name for a PHIR quantum operation.
///
/// Takes the gate name and optional angles (in radians) and returns
/// the name that simulators recognize.
pub fn resolve_sim_name(name: &str, angles: Option<&[f64]>) -> String {
    match name {
        "RZZ" => resolve_rzz(angles),
        "RZ" => resolve_rz(angles),
        "R1XY" => resolve_r1xy(angles),
        _ => name.to_string(),
    }
}

fn resolve_rzz(angles: Option<&[f64]>) -> String {
    if let Some(angs) = angles {
        if angs.len() == 1 {
            let theta = angs[0];
            if isclose(theta, 0.0) {
                return "I".to_string();
            }
            let theta_mod = theta.rem_euclid(TAU);
            if isclose(theta_mod, FRAC_PI_2) {
                return "SZZ".to_string();
            }
            if isclose(theta_mod, PI * 1.5) {
                return "SZZdg".to_string();
            }
        }
    }
    "RZZ".to_string()
}

fn resolve_rz(angles: Option<&[f64]>) -> String {
    if let Some(angs) = angles {
        if angs.len() == 1 {
            let theta = angs[0];
            // Check lookup table first
            if let Some(name) = rz_angle_to_clifford(theta) {
                return name.to_string();
            }
        }
    }
    "RZ".to_string()
}

fn resolve_r1xy(angles: Option<&[f64]>) -> String {
    if let Some(angs) = angles {
        if angs.len() == 2 {
            let theta = angs[0];
            let phi = angs[1];
            // Check lookup table first
            if let Some(name) = r1xy_angles_to_clifford(theta, phi) {
                return name.to_string();
            }
        }
    }
    "R1XY".to_string()
}

/// Look up RZ angle in the Clifford conversion table.
fn rz_angle_to_clifford(theta: f64) -> Option<&'static str> {
    // Check if theta mod tau is close to 0
    let theta_mod = theta.rem_euclid(TAU);
    if isclose(theta_mod, 0.0) || isclose(theta_mod, TAU) {
        return Some("I");
    }

    // Table of known RZ Clifford angles
    let table: &[(f64, &str)] = &[
        (PI, "Z"),
        (FRAC_PI_2, "SZ"),
        (-FRAC_PI_2, "SZdg"),
        (-PI, "Z"),
        (PI * 1.5, "SZdg"),    // 4.712...
        (-PI * 1.5, "SZ"),     // -4.712...
        (TAU, "I"),
        (0.0, "I"),
    ];

    for &(angle, name) in table {
        if isclose(angle, theta) {
            return Some(name);
        }
    }

    None
}

/// Look up R1XY angles in the Clifford conversion table.
fn r1xy_angles_to_clifford(theta: f64, phi: f64) -> Option<&'static str> {
    // Check if theta mod tau is close to 0
    let theta_mod = theta.rem_euclid(TAU);
    if isclose(theta_mod, 0.0) || isclose(theta_mod, TAU) {
        return Some("I");
    }

    // Table from Python: (theta, phi) -> name
    // Includes both positive and negative angle equivalences
    let table: &[(f64, f64, &str)] = &[
        (PI, PI, "X"),
        (PI, FRAC_PI_2, "Y"),
        (PI, 0.0, "X"),
        (PI, -FRAC_PI_2, "Y"),
        (PI, -PI, "X"),
        (FRAC_PI_2, PI, "SXdg"),
        (FRAC_PI_2, FRAC_PI_2, "SY"),
        (FRAC_PI_2, 0.0, "SX"),
        (FRAC_PI_2, -FRAC_PI_2, "SYdg"),
        (FRAC_PI_2, -PI, "SXdg"),
        (-FRAC_PI_2, PI, "SX"),
        (-FRAC_PI_2, FRAC_PI_2, "SYdg"),
        (-FRAC_PI_2, 0.0, "SXdg"),
        (-FRAC_PI_2, -FRAC_PI_2, "SY"),
        (-PI, PI, "X"),
        (-PI, FRAC_PI_2, "Y"),
        (-PI, 0.0, "X"),
        (-PI, -FRAC_PI_2, "Y"),
        (-PI, -PI, "X"),
        // Equivalences for 3pi/2 = -pi/2 (mod 2pi)
        (PI * 1.5, PI, "SX"),
        (PI * 1.5, FRAC_PI_2, "SYdg"),
        (PI * 1.5, 0.0, "SXdg"),
        (PI * 1.5, -FRAC_PI_2, "SY"),
        (PI * 1.5, -PI, "SX"),
        // Equivalences for -3pi/2 = pi/2 (mod 2pi)
        (-PI * 1.5, PI, "SXdg"),
        (-PI * 1.5, FRAC_PI_2, "SY"),
        (-PI * 1.5, 0.0, "SX"),
        (-PI * 1.5, -FRAC_PI_2, "SYdg"),
        (-PI * 1.5, -PI, "SXdg"),
    ];

    for &(t, p, name) in table {
        if isclose(t, theta) && isclose(p, phi) {
            return Some(name);
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rzz_zero_is_identity() {
        assert_eq!(resolve_sim_name("RZZ", Some(&[0.0])), "I");
    }

    #[test]
    fn test_rzz_pi_over_2_is_szz() {
        assert_eq!(resolve_sim_name("RZZ", Some(&[FRAC_PI_2])), "SZZ");
    }

    #[test]
    fn test_rzz_3pi_over_2_is_szzdg() {
        assert_eq!(resolve_sim_name("RZZ", Some(&[PI * 1.5])), "SZZdg");
    }

    #[test]
    fn test_rz_pi_is_z() {
        assert_eq!(resolve_sim_name("RZ", Some(&[PI])), "Z");
    }

    #[test]
    fn test_rz_pi_over_2_is_sz() {
        assert_eq!(resolve_sim_name("RZ", Some(&[FRAC_PI_2])), "SZ");
    }

    #[test]
    fn test_rz_zero_is_identity() {
        assert_eq!(resolve_sim_name("RZ", Some(&[0.0])), "I");
    }

    #[test]
    fn test_r1xy_pi_0_is_x() {
        assert_eq!(resolve_sim_name("R1XY", Some(&[PI, 0.0])), "X");
    }

    #[test]
    fn test_r1xy_pi_pi2_is_y() {
        assert_eq!(resolve_sim_name("R1XY", Some(&[PI, FRAC_PI_2])), "Y");
    }

    #[test]
    fn test_r1xy_pi2_0_is_sx() {
        assert_eq!(resolve_sim_name("R1XY", Some(&[FRAC_PI_2, 0.0])), "SX");
    }

    #[test]
    fn test_passthrough() {
        assert_eq!(resolve_sim_name("H", None), "H");
        assert_eq!(resolve_sim_name("CX", None), "CX");
        assert_eq!(resolve_sim_name("Measure", None), "Measure");
    }
}
