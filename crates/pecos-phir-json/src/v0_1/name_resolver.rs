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
//!
//! Uses `Angle64` for angle comparisons so that equivalent angles
//! (e.g. `3pi/2` and `-pi/2`) are handled automatically via wrapping.

use pecos_core::Angle64;

/// Resolve the simulator name for a PHIR quantum operation.
///
/// Takes the gate name and optional angles (in radians) and returns
/// the name that simulators recognize.
#[must_use]
pub fn resolve_sim_name(name: &str, angles: Option<&[f64]>) -> String {
    match name {
        "RZZ" => resolve_rzz(angles),
        "RZ" => resolve_rz(angles),
        "R1XY" => resolve_r1xy(angles),
        _ => name.to_string(),
    }
}

fn resolve_rzz(angles: Option<&[f64]>) -> String {
    if let Some(angs) = angles
        && angs.len() == 1
    {
        let a = Angle64::from_radians(angs[0]);
        if a == Angle64::ZERO {
            return "I".to_string();
        }
        if a == Angle64::QUARTER_TURN {
            return "SZZ".to_string();
        }
        if a == Angle64::THREE_QUARTERS_TURN {
            return "SZZdg".to_string();
        }
    }
    "RZZ".to_string()
}

fn resolve_rz(angles: Option<&[f64]>) -> String {
    if let Some(angs) = angles
        && angs.len() == 1
        && let Some(name) = rz_angle_to_clifford(Angle64::from_radians(angs[0]))
    {
        return name.to_string();
    }
    "RZ".to_string()
}

fn resolve_r1xy(angles: Option<&[f64]>) -> String {
    if let Some(angs) = angles
        && angs.len() == 2
    {
        let theta = Angle64::from_radians(angs[0]);
        let phi = Angle64::from_radians(angs[1]);
        if let Some(name) = r1xy_angles_to_clifford(theta, phi) {
            return name.to_string();
        }
    }
    "R1XY".to_string()
}

/// Look up RZ angle in the Clifford conversion table.
///
/// With Angle64, equivalent angles like pi and -pi are the same value,
/// so the table only needs canonical entries.
fn rz_angle_to_clifford(angle: Angle64) -> Option<&'static str> {
    if angle == Angle64::ZERO {
        return Some("I");
    }
    // pi/2 -> SZ, pi -> Z, 3pi/4 -> SZdg
    if angle == Angle64::HALF_TURN {
        return Some("Z");
    }
    if angle == Angle64::QUARTER_TURN {
        return Some("SZ");
    }
    if angle == Angle64::THREE_QUARTERS_TURN {
        return Some("SZdg");
    }
    None
}

/// Look up R1XY angles in the Clifford conversion table.
///
/// With Angle64, -pi/2 and 3pi/2 are the same value, so no
/// duplicate entries needed.
fn r1xy_angles_to_clifford(theta: Angle64, phi: Angle64) -> Option<&'static str> {
    if theta == Angle64::ZERO {
        return Some("I");
    }

    let half = Angle64::HALF_TURN;
    let quarter = Angle64::QUARTER_TURN;
    let three_quarter = Angle64::THREE_QUARTERS_TURN;

    // Table: (theta, phi) -> name
    // Only canonical Angle64 values needed -- equivalences are automatic
    let table: &[(Angle64, Angle64, &str)] = &[
        (half, half, "X"),
        (half, quarter, "Y"),
        (half, Angle64::ZERO, "X"),
        (half, three_quarter, "Y"),
        (quarter, half, "SXdg"),
        (quarter, quarter, "SY"),
        (quarter, Angle64::ZERO, "SX"),
        (quarter, three_quarter, "SYdg"),
        (three_quarter, half, "SX"),
        (three_quarter, quarter, "SYdg"),
        (three_quarter, Angle64::ZERO, "SXdg"),
        (three_quarter, three_quarter, "SY"),
    ];

    for &(t, p, name) in table {
        if t == theta && p == phi {
            return Some(name);
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::{FRAC_PI_2, PI};

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
    fn test_rz_neg_pi_is_z() {
        assert_eq!(resolve_sim_name("RZ", Some(&[-PI])), "Z");
    }

    #[test]
    fn test_rz_pi_over_2_is_sz() {
        assert_eq!(resolve_sim_name("RZ", Some(&[FRAC_PI_2])), "SZ");
    }

    #[test]
    fn test_rz_neg_pi_over_2_is_szdg() {
        assert_eq!(resolve_sim_name("RZ", Some(&[-FRAC_PI_2])), "SZdg");
    }

    #[test]
    fn test_rz_3pi_over_2_is_szdg() {
        // 3pi/2 == -pi/2 mod 2pi, both should resolve to SZdg
        assert_eq!(resolve_sim_name("RZ", Some(&[PI * 1.5])), "SZdg");
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
    fn test_r1xy_3pi2_pi2_is_sydg() {
        // 3pi/2 == -pi/2 mod 2pi
        assert_eq!(
            resolve_sim_name("R1XY", Some(&[PI * 1.5, FRAC_PI_2])),
            "SYdg"
        );
    }

    #[test]
    fn test_r1xy_neg_pi2_pi2_is_sydg() {
        assert_eq!(
            resolve_sim_name("R1XY", Some(&[-FRAC_PI_2, FRAC_PI_2])),
            "SYdg"
        );
    }

    #[test]
    fn test_passthrough() {
        assert_eq!(resolve_sim_name("H", None), "H");
        assert_eq!(resolve_sim_name("CX", None), "CX");
        assert_eq!(resolve_sim_name("Measure", None), "Measure");
    }
}
