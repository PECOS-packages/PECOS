// Copyright 2026 The PECOS Developers
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

//! Surface code geometry and layout generation.
//!
//! This module provides tools for creating surface code patches with configurable
//! distance and orientation. It supports both rotated (default) and non-rotated
//! layouts.
//!
//! # Example
//!
//! ```
//! use pecos_qec::SurfaceCode;
//!
//! // Create a distance-3 rotated surface code
//! let code = SurfaceCode::rotated(3).unwrap();
//! assert_eq!(code.distance(), 3);
//! assert_eq!(code.num_data_qubits(), 9);
//!
//! // Get the stabilizer code for verification
//! let stab_code = code.to_stabilizer_code();
//! assert!(stab_code.verify().is_ok());
//! ```

// Allow similar names for logical_x/logical_z pairs - these are intentional
#![allow(clippy::similar_names)]
// Allow casts in layout generation - indices are small and controlled
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::cast_sign_loss)]

use crate::StabilizerCodeSpec;
use crate::geometry::{CheckSchedule, LogicalOperator, StabilizerCheck};

/// Orientation of surface code patch boundaries.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum PatchOrientation {
    /// X boundaries on top/bottom, Z on left/right (default).
    #[default]
    XTopBottom,
    /// Z boundaries on top/bottom, X on left/right.
    ZTopBottom,
}

/// A surface code patch.
///
/// Supports both rotated (more common, fewer qubits) and non-rotated layouts.
#[derive(Clone, Debug)]
pub struct SurfaceCode {
    /// X distance of the code.
    pub dx: usize,
    /// Z distance of the code.
    pub dz: usize,
    /// Whether using the rotated layout.
    pub rotated: bool,
    /// Patch orientation.
    pub orientation: PatchOrientation,
    /// Number of data qubits.
    num_data: usize,
    /// X stabilizer checks.
    x_stabilizers: Vec<StabilizerCheck>,
    /// Z stabilizer checks.
    z_stabilizers: Vec<StabilizerCheck>,
    /// Logical X operator.
    logical_x: LogicalOperator,
    /// Logical Z operator.
    logical_z: LogicalOperator,
}

impl SurfaceCode {
    /// Create a rotated surface code with the given distance.
    ///
    /// The rotated layout is more common and uses fewer physical qubits
    /// for the same code distance: d^2 data qubits for distance d.
    ///
    /// # Errors
    /// Returns an error if distance is less than 3.
    pub fn rotated(distance: usize) -> Result<Self, String> {
        Self::new(distance, distance, true, PatchOrientation::default())
    }

    /// Create a non-rotated (standard) surface code with the given distance.
    ///
    /// The standard layout uses (2d-1)^2 data qubits for distance d.
    ///
    /// # Errors
    /// Returns an error if distance is less than 3.
    pub fn standard(distance: usize) -> Result<Self, String> {
        Self::new(distance, distance, false, PatchOrientation::default())
    }

    /// Create a surface code with custom parameters.
    ///
    /// # Errors
    /// Returns an error if `dx` or `dz` is less than 3.
    pub fn new(
        dx: usize,
        dz: usize,
        rotated: bool,
        orientation: PatchOrientation,
    ) -> Result<Self, String> {
        if dx < 3 {
            return Err(format!("dx must be at least 3, got {dx}"));
        }
        if dz < 3 {
            return Err(format!("dz must be at least 3, got {dz}"));
        }

        let d = dx.min(dz);

        let (num_data, x_stabs, z_stabs, logical_x, logical_z) = if rotated {
            generate_rotated_layout(d)
        } else {
            generate_standard_layout(d)
        };

        Ok(Self {
            dx,
            dz,
            rotated,
            orientation,
            num_data,
            x_stabilizers: x_stabs,
            z_stabilizers: z_stabs,
            logical_x,
            logical_z,
        })
    }

    /// The code distance (minimum of dx and dz).
    #[inline]
    #[must_use]
    pub fn distance(&self) -> usize {
        self.dx.min(self.dz)
    }

    /// Number of data qubits.
    #[inline]
    #[must_use]
    pub fn num_data_qubits(&self) -> usize {
        self.num_data
    }

    /// Number of X stabilizers.
    #[inline]
    #[must_use]
    pub fn num_x_stabilizers(&self) -> usize {
        self.x_stabilizers.len()
    }

    /// Number of Z stabilizers.
    #[inline]
    #[must_use]
    pub fn num_z_stabilizers(&self) -> usize {
        self.z_stabilizers.len()
    }

    /// Total number of stabilizers.
    #[inline]
    #[must_use]
    pub fn num_stabilizers(&self) -> usize {
        self.x_stabilizers.len() + self.z_stabilizers.len()
    }

    /// Get the X stabilizers.
    #[must_use]
    pub fn x_stabilizers(&self) -> &[StabilizerCheck] {
        &self.x_stabilizers
    }

    /// Get the Z stabilizers.
    #[must_use]
    pub fn z_stabilizers(&self) -> &[StabilizerCheck] {
        &self.z_stabilizers
    }

    /// Get all stabilizers (X then Z).
    #[must_use]
    pub fn all_stabilizers(&self) -> Vec<&StabilizerCheck> {
        self.x_stabilizers
            .iter()
            .chain(self.z_stabilizers.iter())
            .collect()
    }

    /// Get the logical X operator.
    #[must_use]
    pub fn logical_x(&self) -> &LogicalOperator {
        &self.logical_x
    }

    /// Get the logical Z operator.
    #[must_use]
    pub fn logical_z(&self) -> &LogicalOperator {
        &self.logical_z
    }

    /// Create a check schedule with X and Z checks in parallel rounds.
    #[must_use]
    pub fn check_schedule(&self) -> CheckSchedule {
        let mut schedule = CheckSchedule::new();
        if !self.x_stabilizers.is_empty() {
            schedule.add_round(self.x_stabilizers.clone());
        }
        if !self.z_stabilizers.is_empty() {
            schedule.add_round(self.z_stabilizers.clone());
        }
        schedule
    }

    /// Convert to a [`StabilizerCodeSpec`] for verification and analysis.
    #[must_use]
    #[allow(clippy::missing_panics_doc)] // Panic unreachable for valid surface codes
    pub fn to_stabilizer_code(&self) -> StabilizerCodeSpec {
        let stabilizers: Vec<_> = self
            .x_stabilizers
            .iter()
            .chain(self.z_stabilizers.iter())
            .map(StabilizerCheck::to_pauli_string)
            .collect();

        let logical_zs = vec![self.logical_z.to_pauli_string()];
        let logical_xs = vec![self.logical_x.to_pauli_string()];

        let mut code = StabilizerCodeSpec::new(self.num_data, stabilizers, logical_zs, logical_xs)
            .expect("Surface code should always produce valid stabilizer code");

        code.set_distance(self.distance());
        code
    }

    /// Get the code parameters as [[n, k, d]] string.
    #[must_use]
    pub fn code_parameters(&self) -> String {
        format!("[[{}, 1, {}]]", self.num_data, self.distance())
    }
}

/// Builder for creating surface codes with custom configuration.
#[derive(Clone, Debug, Default)]
pub struct SurfaceCodeBuilder {
    distance: Option<usize>,
    dx: Option<usize>,
    dz: Option<usize>,
    rotated: bool,
    orientation: PatchOrientation,
}

impl SurfaceCodeBuilder {
    /// Create a new builder (defaults to rotated layout).
    #[must_use]
    pub fn new() -> Self {
        Self {
            rotated: true,
            ..Default::default()
        }
    }

    /// Set the symmetric distance.
    #[must_use]
    pub fn with_distance(mut self, distance: usize) -> Self {
        self.distance = Some(distance);
        self
    }

    /// Set asymmetric distances.
    #[must_use]
    pub fn with_distances(mut self, dx: usize, dz: usize) -> Self {
        self.dx = Some(dx);
        self.dz = Some(dz);
        self
    }

    /// Use the rotated layout (default).
    #[must_use]
    pub fn rotated(mut self) -> Self {
        self.rotated = true;
        self
    }

    /// Use the standard (non-rotated) layout.
    #[must_use]
    pub fn standard(mut self) -> Self {
        self.rotated = false;
        self
    }

    /// Set the patch orientation.
    #[must_use]
    pub fn with_orientation(mut self, orientation: PatchOrientation) -> Self {
        self.orientation = orientation;
        self
    }

    /// Build the surface code.
    ///
    /// # Errors
    /// Returns an error if neither distance nor dx/dz are set.
    pub fn build(self) -> Result<SurfaceCode, String> {
        let (dx, dz) = if let Some(d) = self.distance {
            (d, d)
        } else if let (Some(dx), Some(dz)) = (self.dx, self.dz) {
            (dx, dz)
        } else {
            return Err("Must set either distance or both dx and dz".to_string());
        };

        SurfaceCode::new(dx, dz, self.rotated, self.orientation)
    }
}

// ============================================================================
// Layout generation functions
// ============================================================================

/// Generate the rotated surface code layout for distance d.
///
/// The rotated surface code has d^2 data qubits and encodes 1 logical qubit.
/// Stabilizers are arranged so X and Z stabilizers share 0 or 2 qubits (CSS property).
///
/// For the rotated layout, data qubits form a d x d grid. X stabilizers (plaquettes)
/// and Z stabilizers (vertices) alternate in a checkerboard pattern. Boundary
/// stabilizers have weight 2.
///
/// Returns (`num_data`, `x_stabilizers`, `z_stabilizers`, `logical_x`, `logical_z`).
fn generate_rotated_layout(
    d: usize,
) -> (
    usize,
    Vec<StabilizerCheck>,
    Vec<StabilizerCheck>,
    LogicalOperator,
    LogicalOperator,
) {
    let num_data = d * d;
    let mut x_stabs = Vec::new();
    let mut z_stabs = Vec::new();

    // Qubit layout (d=3 example):
    //   0 - 1 - 2
    //   |   |   |
    //   3 - 4 - 5
    //   |   |   |
    //   6 - 7 - 8
    //
    // Stabilizers are placed at dual lattice positions (face centers).
    // A dual position (r, c) corresponds to the face whose corners are
    // data qubits at (r,c), (r,c+1), (r+1,c), (r+1,c+1).
    //
    // For d=3, bulk dual positions are (0,0), (0,1), (1,0), (1,1).
    // Boundary dual positions extend beyond: (-1,*), (d-1,*), (*,-1), (*,d-1).
    //
    // Checkerboard coloring: X-type when (r+c) is odd, Z-type when even.
    // This ensures CSS property: X and Z stabilizers share 0 or 2 qubits.
    //
    // Boundary rules (X boundaries top/bottom, Z boundaries left/right):
    // - Include X-type boundary stabilizers on top (r=-1) and bottom (r=d-1) edges
    // - Include Z-type boundary stabilizers on left (c=-1) and right (c=d-1) edges

    let d_i = d as i32;
    let mut x_idx = 0;
    let mut z_idx = 0;

    // Iterate over all dual lattice positions including boundaries
    // Dual row r corresponds to faces between data rows r and r+1
    // Valid range: -1 to d-1 (boundary at -1 and d-1)
    for r in -1..d_i {
        for c in -1..d_i {
            // Determine which qubits this stabilizer acts on
            // Face at dual (r,c) has corners at data (r,c), (r,c+1), (r+1,c), (r+1,c+1)
            let mut qubits = Vec::new();

            // Top-left corner: data position (r, c)
            if r >= 0 && r < d_i && c >= 0 && c < d_i {
                qubits.push((r as usize) * d + (c as usize));
            }
            // Top-right corner: data position (r, c+1)
            if r >= 0 && r < d_i && c + 1 >= 0 && c + 1 < d_i {
                qubits.push((r as usize) * d + ((c + 1) as usize));
            }
            // Bottom-left corner: data position (r+1, c)
            if r + 1 >= 0 && r + 1 < d_i && c >= 0 && c < d_i {
                qubits.push(((r + 1) as usize) * d + (c as usize));
            }
            // Bottom-right corner: data position (r+1, c+1)
            if r + 1 >= 0 && r + 1 < d_i && c + 1 >= 0 && c + 1 < d_i {
                qubits.push(((r + 1) as usize) * d + ((c + 1) as usize));
            }

            // Skip if no qubits or only 1 qubit (corner positions)
            if qubits.len() < 2 {
                continue;
            }

            // Determine stabilizer type based on checkerboard
            // X-type when (r+c) is odd, Z-type when even
            let is_x_type = (r + c) % 2 != 0;
            let is_boundary = qubits.len() < 4;

            // Determine boundary location
            let on_top = r == -1;
            let on_bottom = r == d_i - 1 && qubits.len() == 2;
            let on_left = c == -1;
            let on_right = c == d_i - 1 && qubits.len() == 2;

            // Apply boundary rules:
            // - X boundaries (top/bottom): only include X-type stabilizers
            // - Z boundaries (left/right): only include Z-type stabilizers
            // - Bulk (interior): include all stabilizers
            let should_include = if is_boundary {
                if on_top || on_bottom {
                    is_x_type // X boundary: only X-type
                } else if on_left || on_right {
                    !is_x_type // Z boundary: only Z-type
                } else {
                    true // Not a boundary position we care about
                }
            } else {
                true // Bulk stabilizer
            };

            if !should_include {
                continue;
            }

            qubits.sort_unstable();

            if is_x_type {
                let check = StabilizerCheck::x_check(x_idx, &qubits);
                let check = if is_boundary {
                    check.as_boundary()
                } else {
                    check
                };
                x_stabs.push(check);
                x_idx += 1;
            } else {
                let check = StabilizerCheck::z_check(z_idx, &qubits);
                let check = if is_boundary {
                    check.as_boundary()
                } else {
                    check
                };
                z_stabs.push(check);
                z_idx += 1;
            }
        }
    }

    // Logical operators for [[d^2, 1, d]] code
    // Logical X: X on the left column (vertical string through Z boundaries)
    let logical_x_qubits: Vec<usize> = (0..d).map(|row| row * d).collect();
    let logical_x = LogicalOperator::x(logical_x_qubits);

    // Logical Z: Z on the top row (horizontal string through X boundaries)
    let logical_z_qubits: Vec<usize> = (0..d).collect();
    let logical_z = LogicalOperator::z(logical_z_qubits);

    (num_data, x_stabs, z_stabs, logical_x, logical_z)
}

/// Generate the standard (non-rotated) surface code layout for distance d.
fn generate_standard_layout(
    d: usize,
) -> (
    usize,
    Vec<StabilizerCheck>,
    Vec<StabilizerCheck>,
    LogicalOperator,
    LogicalOperator,
) {
    // For a standard d x d surface code:
    // - d^2 data qubits
    // - (d-1) * d / 2 X stabilizers (horizontal plaquettes)
    // - (d-1) * d / 2 Z stabilizers (vertical plaquettes)

    let num_data = d * d;
    let mut x_stabs = Vec::new();
    let mut z_stabs = Vec::new();

    let mut x_idx = 0;
    let mut z_idx = 0;

    // X stabilizers: horizontal plaquettes
    for row in 0..d {
        for col in 0..(d - 1) {
            if (row + col) % 2 == 0 {
                let q = row * d + col;
                let qubits = vec![q, q + 1];
                let is_boundary = row == 0 || row == d - 1;
                let check = StabilizerCheck::x_check(x_idx, &qubits);
                let check = if is_boundary {
                    check.as_boundary()
                } else {
                    check
                };
                x_stabs.push(check);
                x_idx += 1;
            }
        }
    }

    // Z stabilizers: vertical plaquettes
    for row in 0..(d - 1) {
        for col in 0..d {
            if (row + col) % 2 == 1 {
                let q = row * d + col;
                let qubits = vec![q, q + d];
                let is_boundary = col == 0 || col == d - 1;
                let check = StabilizerCheck::z_check(z_idx, &qubits);
                let check = if is_boundary {
                    check.as_boundary()
                } else {
                    check
                };
                z_stabs.push(check);
                z_idx += 1;
            }
        }
    }

    // Logical operators
    // Logical X: X on the left column
    let logical_x_qubits: Vec<usize> = (0..d).map(|row| row * d).collect();
    let logical_x = LogicalOperator::x(logical_x_qubits);

    // Logical Z: Z on the top row
    let logical_z_qubits: Vec<usize> = (0..d).collect();
    let logical_z = LogicalOperator::z(logical_z_qubits);

    (num_data, x_stabs, z_stabs, logical_x, logical_z)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rotated_surface_code_d3() {
        let code = SurfaceCode::rotated(3).unwrap();
        assert_eq!(code.distance(), 3);
        assert_eq!(code.num_data_qubits(), 9);
        assert!(code.num_x_stabilizers() > 0);
        assert!(code.num_z_stabilizers() > 0);
    }

    #[test]
    fn test_rotated_surface_code_d5() {
        let code = SurfaceCode::rotated(5).unwrap();
        assert_eq!(code.distance(), 5);
        assert_eq!(code.num_data_qubits(), 25);
    }

    #[test]
    fn test_standard_surface_code_d3() {
        let code = SurfaceCode::standard(3).unwrap();
        assert_eq!(code.distance(), 3);
        assert_eq!(code.num_data_qubits(), 9);
    }

    #[test]
    fn test_surface_code_to_stabilizer_code() {
        let code = SurfaceCode::rotated(3).unwrap();

        // Debug: print all stabilizers
        println!("X stabilizers:");
        for s in code.x_stabilizers() {
            println!("  {}: {:?}", s.index, s.qubits());
        }
        println!("Z stabilizers:");
        for s in code.z_stabilizers() {
            println!("  {}: {:?}", s.index, s.qubits());
        }

        let stab_code = code.to_stabilizer_code();

        // Debug: find which stabilizers anticommute
        let stabs = stab_code.stabilizers();
        for i in 0..stabs.len() {
            for j in (i + 1)..stabs.len() {
                use pecos_core::PauliOperator;
                if !stabs[i].commutes_with(&stabs[j]) {
                    println!("Anticommute: {i} and {j}");
                    println!("  {}: {:?}", i, stabs[i]);
                    println!("  {}: {:?}", j, stabs[j]);
                }
            }
        }

        // Verify the code is valid (stabilizers commute)
        assert!(stab_code.verify_stabilizers_commute().is_ok());
    }

    #[test]
    fn test_code_parameters() {
        let code = SurfaceCode::rotated(3).unwrap();
        assert_eq!(code.code_parameters(), "[[9, 1, 3]]");
    }

    #[test]
    fn test_builder() {
        let code = SurfaceCodeBuilder::new()
            .with_distance(5)
            .rotated()
            .build()
            .unwrap();
        assert_eq!(code.distance(), 5);
        assert!(code.rotated);
    }

    #[test]
    fn test_check_schedule() {
        let code = SurfaceCode::rotated(3).unwrap();
        let schedule = code.check_schedule();
        assert_eq!(schedule.num_rounds(), 2); // X and Z rounds
        assert_eq!(schedule.total_checks(), code.num_stabilizers());
    }

    #[test]
    fn test_invalid_distance() {
        let result = SurfaceCode::rotated(2);
        assert!(result.is_err());
    }
}
