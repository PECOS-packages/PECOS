//! Angle snapping for tolerance-based canonicalization.
//!
//! When angles come from floating-point sources, they may be close to exact
//! values but not precisely equal. This module provides explicit snapping
//! at the parse boundary.

use pecos_core::Angle64;

/// Result of a successful snap operation.
#[derive(Clone, Debug)]
pub struct SnapResult {
    /// The original angle before snapping
    pub original: Angle64,
    /// The snapped angle (exact value)
    pub snapped: Angle64,
    /// Distance snapped (in turns)
    pub distance: f64,
}

/// Error when snapping fails.
#[derive(Clone, Debug)]
pub struct SnapError {
    /// The angle that couldn't be snapped
    pub angle: Angle64,
    /// The nearest target angle
    pub nearest: Angle64,
    /// Distance to nearest target (in turns)
    pub distance: f64,
    /// The tolerance that was exceeded
    pub tolerance: f64,
}

impl std::fmt::Display for SnapError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Angle {:.6} turns is {:.2e} away from nearest target {:.6} (tolerance: {:.2e})",
            angle_to_turns(self.angle),
            self.distance,
            angle_to_turns(self.nearest),
            self.tolerance
        )
    }
}

impl std::error::Error for SnapError {}

/// Snaps angles to nearest exact canonical values within tolerance.
pub struct AngleSnapper {
    /// Known exact angles to snap to
    targets: Vec<Angle64>,
    /// Maximum distance (in turns) to snap
    tolerance: f64,
}

impl AngleSnapper {
    /// Create a new snapper with custom targets and tolerance.
    #[must_use]
    pub fn new(targets: Vec<Angle64>, tolerance: f64) -> Self {
        Self { targets, tolerance }
    }

    /// Create a snapper for standard angles (multiples of pi/4).
    #[must_use]
    pub fn standard(tolerance: f64) -> Self {
        use Angle64 as A;

        Self {
            targets: vec![
                A::ZERO,
                A::HALF_TURN / 4,                                  // pi/4 (T)
                A::QUARTER_TURN,                                   // pi/2 (S)
                A::HALF_TURN / 4 + A::QUARTER_TURN,                // 3pi/4
                A::HALF_TURN,                                      // pi (Z)
                A::HALF_TURN + A::HALF_TURN / 4,                   // 5pi/4
                A::THREE_QUARTERS_TURN,                            // 3pi/2
                A::HALF_TURN + A::HALF_TURN / 4 + A::QUARTER_TURN, // 7pi/4
            ],
            tolerance,
        }
    }

    /// Create a snapper for Clifford angles only (multiples of pi/2).
    #[must_use]
    pub fn clifford(tolerance: f64) -> Self {
        use Angle64 as A;

        Self {
            targets: vec![
                A::ZERO,
                A::QUARTER_TURN,
                A::HALF_TURN,
                A::THREE_QUARTERS_TURN,
            ],
            tolerance,
        }
    }

    /// Get the tolerance value.
    #[must_use]
    pub fn tolerance(&self) -> f64 {
        self.tolerance
    }

    /// Try to snap an angle to the nearest target.
    ///
    /// Returns `Ok(SnapResult)` if within tolerance, `Err(SnapError)` otherwise.
    pub fn snap(&self, angle: Angle64) -> Result<SnapResult, SnapError> {
        let mut best_target: Option<Angle64> = None;
        let mut best_distance = f64::MAX;

        for &target in &self.targets {
            let distance = angle_distance(angle, target);
            if distance < best_distance {
                best_distance = distance;
                best_target = Some(target);
            }
        }

        let nearest = best_target.unwrap_or(Angle64::ZERO);

        if best_distance <= self.tolerance {
            Ok(SnapResult {
                original: angle,
                snapped: nearest,
                distance: best_distance,
            })
        } else {
            Err(SnapError {
                angle,
                nearest,
                distance: best_distance,
                tolerance: self.tolerance,
            })
        }
    }

    /// Snap or return original (for permissive mode).
    #[must_use]
    pub fn snap_or_keep(&self, angle: Angle64) -> Angle64 {
        self.snap(angle).map(|r| r.snapped).unwrap_or(angle)
    }

    /// Snap or return original, with flag indicating if snapped.
    #[must_use]
    pub fn try_snap(&self, angle: Angle64) -> (Angle64, bool) {
        match self.snap(angle) {
            Ok(result) => (result.snapped, true),
            Err(_) => (angle, false),
        }
    }

    /// Add a custom target angle.
    pub fn add_target(&mut self, target: Angle64) {
        if !self.targets.contains(&target) {
            self.targets.push(target);
        }
    }

    /// Get all target angles.
    #[must_use]
    pub fn targets(&self) -> &[Angle64] {
        &self.targets
    }
}

/// Convert angle to turns (0.0 to 1.0).
fn angle_to_turns(angle: Angle64) -> f64 {
    angle.to_radians() / std::f64::consts::TAU
}

/// Calculate angular distance (shortest path on circle).
fn angle_distance(a: Angle64, b: Angle64) -> f64 {
    let diff = (angle_to_turns(a) - angle_to_turns(b)).abs();
    diff.min(1.0 - diff) // Handle wraparound
}

/// Snapping policy for circuit parsing.
#[derive(Clone, Debug, Default)]
pub enum SnapPolicy {
    /// No snapping - angles must be exact
    #[default]
    Exact,
    /// Snap within tolerance, fail if no target close enough
    SnapOrFail { tolerance: f64 },
    /// Snap within tolerance, keep original if no target close enough
    SnapOrKeep { tolerance: f64 },
}

impl SnapPolicy {
    /// Create a `SnapOrFail` policy with the given tolerance.
    #[must_use]
    pub fn snap_or_fail(tolerance: f64) -> Self {
        Self::SnapOrFail { tolerance }
    }

    /// Create a `SnapOrKeep` policy with the given tolerance.
    #[must_use]
    pub fn snap_or_keep(tolerance: f64) -> Self {
        Self::SnapOrKeep { tolerance }
    }

    /// Apply this policy to an angle.
    pub fn apply(&self, angle: Angle64, snapper: &AngleSnapper) -> Result<Angle64, SnapError> {
        match self {
            Self::Exact => Ok(angle),
            Self::SnapOrFail { .. } => snapper.snap(angle).map(|r| r.snapped),
            Self::SnapOrKeep { .. } => Ok(snapper.snap_or_keep(angle)),
        }
    }
}
