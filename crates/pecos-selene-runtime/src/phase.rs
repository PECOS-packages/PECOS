//! Bounded phase arithmetic shared with host-side virtual-Z predictions.

/// Reduce a finite angle in radians to [-pi, pi], preserving small angles exactly.
///
/// Trigonometric argument reduction handles huge inputs without treating the
/// rounded `f64` value of tau as an exact period. Reduce each operand before
/// adding or subtracting angles, so a huge operand cannot erase a small one.
/// Non-finite inputs return NaN; gate entry points reject them before mutation.
#[must_use]
pub fn normalize_phase(angle: f64) -> f64 {
    if angle.abs() <= std::f64::consts::PI {
        angle
    } else {
        let (sin, cos) = angle.sin_cos();
        sin.atan2(cos)
    }
}
