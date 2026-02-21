//! Trigonometric functions for [`Angle<T>`].
//!
//! Forward functions ([`sin`](Angle::sin), [`cos`](Angle::cos),
//! [`sin_cos`](Angle::sin_cos), [`tan`](Angle::tan)) exploit the fixed-point
//! representation to skip libm's range reduction. The fractional bits are split
//! into quadrant, half-quadrant, and remainder, providing an exact reduction to
//! \[0, pi/4\]. Minimax polynomials (degree 11 for sin, degree 12 for cos) give
//! near-f64 precision.
//!
//! Inverse functions ([`asin`](Angle::asin), [`acos`](Angle::acos),
//! [`atan`](Angle::atan), [`atan2`](Angle::atan2)) wrap the stdlib and convert
//! back via [`from_radians`](Angle::from_radians).

use super::Angle;
use num_complex::Complex64;
use num_traits::{
    Bounded, FromPrimitive, ToPrimitive, Unsigned, WrappingAdd, WrappingNeg, WrappingSub, Zero,
};
use std::ops::Rem;

// ---------------------------------------------------------------------------
// Minimax polynomial coefficients (Cephes / musl-libc)
// ---------------------------------------------------------------------------

// sin(x) = x + x^3 * (S1 + x^2 * (S2 + x^2 * (S3 + x^2 * (S4 + x^2 * (S5 + x^2 * S6)))))
//
// Coefficients from Cephes/musl-libc; preserved as published.
#[allow(clippy::excessive_precision)]
const S1: f64 = -1.666_666_666_666_663_24e-01;
#[allow(clippy::excessive_precision)]
const S2: f64 = 8.333_333_333_322_489_46e-03;
#[allow(clippy::excessive_precision)]
const S3: f64 = -1.984_126_982_985_794_93e-04;
#[allow(clippy::excessive_precision)]
const S4: f64 = 2.755_731_370_707_006_77e-06;
#[allow(clippy::excessive_precision)]
const S5: f64 = -2.505_076_025_340_686_34e-08;
#[allow(clippy::excessive_precision)]
const S6: f64 = 1.589_690_995_211_550_10e-10;

// cos(x) = 1 - x^2/2 + x^4 * (C1 + x^2 * (C2 + x^2 * (C3 + x^2 * (C4 + x^2 * (C5 + x^2 * C6)))))
//
// Coefficients from Cephes/musl-libc; preserved as published.
#[allow(clippy::excessive_precision)]
const C1: f64 = 4.166_666_666_666_660_19e-02;
#[allow(clippy::excessive_precision)]
const C2: f64 = -1.388_888_888_887_410_96e-03;
#[allow(clippy::excessive_precision)]
const C3: f64 = 2.480_158_728_947_672_94e-05;
#[allow(clippy::excessive_precision)]
const C4: f64 = -2.755_731_435_139_066_33e-07;
#[allow(clippy::excessive_precision)]
const C5: f64 = 2.087_572_321_298_174_83e-09;
#[allow(clippy::excessive_precision)]
const C6: f64 = -1.135_964_755_778_819_48e-11;

// ---------------------------------------------------------------------------
// Polynomial evaluation helpers
// ---------------------------------------------------------------------------

/// Fused multiply-add that uses hardware FMA when available, otherwise falls
/// back to `a * b + c`. The software `fma()` in libm is correct but very slow
/// (~1.5 ns per call), so avoiding it when FMA is not compiled in gives a
/// large speedup with negligible precision loss for our polynomials.
#[allow(clippy::inline_always)]
#[inline(always)]
fn fma(a: f64, b: f64, c: f64) -> f64 {
    #[cfg(target_feature = "fma")]
    {
        a.mul_add(b, c)
    }
    #[cfg(not(target_feature = "fma"))]
    {
        a * b + c
    }
}

/// Evaluate the minimax sin polynomial on [0, pi/4].
///
/// With hardware FMA: uses Horner's method (fewer total instructions, each step
/// is a single FMA instruction).
///
/// Without FMA: uses Estrin's scheme (evaluates pairs of terms in parallel,
/// reducing critical-path latency when each step is mul+add = 2 instructions).
#[inline]
fn poly_sin(x: f64) -> f64 {
    let x2 = x * x;
    let x3 = x * x2;

    #[cfg(target_feature = "fma")]
    {
        // Horner: 6 sequential FMAs, minimal total instruction count.
        let r = fma(S6, x2, S5);
        let r = fma(r, x2, S4);
        let r = fma(r, x2, S3);
        let r = fma(r, x2, S2);
        let r = fma(r, x2, S1);
        fma(x3, r, x)
    }
    #[cfg(not(target_feature = "fma"))]
    {
        // Estrin: 3 levels of parallelism, shorter dependency chain.
        let x4 = x2 * x2;
        let q0 = fma(S2, x2, S1);
        let q1 = fma(S4, x2, S3);
        let q2 = fma(S6, x2, S5);
        let r0 = fma(q1, x4, q0);
        let x8 = x4 * x4;
        let p = fma(q2, x8, r0);
        fma(x3, p, x)
    }
}

/// Evaluate the minimax cos polynomial on [0, pi/4].
///
/// Strategy selection matches `poly_sin`: Horner with FMA, Estrin without.
#[inline]
fn poly_cos(x: f64) -> f64 {
    let x2 = x * x;
    let x4 = x2 * x2;

    #[cfg(target_feature = "fma")]
    {
        let r = fma(C6, x2, C5);
        let r = fma(r, x2, C4);
        let r = fma(r, x2, C3);
        let r = fma(r, x2, C2);
        let r = fma(r, x2, C1);
        fma(x4, r, 1.0 - 0.5 * x2)
    }
    #[cfg(not(target_feature = "fma"))]
    {
        let q0 = fma(C2, x2, C1);
        let q1 = fma(C4, x2, C3);
        let q2 = fma(C6, x2, C5);
        let r0 = fma(q1, x4, q0);
        let x8 = x4 * x4;
        let p = fma(q2, x8, r0);
        fma(x4, p, 1.0 - 0.5 * x2)
    }
}

// ---------------------------------------------------------------------------
// Octant reduction
// ---------------------------------------------------------------------------

/// Result of octant-based range reduction.
struct Octant {
    /// Reduced angle in [0, pi/4].
    x: f64,
    /// Whether the half-quadrant (3rd bit) was set.
    half: bool,
    /// Quadrant index (0-3) from top 2 bits.
    quadrant: u32,
}

/// Reduce a u64 fraction to an angle in [0, pi/4] plus octant metadata.
///
/// When the half-quadrant bit is set, the complement is computed in the
/// integer domain (`max - remainder`) before converting to float. This
/// avoids catastrophic cancellation that would occur with a float-domain
/// `pi/4 - x_raw` subtraction when `x_raw ≈ pi/4`.
#[inline]
#[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
fn reduce_u64(frac: u64, bits: u32) -> Octant {
    // Top 2 bits give quadrant (0-3), always fits in u32.
    let quadrant = (frac >> (bits - 2)) as u32;
    let half = ((frac >> (bits - 3)) & 1) != 0;
    let rem_bits = bits - 3;
    let remainder = frac & ((1u64 << rem_bits) - 1);

    // Integer-domain complement to avoid floating-point cancellation.
    let r = if half {
        ((1u64 << rem_bits) - 1) - remainder
    } else {
        remainder
    };

    // Precision loss converting to f64 is inherent (f64 has 53-bit mantissa).
    let inv_scale = std::f64::consts::FRAC_PI_4 / (1u64 << rem_bits) as f64;
    let x = (r as f64) * inv_scale;

    Octant { x, half, quadrant }
}

/// Reduce a u128 fraction to an angle in [0, pi/4] plus octant metadata.
///
/// Same integer-complement strategy as `reduce_u64` (see its doc comment).
#[inline]
#[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
fn reduce_u128(frac: u128) -> Octant {
    let quadrant = (frac >> 126) as u32;
    let half = ((frac >> 125) & 1) != 0;
    let remainder = frac & ((1u128 << 125) - 1);

    // Integer-domain complement to avoid floating-point cancellation.
    let r = if half {
        ((1u128 << 125) - 1) - remainder
    } else {
        remainder
    };

    // 125-bit value: split into high (53-bit) and mid (53-bit) parts.
    // The bottom 19 bits are below f64 precision.
    let split = 53u32;
    let discard = 125 - 2 * split; // 19 bits
    let hi = (r >> (split + discard)) as f64;
    let mid = ((r >> discard) as u64 & ((1u64 << split) - 1)) as f64;
    let inv_scale_hi = std::f64::consts::FRAC_PI_4 / (1u128 << (125 - split - discard)) as f64;
    let inv_scale_mid = std::f64::consts::FRAC_PI_4 / (1u128 << (125 - discard)) as f64;
    let x = fma(hi, inv_scale_hi, mid * inv_scale_mid);

    Octant { x, half, quadrant }
}

/// Reduce any supported fraction to octant form.
/// Uses native u64 arithmetic for types up to 64 bits; u128 only for u128.
#[inline]
#[allow(clippy::cast_possible_truncation)]
fn reduce<T: ToPrimitive + Copy>(fraction: T) -> Octant {
    let bits = std::mem::size_of::<T>() * 8;
    if bits <= 64 {
        reduce_u64(
            fraction
                .to_u64()
                .expect("Failed to convert fraction to u64"),
            bits as u32, // size_of * 8 always fits in u32
        )
    } else {
        reduce_u128(
            fraction
                .to_u128()
                .expect("Failed to convert fraction to u128"),
        )
    }
}

/// Conditionally negate an f64 by XOR-ing the sign bit. Branchless.
#[inline]
fn apply_sign(val: f64, negate: bool) -> f64 {
    f64::from_bits(val.to_bits() ^ (u64::from(negate) << 63))
}

// ---------------------------------------------------------------------------
// Trigonometric methods on Angle<T>
// ---------------------------------------------------------------------------

impl<T> Angle<T>
where
    T: Unsigned
        + Copy
        + ToPrimitive
        + FromPrimitive
        + Zero
        + Bounded
        + WrappingAdd
        + WrappingSub
        + WrappingNeg
        + Rem<Output = T>,
{
    /// Returns `(sin(theta), cos(theta))` using octant-based range reduction.
    ///
    /// This avoids the expensive range-reduction step in libm by exploiting
    /// the fixed-point representation: the top 3 bits give the octant directly,
    /// and the remaining bits are divided by a power of 2 (exact in f64) to
    /// produce the reduced angle in \[0, pi/4\].
    #[inline]
    pub fn sin_cos(&self) -> (f64, f64) {
        let oct = reduce(self.fraction);

        let mut s = poly_sin(oct.x);
        let mut c = poly_cos(oct.x);

        if oct.half {
            std::mem::swap(&mut s, &mut c);
        }

        // Quadrant sign + swap mapping:
        //   Q0: ( s,  c)    Q1: ( c, -s)
        //   Q2: (-s, -c)    Q3: (-c,  s)
        let (sin_base, cos_base) = if oct.quadrant & 1 != 0 {
            (c, s)
        } else {
            (s, c)
        };
        let sin_val = apply_sign(sin_base, oct.quadrant >= 2);
        let cos_val = apply_sign(cos_base, oct.quadrant == 1 || oct.quadrant == 2);
        (sin_val, cos_val)
    }

    /// Returns the sine of the angle.
    #[inline]
    pub fn sin(&self) -> f64 {
        self.sin_cos().0
    }

    /// Returns the cosine of the angle.
    #[inline]
    pub fn cos(&self) -> f64 {
        self.sin_cos().1
    }

    /// Returns the tangent of the angle.
    #[inline]
    pub fn tan(&self) -> f64 {
        let (s, c) = self.sin_cos();
        s / c
    }

    /// Returns the angle whose sine is `value`.
    ///
    /// # Panics
    /// Panics if the conversion from radians fails.
    #[inline]
    #[must_use]
    pub fn asin(value: f64) -> Self {
        Self::from_radians(value.asin())
    }

    /// Returns the angle whose cosine is `value`.
    ///
    /// # Panics
    /// Panics if the conversion from radians fails.
    #[inline]
    #[must_use]
    pub fn acos(value: f64) -> Self {
        Self::from_radians(value.acos())
    }

    /// Returns the angle whose tangent is `value`.
    ///
    /// # Panics
    /// Panics if the conversion from radians fails.
    #[inline]
    #[must_use]
    pub fn atan(value: f64) -> Self {
        Self::from_radians(value.atan())
    }

    /// Returns the angle whose tangent is `y / x`, using signs to determine
    /// the quadrant.
    ///
    /// # Panics
    /// Panics if the conversion from radians fails.
    #[inline]
    #[must_use]
    pub fn atan2(y: f64, x: f64) -> Self {
        Self::from_radians(y.atan2(x))
    }

    /// Returns `(sin(theta/2), cos(theta/2))`.
    ///
    /// Common in quantum computing where rotation gates use half-angle
    /// components (e.g., `cos(theta/2)` and `sin(theta/2)` for RX, RY gates).
    /// Halving the fixed-point fraction is exact (integer division by 2).
    ///
    /// # Panics
    /// Panics if `T` cannot represent the value 2 (unreachable for all
    /// standard unsigned integer types).
    #[inline]
    pub fn half_angle_sin_cos(&self) -> (f64, f64) {
        let half = Self::new(self.fraction / T::from_u32(2).expect("2 must be representable"));
        half.sin_cos()
    }

    /// Returns `e^(i*theta) = cos(theta) + i*sin(theta)` as a `Complex64`.
    ///
    /// Euler's formula, commonly used in quantum gate matrices and signal
    /// processing. Uses [`sin_cos`](Self::sin_cos) internally.
    #[inline]
    pub fn cis(&self) -> Complex64 {
        let (s, c) = self.sin_cos();
        Complex64::new(c, s)
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::super::{Angle8, Angle16, Angle32, Angle64, Angle128};
    use rand::RngExt;
    use std::f64::consts::{FRAC_PI_3, FRAC_PI_4, FRAC_PI_6};

    const TOL: f64 = 1e-15;
    const TOL_LOOSE: f64 = 1e-10;

    // -- Cardinal angles --------------------------------------------------

    #[test]
    fn sin_cos_zero() {
        let (s, c) = Angle64::ZERO.sin_cos();
        assert!(s.abs() < TOL, "sin(0) = {s}");
        assert!((c - 1.0).abs() < TOL, "cos(0) = {c}");
    }

    #[test]
    fn sin_cos_quarter_turn() {
        let (s, c) = Angle64::QUARTER_TURN.sin_cos();
        assert!((s - 1.0).abs() < TOL, "sin(pi/2) = {s}");
        assert!(c.abs() < TOL, "cos(pi/2) = {c}");
    }

    #[test]
    fn sin_cos_half_turn() {
        let (s, c) = Angle64::HALF_TURN.sin_cos();
        assert!(s.abs() < TOL, "sin(pi) = {s}");
        assert!((c + 1.0).abs() < TOL, "cos(pi) = {c}");
    }

    #[test]
    fn sin_cos_three_quarters_turn() {
        let (s, c) = Angle64::THREE_QUARTERS_TURN.sin_cos();
        assert!((s + 1.0).abs() < TOL, "sin(3pi/2) = {s}");
        assert!(c.abs() < TOL, "cos(3pi/2) = {c}");
    }

    // -- Known values -----------------------------------------------------

    #[test]
    fn sin_cos_pi_over_4() {
        let angle = Angle64::from_radians(FRAC_PI_4);
        let (s, c) = angle.sin_cos();
        let expected = std::f64::consts::FRAC_1_SQRT_2;
        assert!(
            (s - expected).abs() < TOL_LOOSE,
            "sin(pi/4) = {s}, expected {expected}"
        );
        assert!(
            (c - expected).abs() < TOL_LOOSE,
            "cos(pi/4) = {c}, expected {expected}"
        );
    }

    #[test]
    fn sin_cos_pi_over_6() {
        let angle = Angle64::from_radians(FRAC_PI_6);
        let (s, c) = angle.sin_cos();
        let sqrt3_over_2 = 3.0_f64.sqrt() / 2.0;
        assert!((s - 0.5).abs() < TOL_LOOSE, "sin(pi/6) = {s}, expected 0.5");
        assert!(
            (c - sqrt3_over_2).abs() < TOL_LOOSE,
            "cos(pi/6) = {c}, expected {sqrt3_over_2}"
        );
    }

    #[test]
    fn sin_cos_pi_over_3() {
        let angle = Angle64::from_radians(FRAC_PI_3);
        let (s, c) = angle.sin_cos();
        let sqrt3_over_2 = 3.0_f64.sqrt() / 2.0;
        assert!(
            (s - sqrt3_over_2).abs() < TOL_LOOSE,
            "sin(pi/3) = {s}, expected {sqrt3_over_2}"
        );
        assert!((c - 0.5).abs() < TOL_LOOSE, "cos(pi/3) = {c}, expected 0.5");
    }

    // -- Standalone sin/cos match sin_cos ---------------------------------

    #[test]
    fn standalone_sin_matches_sin_cos() {
        let mut rng = rand::rng();
        for _ in 0..10_000 {
            let frac: u64 = rng.random();
            let angle = Angle64::new(frac);
            let (sc_sin, _) = angle.sin_cos();
            let standalone = angle.sin();
            assert!(
                (sc_sin - standalone).abs() < 1e-15,
                "sin mismatch: sin_cos={sc_sin}, sin()={standalone}, frac={frac}"
            );
        }
    }

    #[test]
    fn standalone_cos_matches_sin_cos() {
        let mut rng = rand::rng();
        for _ in 0..10_000 {
            let frac: u64 = rng.random();
            let angle = Angle64::new(frac);
            let (_, sc_cos) = angle.sin_cos();
            let standalone = angle.cos();
            assert!(
                (sc_cos - standalone).abs() < 1e-15,
                "cos mismatch: sin_cos={sc_cos}, cos()={standalone}, frac={frac}"
            );
        }
    }

    // -- Pythagorean identity ---------------------------------------------

    #[test]
    fn pythagorean_identity() {
        let mut rng = rand::rng();
        for _ in 0..10_000 {
            let frac: u64 = rng.random();
            let angle = Angle64::new(frac);
            let (s, c) = angle.sin_cos();
            let sum = s.mul_add(s, c * c);
            assert!(
                (sum - 1.0).abs() < 1e-14,
                "sin^2+cos^2 = {sum} for fraction {frac}"
            );
        }
    }

    #[test]
    fn pythagorean_identity_standalone() {
        let mut rng = rand::rng();
        for _ in 0..10_000 {
            let frac: u64 = rng.random();
            let angle = Angle64::new(frac);
            let s = angle.sin();
            let c = angle.cos();
            let sum = s.mul_add(s, c * c);
            assert!(
                (sum - 1.0).abs() < 1e-14,
                "standalone sin^2+cos^2 = {sum} for fraction {frac}"
            );
        }
    }

    // -- Symmetry ---------------------------------------------------------

    #[test]
    fn sin_odd_symmetry() {
        for &rad in &[0.1, 1.0, 2.5] {
            let a = Angle64::from_radians(rad);
            let neg = Angle64::ZERO - a;
            assert!(
                (a.sin() + neg.sin()).abs() < 1e-14,
                "sin(-x) should equal -sin(x) for x={rad}"
            );
        }
    }

    #[test]
    fn cos_even_symmetry() {
        for &rad in &[0.1, 1.0, 2.5] {
            let a = Angle64::from_radians(rad);
            let neg = Angle64::ZERO - a;
            assert!(
                (a.cos() - neg.cos()).abs() < 1e-14,
                "cos(-x) should equal cos(x) for x={rad}"
            );
        }
    }

    // -- Octant boundary continuity ---------------------------------------

    #[test]
    fn octant_boundary_continuity() {
        let octant_size: u64 = 1 << (64 - 3);
        for octant in 1..8u64 {
            let boundary = octant * octant_size;
            let below = Angle64::new(boundary - 1);
            let at = Angle64::new(boundary);
            let (s1, c1) = below.sin_cos();
            let (s2, c2) = at.sin_cos();
            assert!(
                (s1 - s2).abs() < 1e-10,
                "sin discontinuity at octant {octant}: {s1} vs {s2}"
            );
            assert!(
                (c1 - c2).abs() < 1e-10,
                "cos discontinuity at octant {octant}: {c1} vs {c2}"
            );
        }
    }

    // -- Small angles -----------------------------------------------------

    #[test]
    fn small_angles() {
        for k in 1..20 {
            let frac = 1u64 << k;
            let angle = Angle64::new(frac);
            let theta = angle.to_radians();
            let (s, c) = angle.sin_cos();
            assert!(
                (s - theta.sin()).abs() < 1e-12,
                "sin mismatch for small angle (k={k})"
            );
            assert!(
                (c - theta.cos()).abs() < 1e-12,
                "cos mismatch for small angle (k={k})"
            );
        }
    }

    // -- tan near pi/2 ----------------------------------------------------

    #[test]
    fn tan_near_pi_over_2() {
        let just_below = Angle64::QUARTER_TURN - Angle64::new(1);
        let just_above = Angle64::QUARTER_TURN + Angle64::new(1);
        assert!(
            just_below.tan() > 1e10,
            "tan should be large just below pi/2"
        );
        assert!(
            just_above.tan() < -1e10,
            "tan should be large negative just above pi/2"
        );
    }

    // -- Inverse round-trips ----------------------------------------------

    #[test]
    fn asin_round_trip() {
        let angle = Angle64::from_radians(0.3);
        let recovered = Angle64::asin(angle.sin());
        assert!(
            (recovered.to_radians() - 0.3).abs() < TOL_LOOSE,
            "asin round-trip failed: got {}",
            recovered.to_radians()
        );
    }

    #[test]
    fn acos_round_trip() {
        let angle = Angle64::from_radians(1.0);
        let recovered = Angle64::acos(angle.cos());
        assert!(
            (recovered.to_radians() - 1.0).abs() < TOL_LOOSE,
            "acos round-trip failed: got {}",
            recovered.to_radians()
        );
    }

    #[test]
    fn atan_round_trip() {
        let angle = Angle64::from_radians(0.5);
        let recovered = Angle64::atan(angle.tan());
        assert!(
            (recovered.to_radians() - 0.5).abs() < TOL_LOOSE,
            "atan round-trip failed: got {}",
            recovered.to_radians()
        );
    }

    #[test]
    fn atan2_basic() {
        let angle = Angle64::atan2(1.0, 1.0);
        assert!(
            (angle.to_radians() - FRAC_PI_4).abs() < TOL_LOOSE,
            "atan2(1,1) = {}, expected pi/4",
            angle.to_radians()
        );
    }

    // -- Cross-type -------------------------------------------------------

    #[test]
    fn sin_cos_all_types() {
        macro_rules! check_quarter {
            ($ty:ty, $tol:expr) => {
                let (s, c) = <$ty>::QUARTER_TURN.sin_cos();
                assert!(
                    (s - 1.0).abs() < $tol,
                    "{}: sin(pi/2) = {s}",
                    stringify!($ty)
                );
                assert!(c.abs() < $tol, "{}: cos(pi/2) = {c}", stringify!($ty));
            };
        }
        check_quarter!(Angle8, TOL);
        check_quarter!(Angle16, TOL);
        check_quarter!(Angle32, TOL);
        check_quarter!(Angle64, TOL);
        check_quarter!(Angle128, TOL);
    }

    #[test]
    fn sin_cos_half_turn_all_types() {
        macro_rules! check_half {
            ($ty:ty) => {
                let (s, c) = <$ty>::HALF_TURN.sin_cos();
                assert!(s.abs() < TOL, "{}: sin(pi) = {s}", stringify!($ty));
                assert!((c + 1.0).abs() < TOL, "{}: cos(pi) = {c}", stringify!($ty));
            };
        }
        check_half!(Angle8);
        check_half!(Angle16);
        check_half!(Angle32);
        check_half!(Angle64);
        check_half!(Angle128);
    }

    // -- half_angle_sin_cos -----------------------------------------------

    #[test]
    fn half_angle_matches_manual_halving() {
        // half_angle_sin_cos(HALF_TURN) should equal sin_cos(QUARTER_TURN)
        let (s, c) = Angle64::HALF_TURN.half_angle_sin_cos();
        let (s_ref, c_ref) = Angle64::QUARTER_TURN.sin_cos();
        assert!((s - s_ref).abs() < TOL, "sin mismatch: {s} vs {s_ref}");
        assert!((c - c_ref).abs() < TOL, "cos mismatch: {c} vs {c_ref}");
    }

    #[test]
    fn half_angle_quarter_turn() {
        // half_angle_sin_cos(QUARTER_TURN) = sin_cos(pi/4)
        let (s, c) = Angle64::QUARTER_TURN.half_angle_sin_cos();
        let expected = std::f64::consts::FRAC_1_SQRT_2;
        assert!(
            (s - expected).abs() < TOL,
            "sin(pi/4) = {s}, expected {expected}"
        );
        assert!(
            (c - expected).abs() < TOL,
            "cos(pi/4) = {c}, expected {expected}"
        );
    }

    #[test]
    fn half_angle_random_matches() {
        let mut rng = rand::rng();
        for _ in 0..10_000 {
            let frac: u64 = rng.random();
            let angle = Angle64::new(frac);
            let (s, c) = angle.half_angle_sin_cos();
            let halved = Angle64::new(frac / 2);
            let (s_ref, c_ref) = halved.sin_cos();
            assert!((s - s_ref).abs() < TOL, "sin mismatch for frac={frac}");
            assert!((c - c_ref).abs() < TOL, "cos mismatch for frac={frac}");
        }
    }

    // -- cis --------------------------------------------------------------

    #[test]
    fn cis_zero() {
        let z = Angle64::ZERO.cis();
        assert!((z.re - 1.0).abs() < TOL, "cis(0) real = {}", z.re);
        assert!(z.im.abs() < TOL, "cis(0) imag = {}", z.im);
    }

    #[test]
    fn cis_quarter_turn() {
        // e^(i*pi/2) = i
        let z = Angle64::QUARTER_TURN.cis();
        assert!(z.re.abs() < TOL, "cis(pi/2) real = {}", z.re);
        assert!((z.im - 1.0).abs() < TOL, "cis(pi/2) imag = {}", z.im);
    }

    #[test]
    fn cis_half_turn() {
        // e^(i*pi) = -1
        let z = Angle64::HALF_TURN.cis();
        assert!((z.re + 1.0).abs() < TOL, "cis(pi) real = {}", z.re);
        assert!(z.im.abs() < TOL, "cis(pi) imag = {}", z.im);
    }

    #[test]
    fn cis_unit_magnitude() {
        let mut rng = rand::rng();
        for _ in 0..10_000 {
            let frac: u64 = rng.random();
            let z = Angle64::new(frac).cis();
            let mag_sq = z.re * z.re + z.im * z.im;
            assert!(
                (mag_sq - 1.0).abs() < 1e-14,
                "|cis|^2 = {mag_sq} for frac={frac}"
            );
        }
    }

    // -- Comparison with stdlib -------------------------------------------

    #[test]
    fn matches_stdlib() {
        let mut rng = rand::rng();
        for _ in 0..10_000 {
            let frac: u64 = rng.random();
            let angle = Angle64::new(frac);
            let theta = angle.to_radians();
            let (s, c) = angle.sin_cos();
            assert!(
                (s - theta.sin()).abs() < 1e-12,
                "sin mismatch: ours={s}, stdlib={}, frac={frac}",
                theta.sin()
            );
            assert!(
                (c - theta.cos()).abs() < 1e-12,
                "cos mismatch: ours={c}, stdlib={}, frac={frac}",
                theta.cos()
            );
        }
    }

    // -- Accuracy: max error across all octants ---------------------------

    #[test]
    fn max_ulp_error_vs_stdlib() {
        // Walk through each octant with fine steps and track max error.
        let octant_size: u64 = 1 << (64 - 3);
        let step = octant_size / 10_000;
        let mut max_sin_err: f64 = 0.0;
        let mut max_cos_err: f64 = 0.0;

        for octant in 0..8u64 {
            let base = octant * octant_size;
            for i in 0..10_000u64 {
                let frac = base + i * step;
                let angle = Angle64::new(frac);
                let theta = angle.to_radians();
                let (s, c) = angle.sin_cos();
                max_sin_err = max_sin_err.max((s - theta.sin()).abs());
                max_cos_err = max_cos_err.max((c - theta.cos()).abs());
            }
        }

        assert!(
            max_sin_err < 1e-12,
            "max sin error {max_sin_err:.3e} exceeds threshold"
        );
        assert!(
            max_cos_err < 1e-12,
            "max cos error {max_cos_err:.3e} exceeds threshold"
        );
    }
}
