//! Ross--Selinger one- and two-dimensional grid-candidate machinery.

use std::cmp::Ordering;

use num_bigint::BigInt;
use num_traits::{One, Zero};

use crate::interval::{
    DyadicInterval, IntervalOrdering, retry_with_precision, sqrt_interval, target_phase,
};
use crate::{DOmega, SynthError, ZOmega, ZSqrt2};

const UPRIGHT_SKEW: u8 = 15;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum CandidateBranch {
    Unshifted,
    Shifted,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct GridCandidate {
    pub(crate) u: DOmega,
    pub(crate) branch: CandidateBranch,
    pub(crate) k: u32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum GridError {
    InvalidEpsilon,
    ExponentOverflow,
    Synthesis(SynthError),
}

impl From<SynthError> for GridError {
    fn from(value: SynthError) -> Self {
        Self::Synthesis(value)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct Epsilon {
    numerator: u64,
    log_denominator: u32,
}

impl Epsilon {
    fn new(numerator: u64, log_denominator: u32) -> Result<Self, GridError> {
        if numerator == 0 {
            return Err(GridError::InvalidEpsilon);
        }
        let less_than_two = if log_denominator >= 63 {
            true
        } else {
            let comparison_exponent = u32::try_from(u64::from(log_denominator) + 1)
                .map_err(|_| GridError::ExponentOverflow)?;
            u128::from(numerator) < (1_u128 << comparison_exponent)
        };
        if !less_than_two {
            return Err(GridError::InvalidEpsilon);
        }
        u32::try_from(
            u64::from(log_denominator)
                .checked_mul(2)
                .and_then(|value| value.checked_add(1))
                .ok_or(GridError::ExponentOverflow)?,
        )
        .map_err(|_| GridError::ExponentOverflow)?;
        Ok(Self {
            numerator,
            log_denominator,
        })
    }

    fn h(&self, precision: u32) -> Result<DyadicInterval, GridError> {
        let exponent = u64::from(self.log_denominator)
            .checked_mul(2)
            .and_then(|value| value.checked_add(1))
            .ok_or(GridError::ExponentOverflow)?;
        let exponent = u32::try_from(exponent).map_err(|_| GridError::ExponentOverflow)?;
        let numerator = BigInt::from(self.numerator) * BigInt::from(self.numerator);
        Ok(DyadicInterval::exact(numerator, exponent).at_precision(precision))
    }
}

#[derive(Clone)]
struct CapGeometry {
    z: [DyadicInterval; 2],
    perpendicular: [DyadicInterval; 2],
    h: DyadicInterval,
    d: DyadicInterval,
    transverse_squared: DyadicInterval,
}

impl CapGeometry {
    fn new(angle_fraction: u64, epsilon: &Epsilon, precision: u32) -> Result<Self, GridError> {
        let (zx, zy) = target_phase(angle_fraction, precision)?;
        let h = epsilon.h(precision)?;
        let one = DyadicInterval::integer(1_u8, precision);
        let two = DyadicInterval::integer(2_u8, precision);
        let d = one.sub(&h);
        let transverse_squared = match h.compare(&one) {
            IntervalOrdering::DefinitelyLess => h.mul(&two.sub(&h)),
            IntervalOrdering::DefinitelyGreaterOrEqual => h.clone(),
            IntervalOrdering::Straddles => {
                if h.is_exact() && h.lo_numerator() == one.lo_numerator() {
                    h.clone()
                } else {
                    return Err(GridError::Synthesis(SynthError::Inconclusive { precision }));
                }
            }
        };
        Ok(Self {
            z: [zx.clone(), zy.clone()],
            perpendicular: [zy.neg(), zx],
            h,
            d,
            transverse_squared,
        })
    }

    fn state(&self, precision: u32) -> SymmetricMatrix {
        // Write cap points in the orthonormal (z,z_perp) frame as (x,y),
        // put h=epsilon^2/2, d=1-h, and t=1-x. For 0<h<=1 use
        //
        //   (x-d)^2/h^2 + y^2/[h(2-h)] <= 1.
        //
        // On the circular cap, 0<=t<=h and y^2<=t(2-t). The ellipse's
        // permitted y^2 exceeds this by
        // `2(1-h)(h-t)t/h >= 0`, proving closed-cap containment. For
        // 1<=h<2 use transverse axis squared h; its permitted y^2 is
        // `2t-t^2/h >= t(2-t)`. All coefficients below are outward
        // intervals, so replacing exact products by their enclosures can only
        // enlarge the enumerated bounding box; every emitted point is checked
        // against the exact closed cap afterward.
        //
        // The narrow ellipse has area `pi*h^(3/2)*sqrt(2-h)`, while
        // `area(cap)=2*integral_0^h sqrt(t(2-t))dt` is at least
        // `(4/3)h^(3/2)sqrt(2-h)`, hence inflation <=3pi/4<3. For h>=1,
        // circular-segment symmetry and concavity give area(cap)>=pi*h/2;
        // the wide ellipse ratio is <=2sqrt(h)<=2sqrt(2)<3. Thus C=3 is
        // uniform and epsilon-independent. This is the cap-specific
        // enclosing ellipse used with Ross--Selinger, arXiv:1403.2975v3,
        // Proposition 5.17 and Theorem 5.18.
        let h_squared = self.h.square();
        let zx_squared = self.z[0].square();
        let zy_squared = self.z[1].square();
        let a = self
            .transverse_squared
            .mul_to(&zx_squared, precision)
            .add(&h_squared.mul_to(&zy_squared, precision));
        let d = self
            .transverse_squared
            .mul_to(&zy_squared, precision)
            .add(&h_squared.mul_to(&zx_squared, precision));
        let b = self
            .transverse_squared
            .sub(&h_squared)
            .mul_to(&self.z[0], precision)
            .mul_to(&self.z[1], precision);
        SymmetricMatrix { a, b, d }
    }
}

/// Enumerates both Ross--Selinger branches through `max_k`, in increasing
/// denominator exponent, with the unshifted branch before the shifted branch.
pub(crate) fn candidate_stream(
    angle_fraction: u64,
    epsilon_numerator: u64,
    epsilon_log_denominator: u32,
    max_k: u32,
    base_precision: u32,
    max_precision: u32,
) -> Result<Vec<GridCandidate>, GridError> {
    let epsilon = Epsilon::new(epsilon_numerator, epsilon_log_denominator)?;
    retry_with_precision(
        base_precision,
        max_precision,
        |precision| match enumerate_at_precision(angle_fraction, &epsilon, max_k, precision) {
            Ok(candidates) => Ok(Some(candidates)),
            Err(GridError::Synthesis(SynthError::Inconclusive { .. })) => Ok(None),
            Err(error) => Err(error),
        },
    )
}

fn enumerate_at_precision(
    angle_fraction: u64,
    epsilon: &Epsilon,
    max_k: u32,
    precision: u32,
) -> Result<Vec<GridCandidate>, GridError> {
    let cap = CapGeometry::new(angle_fraction, epsilon, precision)?;
    let upright = upright_operator(&cap, precision)?;
    let mut candidates = Vec::new();
    for k in 0..=max_k {
        for branch in [CandidateBranch::Unshifted, CandidateBranch::Shifted] {
            let points = enumerate_branch_k(&cap, &upright, branch, k, precision)?;
            candidates.extend(points);
        }
    }
    Ok(candidates)
}

/// Ross and Selinger, arXiv:1403.2975v3, Proposition 4.5: rescale by powers
/// of the fundamental unit until the first interval has width in
/// `[lambda^-1, 1)`, then enumerate its `sqrt(2)` coordinate. Multiplication
/// by `lambda` changes the conjugate interval by `-lambda^-1`, so the width
/// product is invariant. Outward interval endpoints can only add trials; the
/// final exact algebraic checks below remove them and retain closed boundaries.
fn solve_one_dimensional(
    x: &DyadicInterval,
    y: &DyadicInterval,
    precision: u32,
) -> Result<Vec<ZSqrt2>, GridError> {
    let mut exponent = 0_i64;
    let mut scale = ZSqrt2::one();
    let width = DyadicInterval::exact(x.width_numerator(), x.precision());
    if width.lo_numerator().is_zero() {
        let lower = x.ceil_lower();
        if lower == x.floor_upper() {
            let candidate = ZSqrt2::new(lower, BigInt::zero());
            if contains_zsqrt(y, &candidate) {
                return Ok(vec![candidate]);
            }
        }
        return Ok(Vec::new());
    }
    let one = DyadicInterval::integer(1_u8, precision);
    let inverse_lambda = evaluate_zsqrt(&ZSqrt2::new(BigInt::from(-1), BigInt::one()), precision);

    let mut normalized = false;
    for _ in 0..4096_u32 {
        let scaled_width = width
            .at_precision(precision)
            .mul(&evaluate_zsqrt(&scale, precision));
        if scaled_width.compare(&one) != IntervalOrdering::DefinitelyLess {
            if scaled_width.is_exact() && scaled_width == one {
                exponent = exponent.checked_sub(1).ok_or(GridError::ExponentOverflow)?;
                scale = &scale * &lambda_pow(-1);
                continue;
            }
            if scaled_width.compare(&one) == IntervalOrdering::Straddles {
                return Err(GridError::Synthesis(SynthError::Inconclusive { precision }));
            }
            exponent = exponent.checked_sub(1).ok_or(GridError::ExponentOverflow)?;
            scale = &scale * &lambda_pow(-1);
            continue;
        }
        match scaled_width.compare(&inverse_lambda) {
            IntervalOrdering::DefinitelyLess => {
                exponent = exponent.checked_add(1).ok_or(GridError::ExponentOverflow)?;
                scale = &scale * &lambda_pow(1);
            }
            IntervalOrdering::DefinitelyGreaterOrEqual => {
                normalized = true;
                break;
            }
            IntervalOrdering::Straddles => {
                return Err(GridError::Synthesis(SynthError::Inconclusive { precision }));
            }
        }
    }
    if !normalized {
        return Err(GridError::ExponentOverflow);
    }

    let transformed_x = scale_real_interval(x, &scale, precision);
    let bullet_scale = scale.sqrt2_conjugate();
    let transformed_y = scale_real_interval(y, &bullet_scale, precision);
    let sqrt2_over_four =
        evaluate_zsqrt(&ZSqrt2::sqrt2(), precision).div_positive_int(&BigInt::from(4_u8));
    let b_bounds = DyadicInterval::new(
        transformed_x.lo_numerator() - transformed_y.hi_numerator(),
        transformed_x.hi_numerator() - transformed_y.lo_numerator(),
        precision,
    )
    .mul(&sqrt2_over_four);
    let mut b = b_bounds.ceil_lower();
    let b_max = b_bounds.floor_upper();
    let inverse_scale = lambda_pow(exponent.checked_neg().ok_or(GridError::ExponentOverflow)?);
    let mut solutions = Vec::new();

    while b <= b_max {
        let b_sqrt2 = evaluate_zsqrt(&ZSqrt2::new(BigInt::zero(), b.clone()), precision);
        let a_bounds = DyadicInterval::new(
            transformed_x.lo_numerator() - b_sqrt2.hi_numerator(),
            transformed_x.hi_numerator() - b_sqrt2.lo_numerator(),
            precision,
        );
        let mut a = a_bounds.ceil_lower();
        let a_max = a_bounds.floor_upper();
        while a <= a_max {
            let transformed = ZSqrt2::new(a.clone(), b.clone());
            let original = &inverse_scale * &transformed;
            if contains_zsqrt(x, &original) && contains_zsqrt(y, &original.sqrt2_conjugate()) {
                solutions.push(original);
            }
            a += 1_u8;
        }
        b += 1_u8;
    }
    solutions.sort_by(|left, right| {
        left.rational_part()
            .cmp(right.rational_part())
            .then_with(|| left.sqrt2_part().cmp(right.sqrt2_part()))
    });
    solutions.dedup();
    Ok(solutions)
}

fn contains_zsqrt(interval: &DyadicInterval, value: &ZSqrt2) -> bool {
    compare_zsqrt_to_dyadic(value, interval.lo_numerator(), interval.precision()) != Ordering::Less
        && compare_zsqrt_to_dyadic(value, interval.hi_numerator(), interval.precision())
            != Ordering::Greater
}

fn compare_zsqrt_to_dyadic(value: &ZSqrt2, numerator: &BigInt, precision: u32) -> Ordering {
    let scale = BigInt::one() << shift_count(precision);
    ZSqrt2::new(
        value.rational_part() * &scale - numerator,
        value.sqrt2_part() * scale,
    )
    .cmp(&ZSqrt2::zero())
}

fn scale_real_interval(
    interval: &DyadicInterval,
    scalar: &ZSqrt2,
    precision: u32,
) -> DyadicInterval {
    interval
        .at_precision(precision)
        .mul(&evaluate_zsqrt(scalar, precision))
}

#[derive(Clone)]
struct SymmetricMatrix {
    a: DyadicInterval,
    b: DyadicInterval,
    d: DyadicInterval,
}

impl SymmetricMatrix {
    fn identity(precision: u32) -> Self {
        Self {
            a: DyadicInterval::integer(1_u8, precision),
            b: DyadicInterval::integer(0_u8, precision),
            d: DyadicInterval::integer(1_u8, precision),
        }
    }

    fn determinant(&self) -> DyadicInterval {
        self.a.mul(&self.d).sub(&self.b.square())
    }

    fn transform(&self, operator: &GridOperator, bullet: bool, precision: u32) -> Self {
        let entries = operator.intervals(bullet, precision);
        let [g00, g01, g10, g11] = entries;
        let a = self
            .a
            .mul(&g00.square())
            .add(
                &self
                    .b
                    .mul(&g00)
                    .mul(&g10)
                    .mul(&DyadicInterval::integer(2_u8, precision)),
            )
            .add(&self.d.mul(&g10.square()));
        let b = self
            .a
            .mul(&g00)
            .mul(&g01)
            .add(&self.b.mul(&g00.mul(&g11).add(&g01.mul(&g10))))
            .add(&self.d.mul(&g10).mul(&g11));
        let d = self
            .a
            .mul(&g01.square())
            .add(
                &self
                    .b
                    .mul(&g01)
                    .mul(&g11)
                    .mul(&DyadicInterval::integer(2_u8, precision)),
            )
            .add(&self.d.mul(&g11.square()));
        Self { a, b, d }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct GridOperator {
    entries: [DOmega; 4],
}

impl GridOperator {
    fn identity() -> Self {
        Self::new(DOmega::one(), DOmega::zero(), DOmega::zero(), DOmega::one())
    }

    fn new(g00: DOmega, g01: DOmega, g10: DOmega, g11: DOmega) -> Self {
        Self {
            entries: [g00, g01, g10, g11],
        }
    }

    fn intervals(&self, bullet: bool, precision: u32) -> [DyadicInterval; 4] {
        std::array::from_fn(|index| {
            let value = if bullet {
                self.entries[index].sqrt2_conjugate()
            } else {
                self.entries[index].clone()
            };
            evaluate_real_domega(&value, precision)
        })
    }

    fn mul(&self, rhs: &Self) -> Self {
        let [a, b, c, d] = &self.entries;
        let [e, f, g, h] = &rhs.entries;
        Self::new(
            &(a * e) + &(b * g),
            &(a * f) + &(b * h),
            &(c * e) + &(d * g),
            &(c * f) + &(d * h),
        )
    }

    fn inverse(&self) -> Self {
        let [a, b, c, d] = &self.entries;
        let determinant = &(a * d) - &(b * c);
        let one = DOmega::one();
        if determinant == one {
            Self::new(d.clone(), -b, -c, a.clone())
        } else {
            assert_eq!(determinant, -one, "uprighting operator must be special");
            Self::new(-d, b.clone(), c.clone(), -a)
        }
    }

    fn bullet(&self) -> Self {
        Self {
            entries: self.entries.clone().map(|entry| entry.sqrt2_conjugate()),
        }
    }

    fn shifted_conjugate(&self, exponent: i64) -> Self {
        let positive = domega_from_zsqrt(&lambda_pow(exponent));
        let negative = domega_from_zsqrt(&lambda_pow(-exponent));
        Self::new(
            &positive * &self.entries[0],
            self.entries[1].clone(),
            self.entries[2].clone(),
            &negative * &self.entries[3],
        )
    }

    fn apply_zomega(&self, value: &ZOmega) -> ZOmega {
        let value = DOmega::from(value.clone());
        let (x, y) = complex_parts(&value);
        let real = &(&self.entries[0] * &x) + &(&self.entries[1] * &y);
        let imaginary = &(&self.entries[2] * &x) + &(&self.entries[3] * &y);
        let output = &real + &(&imaginary * &DOmega::from(ZOmega::i()));
        assert_eq!(
            output.least_denominator_exponent(),
            0,
            "grid operator must preserve Z[omega]"
        );
        output.numerator().clone()
    }

    fn lex_key(&self) -> Vec<BigInt> {
        // Ross--Selinger, arXiv:1403.2975v3, Section 5 ("Grid operators")
        // writes every grid-operator entry uniquely as `a + a'/sqrt(2)`.
        // These are the integer coordinates used for the specified row-major
        // lexicographic tie-break.
        let mut key = Vec::with_capacity(8);
        for entry in &self.entries {
            let coordinates = entry.numerator().coordinates();
            assert!(coordinates[2].is_zero() && coordinates[1] == -&coordinates[3]);
            match entry.least_denominator_exponent() {
                0 => {
                    key.push(coordinates[0].clone());
                    key.push(BigInt::from(2_u8) * &coordinates[1]);
                }
                1 => {
                    key.push(coordinates[1].clone());
                    key.push(coordinates[0].clone());
                }
                _ => panic!("grid-operator entry must be in Z[1/sqrt(2)]"),
            }
        }
        key
    }
}

fn upright_operator(cap: &CapGeometry, precision: u32) -> Result<GridOperator, GridError> {
    // Ross and Selinger, arXiv:1403.2975v3, Section 5 and Appendix A:
    // special grid operators act on the paired ellipse matrices as
    // `(G^T D G, G^{bullet T} Delta G^bullet)`. Repeated Step-Lemma
    // reductions to skew <= 15 make both transformed ellipses at least
    // 1/6-upright (Theorem 5.16). We test the reduction inequality itself
    // with outward intervals and choose the lexicographically smallest
    // row-major canonical integer-coordinate matrix whenever several of
    // Appendix A's operators work.
    let original_cap = cap.state(precision);
    let original_disk = SymmetricMatrix::identity(precision);
    let mut reduction = GridOperator::identity();

    for _ in 0..4096_u32 {
        let current_cap = original_cap.transform(&reduction, false, precision);
        let current_disk = original_disk.transform(&reduction, true, precision);
        match skew_at_most(&current_cap, &current_disk, UPRIGHT_SKEW) {
            IntervalOrdering::DefinitelyGreaterOrEqual => return Ok(reduction.inverse()),
            IntervalOrdering::Straddles => {
                return Err(GridError::Synthesis(SynthError::Inconclusive { precision }));
            }
            IntervalOrdering::DefinitelyLess => {}
        }

        let shift = bias_shift(&current_cap, &current_disk, precision)?;
        let shifted_cap = shift_state(&current_cap, shift, false, precision);
        let shifted_disk = shift_state(&current_disk, shift, true, precision);
        let mut trials = appendix_trials(&shifted_cap, &shifted_disk, precision)?;
        for trial in &mut trials {
            *trial = trial.shifted_conjugate(shift);
        }
        trials.sort_by_key(GridOperator::lex_key);
        trials.dedup();

        let mut next = None;
        for trial in trials {
            let cap_after = current_cap.transform(&trial, false, precision);
            let disk_after = current_disk.transform(&trial, true, precision);
            match reduces_skew_by_ten_percent(&current_cap, &current_disk, &cap_after, &disk_after)
            {
                IntervalOrdering::DefinitelyLess => {}
                IntervalOrdering::DefinitelyGreaterOrEqual => {
                    next = Some(trial);
                    break;
                }
                // A lexicographically earlier trial cannot be skipped merely
                // because the current enclosure cannot decide it: doing so
                // would make the selected operator depend on base precision.
                IntervalOrdering::Straddles => {
                    return Err(GridError::Synthesis(SynthError::Inconclusive { precision }));
                }
            }
        }
        let Some(next) = next else {
            return Err(GridError::Synthesis(SynthError::Inconclusive { precision }));
        };
        reduction = reduction.mul(&next);
    }
    Err(GridError::ExponentOverflow)
}

/// Returns `DefinitelyGreaterOrEqual` when `skew <= bound`.
fn skew_at_most(first: &SymmetricMatrix, second: &SymmetricMatrix, bound: u8) -> IntervalOrdering {
    // Each matrix may have an arbitrary positive scalar. The normalized
    // anti-diagonal square is `b^2/det`; combine the two fractions exactly.
    let first_det = first.determinant();
    let second_det = second.determinant();
    let left = first
        .b
        .square()
        .mul(&second_det)
        .add(&second.b.square().mul(&first_det));
    let right = first_det
        .mul(&second_det)
        .mul(&DyadicInterval::integer(bound, left.precision()));
    right.compare(&left)
}

fn reduces_skew_by_ten_percent(
    old_first: &SymmetricMatrix,
    old_second: &SymmetricMatrix,
    new_first: &SymmetricMatrix,
    new_second: &SymmetricMatrix,
) -> IntervalOrdering {
    let old_first_det = old_first.determinant();
    let old_second_det = old_second.determinant();
    let new_first_det = new_first.determinant();
    let new_second_det = new_second.determinant();
    let old_num = old_first
        .b
        .square()
        .mul(&old_second_det)
        .add(&old_second.b.square().mul(&old_first_det));
    let old_den = old_first_det.mul(&old_second_det);
    let new_num = new_first
        .b
        .square()
        .mul(&new_second_det)
        .add(&new_second.b.square().mul(&new_first_det));
    let new_den = new_first_det.mul(&new_second_det);
    let precision = old_num.precision();
    let left = new_num
        .mul(&old_den)
        .mul(&DyadicInterval::integer(10_u8, precision));
    let right = old_num
        .mul(&new_den)
        .mul(&DyadicInterval::integer(9_u8, precision));
    right.compare(&left)
}

fn bias_shift(
    first: &SymmetricMatrix,
    second: &SymmetricMatrix,
    precision: u32,
) -> Result<i64, GridError> {
    let lambda_squared = evaluate_zsqrt(&(&lambda_pow(1) * &lambda_pow(1)), precision);
    let mut shift = 0_i64;
    for _ in 0..4096_u32 {
        let shifted_first = shift_state(first, shift, false, precision);
        let shifted_second = shift_state(second, shift, true, precision);
        let ratio_num = shifted_second.d.mul(&shifted_first.a);
        let ratio_den = shifted_second.a.mul(&shifted_first.d);
        match ratio_num.compare(&lambda_squared.mul(&ratio_den)) {
            IntervalOrdering::DefinitelyGreaterOrEqual => {
                shift = shift.checked_sub(1).ok_or(GridError::ExponentOverflow)?;
                continue;
            }
            IntervalOrdering::Straddles => {
                return Err(GridError::Synthesis(SynthError::Inconclusive { precision }));
            }
            IntervalOrdering::DefinitelyLess => {}
        }
        match lambda_squared.mul(&ratio_num).compare(&ratio_den) {
            IntervalOrdering::DefinitelyLess => {
                shift = shift.checked_add(1).ok_or(GridError::ExponentOverflow)?;
            }
            IntervalOrdering::DefinitelyGreaterOrEqual => return Ok(shift),
            IntervalOrdering::Straddles => {
                return Err(GridError::Synthesis(SynthError::Inconclusive { precision }));
            }
        }
    }
    Err(GridError::ExponentOverflow)
}

fn shift_state(
    state: &SymmetricMatrix,
    exponent: i64,
    bullet_side: bool,
    precision: u32,
) -> SymmetricMatrix {
    let positive = evaluate_zsqrt(&lambda_pow(exponent), precision);
    let negative = evaluate_zsqrt(&lambda_pow(-exponent), precision);
    let odd_negative = !exponent.unsigned_abs().is_multiple_of(2);
    if bullet_side {
        SymmetricMatrix {
            a: state.a.mul(&negative),
            b: if odd_negative {
                state.b.neg()
            } else {
                state.b.clone()
            },
            d: state.d.mul(&positive),
        }
    } else {
        SymmetricMatrix {
            a: state.a.mul(&positive),
            b: state.b.clone(),
            d: state.d.mul(&negative),
        }
    }
}

fn appendix_trials(
    first: &SymmetricMatrix,
    second: &SymmetricMatrix,
    precision: u32,
) -> Result<Vec<GridOperator>, GridError> {
    let identity = GridOperator::identity();
    let x = operator_x();
    let z = operator_z();
    let symmetries = [identity, x.clone(), z.clone(), x.mul(&z)];
    let r = operator_r();
    let k = operator_k();
    let k_bullet = k.bullet();
    let mut trials = Vec::new();

    for symmetry in symmetries {
        let sym_first = first.transform(&symmetry, false, precision);
        let sym_second = second.transform(&symmetry, true, precision);
        trials.push(symmetry.mul(&r));
        trials.push(symmetry.mul(&k));
        trials.push(symmetry.mul(&k_bullet));
        for n in shear_candidates(&sym_first, &sym_second, 4_u8, precision)? {
            trials.push(symmetry.mul(&operator_a(&n)));
        }
        for n in shear_candidates(&sym_first, &sym_second, 2_u8, precision)? {
            trials.push(symmetry.mul(&operator_b(&n)));
        }
    }
    Ok(trials)
}

fn shear_candidates(
    first: &SymmetricMatrix,
    second: &SymmetricMatrix,
    factor: u8,
    precision: u32,
) -> Result<Vec<BigInt>, GridError> {
    let factor_interval = DyadicInterval::integer(factor, precision);
    let holds = |n: &BigInt| {
        let n_squared = DyadicInterval::integer(n * n, precision);
        let multiplier = factor_interval.mul(&n_squared);
        let first_comparison = multiplier.mul(&first.a).compare(&first.d);
        let second_comparison = multiplier.mul(&second.a).compare(&second.d);
        (first_comparison, second_comparison)
    };
    let mut high = BigInt::one();
    for _ in 0..4096_u32 {
        let comparisons = holds(&high);
        if comparisons.0 == IntervalOrdering::DefinitelyLess
            && comparisons.1 == IntervalOrdering::DefinitelyLess
        {
            high <<= 1_usize;
        } else {
            break;
        }
    }
    let mut low = &high >> 1_usize;
    while &high - &low > BigInt::one() {
        let middle = (&high + &low) >> 1_usize;
        let comparisons = holds(&middle);
        if comparisons.0 == IntervalOrdering::DefinitelyLess
            && comparisons.1 == IntervalOrdering::DefinitelyLess
        {
            low = middle;
        } else if comparisons.0 == IntervalOrdering::DefinitelyGreaterOrEqual
            || comparisons.1 == IntervalOrdering::DefinitelyGreaterOrEqual
        {
            high = middle;
        } else {
            return Err(GridError::Synthesis(SynthError::Inconclusive { precision }));
        }
    }
    let center = low.max(BigInt::one());
    Ok([
        (&center - BigInt::one()).max(BigInt::one()),
        center.clone(),
        center + BigInt::one(),
    ]
    .into_iter()
    .collect())
}

fn operator_r() -> GridOperator {
    let inv_sqrt2 = DOmega::new(ZOmega::one(), 1);
    GridOperator::new(inv_sqrt2.clone(), -&inv_sqrt2, inv_sqrt2.clone(), inv_sqrt2)
}

fn operator_a(exponent: &BigInt) -> GridOperator {
    GridOperator::new(
        DOmega::one(),
        DOmega::from(ZOmega::from(-BigInt::from(2_u8) * exponent)),
        DOmega::zero(),
        DOmega::one(),
    )
}

fn operator_b(exponent: &BigInt) -> GridOperator {
    GridOperator::new(
        DOmega::one(),
        domega_from_zsqrt(&ZSqrt2::new(BigInt::zero(), exponent.clone())),
        DOmega::zero(),
        DOmega::one(),
    )
}

fn operator_k() -> GridOperator {
    GridOperator::new(
        DOmega::new(
            zsqrt_to_zomega(&ZSqrt2::new(BigInt::one(), BigInt::from(-1))),
            1,
        ),
        -DOmega::new(ZOmega::one(), 1),
        DOmega::new(zsqrt_to_zomega(&lambda_pow(1)), 1),
        DOmega::new(ZOmega::one(), 1),
    )
}

fn operator_x() -> GridOperator {
    GridOperator::new(DOmega::zero(), DOmega::one(), DOmega::one(), DOmega::zero())
}

fn operator_z() -> GridOperator {
    GridOperator::new(
        DOmega::one(),
        DOmega::zero(),
        DOmega::zero(),
        -DOmega::one(),
    )
}

fn enumerate_branch_k(
    cap: &CapGeometry,
    upright: &GridOperator,
    branch: CandidateBranch,
    k: u32,
    precision: u32,
) -> Result<Vec<GridCandidate>, GridError> {
    // Ross and Selinger, arXiv:1403.2975v3, Lemmas 9.6--9.7 and
    // Algorithm 9.8: for the shifted branch, u'=delta*u ranges over the cap
    // scaled by |delta| (squared 2+sqrt(2)), while bullet(u') ranges over the
    // disk scaled by |bullet(delta)| (squared 2-sqrt(2)).
    let (scale_squared, bullet_radius_squared) = match branch {
        CandidateBranch::Unshifted => (ZSqrt2::one(), ZSqrt2::one()),
        CandidateBranch::Shifted => (
            ZSqrt2::new(BigInt::from(2_u8), BigInt::one()),
            ZSqrt2::new(BigInt::from(2_u8), BigInt::from(-1)),
        ),
    };
    let integer_scale_squared = BigInt::one() << shift_count(k);
    let a_box = transformed_cap_box(
        cap,
        upright,
        &scale_squared,
        &integer_scale_squared,
        precision,
    );
    let b_box = transformed_disk_box(
        &upright.bullet(),
        &bullet_radius_squared,
        &integer_scale_squared,
        precision,
    );
    let inverse = upright.inverse();
    let mut output = Vec::new();

    // Ross and Selinger, arXiv:1403.2975v3, Lemma 5.5: Z[omega] is
    // the disjoint union of `alpha + beta i` and
    // `alpha + beta i + omega`, with alpha,beta in Z[sqrt(2)].
    // The plain coset is deliberately emitted first.
    for shifted_coset in [false, true] {
        let offset =
            evaluate_zsqrt(&ZSqrt2::sqrt2(), precision).div_positive_int(&BigInt::from(2_u8));
        let (ax, ay, bx, by) = if shifted_coset {
            (
                a_box[0].sub(&offset),
                a_box[1].sub(&offset),
                b_box[0].add(&offset),
                b_box[1].add(&offset),
            )
        } else {
            (
                a_box[0].clone(),
                a_box[1].clone(),
                b_box[0].clone(),
                b_box[1].clone(),
            )
        };
        let alphas = solve_one_dimensional(&ax, &bx, precision)?;
        let betas = solve_one_dimensional(&ay, &by, precision)?;
        for alpha in &alphas {
            for beta in &betas {
                let transformed = zomega_from_parts(alpha, beta, shifted_coset);
                let numerator = inverse.apply_zomega(&transformed);
                let shifted_value = DOmega::new(numerator, k);
                if shifted_value.least_denominator_exponent() != k {
                    continue;
                }
                let Some(u) = exact_branch_test(cap, &shifted_value, branch, precision)? else {
                    continue;
                };
                output.push(GridCandidate { u, branch, k });
            }
        }
    }
    Ok(output)
}

fn transformed_cap_box(
    cap: &CapGeometry,
    operator: &GridOperator,
    branch_scale_squared: &ZSqrt2,
    integer_scale_squared: &BigInt,
    precision: u32,
) -> [DyadicInterval; 2] {
    let entries = operator.intervals(false, precision);
    let scale_squared = evaluate_zsqrt(branch_scale_squared, precision).mul(
        &DyadicInterval::integer(integer_scale_squared.clone(), precision),
    );
    let scale = sqrt_interval(&scale_squared);
    let center_base = [cap.d.mul(&cap.z[0]), cap.d.mul(&cap.z[1])];
    std::array::from_fn(|row| {
        let left = &entries[row * 2];
        let right = &entries[row * 2 + 1];
        let center = left
            .mul(&center_base[0])
            .add(&right.mul(&center_base[1]))
            .mul(&scale);
        let along = left.mul(&cap.z[0]).add(&right.mul(&cap.z[1]));
        let across = left
            .mul(&cap.perpendicular[0])
            .add(&right.mul(&cap.perpendicular[1]));
        let extent_squared = cap
            .h
            .square()
            .mul(&along.square())
            .add(&cap.transverse_squared.mul(&across.square()))
            .mul(&scale_squared);
        let extent = sqrt_interval(&extent_squared);
        DyadicInterval::new(
            center.lo_numerator() - extent.hi_numerator(),
            center.hi_numerator() + extent.hi_numerator(),
            precision,
        )
    })
}

fn transformed_disk_box(
    operator: &GridOperator,
    radius_squared: &ZSqrt2,
    integer_scale_squared: &BigInt,
    precision: u32,
) -> [DyadicInterval; 2] {
    let entries = operator.intervals(false, precision);
    let scale_squared = evaluate_zsqrt(radius_squared, precision).mul(&DyadicInterval::integer(
        integer_scale_squared.clone(),
        precision,
    ));
    std::array::from_fn(|row| {
        let extent_squared = entries[row * 2]
            .square()
            .add(&entries[row * 2 + 1].square())
            .mul(&scale_squared);
        let extent = sqrt_interval(&extent_squared);
        DyadicInterval::new(
            -extent.hi_numerator(),
            extent.hi_numerator().clone(),
            precision,
        )
    })
}

fn exact_branch_test(
    cap: &CapGeometry,
    shifted_value: &DOmega,
    branch: CandidateBranch,
    precision: u32,
) -> Result<Option<DOmega>, GridError> {
    // Ross and Selinger, arXiv:1403.2975v3, equations (13)--(14): the
    // epsilon region is the intersection of the unit disk and the half-plane
    // Re(z^dagger u) >= 1-epsilon^2/2. The packet requires the closed form,
    // so equality is accepted by the interval comparison below.
    let (radius_squared, bullet_radius_squared) = match branch {
        CandidateBranch::Unshifted => (ZSqrt2::one(), ZSqrt2::one()),
        CandidateBranch::Shifted => (
            ZSqrt2::new(BigInt::from(2_u8), BigInt::one()),
            ZSqrt2::new(BigInt::from(2_u8), BigInt::from(-1)),
        ),
    };
    if !norm_at_most(shifted_value, &radius_squared)
        || !norm_at_most(&shifted_value.sqrt2_conjugate(), &bullet_radius_squared)
    {
        return Ok(None);
    }
    let (x, y) = evaluate_complex_domega(shifted_value, precision);
    let dot = cap.z[0].mul(&x).add(&cap.z[1].mul(&y));
    let rhs = if branch == CandidateBranch::Unshifted {
        cap.d.clone()
    } else {
        cap.d
            .mul(&sqrt_interval(&evaluate_zsqrt(&radius_squared, precision)))
    };
    match dot.compare(&rhs) {
        IntervalOrdering::DefinitelyLess => Ok(None),
        IntervalOrdering::DefinitelyGreaterOrEqual => {
            let u = if branch == CandidateBranch::Unshifted {
                shifted_value.clone()
            } else {
                // Ross and Selinger, arXiv:1403.2975v3, Lemmas 9.6--9.7
                // and Algorithm 9.8: enumerate u'=delta*u in the scaled
                // regions, recover u with delta^-1=(omega-i)/sqrt(2), but
                // retain k=lde(u'), not lde(u).
                let delta_inverse = DOmega::new(&ZOmega::omega() - &ZOmega::i(), 1);
                shifted_value * &delta_inverse
            };
            Ok(Some(u))
        }
        IntervalOrdering::Straddles => {
            Err(GridError::Synthesis(SynthError::Inconclusive { precision }))
        }
    }
}

fn norm_at_most(value: &DOmega, bound: &ZSqrt2) -> bool {
    let numerator_norm = value.numerator().norm_squared();
    let denominator = BigInt::one() << shift_count(value.least_denominator_exponent());
    numerator_norm <= bound * &ZSqrt2::new(denominator, BigInt::zero())
}

fn evaluate_zsqrt(value: &ZSqrt2, precision: u32) -> DyadicInterval {
    let rational = DyadicInterval::integer(value.rational_part().clone(), precision);
    let irrational = sqrt_interval(&DyadicInterval::integer(2_u8, precision)).mul(
        &DyadicInterval::integer(value.sqrt2_part().clone(), precision),
    );
    rational.add(&irrational)
}

fn evaluate_real_domega(value: &DOmega, precision: u32) -> DyadicInterval {
    let (real, imaginary) = evaluate_complex_domega(value, precision);
    debug_assert!(imaginary.contains_zero());
    real
}

fn evaluate_complex_domega(value: &DOmega, precision: u32) -> (DyadicInterval, DyadicInterval) {
    let coordinates = value.numerator().coordinates();
    let sqrt2_over_two = sqrt_interval(&DyadicInterval::integer(2_u8, precision))
        .div_positive_int(&BigInt::from(2_u8));
    let real = DyadicInterval::integer(coordinates[0].clone(), precision).add(
        &DyadicInterval::integer(&coordinates[1] - &coordinates[3], precision).mul(&sqrt2_over_two),
    );
    let imaginary = DyadicInterval::integer(coordinates[2].clone(), precision).add(
        &DyadicInterval::integer(&coordinates[1] + &coordinates[3], precision).mul(&sqrt2_over_two),
    );
    let half_exponent = value.least_denominator_exponent() / 2;
    let divisor = BigInt::one() << shift_count(half_exponent);
    let mut real = real.div_positive_int(&divisor);
    let mut imaginary = imaginary.div_positive_int(&divisor);
    if !value.least_denominator_exponent().is_multiple_of(2) {
        real = real.mul(&sqrt2_over_two);
        imaginary = imaginary.mul(&sqrt2_over_two);
    }
    (real, imaginary)
}

fn complex_parts(value: &DOmega) -> (DOmega, DOmega) {
    let half = DOmega::new(ZOmega::one(), 2);
    let conjugate = value.conjugate();
    let real = &(value + &conjugate) * &half;
    let minus_i = -ZOmega::i();
    let imaginary = &(&(value - &conjugate) * &DOmega::from(minus_i)) * &half;
    (real, imaginary)
}

fn zomega_from_parts(alpha: &ZSqrt2, beta: &ZSqrt2, shifted: bool) -> ZOmega {
    let mut value = ZOmega::new(
        alpha.rational_part().clone(),
        alpha.sqrt2_part() + beta.sqrt2_part(),
        beta.rational_part().clone(),
        -alpha.sqrt2_part() + beta.sqrt2_part(),
    );
    if shifted {
        value = &value + &ZOmega::omega();
    }
    value
}

fn zsqrt_to_zomega(value: &ZSqrt2) -> ZOmega {
    ZOmega::new(
        value.rational_part().clone(),
        value.sqrt2_part().clone(),
        BigInt::zero(),
        -value.sqrt2_part(),
    )
}

fn domega_from_zsqrt(value: &ZSqrt2) -> DOmega {
    DOmega::from(zsqrt_to_zomega(value))
}

fn lambda_pow(exponent: i64) -> ZSqrt2 {
    let base = if exponent >= 0 {
        ZSqrt2::new(BigInt::one(), BigInt::one())
    } else {
        ZSqrt2::new(BigInt::from(-1), BigInt::one())
    };
    let mut result = ZSqrt2::one();
    for _ in 0..exponent.unsigned_abs() {
        result = &result * &base;
    }
    result
}

fn shift_count(value: u32) -> usize {
    usize::try_from(u64::from(value)).expect("u32 shift count fits usize")
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::*;

    fn candidate_key(candidate: &GridCandidate) -> (u32, bool, u32, [BigInt; 4]) {
        (
            candidate.k,
            candidate.branch == CandidateBranch::Shifted,
            candidate.u.least_denominator_exponent(),
            candidate.u.numerator().coordinates().clone(),
        )
    }

    fn shifted_value(candidate: &GridCandidate) -> DOmega {
        if candidate.branch == CandidateBranch::Unshifted {
            candidate.u.clone()
        } else {
            let delta = &ZOmega::one() + &ZOmega::omega();
            &candidate.u * &DOmega::from(delta)
        }
    }

    fn coset_order_key(value: &ZOmega) -> (bool, BigInt, BigInt, BigInt, BigInt) {
        let coordinates = value.coordinates();
        let parity = (&coordinates[1] - &coordinates[3]) & BigInt::one();
        let shifted = !parity.is_zero();
        let offset = BigInt::from(u8::from(shifted));
        (
            shifted,
            coordinates[0].clone(),
            (&coordinates[1] - &coordinates[3] - &offset) / BigInt::from(2_u8),
            coordinates[2].clone(),
            (&coordinates[1] + &coordinates[3] - offset) / BigInt::from(2_u8),
        )
    }

    #[test]
    fn one_dimensional_solver_matches_bounded_brute_force_and_order() {
        let precision = 96;
        let boxes = [
            (
                DyadicInterval::new(
                    BigInt::from(-3_i8) << 95_usize,
                    BigInt::from(5_i8) << 94_usize,
                    precision,
                ),
                DyadicInterval::new(
                    BigInt::from(-5_i8) << 94_usize,
                    BigInt::from(3_i8) << 95_usize,
                    precision,
                ),
            ),
            (
                DyadicInterval::new(
                    BigInt::from(-2_i8) << 96_usize,
                    BigInt::from(2_i8) << 96_usize,
                    precision,
                ),
                DyadicInterval::new(
                    BigInt::from(-1_i8) << 96_usize,
                    BigInt::from(3_i8) << 96_usize,
                    precision,
                ),
            ),
            (
                DyadicInterval::new(
                    BigInt::from(-7_i8) << 93_usize,
                    BigInt::from(9_i8) << 93_usize,
                    precision,
                ),
                DyadicInterval::new(
                    BigInt::from(-11_i8) << 93_usize,
                    BigInt::from(5_i8) << 93_usize,
                    precision,
                ),
            ),
        ];
        for (x, y) in boxes {
            let actual = solve_one_dimensional(&x, &y, precision).unwrap();
            let mut expected = Vec::new();
            for a in -12_i64..=12 {
                for b in -12_i64..=12 {
                    let value = ZSqrt2::new(BigInt::from(a), BigInt::from(b));
                    if contains_zsqrt(&x, &value) && contains_zsqrt(&y, &value.sqrt2_conjugate()) {
                        expected.push(value);
                    }
                }
            }
            expected.sort_by(|left, right| {
                left.rational_part()
                    .cmp(right.rational_part())
                    .then_with(|| left.sqrt2_part().cmp(right.sqrt2_part()))
            });
            assert_eq!(actual, expected);
        }
    }

    #[test]
    fn cap_ellipse_has_uniform_area_bound_on_epsilon_ladder() {
        // For h<=1 the cap area is
        // `2 integral_0^h sqrt(t(2-t)) dt`, at least
        // `(4/3) h^(3/2) sqrt(2-h)`. The ellipse area is
        // `pi h^(3/2) sqrt(2-h)`, so its inflation is at most 3pi/4<3.
        // For 1<=h<2, the wide-cap ellipse has area pi*h^(3/2), while
        // concavity/symmetry of the circular-segment area gives
        // `area(cap)>=pi*h/2`; its ratio is at most 2sqrt(2)<3.
        // This is the cap-specific containment/area construction required by
        // Ross--Selinger Proposition 5.17 and Theorem 5.18; the proof is also
        // reproduced beside `cap.state` and `transformed_cap_box`.
        let precision = 256;
        let pi = crate::interval::pi_interval(precision);
        for log_denominator in 4_u32..=40 {
            let epsilon = Epsilon::new(1, log_denominator).unwrap();
            let h = epsilon.h(precision).unwrap();
            let two = DyadicInterval::integer(2_u8, precision);
            let transverse_squared = h.mul(&two.sub(&h));
            let shared_factor = h.mul(&sqrt_interval(&transverse_squared));
            let ellipse_upper = pi.mul(&shared_factor);
            let cap_lower = shared_factor
                .mul(&DyadicInterval::integer(4_u8, precision))
                .div_positive_int(&BigInt::from(3_u8));
            let three_cap_lower = cap_lower.mul(&DyadicInterval::integer(3_u8, precision));
            assert_eq!(
                three_cap_lower.compare(&ellipse_upper),
                IntervalOrdering::DefinitelyGreaterOrEqual
            );
            assert_eq!(
                h.compare(&DyadicInterval::integer(1_u8, precision)),
                IntervalOrdering::DefinitelyLess
            );

            let h_squared = h.square();
            let right = h_squared.mul(&transverse_squared);
            for sample in 1_u8..16 {
                let t = h
                    .mul(&DyadicInterval::integer(sample, precision))
                    .div_positive_int(&BigInt::from(16_u8));
                let x_minus_center = h.sub(&t);
                let y_squared = t.mul(&two.sub(&t));
                let left = x_minus_center
                    .square()
                    .mul(&transverse_squared)
                    .add(&y_squared.mul(&h_squared));
                assert_eq!(
                    right.compare(&left),
                    IntervalOrdering::DefinitelyGreaterOrEqual
                );
            }
        }
    }

    #[test]
    fn two_dimensional_stream_matches_bounded_brute_force_for_both_branches() {
        let precision = 192;
        let max_k = 2;
        let angle_fraction = 1_u64 << 61;
        let actual = candidate_stream(angle_fraction, 1, 1, max_k, 96, precision).unwrap();
        let epsilon = Epsilon::new(1, 1).unwrap();
        let cap = CapGeometry::new(angle_fraction, &epsilon, precision).unwrap();
        let mut expected = Vec::new();
        for k in 0..=max_k {
            for branch in [CandidateBranch::Unshifted, CandidateBranch::Shifted] {
                for x0 in -6_i64..=6 {
                    for x1 in -6_i64..=6 {
                        for x2 in -6_i64..=6 {
                            for x3 in -6_i64..=6 {
                                let numerator = ZOmega::new(
                                    BigInt::from(x0),
                                    BigInt::from(x1),
                                    BigInt::from(x2),
                                    BigInt::from(x3),
                                );
                                let value = DOmega::new(numerator, k);
                                if value.least_denominator_exponent() != k {
                                    continue;
                                }
                                if let Some(u) =
                                    exact_branch_test(&cap, &value, branch, precision).unwrap()
                                {
                                    expected.push(GridCandidate { u, branch, k });
                                }
                            }
                        }
                    }
                }
            }
        }
        let actual_set: BTreeSet<_> = actual.iter().map(candidate_key).collect();
        let expected_set: BTreeSet<_> = expected.iter().map(candidate_key).collect();
        assert_eq!(actual_set, expected_set);

        let upright = upright_operator(&cap, precision).unwrap();
        expected.sort_by_key(|candidate| {
            let value = shifted_value(candidate);
            (
                candidate.k,
                candidate.branch == CandidateBranch::Shifted,
                coset_order_key(&upright.apply_zomega(value.numerator())),
            )
        });
        assert_eq!(
            actual.iter().map(candidate_key).collect::<Vec<_>>(),
            expected.iter().map(candidate_key).collect::<Vec<_>>()
        );

        for candidate in actual
            .iter()
            .filter(|candidate| candidate.branch == CandidateBranch::Shifted)
        {
            assert_eq!(
                shifted_value(candidate).least_denominator_exponent(),
                candidate.k
            );
        }

        for k in 0..=max_k {
            for branch in [CandidateBranch::Unshifted, CandidateBranch::Shifted] {
                let keys: Vec<_> = actual
                    .iter()
                    .filter(|candidate| candidate.k == k && candidate.branch == branch)
                    .map(|candidate| {
                        let value = shifted_value(candidate);
                        let transformed = upright.apply_zomega(value.numerator());
                        coset_order_key(&transformed)
                    })
                    .collect();
                assert!(keys.windows(2).all(|pair| pair[0] <= pair[1]));
            }
        }
    }

    #[test]
    fn candidate_stream_is_deterministic_and_branch_k_is_preserved() {
        let first = candidate_stream(0, 1, 2, 3, 96, 384).unwrap();
        let second = candidate_stream(0, 1, 2, 3, 96, 384).unwrap();
        assert_eq!(first, second);
        let set: BTreeSet<_> = first.iter().map(candidate_key).collect();
        assert_eq!(set.len(), first.len());
    }
}
