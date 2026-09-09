//! Exact number rings used by Clifford+T synthesis.

mod domega;
mod zomega;
mod zsqrt2;

pub use domega::DOmega;
pub use zomega::ZOmega;
pub use zsqrt2::ZSqrt2;

#[cfg(test)]
mod test_support {
    use std::fmt::Debug;
    use std::ops::{Add, Mul, Neg, Sub};

    use num_traits::{One, Zero};

    pub struct Lcg(u64);

    impl Lcg {
        pub const fn new(seed: u64) -> Self {
            Self(seed)
        }

        fn next_u64(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            self.0
        }

        pub fn next_i64(&mut self, radius: u64) -> i64 {
            let width = radius
                .checked_mul(2)
                .and_then(|value| value.checked_add(1))
                .expect("test sampling radius is too large");
            let value = self.next_u64() % width;
            i64::try_from(value).expect("sample fits in i64")
                - i64::try_from(radius).expect("radius fits in i64")
        }
    }

    pub fn assert_commutative_ring<T>(a: &T, b: &T, c: &T)
    where
        T: Add<Output = T>
            + Clone
            + Debug
            + Eq
            + Mul<Output = T>
            + Neg<Output = T>
            + One
            + Sub<Output = T>
            + Zero,
    {
        assert_eq!(
            (a.clone() + b.clone()) + c.clone(),
            a.clone() + (b.clone() + c.clone())
        );
        assert_eq!(a.clone() + b.clone(), b.clone() + a.clone());
        assert_eq!(
            (a.clone() * b.clone()) * c.clone(),
            a.clone() * (b.clone() * c.clone())
        );
        assert_eq!(a.clone() * b.clone(), b.clone() * a.clone());
        assert_eq!(
            a.clone() * (b.clone() + c.clone()),
            a.clone() * b.clone() + a.clone() * c.clone()
        );
        assert_eq!(a.clone() + (-a.clone()), T::zero());
        assert_eq!(a.clone() - a.clone(), T::zero());
        assert_eq!(a.clone() * T::one(), a.clone());
        assert_eq!(T::one() * a.clone(), a.clone());
    }
}
