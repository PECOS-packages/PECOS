// Copyright 2025 The PECOS Developers
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

use core::convert::Infallible;
use rand_core::{Rng, SeedableRng, TryRng};
use std::fmt::{self, Debug};

/// A deterministic random number generator that cycles through a predefined sequence of values
///
/// This RNG is useful for testing scenarios where you want complete control over
/// the random numbers that will be generated in a specific sequence.
///
/// `ReplayingRng` pairs well with `RecordingRng` when recording and replaying
/// random sequences, making it ideal for deterministic testing.
///
/// # Example
///
/// ```
/// use pecos_core::rng::ReplayingRng;
/// use rand_core::Rng;
///
/// // Create an RNG with a predefined sequence
/// let mut rng = ReplayingRng::from_values(vec![42, 123, 7, 99]);
///
/// // Generated values will follow the sequence
/// assert_eq!(rng.next_u64(), 42);
/// assert_eq!(rng.next_u64(), 123);
/// assert_eq!(rng.next_u64(), 7);
/// assert_eq!(rng.next_u64(), 99);
/// // After reaching the end, it loops back to the beginning
/// assert_eq!(rng.next_u64(), 42);
/// ```
#[derive(Clone)]
pub struct ReplayingRng {
    /// The predefined sequence of values to return
    values: Vec<u64>,
    /// The current position in the sequence
    position: usize,
    /// Optional bytes for `fill_bytes` operation
    bytes: Option<Vec<u8>>,
}

impl ReplayingRng {
    /// Create a new `ReplayingRng` from a vector of u64 values
    ///
    /// # Arguments
    /// * `values` - The sequence of values the RNG will cycle through
    ///
    /// # Panics
    /// Panics if `values` is empty
    #[must_use]
    pub fn from_values(values: Vec<u64>) -> Self {
        assert!(
            !values.is_empty(),
            "ReplayingRng requires at least one value"
        );

        Self {
            values,
            position: 0,
            bytes: None,
        }
    }

    /// Create a new `ReplayingRng` from a vector of u64 values and bytes for `fill_bytes` operations
    ///
    /// # Arguments
    /// * `values` - The sequence of values the RNG will cycle through
    /// * `bytes` - The bytes to use for `fill_bytes` operations
    ///
    /// # Panics
    /// Panics if `values` is empty
    #[must_use]
    pub fn from_values_and_bytes(values: Vec<u64>, bytes: Vec<u8>) -> Self {
        assert!(
            !values.is_empty(),
            "ReplayingRng requires at least one value"
        );

        Self {
            values,
            position: 0,
            bytes: Some(bytes),
        }
    }

    /// Get the next value in the sequence and advance the position
    fn next_value(&mut self) -> u64 {
        let value = self.values[self.position];
        self.position = (self.position + 1) % self.values.len();
        value
    }
}

impl Debug for ReplayingRng {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Limit the output for large vectors
        const MAX_DISPLAY: usize = 4;

        write!(
            f,
            "ReplayingRng {{ position: {}/{}, values: [",
            self.position,
            self.values.len()
        )?;

        if self.values.is_empty() {
            write!(f, "]")?;
        } else {
            let display_len = std::cmp::min(self.values.len(), MAX_DISPLAY);

            for i in 0..display_len {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{}", self.values[i])?;
            }

            if self.values.len() > MAX_DISPLAY {
                write!(f, ", ... ({} more)]", self.values.len() - MAX_DISPLAY)?;
            } else {
                write!(f, "]")?;
            }
        }

        if let Some(bytes) = &self.bytes {
            write!(f, ", bytes: [{} bytes]", bytes.len())?;
        }

        write!(f, " }}")
    }
}

impl TryRng for ReplayingRng {
    type Error = Infallible;

    fn try_next_u32(&mut self) -> Result<u32, Self::Error> {
        // Get the next u64 value and truncate it to u32
        #[allow(clippy::cast_possible_truncation)]
        Ok((self.next_value() & 0xFFFF_FFFF) as u32)
    }

    fn try_next_u64(&mut self) -> Result<u64, Self::Error> {
        Ok(self.next_value())
    }

    fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), Self::Error> {
        if let Some(bytes) = &self.bytes {
            // Special handling when we have recorded bytes
            // The next two values will be start index and length
            if self.position + 1 < self.values.len() {
                let start = usize::try_from(self.values[self.position])
                    .unwrap_or_else(|_| panic!("start index value too large for platform"));
                let len = usize::try_from(self.values[self.position + 1])
                    .unwrap_or_else(|_| panic!("length value too large for platform"));
                self.position = (self.position + 2) % self.values.len();

                // Check if we have enough bytes
                if start + len <= bytes.len() && len <= dest.len() {
                    dest[..len].copy_from_slice(&bytes[start..start + len]);
                    return Ok(());
                }
            }
        }

        // Fall back to the original implementation if we can't use recorded bytes
        let mut i = 0;
        while i < dest.len() {
            let random_val = self.next_u32();
            let random_bytes = random_val.to_le_bytes();
            let remaining = dest.len() - i;
            let len = std::cmp::min(random_bytes.len(), remaining);
            dest[i..i + len].copy_from_slice(&random_bytes[..len]);
            i += len;
        }
        Ok(())
    }
}

impl SeedableRng for ReplayingRng {
    type Seed = [u8; 8];

    fn from_seed(seed: Self::Seed) -> Self {
        // Create a simple RNG with a sequence based on the seed
        let seed_val = u64::from_le_bytes(seed);

        // Create a somewhat interesting sequence from the seed
        let values = vec![
            seed_val,
            seed_val.wrapping_add(1),
            seed_val.wrapping_mul(3),
            seed_val.wrapping_add(5),
        ];

        Self {
            values,
            position: 0,
            bytes: None,
        }
    }

    fn seed_from_u64(seed: u64) -> Self {
        // Create a simple deterministic sequence from the seed
        Self::from_seed(seed.to_le_bytes())
    }

    fn from_rng<R: Rng + ?Sized>(rng: &mut R) -> Self {
        // Generate a small set of random values for our sequence
        let mut values = Vec::with_capacity(5);
        for _ in 0..5 {
            values.push(rng.next_u64());
        }

        Self {
            values,
            position: 0,
            bytes: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector_rng_cycles() {
        let values = vec![42, 123, 7];
        let mut rng = ReplayingRng::from_values(values.clone());

        // First pass through the sequence
        for &val in &values {
            assert_eq!(rng.next_u64(), val);
        }

        // Second pass should repeat
        for &val in &values {
            assert_eq!(rng.next_u64(), val);
        }
    }

    #[test]
    fn test_vector_rng_next_u32() {
        let values = vec![0x1122_3344_5566_7788, 0x99AA_BBCC_DDEE_FF00];
        let mut rng = ReplayingRng::from_values(values);

        assert_eq!(rng.next_u32(), 0x5566_7788);
        assert_eq!(rng.next_u32(), 0xDDEE_FF00);
    }

    #[test]
    fn test_seed_from_u64() {
        let seed = 42;
        let mut rng1 = ReplayingRng::seed_from_u64(seed);
        let mut rng2 = ReplayingRng::seed_from_u64(seed);

        // Both RNGs should produce the same sequence
        for _ in 0..10 {
            assert_eq!(rng1.next_u64(), rng2.next_u64());
        }

        // A different seed should produce different values
        let mut rng3 = ReplayingRng::seed_from_u64(seed + 1);
        assert_ne!(rng1.next_u64(), rng3.next_u64());
    }

    #[test]
    #[should_panic(expected = "ReplayingRng requires at least one value")]
    fn test_empty_vector_panics() {
        let _rng = ReplayingRng::from_values(vec![]);
    }

    #[test]
    fn test_fill_bytes_with_recorded() {
        let values = vec![0, 4, 4, 4]; // Start index, length, unused values
        let bytes = vec![10, 20, 30, 40, 50]; // The recorded bytes
        let mut rng = ReplayingRng::from_values_and_bytes(values, bytes);

        let mut buffer = [0u8; 10];
        rng.fill_bytes(&mut buffer[..4]); // Only fill the first 4 bytes

        assert_eq!(buffer, [10, 20, 30, 40, 0, 0, 0, 0, 0, 0]);
    }
}
