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

use rand::{RngCore, SeedableRng};
use std::fmt::{self, Debug};
use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::Path;

/// A wrapper RNG that records all calls to the underlying RNG methods
///
/// This wrapper allows capturing raw random values from any RNG that implements
/// the `RngCore` trait, making it possible to replay the exact same sequence later
/// using a `ReplayingRng`.
///
/// # Example
///
/// ```
/// use pecos_core::rng::{RecordingRng, ReplayingRng};
/// use pecos_rng::{PecosRng, Rng, SeedableRng};
///
/// // Create a recording wrapper around PecosRng
/// let rng = PecosRng::seed_from_u64(42);
/// let mut recording_rng = RecordingRng::new(rng);
///
/// // Generate some random values
/// let float = recording_rng.random::<f64>();
/// let int = recording_rng.random_range(1..100);
///
/// // Extract the recorded values
/// let recorded_values = recording_rng.take_recorded_values();
///
/// // Later, replay the exact same sequence
/// let mut replay_rng = ReplayingRng::from_values(recorded_values);
/// let replay_float = replay_rng.random::<f64>();
/// let replay_int = replay_rng.random_range(1..100);
///
/// assert_eq!(float, replay_float);
/// assert_eq!(int, replay_int);
/// ```
pub struct RecordingRng<R: RngCore> {
    /// The underlying RNG being wrapped
    inner: R,
    /// The recorded raw values from all RNG method calls
    recorded_values: Vec<u64>,
    /// The recorded bytes from `fill_bytes` calls
    recorded_bytes: Vec<u8>,
}

impl<R: RngCore> RecordingRng<R> {
    /// Create a new `RecordingRng` wrapping the provided RNG
    pub fn new(inner: R) -> Self {
        Self {
            inner,
            recorded_values: Vec::new(),
            recorded_bytes: Vec::new(),
        }
    }

    /// Create a new `RecordingRng` with a pre-allocated capacity
    pub fn with_capacity(inner: R, capacity: usize) -> Self {
        Self {
            inner,
            recorded_values: Vec::with_capacity(capacity),
            recorded_bytes: Vec::new(),
        }
    }

    /// Get a reference to the recorded values
    pub fn recorded_values(&self) -> &[u64] {
        &self.recorded_values
    }

    /// Get a reference to the recorded bytes
    pub fn recorded_bytes(&self) -> &[u8] {
        &self.recorded_bytes
    }

    /// Extract the recorded values, consuming the `RecordingRng`
    pub fn take_recorded_values(self) -> Vec<u64> {
        self.recorded_values
    }

    /// Extract the recorded values and bytes, consuming the `RecordingRng`
    pub fn take_all_recordings(self) -> (Vec<u64>, Vec<u8>) {
        (self.recorded_values, self.recorded_bytes)
    }

    /// Save the recorded values to a file
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be written
    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> io::Result<()> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);

        // Write the number of values as a header
        let len = self.recorded_values.len() as u64;
        writer.write_all(&len.to_le_bytes())?;

        // Write each value
        for value in &self.recorded_values {
            writer.write_all(&value.to_le_bytes())?;
        }

        // Write the number of bytes
        let bytes_len = self.recorded_bytes.len() as u64;
        writer.write_all(&bytes_len.to_le_bytes())?;

        // Write the bytes
        writer.write_all(&self.recorded_bytes)?;

        Ok(())
    }

    /// Load recorded values from a file
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read or has an invalid format
    pub fn load_from_file<P: AsRef<Path>>(path: P) -> io::Result<(Vec<u64>, Vec<u8>)> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);

        // Read the number of values
        let mut len_bytes = [0u8; 8];
        reader.read_exact(&mut len_bytes)?;
        let len = usize::try_from(u64::from_le_bytes(len_bytes)).map_err(|_| {
            io::Error::new(io::ErrorKind::InvalidData, "length too large for platform")
        })?;

        // Read each value
        let mut values = Vec::with_capacity(len);
        for _ in 0..len {
            let mut value_bytes = [0u8; 8];
            reader.read_exact(&mut value_bytes)?;
            values.push(u64::from_le_bytes(value_bytes));
        }

        // Read the number of bytes
        let mut bytes_len_bytes = [0u8; 8];
        reader.read_exact(&mut bytes_len_bytes)?;
        let bytes_len = usize::try_from(u64::from_le_bytes(bytes_len_bytes)).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "bytes length too large for platform",
            )
        })?;

        // Read the bytes
        let mut bytes = vec![0; bytes_len];
        reader.read_exact(&mut bytes)?;

        Ok((values, bytes))
    }

    /// Get a reference to the inner RNG
    pub fn inner(&self) -> &R {
        &self.inner
    }

    /// Get a mutable reference to the inner RNG
    pub fn inner_mut(&mut self) -> &mut R {
        &mut self.inner
    }

    /// Unwrap the `RecordingRng`, returning the inner RNG and recorded values
    pub fn into_inner(self) -> (R, Vec<u64>, Vec<u8>) {
        (self.inner, self.recorded_values, self.recorded_bytes)
    }
}

impl<R: RngCore> RngCore for RecordingRng<R> {
    fn next_u32(&mut self) -> u32 {
        let value = self.inner.next_u32();
        self.recorded_values.push(u64::from(value));
        value
    }

    fn next_u64(&mut self) -> u64 {
        let value = self.inner.next_u64();
        self.recorded_values.push(value);
        value
    }

    fn fill_bytes(&mut self, dest: &mut [u8]) {
        // Fill the bytes using the inner RNG
        self.inner.fill_bytes(dest);

        // Record the actual bytes for exact replay
        let start = self.recorded_bytes.len();
        self.recorded_bytes.extend_from_slice(dest);

        // Also record the byte range in the u64 values (start and length)
        // This allows ReplayingRng to know which portion of recorded_bytes to use
        self.recorded_values
            .push(u64::try_from(start).expect("recorded_bytes length exceeds u64 capacity"));
        self.recorded_values.push(
            u64::try_from(dest.len()).expect("destination buffer length exceeds u64 capacity"),
        );
    }
}

impl<R: Debug + RngCore> Debug for RecordingRng<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RecordingRng")
            .field("inner", &self.inner)
            .field(
                "recorded_values",
                &format!("[{} items]", self.recorded_values.len()),
            )
            .field(
                "recorded_bytes",
                &format!("[{} bytes]", self.recorded_bytes.len()),
            )
            .finish()
    }
}

// If the inner RNG is seedable, delegate to it
impl<R: SeedableRng + RngCore> SeedableRng for RecordingRng<R> {
    type Seed = R::Seed;

    fn from_seed(seed: Self::Seed) -> Self {
        Self {
            inner: R::from_seed(seed),
            recorded_values: Vec::new(),
            recorded_bytes: Vec::new(),
        }
    }

    fn seed_from_u64(seed: u64) -> Self {
        Self {
            inner: R::seed_from_u64(seed),
            recorded_values: Vec::new(),
            recorded_bytes: Vec::new(),
        }
    }

    fn from_rng(rng: &mut impl RngCore) -> Self {
        Self {
            inner: R::from_rng(rng),
            recorded_values: Vec::new(),
            recorded_bytes: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::ReplayingRng;
    use rand::SeedableRng;
    use rand_xoshiro::Xoshiro256PlusPlus;

    #[test]
    fn test_recording_basic() {
        let base_rng = Xoshiro256PlusPlus::seed_from_u64(42);
        let mut recording_rng = RecordingRng::new(base_rng);

        // Generate some values
        let _ = recording_rng.next_u32();
        let _ = recording_rng.next_u64();

        // Check that values were recorded
        assert_eq!(recording_rng.recorded_values().len(), 2);
    }

    #[test]
    fn test_recording_and_replay() {
        let base_rng = Xoshiro256PlusPlus::seed_from_u64(42);
        let mut recording_rng = RecordingRng::new(base_rng);

        // Generate a sequence of values
        let val1 = recording_rng.next_u32();
        let val2 = recording_rng.next_u64();
        let val3 = recording_rng.next_u32();

        // Get the recorded values
        let recorded_values = recording_rng.recorded_values().to_vec();

        // Create a ReplayingRng for replay
        let mut replay_rng = ReplayingRng::from_values(recorded_values);

        // Replay the sequence
        assert_eq!(replay_rng.next_u32(), val1);
        assert_eq!(replay_rng.next_u64(), val2);
        assert_eq!(replay_rng.next_u32(), val3);
    }

    #[test]
    fn test_seedable() {
        // Test the SeedableRng implementation
        let mut rng1 = RecordingRng::<Xoshiro256PlusPlus>::seed_from_u64(42);
        let mut rng2 = RecordingRng::<Xoshiro256PlusPlus>::seed_from_u64(42);

        // Both RNGs should produce the same sequence
        assert_eq!(rng1.next_u64(), rng2.next_u64());
    }

    #[test]
    fn test_fill_bytes() {
        let base_rng = Xoshiro256PlusPlus::seed_from_u64(42);
        let mut recording_rng = RecordingRng::new(base_rng);

        // Create a buffer and fill it with random bytes
        let mut buffer = [0u8; 16];
        recording_rng.fill_bytes(&mut buffer);

        // Get the recorded values and bytes
        let recorded_values = recording_rng.recorded_values().to_vec();
        let recorded_bytes = recording_rng.recorded_bytes().to_vec();

        // Replay using ReplayingRng
        let mut replay_rng = ReplayingRng::from_values_and_bytes(recorded_values, recorded_bytes);

        // Create a new buffer for the replay
        let mut replay_buffer = [0u8; 16];
        replay_rng.fill_bytes(&mut replay_buffer);

        // This is a simplified version of what ReplayingRng would do
        assert_eq!(buffer, replay_buffer);
    }
}
