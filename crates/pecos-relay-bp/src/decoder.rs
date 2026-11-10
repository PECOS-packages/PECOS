//! Relay BP and min-sum BP decoder implementations

use crate::config::{MinSumConfig, RelayConfig};
use crate::errors::{RelayBpError, Result, native_message};
use ndarray::{Array1, ArrayView1, ArrayView2};
use pecos_bp::{BpGraph, RelayBp};
use pecos_decoder_core::DecoderError;

/// Decoding result from Relay BP / min-sum BP
#[derive(Debug, Clone)]
pub struct DecodingResult {
    /// The decoded error vector
    pub decoding: Array1<u8>,
    /// Whether the decoder converged
    pub converged: bool,
    /// Number of iterations used
    pub iterations: usize,
}

/// Relay BP ensemble decoder
///
/// A thin facade over PECOS native memory-BP. Results are a deterministic
/// function of (seed, syndrome): decode order, batch splitting and worker
/// count cannot change them.
pub struct RelayBpDecoder {
    inner: RelayBp,
    syndrome: Vec<u8>,
    num_checks: usize,
    num_bits: usize,
    alpha: Option<f64>,
    seed: u64,
}

impl RelayBpDecoder {
    /// Create a new Relay BP decoder from a dense check matrix
    ///
    /// # Errors
    ///
    /// Returns [`RelayBpError::InvalidMatrix`] if the check matrix is invalid
    /// or the error priors are invalid. Returns
    /// [`RelayBpError::Configuration`] if tuning parameters or explicit gammas are invalid.
    pub fn new(
        check_matrix: &ArrayView2<u8>,
        error_priors: &[f64],
        relay_config: &RelayConfig,
    ) -> Result<Self> {
        let graph = BpGraph::from_dense_check_matrix(*check_matrix, error_priors)
            .map_err(|error| RelayBpError::InvalidMatrix(native_message(error)))?;
        let inner = RelayBp::new(graph, relay_config.to_native(check_matrix.ncols())?)
            .map_err(|error| RelayBpError::Configuration(native_message(error)))?;
        Ok(Self {
            inner,
            syndrome: Vec::with_capacity(check_matrix.nrows()),
            num_checks: check_matrix.nrows(),
            num_bits: check_matrix.ncols(),
            alpha: relay_config.alpha,
            seed: relay_config.seed,
        })
    }

    /// Create a builder for configuring a new Relay BP decoder
    #[must_use]
    pub fn builder<'a>(check_matrix: &ArrayView2<'a, u8>) -> crate::builder::RelayBpBuilder<'a> {
        crate::builder::RelayBpBuilder::new(check_matrix)
    }

    /// Decode a syndrome vector
    ///
    /// # Errors
    ///
    /// Returns [`RelayBpError::InvalidSyndrome`] if the syndrome length doesn't
    /// match the number of checks, or an entry is not binary.
    pub fn decode(&mut self, syndrome: &ArrayView1<u8>) -> Result<DecodingResult> {
        copy_syndrome(syndrome, self.num_checks, &mut self.syndrome)?;
        let seed = syndrome_seed(self.seed, &self.syndrome);
        self.inner
            .decode(&self.syndrome, seed)
            .map(result)
            .map_err(decode_error)
    }

    /// Get the number of checks (rows in parity check matrix)
    #[must_use]
    pub fn check_count(&self) -> usize {
        self.num_checks
    }

    /// Get the number of bits (columns in parity check matrix)
    #[must_use]
    pub fn bit_count(&self) -> usize {
        self.num_bits
    }

    /// Get the optional min-sum scaling factor; `Some(0.0)` selects dynamic scaling.
    #[must_use]
    pub fn alpha(&self) -> Option<f64> {
        self.alpha
    }

    /// Get the random seed used for relay parameter sampling.
    #[must_use]
    pub fn seed(&self) -> u64 {
        self.seed
    }
}

/// Min-sum BP decoder
///
/// Runs one PECOS native memory-BP leg. With gamma0=None this is plain
/// min-sum belief propagation without memory or relay legs.
pub struct MinSumBpDecoder {
    inner: RelayBp,
    syndrome: Vec<u8>,
    num_checks: usize,
    num_bits: usize,
    alpha: Option<f64>,
    max_iter: usize,
}

impl MinSumBpDecoder {
    /// Create a new min-sum BP decoder from a dense check matrix
    ///
    /// # Errors
    ///
    /// Returns [`RelayBpError::InvalidMatrix`] for an invalid matrix or error priors,
    /// or [`RelayBpError::Configuration`] for invalid tuning parameters (including
    /// zero `max_iter`).
    pub fn new(check_matrix: &ArrayView2<u8>, config: &MinSumConfig) -> Result<Self> {
        let graph = BpGraph::from_dense_check_matrix(*check_matrix, &config.error_priors)
            .map_err(|error| RelayBpError::InvalidMatrix(native_message(error)))?;
        let inner = RelayBp::new(graph, config.to_native()?)
            .map_err(|error| RelayBpError::Configuration(native_message(error)))?;
        Ok(Self {
            inner,
            syndrome: Vec::with_capacity(check_matrix.nrows()),
            num_checks: check_matrix.nrows(),
            num_bits: check_matrix.ncols(),
            alpha: config.alpha,
            max_iter: config.max_iter,
        })
    }

    /// Create a builder for configuring a new min-sum BP decoder
    #[must_use]
    pub fn builder<'a>(check_matrix: &ArrayView2<'a, u8>) -> crate::builder::MinSumBpBuilder<'a> {
        crate::builder::MinSumBpBuilder::new(check_matrix)
    }

    /// Decode a syndrome vector
    ///
    /// # Errors
    ///
    /// Returns [`RelayBpError::InvalidSyndrome`] if the syndrome length doesn't
    /// match the number of checks, or an entry is not binary.
    pub fn decode(&mut self, syndrome: &ArrayView1<u8>) -> Result<DecodingResult> {
        copy_syndrome(syndrome, self.num_checks, &mut self.syndrome)?;
        self.inner
            .decode(&self.syndrome, 0)
            .map(result)
            .map_err(decode_error)
    }

    /// Get the number of checks (rows in parity check matrix)
    #[must_use]
    pub fn check_count(&self) -> usize {
        self.num_checks
    }

    /// Get the number of bits (columns in parity check matrix)
    #[must_use]
    pub fn bit_count(&self) -> usize {
        self.num_bits
    }

    /// Get the maximum number of BP iterations.
    #[must_use]
    pub fn max_iter(&self) -> usize {
        self.max_iter
    }

    /// Get the optional min-sum scaling factor; `Some(0.0)` selects dynamic scaling.
    #[must_use]
    pub fn alpha(&self) -> Option<f64> {
        self.alpha
    }
}

fn result(outcome: pecos_bp::RelayOutcome) -> DecodingResult {
    DecodingResult {
        decoding: Array1::from(outcome.correction),
        converged: outcome.converged,
        iterations: outcome.total_iterations,
    }
}

fn decode_error(error: DecoderError) -> RelayBpError {
    match error {
        DecoderError::InvalidSyndrome(message) => RelayBpError::InvalidSyndrome(message),
        DecoderError::InvalidDimensions { expected, actual } => RelayBpError::InvalidSyndrome(
            format!("syndrome length {actual} does not match check count {expected}"),
        ),
        error => RelayBpError::DecodingFailed(native_message(error)),
    }
}

fn copy_syndrome(view: &ArrayView1<u8>, checks: usize, buffer: &mut Vec<u8>) -> Result<()> {
    if view.len() != checks {
        return Err(RelayBpError::InvalidSyndrome(format!(
            "syndrome length {} does not match check count {checks}",
            view.len()
        )));
    }
    buffer.clear();
    for (index, &value) in view.iter().enumerate() {
        if value > 1 {
            return Err(RelayBpError::InvalidSyndrome(format!(
                "syndrome[{index}] must be binary, got {value}"
            )));
        }
        buffer.push(value);
    }
    Ok(())
}

/// `SplitMix64` finalizer fold over the seed, length and little-endian bit words.
fn syndrome_seed(seed: u64, syndrome: &[u8]) -> u64 {
    fn mix(mut z: u64) -> u64 {
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    const G: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut h = mix(seed.wrapping_add(G));
    h = mix(h ^ syndrome.len() as u64);
    for chunk in syndrome.chunks(64) {
        let word = chunk
            .iter()
            .enumerate()
            .fold(0, |word, (bit, &value)| word | (u64::from(value) << bit));
        h = mix(h.wrapping_add(G) ^ word);
    }
    h
}

#[cfg(test)]
mod tests {
    use super::syndrome_seed;

    #[test]
    fn facade_passes_the_syndrome_seed_to_random_relay_legs() {
        use super::RelayBpDecoder;
        use pecos_bp::{Alpha, BpGraph, RelayBp, Schedule, StopRule};
        let matrix = ndarray::arr2(&[
            [1, 0, 1, 0, 1, 0, 1],
            [0, 1, 1, 0, 0, 1, 1],
            [0, 0, 0, 1, 1, 1, 1],
        ]);
        let syndrome = ndarray::arr1(&[0, 1, 0]);
        let seed = 31;
        let mut facade = RelayBpDecoder::builder(&matrix.view())
            .error_priors(&[0.05; 7])
            .pre_iter(1)
            .num_sets(3)
            .set_max_iter(2)
            .gamma0(Some(0.65))
            .seed(seed)
            .build()
            .unwrap();
        // Configure the native oracle independently of facade conversion.
        let graph = BpGraph::from_dense_check_matrix(matrix.view(), &[0.05; 7]).unwrap();
        let mut native = RelayBp::new(
            graph,
            pecos_bp::RelayConfig {
                schedule: Schedule::Flooding,
                alpha: Alpha::Constant(1.0),
                gamma0: 0.65,
                pre_iterations: 1,
                num_legs: 3,
                leg_iterations: 2,
                gamma_range: (-0.24, 0.66),
                stop_rule: StopRule::Converged(1),
                explicit_gammas: None,
            },
        )
        .unwrap();
        let mixed = native
            .decode(&[0, 1, 0], syndrome_seed(seed, &[0, 1, 0]))
            .unwrap();
        let raw = native.decode(&[0, 1, 0], seed).unwrap();
        let actual = facade.decode(&syndrome.view()).unwrap();
        assert_eq!(actual.decoding.to_vec(), mixed.correction);
        assert_eq!(actual.converged, mixed.converged);
        assert_eq!(actual.iterations, mixed.total_iterations);
        assert!(actual.iterations > 1, "the shot must reach relay legs");
        assert_ne!(mixed.correction, raw.correction);
        assert!(mixed.converged);
        assert!(!raw.converged);
    }

    #[test]
    fn syndrome_seed_golden_vectors() {
        for (seed, syndrome, expected) in [
            (0, vec![0], 0x85c6_1a30_0ec7_0fa1),
            (0, vec![0, 0], 0x5b79_f476_df1f_fa2a),
            (0, vec![1], 0xd0b1_009a_d9ae_d4b2),
            (1, vec![1], 0x90be_2113_79f9_f704),
            (0, vec![1; 63], 0x22f3_fa79_4c04_a8f0),
            (0, vec![1; 64], 0x6fa9_c150_2903_aa2a),
            (0, vec![1; 65], 0xb8d5_2b7d_a6e5_897a),
            (
                42,
                (0..1008).map(|i| u8::from(i % 3 == 0)).collect(),
                0xc1d5_4f77_7bc2_3df3,
            ),
        ] {
            assert_eq!(syndrome_seed(seed, &syndrome), expected);
        }
    }

    #[test]
    fn syndrome_seed_uses_every_bit_length_and_seed() {
        let mut syndrome = vec![0; 130];
        let baseline = syndrome_seed(0, &syndrome);
        for bit in 0..130 {
            syndrome[bit] = 1;
            assert_ne!(syndrome_seed(0, &syndrome), baseline, "bit {bit}");
            syndrome[bit] = 0;
        }
        assert_ne!(syndrome_seed(0, &[0]), syndrome_seed(0, &[0, 0]));
        assert_ne!(baseline, syndrome_seed(1, &syndrome));
    }
}
