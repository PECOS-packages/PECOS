//! Monte Carlo simulations for statistical validation of decoder performance

use ndarray::{Array1, Array2};
use pecos_ldpc_decoders::*;
use rand::RngExt;

/// Run a Monte Carlo simulation for a decoder
#[allow(clippy::cast_precision_loss)] // Acceptable for Monte Carlo statistics
fn monte_carlo_simulation<F>(
    pcm: &Array2<u8>,
    sparse_pcm: &SparseMatrix,
    error_rate: f64,
    num_trials: usize,
    mut create_decoder: F,
) -> MonteCarloResult
where
    F: FnMut(&SparseMatrix) -> Box<dyn DecoderTrait>,
{
    let n = pcm.ncols();
    let mut rng = rand::rng();

    let mut total_errors = 0;
    let mut decoder_failures = 0;
    let mut total_iterations = 0;

    for _ in 0..num_trials {
        // Generate random error
        let mut error = Array1::zeros(n);
        let mut error_weight = 0;
        for i in 0..n {
            if rng.random::<f64>() < error_rate {
                error[i] = 1;
                error_weight += 1;
            }
        }
        total_errors += error_weight;

        // Calculate syndrome
        let syndrome = pcm.dot(&error).mapv(|x| x % 2);

        // Decode
        let mut decoder = create_decoder(sparse_pcm);
        let result = decoder.decode(&syndrome.view()).unwrap();
        total_iterations += result.iterations;

        // Check if decoding is correct
        let decoded_syndrome = pcm.dot(&result.decoding).mapv(|x| x % 2);
        if decoded_syndrome != syndrome {
            decoder_failures += 1;
        }
    }

    // For Monte Carlo simulations, casting to f64 is acceptable as we need floating point division
    let num_trials_f64 = num_trials as f64;

    MonteCarloResult {
        failure_rate: f64::from(decoder_failures) / num_trials_f64,
        avg_iterations: total_iterations as f64 / num_trials_f64,
        avg_error_weight: f64::from(total_errors) / num_trials_f64,
    }
}

#[derive(Debug)]
struct MonteCarloResult {
    failure_rate: f64,
    avg_iterations: f64,
    avg_error_weight: f64,
}

/// Create a simple repetition code
fn repetition_code(n: usize) -> (Array2<u8>, SparseMatrix) {
    let mut pcm = Array2::zeros((n - 1, n));
    for i in 0..n - 1 {
        pcm[[i, i]] = 1;
        pcm[[i, i + 1]] = 1;
    }
    let sparse = SparseMatrix::from_dense(&pcm.view());
    (pcm, sparse)
}

#[test]
fn test_bp_osd_error_rate_curve() {
    let (pcm, sparse_pcm) = repetition_code(20);
    let error_rates = vec![0.01, 0.05, 0.1];
    let num_trials = 1000;

    println!("\nBP+OSD Monte Carlo Results (Rep(20)):");
    println!("Error Rate | Failure Rate | Avg Iterations");
    println!("-----------|--------------|---------------");

    for &error_rate in &error_rates {
        let result = monte_carlo_simulation(&pcm, &sparse_pcm, error_rate, num_trials, |sparse| {
            Box::new(
                BpOsdDecoder::new(
                    sparse,
                    Some(error_rate),
                    None,
                    50,
                    BpMethod::ProductSum,
                    BpSchedule::Parallel,
                    0.625,
                    OsdMethod::OsdCs,
                    15,
                    InputVectorType::Syndrome,
                    None,
                    None,
                    None,
                )
                .unwrap(),
            )
        });

        println!(
            "{:10.3} | {:12.4} | {:14.2}",
            error_rate, result.failure_rate, result.avg_iterations
        );

        // Basic sanity check - decoder should work well at low error rates
        if error_rate <= 0.05 {
            assert!(
                result.failure_rate < 0.01,
                "BP+OSD failure rate too high at error rate {}: {}",
                error_rate,
                result.failure_rate
            );
        }
    }
}

#[test]
fn test_decoder_comparison() {
    let (pcm, sparse_pcm) = repetition_code(15);
    let error_rate = 0.05;
    let num_trials = 500;

    // Compare different decoders
    let bp_osd_result =
        monte_carlo_simulation(&pcm, &sparse_pcm, error_rate, num_trials, |sparse| {
            Box::new(
                BpOsdDecoder::new(
                    sparse,
                    Some(error_rate),
                    None,
                    30,
                    BpMethod::ProductSum,
                    BpSchedule::Parallel,
                    0.625,
                    OsdMethod::OsdCs,
                    10,
                    InputVectorType::Syndrome,
                    None,
                    None,
                    None,
                )
                .unwrap(),
            )
        });

    let lsd_decoder_result =
        monte_carlo_simulation(&pcm, &sparse_pcm, error_rate, num_trials, |sparse| {
            Box::new(
                BpLsdDecoder::new(
                    sparse,
                    Some(error_rate),
                    None,
                    30,
                    BpMethod::ProductSum,
                    BpSchedule::Parallel,
                    0.625,
                    OsdMethod::OsdCs,
                    10,
                    1,
                    InputVectorType::Syndrome,
                    None,
                    None,
                    None,
                )
                .unwrap(),
            )
        });

    let flip_result = monte_carlo_simulation(&pcm, &sparse_pcm, error_rate, num_trials, |sparse| {
        Box::new(FlipDecoder::new(sparse, 100, 10, 42).unwrap())
    });

    println!("\nDecoder Comparison at {error_rate} error rate:");
    println!("Decoder  | Failure Rate | Avg Iterations");
    println!("---------|--------------|---------------");
    println!(
        "BP+OSD   | {:12.4} | {:14.2}",
        bp_osd_result.failure_rate, bp_osd_result.avg_iterations
    );
    println!(
        "BP+LSD   | {:12.4} | {:14.2}",
        lsd_decoder_result.failure_rate, lsd_decoder_result.avg_iterations
    );
    println!(
        "Flip     | {:12.4} | {:14.2}",
        flip_result.failure_rate, flip_result.avg_iterations
    );

    // BP-based decoders should outperform simple flip decoder
    assert!(
        bp_osd_result.failure_rate <= flip_result.failure_rate,
        "BP+OSD should perform at least as well as Flip decoder"
    );
}

#[test]
fn test_medium_code_performance() {
    // Use a medium-sized code that runs faster
    let n = 40;
    let m = 20;
    let mut rng = rand::rng();
    let mut pcm = Array2::zeros((m, n));

    // Create a more structured LDPC code for better performance
    for i in 0..m {
        // Each check connects to exactly 3 bits (regular LDPC)
        let mut connected = std::collections::BTreeSet::new();
        while connected.len() < 3 {
            let j = rng.random_range(0..n);
            if connected.insert(j) {
                pcm[[i, j]] = 1;
            }
        }
    }

    let sparse_pcm = SparseMatrix::from_dense(&pcm.view());
    let error_rate = 0.02;
    let num_trials = 50; // Reduced trials for faster execution

    let result = monte_carlo_simulation(&pcm, &sparse_pcm, error_rate, num_trials, |sparse| {
        Box::new(
            BpOsdDecoder::new(
                sparse,
                Some(error_rate),
                None,
                50, // Reduced iterations
                BpMethod::MinimumSum,
                BpSchedule::Parallel,
                0.625,
                OsdMethod::OsdCs,
                3, // Reduced OSD order
                InputVectorType::Syndrome,
                None,
                None,
                None,
            )
            .unwrap(),
        )
    });

    println!("\nMedium code ({m} x {n}) performance:");
    println!("Failure rate: {:.4}", result.failure_rate);
    println!("Avg iterations: {:.2}", result.avg_iterations);
    println!("Avg error weight: {:.2}", result.avg_error_weight);

    // Ensure decoder is working reasonably well
    assert!(
        result.failure_rate < 0.1,
        "Failure rate too high: {}",
        result.failure_rate
    );
}

#[test]
#[ignore = "Run with --ignored for extensive performance testing"]
fn test_large_code_performance_extensive() {
    // Keep the original large test as an ignored extensive test
    let n = 100;
    let m = 50;
    let mut rng = rand::rng();
    let mut pcm = Array2::zeros((m, n));

    // Create a random sparse matrix
    for i in 0..m {
        for _ in 0..4 {
            let j = rng.random_range(0..n);
            pcm[[i, j]] = 1;
        }
    }

    let sparse_pcm = SparseMatrix::from_dense(&pcm.view());
    let error_rate = 0.02;
    let num_trials = 100;

    let result = monte_carlo_simulation(&pcm, &sparse_pcm, error_rate, num_trials, |sparse| {
        Box::new(
            BpOsdDecoder::new(
                sparse,
                Some(error_rate),
                None,
                100,
                BpMethod::MinimumSum,
                BpSchedule::Parallel,
                0.625,
                OsdMethod::OsdCs,
                5,
                InputVectorType::Syndrome,
                None,
                None,
                None,
            )
            .unwrap(),
        )
    });

    println!("\nLarge code ({m} x {n}) extensive performance:");
    println!("Failure rate: {:.4}", result.failure_rate);
    println!("Avg iterations: {:.2}", result.avg_iterations);
    println!("Avg error weight: {:.2}", result.avg_error_weight);
}

// Trait for generic decoder testing
trait DecoderTrait {
    fn decode(&mut self, syndrome: &ndarray::ArrayView1<u8>) -> Result<DecodingResult>;
}

impl DecoderTrait for BpOsdDecoder {
    fn decode(&mut self, syndrome: &ndarray::ArrayView1<u8>) -> Result<DecodingResult> {
        self.decode(syndrome)
    }
}

impl DecoderTrait for BpLsdDecoder {
    fn decode(&mut self, syndrome: &ndarray::ArrayView1<u8>) -> Result<DecodingResult> {
        self.decode(syndrome)
    }
}

impl DecoderTrait for FlipDecoder {
    fn decode(&mut self, syndrome: &ndarray::ArrayView1<u8>) -> Result<DecodingResult> {
        self.decode(syndrome)
    }
}

impl DecoderTrait for BeliefFindDecoder {
    fn decode(&mut self, syndrome: &ndarray::ArrayView1<u8>) -> Result<DecodingResult> {
        self.decode(syndrome)
    }
}
