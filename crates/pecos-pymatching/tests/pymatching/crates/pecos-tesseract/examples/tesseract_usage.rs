//! Example of using the Tesseract decoder for quantum error correction

use ndarray::Array1;
use pecos_tesseract::{TesseractConfig, TesseractDecoder};

#[allow(clippy::too_many_lines)] // Example demonstrating various features
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Tesseract Decoder Example");
    println!("========================\n");

    // Example 1: Simple DEM with a few error mechanisms
    println!("Example 1: Simple error model");
    println!("----------------------------");

    let simple_dem = r"
error(0.1) D0 D1
error(0.05) D1 D2
error(0.02) D0 D2 L0
    ";

    let config = TesseractConfig::default();
    let mut decoder = TesseractDecoder::new(simple_dem, config)?;

    println!(
        "Created decoder with {} detectors and {} errors",
        decoder.num_detectors(),
        decoder.num_errors()
    );

    // Decode a simple detection pattern
    let detections = Array1::from_vec(vec![0, 1]); // Detectors 0 and 1 triggered
    let result = decoder.decode_detections(&detections.view())?;

    println!("Detection pattern: {detections:?}");
    println!("Predicted errors: {:?}", result.predicted_errors);
    println!("Observables mask: 0x{:x}", result.observables_mask);
    println!("Decoding cost: {:.3}", result.cost);
    println!("Low confidence: {}\n", result.low_confidence);

    // Example 2: Using optimized configuration for performance
    println!("Example 2: Performance-optimized configuration");
    println!("---------------------------------------------");

    let surface_code_dem = r"
error(0.001) D0 D1
error(0.001) D1 D2
error(0.001) D2 D3
error(0.001) D3 D0
error(0.0005) D0 D2 L0
error(0.0005) D1 D3 L0
    ";

    let fast_config = TesseractConfig::fast();
    println!(
        "Fast config - beam size: {}, beam climbing: {}",
        fast_config.det_beam, fast_config.beam_climbing
    );

    let mut fast_decoder = TesseractDecoder::new(surface_code_dem, fast_config)?;

    // Test multiple detection patterns
    let test_patterns = [vec![0], vec![0, 1], vec![0, 2], vec![1, 2, 3]];

    for (i, pattern) in test_patterns.iter().enumerate() {
        let detections = Array1::from_vec(pattern.clone());
        let result = fast_decoder.decode_detections(&detections.view())?;

        println!(
            "Pattern {}: {:?} -> errors: {:?}, cost: {:.3}",
            i + 1,
            pattern,
            result.predicted_errors.as_slice().unwrap(),
            result.cost
        );
    }

    // Example 3: Accuracy-focused configuration
    println!("\nExample 3: Accuracy-focused configuration");
    println!("----------------------------------------");

    let accurate_config = TesseractConfig::accurate();
    println!(
        "Accurate config - beam size: {}, beam climbing: {}",
        accurate_config.det_beam, accurate_config.beam_climbing
    );

    let mut accurate_decoder = TesseractDecoder::new(surface_code_dem, accurate_config)?;

    // Test the same patterns with accuracy-focused decoder
    for (i, pattern) in test_patterns.iter().enumerate() {
        let detections = Array1::from_vec(pattern.clone());
        let result = accurate_decoder.decode_detections(&detections.view())?;

        println!(
            "Pattern {}: {:?} -> errors: {:?}, cost: {:.3}",
            i + 1,
            pattern,
            result.predicted_errors.as_slice().unwrap(),
            result.cost
        );
    }

    // Example 4: Error analysis
    println!("\nExample 4: Error mechanism analysis");
    println!("----------------------------------");

    for i in 0..fast_decoder.num_errors() {
        if let Some(error_info) = fast_decoder.get_error_info(i) {
            println!(
                "Error {}: prob={:.4}, cost={:.3}, detectors={:?}, obs=0x{:x}",
                i,
                error_info.probability,
                error_info.cost,
                error_info.detectors,
                error_info.observables
            );
        }
    }

    // Example 5: Custom configuration
    println!("\nExample 5: Custom configuration");
    println!("------------------------------");

    let custom_config = TesseractConfig {
        det_beam: 50,
        beam_climbing: true,
        no_revisit_dets: false,
        at_most_two_errors_per_detector: true,
        verbose: false,
        pqlimit: 10000,
        det_penalty: 0.05,
    };

    let mut custom_decoder = TesseractDecoder::new(surface_code_dem, custom_config)?;

    let heavy_pattern = vec![0, 1, 2, 3];
    let detections = Array1::from_vec(heavy_pattern);
    let result = custom_decoder.decode_detections(&detections.view())?;

    println!("Heavy detection pattern: {detections:?}");
    println!(
        "Custom decoder result: errors={:?}, cost={:.3}",
        result.predicted_errors.as_slice().unwrap(),
        result.cost
    );

    // Show decoder configuration
    println!("\nDecoder configuration:");
    println!("  Detector beam: {}", custom_decoder.det_beam());
    println!("  Beam climbing: {}", custom_decoder.beam_climbing());
    println!(
        "  No revisit detectors: {}",
        custom_decoder.no_revisit_dets()
    );
    println!(
        "  At most two errors per detector: {}",
        custom_decoder.at_most_two_errors_per_detector()
    );
    println!("  Priority queue limit: {}", custom_decoder.pqlimit());
    println!("  Detector penalty: {:.3}", custom_decoder.det_penalty());

    Ok(())
}
