//! Example of using the Chromobius decoder

fn main() -> Result<(), Box<dyn std::error::Error>> {
    use pecos_chromobius::{ChromobiusConfig, ChromobiusDecoder};

    println!("Chromobius decoder example");
    println!("=========================");

    // Create a simple detector error model with color/basis annotations
    // The 4th coordinate encodes color and basis:
    // 0: basis=X, color=R
    // 1: basis=X, color=G
    // 2: basis=X, color=B
    // 3: basis=Z, color=R
    // 4: basis=Z, color=G
    // 5: basis=Z, color=B
    let dem = r"
# Simple color code error model
error(0.1) D0 D1
error(0.1) D1 D2 L0
error(0.1) D2 D3
detector(0, 0, 0, 0) D0
detector(1, 0, 0, 1) D1
detector(2, 0, 0, 2) D2
detector(3, 0, 0, 0) D3
    "
    .trim();

    // Create decoder with default configuration
    let config = ChromobiusConfig::default();
    let mut decoder = ChromobiusDecoder::new(dem, config)?;

    println!("Created decoder with:");
    println!("  {} detectors", decoder.num_detectors());
    println!("  {} observables", decoder.num_observables());

    // Example 1: Decode some detection events
    println!("\nExample 1: Basic decoding");
    println!("-------------------------");

    // Create bit-packed detection events
    // For 4 detectors, we need 1 byte
    // Set detectors 0 and 1 as triggered
    let detection_events = vec![0b0000_0011_u8];

    let result = decoder.decode_detection_events(&detection_events)?;
    println!("Detection pattern: 0b{:08b}", detection_events[0]);
    println!("Predicted observables: 0x{:x}", result.observables);

    // Example 2: Decode with weight information
    println!("\nExample 2: Decoding with weight");
    println!("-------------------------------");

    // Different detection pattern
    let detection_events = vec![0b0000_0110_u8]; // Detectors 1 and 2

    let result = decoder.decode_detection_events_with_weight(&detection_events)?;
    println!("Detection pattern: 0b{:08b}", detection_events[0]);
    println!("Predicted observables: 0x{:x}", result.observables);
    println!("Solution weight: {:.3}", result.weight.unwrap());

    // Example 3: No detections (trivial case)
    println!("\nExample 3: No detections");
    println!("------------------------");

    let detection_events = vec![0b0000_0000_u8];
    let result = decoder.decode_detection_events(&detection_events)?;
    println!("Detection pattern: 0b{:08b}", detection_events[0]);
    println!("Predicted observables: 0x{:x}", result.observables);

    Ok(())
}
