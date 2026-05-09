// Simple profiling harness for the UF decoder.
// Run with: cargo run --release --example profile_decode -p pecos-uf-decoder

use pecos_decoder_core::dem::DemMatchingGraph;
use pecos_uf_decoder::{UfDecoder, UfDecoderConfig};
use std::time::Instant;

const D3_DEM: &str =
    include_str!("../../../examples/surface_code_circuits/surface_code_d3_z_stim.dem");
const D5_DEM: &str =
    include_str!("../../../examples/surface_code_circuits/surface_code_d5_z_stim.dem");

fn shots_as_f64(num_shots: usize) -> f64 {
    f64::from(u32::try_from(num_shots).expect("profile shot count fits in u32"))
}

fn profile_decoder(name: &str, dem: &str, num_shots: usize) {
    let graph = DemMatchingGraph::from_dem_str(dem).unwrap();
    let mut dec = UfDecoder::from_matching_graph(&graph, UfDecoderConfig::fast());
    let num_det = graph.num_detectors;

    // Generate random syndromes
    let mut rng = fastrand::Rng::with_seed(42);
    let syndromes: Vec<Vec<u8>> = (0..num_shots)
        .map(|_| (0..num_det).map(|_| u8::from(rng.f64() < 0.05)).collect())
        .collect();

    // Warm up
    for syn in &syndromes[..100.min(num_shots)] {
        let _ = dec.decode_syndrome(syn);
    }

    // Time the full batch
    let t0 = Instant::now();
    let mut errors = 0u64;
    for syn in &syndromes {
        let obs = dec.decode_syndrome(syn);
        errors += obs;
    }
    let elapsed = t0.elapsed();

    let shots = shots_as_f64(num_shots);
    let per_shot_ns = elapsed.as_secs_f64() * 1.0e9 / shots;
    let throughput = shots / elapsed.as_secs_f64();
    println!(
        "{name:8}: {num_det:3} det, {per_shot_ns:8.0} ns/shot ({:.0} kshots/s), errors={errors}",
        throughput / 1000.0
    );
}

fn profile_phases(name: &str, dem: &str, num_shots: usize) {
    let graph = DemMatchingGraph::from_dem_str(dem).unwrap();
    let num_det = graph.num_detectors;
    let num_edges = graph.edges.len();

    let mut rng = fastrand::Rng::with_seed(42);
    let syndromes: Vec<Vec<u8>> = (0..num_shots)
        .map(|_| (0..num_det).map(|_| u8::from(rng.f64() < 0.05)).collect())
        .collect();

    // Phase 1: measure reset + syndrome loading only
    let mut dec = UfDecoder::from_matching_graph(&graph, UfDecoderConfig::fast());
    let t0 = Instant::now();
    for syn in &syndromes {
        dec.syndrome_validate(syn); // reset + grow (no peel)
    }
    let grow_time = t0.elapsed();

    // Phase 2: measure full decode (reset + grow + peel)
    let t0 = Instant::now();
    for syn in &syndromes {
        let _ = dec.decode_syndrome(syn);
    }
    let total_time = t0.elapsed();

    let shots = shots_as_f64(num_shots);
    let grow_ns = grow_time.as_secs_f64() * 1.0e9 / shots;
    let total_ns = total_time.as_secs_f64() * 1.0e9 / shots;
    let peel_ns = total_ns - grow_ns;

    println!(
        "{name}: {num_det} det, {num_edges} edges | total {total_ns:.0} ns = grow {grow_ns:.0} ns ({:.0}%) + peel {peel_ns:.0} ns ({:.0}%)",
        grow_ns / total_ns * 100.0,
        peel_ns / total_ns * 100.0,
    );

    // Also profile BP+UF if available
    if let Ok(mut bp_dec) =
        pecos_uf_decoder::BpUfDecoder::from_dem(dem, pecos_uf_decoder::BpUfConfig::default())
    {
        use pecos_decoder_core::ObservableDecoder;
        let t0 = Instant::now();
        for syn in &syndromes {
            let _ = bp_dec.decode_to_observables(syn);
        }
        let bp_total = t0.elapsed();
        let bp_ns = bp_total.as_secs_f64() * 1.0e9 / shots;
        let bp_only = bp_ns - total_ns; // approximate BP overhead
        println!(
            "  BP+UF: {bp_ns:.0} ns/shot total (BP overhead ~{bp_only:.0} ns = {:.0}%)",
            bp_only / bp_ns * 100.0,
        );
    }
}

const D7_DEM: &str =
    include_str!("../../../examples/surface_code_circuits/surface_code_d7_z_stim.dem");

fn main() {
    let num_shots = 100_000;

    println!("=== UF Decoder Profiling ({num_shots} shots) ===");
    println!();

    profile_decoder("d3", D3_DEM, num_shots);
    profile_decoder("d5", D5_DEM, num_shots);
    profile_decoder("d7", D7_DEM, num_shots);

    println!();
    println!("=== Phase breakdown ===");
    profile_phases("d3", D3_DEM, num_shots);
    profile_phases("d5", D5_DEM, num_shots);
    profile_phases("d7", D7_DEM, num_shots);

    // Also profile with balanced config (Prim MST)
    println!();
    println!("=== Balanced (Prim MST) ===");
    let graph = DemMatchingGraph::from_dem_str(D5_DEM).unwrap();
    let mut dec = UfDecoder::from_matching_graph(&graph, UfDecoderConfig::balanced());
    let num_det = graph.num_detectors;

    let mut rng = fastrand::Rng::with_seed(42);
    let syndromes: Vec<Vec<u8>> = (0..num_shots)
        .map(|_| (0..num_det).map(|_| u8::from(rng.f64() < 0.05)).collect())
        .collect();

    let t0 = Instant::now();
    for syn in &syndromes {
        let _ = dec.decode_syndrome(syn);
    }
    let elapsed = t0.elapsed();
    let per_shot_ns = elapsed.as_secs_f64() * 1.0e9 / shots_as_f64(num_shots);
    println!("d5-bal : {num_det:3} det, {per_shot_ns:8.0} ns/shot");
}
