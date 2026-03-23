use pecos_core::{Angle64, QubitId};
use pecos_neo::GateType;
use pecos_neo::noise::composite::channel::CompositeChannel;
use pecos_neo::noise::composite::prelude::*;
use pecos_neo::noise::{NoiseChannel, NoiseContext, NoiseEvent};
use pecos_random::PecosRng;
use std::time::Instant;

fn bench_scale(num_qubits: usize, prob: f64, iterations: usize) {
    let qubits: Vec<QubitId> = (0..num_qubits).map(QubitId).collect();
    let angles: Vec<Angle64> = vec![];

    let channel = CompositeChannel::new("test", pauli())
        .with_probability(prob)
        .with_filter(CompositeEventFilter::SingleQubitGate);

    let event = NoiseEvent::AfterGate {
        gate_type: GateType::H,
        qubits: &qubits,
        angles: &angles,
        gate_id: None,
    };

    let mut ctx = NoiseContext::new();
    let mut rng = PecosRng::seed_from_u64(42);

    // Warmup
    for _ in 0..10 {
        let _ = channel.apply(&event, &mut ctx, &mut rng);
    }

    let start = Instant::now();
    for _ in 0..iterations {
        let _ = channel.apply(&event, &mut ctx, &mut rng);
    }
    let elapsed = start.elapsed();

    let per_iter = elapsed.as_nanos() as f64 / iterations as f64;
    let expected_events = (num_qubits as f64 * prob) as usize;
    println!(
        "{num_qubits:>12} qubits, p={prob:.0e}: {per_iter:>12.1} ns/iter (~{expected_events:>5} events)"
    );
}

fn main() {
    println!("=== CompositeChannel.with_probability() Scale Test ===\n");

    println!("At p=1e-4:");
    bench_scale(100_000, 1e-4, 10000);
    bench_scale(1_000_000, 1e-4, 1000);
    bench_scale(10_000_000, 1e-4, 100);

    println!("\nAt p=1e-5:");
    bench_scale(1_000_000, 1e-5, 10000);
    bench_scale(10_000_000, 1e-5, 1000);
    bench_scale(100_000_000, 1e-5, 100);

    println!("\nAt p=1e-6:");
    bench_scale(10_000_000, 1e-6, 1000);
    bench_scale(100_000_000, 1e-6, 100);
}
