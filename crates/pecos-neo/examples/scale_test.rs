use pecos_neo::noise::flow::prelude::*;
use pecos_neo::noise::flow::channel::FlowChannel;
use pecos_neo::noise::{NoiseChannel, NoiseContext, NoiseEvent};
use pecos_neo::GateType;
use pecos_core::{QubitId, Angle64};
use pecos_rng::PecosRng;
use std::time::Instant;

fn bench_scale(num_qubits: usize, prob: f64, iterations: usize) {
    let qubits: Vec<QubitId> = (0..num_qubits).map(QubitId).collect();
    let angles: Vec<Angle64> = vec![];
    
    let channel = FlowChannel::new("test", pauli())
        .with_probability(prob)
        .with_filter(FlowEventFilter::SingleQubitGate);
    
    let event = NoiseEvent::AfterGate {
        gate_type: GateType::H,
        qubits: &qubits,
        angles: &angles,
    gate_id: None, };
    
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
    println!("{:>12} qubits, p={:.0e}: {:>12.1} ns/iter (~{:>5} events)", 
             num_qubits, prob, per_iter, expected_events);
}

fn main() {
    println!("=== FlowChannel.with_probability() Scale Test ===\n");
    
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
