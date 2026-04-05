use pecos_core::QubitId;
use pecos_simulators::CliffordGateable;
use std::time::Instant;

#[allow(clippy::cast_precision_loss)] // profiling calculation
fn main() {
    type Sim = pecos_simulators::CHForm;
    type GateFn<S> = Box<dyn Fn(&mut S)>;

    let nq: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(1000);
    let iters: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(100);

    let mut sim = Sim::new(nq);
    for q in 0..nq {
        sim.h(&[QubitId(q)]);
    }
    if nq > 1 {
        sim.cx(&[(QubitId(0), QubitId(1))]);
    }

    let gates: Vec<(&str, GateFn<Sim>, usize)> = vec![
        // Single-qubit gates
        (
            "Z",
            Box::new(move |s: &mut Sim| {
                for q in 0..nq {
                    s.z(&[QubitId(q)]);
                }
            }),
            nq,
        ),
        (
            "S",
            Box::new(move |s: &mut Sim| {
                for q in 0..nq {
                    s.sz(&[QubitId(q)]);
                }
            }),
            nq,
        ),
        (
            "Sdg",
            Box::new(move |s: &mut Sim| {
                for q in 0..nq {
                    s.szdg(&[QubitId(q)]);
                }
            }),
            nq,
        ),
        (
            "H",
            Box::new(move |s: &mut Sim| {
                for q in 0..nq {
                    s.h(&[QubitId(q)]);
                }
            }),
            nq,
        ),
        (
            "X",
            Box::new(move |s: &mut Sim| {
                for q in 0..nq {
                    s.x(&[QubitId(q)]);
                }
            }),
            nq,
        ),
        (
            "Y",
            Box::new(move |s: &mut Sim| {
                for q in 0..nq {
                    s.y(&[QubitId(q)]);
                }
            }),
            nq,
        ),
        (
            "SX",
            Box::new(move |s: &mut Sim| {
                for q in 0..nq {
                    s.sx(&[QubitId(q)]);
                }
            }),
            nq,
        ),
        (
            "SXdg",
            Box::new(move |s: &mut Sim| {
                for q in 0..nq {
                    s.sxdg(&[QubitId(q)]);
                }
            }),
            nq,
        ),
        (
            "SY",
            Box::new(move |s: &mut Sim| {
                for q in 0..nq {
                    s.sy(&[QubitId(q)]);
                }
            }),
            nq,
        ),
        (
            "SYdg",
            Box::new(move |s: &mut Sim| {
                for q in 0..nq {
                    s.sydg(&[QubitId(q)]);
                }
            }),
            nq,
        ),
        // Two-qubit gates
        (
            "CX",
            Box::new(move |s: &mut Sim| {
                for q in (0..nq - 1).step_by(2) {
                    s.cx(&[(QubitId(q), QubitId(q + 1))]);
                }
            }),
            nq / 2,
        ),
        (
            "CZ",
            Box::new(move |s: &mut Sim| {
                for q in (0..nq - 1).step_by(2) {
                    s.cz(&[(QubitId(q), QubitId(q + 1))]);
                }
            }),
            nq / 2,
        ),
        (
            "CY",
            Box::new(move |s: &mut Sim| {
                for q in (0..nq - 1).step_by(2) {
                    s.cy(&[(QubitId(q), QubitId(q + 1))]);
                }
            }),
            nq / 2,
        ),
        (
            "SZZ",
            Box::new(move |s: &mut Sim| {
                for q in (0..nq - 1).step_by(2) {
                    s.szz(&[(QubitId(q), QubitId(q + 1))]);
                }
            }),
            nq / 2,
        ),
        (
            "SZZdg",
            Box::new(move |s: &mut Sim| {
                for q in (0..nq - 1).step_by(2) {
                    s.szzdg(&[(QubitId(q), QubitId(q + 1))]);
                }
            }),
            nq / 2,
        ),
        (
            "SXX",
            Box::new(move |s: &mut Sim| {
                for q in (0..nq - 1).step_by(2) {
                    s.sxx(&[(QubitId(q), QubitId(q + 1))]);
                }
            }),
            nq / 2,
        ),
        (
            "SYY",
            Box::new(move |s: &mut Sim| {
                for q in (0..nq - 1).step_by(2) {
                    s.syy(&[(QubitId(q), QubitId(q + 1))]);
                }
            }),
            nq / 2,
        ),
    ];

    eprintln!("Gate costs at n={nq} ({iters} iters):");
    for (name, gate_fn, ngates) in &gates {
        let t0 = Instant::now();
        for _ in 0..iters {
            gate_fn(&mut sim);
        }
        let total = t0.elapsed();
        #[allow(clippy::cast_possible_truncation)] // iters is small benchmark count
        let avg = total / iters as u32;
        eprintln!(
            "  {:>5} x{:<5}: {:>8.1?}  ({:>5.0}ns/gate)",
            name,
            ngates,
            avg,
            total.as_nanos() as f64 / (iters * ngates) as f64
        );
    }
}
