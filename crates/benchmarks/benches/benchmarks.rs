use criterion::{criterion_group, criterion_main, Criterion};

mod modules {
    pub mod element_ops;
    // TODO: pub mod hadamard_ops;
    // TODO: pub mod pauli_ops;
    pub mod set_ops;
}

use modules::{element_ops, set_ops};

fn all_benchmarks(c: &mut Criterion) {
    element_ops::benchmarks(c);
    set_ops::benchmarks(c);
    // TODO: pauli_ops::benchmarks(c);
    // TODO: hadamard_ops::benchmarks(c);
}

criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(100).measurement_time(core::time::Duration::from_secs(10));
    targets = all_benchmarks
}
criterion_main!(benches);
