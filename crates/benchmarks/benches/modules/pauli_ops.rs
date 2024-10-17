use criterion::{black_box, measurement::Measurement, BenchmarkGroup, Criterion};
use pecos_core::{IndexableElement, Set, VecSet};
use pecos_qsims::{CliffordSimulator, PauliProp};

pub fn benchmarks<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("Pauli Operations");
    bench_pauli_prop_init::<VecSet<usize>, usize, M>(&mut group, "usize");
    bench_pauli_prop_init::<VecSet<u16>, u16, M>(&mut group, "u16");
    group.finish();
}

fn bench_pauli_prop_init<T, E, M: Measurement>(group: &mut BenchmarkGroup<M>, type_name: &str)
where
    T: for<'a> Set<'a, Element = E>,
    E: IndexableElement + From<u16>,
{
    group.bench_function(format!("pauli_prop_init_{type_name}"), |b| {
        b.iter(|| {
            let mut pauli = PauliProp::<T, E>::new(100);
            for i in 0..100_u16 {
                pauli.insert_x(E::from(i));
                pauli.insert_z(E::from(i));
            }
            black_box(pauli)
        });
    });
}
