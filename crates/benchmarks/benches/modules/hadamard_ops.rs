use criterion::{measurement::Measurement, BenchmarkGroup, BenchmarkId, Criterion};
use pecos_core::{IndexableElement, Set, VecSet};
use pecos_qsims::{CliffordSimulator, PauliProp};

pub fn benchmarks<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("Hadamard Operations");
    let sizes = [10, 100, 1000];
    run_hadamard_benchmark::<VecSet<usize>, usize, M>(&mut group, "usize", &sizes);
    run_hadamard_benchmark::<VecSet<u16>, u16, M>(&mut group, "u16", &sizes);
    run_hadamard_v2_benchmark::<VecSet<usize>, usize, M>(&mut group, "usize", &sizes);
    run_hadamard_v2_benchmark::<VecSet<u16>, u16, M>(&mut group, "u16", &sizes);
    bench_hadamard::<VecSet<usize>, usize, M>(&mut group, "usize");
    bench_hadamard::<VecSet<u16>, u16, M>(&mut group, "u16");
    group.finish();
}

fn run_hadamard_benchmark<T, E, M: Measurement>(
    group: &mut BenchmarkGroup<M>,
    name: &str,
    sizes: &[usize],
) where
    T: for<'a> Set<'a, Element = E>,
    E: IndexableElement,
{
    for &size in sizes {
        group.bench_with_input(
            BenchmarkId::new("Hadamard", format!("{name}_{size}")),
            &size,
            |b, &size| {
                let mut pauli = PauliProp::<T, E>::new(size);
                b.iter(|| {
                    for i in 0..size {
                        pauli.h(E::from_usize(i));
                    }
                });
            },
        );
    }
}

fn run_hadamard_v2_benchmark<T, E, M: Measurement>(
    group: &mut BenchmarkGroup<M>,
    name: &str,
    sizes: &[usize],
) where
    T: for<'a> Set<'a, Element = E>,
    E: IndexableElement,
{
    for &size in sizes {
        group.bench_with_input(
            BenchmarkId::new("Hadamard_v2", format!("{name}_{size}")),
            &size,
            |b, &size| {
                let mut pauli = PauliProp::<T, E>::new(size);
                b.iter(|| {
                    for i in 0..size {
                        pauli.h_v2(E::from_usize(i));
                    }
                });
            },
        );
    }
}

fn bench_hadamard<T, E, M: Measurement>(group: &mut BenchmarkGroup<M>, type_name: &str)
where
    T: for<'a> Set<'a, Element = E>,
    E: IndexableElement + From<u16>,
{
    group.bench_function(format!("hadamard_{type_name}"), |b| {
        let mut pauli = PauliProp::<T, E>::new(100);
        for i in 0..100_u16 {
            pauli.insert_x(E::from(i));
            pauli.insert_z(E::from(i));
        }
        b.iter(|| {
            for i in 0..100_u16 {
                pauli.h(E::from(i));
            }
        });
    });
}
