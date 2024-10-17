use criterion::{black_box, measurement::Measurement, BenchmarkGroup, Criterion};
use pecos_core::IndexableElement;

pub fn benchmarks<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("Element Operations");
    bench_conversion::<usize, M>(&mut group, "usize");
    bench_conversion::<u16, M>(&mut group, "u16");
    bench_memory_access::<usize, M>(&mut group, "usize");
    bench_memory_access::<u16, M>(&mut group, "u16");
    group.finish();
}

fn bench_conversion<E: IndexableElement + From<u16>, M: Measurement>(
    group: &mut BenchmarkGroup<M>,
    type_name: &str,
) {
    group.bench_function(format!("conversion_{type_name}"), |b| {
        b.iter(|| {
            let mut sum = 0_usize;
            for i in 0..1000_u16 {
                sum += E::from(i).to_usize();
            }
            black_box(sum)
        });
    });
}

#[allow(clippy::cast_possible_truncation)]
fn bench_memory_access<E: IndexableElement + From<u16>, M: Measurement>(
    group: &mut BenchmarkGroup<M>,
    type_name: &str,
) {
    group.bench_function(format!("memory_access_{type_name}"), |b| {
        let mut vec = Vec::new();
        for i in 0..1000_u16 {
            vec.push(E::from(i));
        }
        b.iter(|| {
            let mut sum = E::from(0_u16);
            for &item in &vec {
                sum = E::from((sum.to_usize() + item.to_usize()) as u16);
            }
            black_box(sum)
        });
    });
}
