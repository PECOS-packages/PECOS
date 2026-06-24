// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under
// the License.

//! Engines-vs-neo stack baselines for the transition validation gate.
//!
//! Measures end-to-end `sim(qasm).run(shots)` (parse + build + execute) on
//! both stacks over a standard circuit set. Run with:
//! `cargo bench -p pecos --features neo --bench stack_comparison`

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use pecos::{SimStack, sim};
use pecos_programs::Qasm;
use std::fmt::Write;

const SHOTS: usize = 1000;

fn bell() -> String {
    r#"
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[2];
    creg c[2];
    h q[0];
    cx q[0], q[1];
    measure q -> c;
    "#
    .to_string()
}

fn ghz(n: usize) -> String {
    let mut s =
        format!("OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[{n}];\ncreg c[{n}];\nh q[0];\n");
    for i in 1..n {
        let _ = writeln!(s, "cx q[{}], q[{i}];", i - 1);
    }
    s.push_str("measure q -> c;\n");
    s
}

/// Chain of measure-and-correct rounds: exercises the feedback path.
fn feedback_chain(rounds: usize) -> String {
    let mut s = String::from("OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[2];\n");
    for r in 0..rounds {
        let _ = writeln!(s, "creg c{r}[1];");
    }
    for r in 0..rounds {
        let _ = writeln!(
            s,
            "h q[0];\nmeasure q[0] -> c{r}[0];\nif (c{r} == 1) x q[0];"
        );
    }
    s
}

/// Layered Clifford circuit: H + S on every qubit, CX ladder, repeated.
fn clifford_layers(n: usize, depth: usize) -> String {
    let mut s = format!("OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[{n}];\ncreg c[{n}];\n");
    for _ in 0..depth {
        for i in 0..n {
            let _ = writeln!(s, "h q[{i}];\ns q[{i}];");
        }
        for i in (0..n - 1).step_by(2) {
            let _ = writeln!(s, "cx q[{}], q[{}];", i, i + 1);
        }
    }
    s.push_str("measure q -> c;\n");
    s
}

fn run_stack(qasm: &str, stack: SimStack, noisy: bool) {
    let mut builder = sim(Qasm::from_string(qasm)).stack(stack).seed(42);
    if noisy {
        builder = builder.noise(pecos_engines::DepolarizingNoise { p: 0.001 });
    }
    let results = builder.shots(SHOTS).run().expect("run");
    assert_eq!(results.shots.len(), SHOTS);
}

fn bench_stacks(c: &mut Criterion) {
    let cases: Vec<(&str, String, bool)> = vec![
        ("bell", bell(), false),
        ("bell_noisy", bell(), true),
        ("ghz10", ghz(10), false),
        ("feedback16", feedback_chain(16), false),
        ("clifford_12q_x8", clifford_layers(12, 8), false),
        ("clifford_12q_x8_noisy", clifford_layers(12, 8), true),
    ];

    let mut group = c.benchmark_group(format!("sim_run_{SHOTS}_shots"));
    group.sample_size(10);
    for (name, qasm, noisy) in &cases {
        for (stack_name, stack) in [("engines", SimStack::Engines), ("neo", SimStack::Neo)] {
            group.bench_with_input(
                BenchmarkId::new(*name, stack_name),
                &(qasm.as_str(), stack, *noisy),
                |b, &(qasm, stack, noisy)| b.iter(|| run_stack(qasm, stack, noisy)),
            );
        }
    }
    group.finish();
}

criterion_group!(benches, bench_stacks);
criterion_main!(benches);
