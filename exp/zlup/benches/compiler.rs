//! Performance benchmarks for the Zlup compiler.
//!
//! Run with: cargo bench --bench compiler

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use zlup::codegen::{QasmCodegen, SlrCodegen};
use zlup::semantic::SemanticAnalyzer;

// =============================================================================
// Test Programs
// =============================================================================

/// Minimal Bell state program (2 qubits, 3 operations)
const SMALL_PROGRAM: &str = r#"
pub fn main() -> unit {
    q := qalloc(2);
    pz q;
    h q[0];
    cx (q[0], q[1]);
    results: [2]u1 = mz([2]u1) [q[0], q[1]];
    return unit;
}
"#;

/// Medium-sized GHZ state program (4 qubits, 6 operations)
const MEDIUM_PROGRAM: &str = r#"
pub fn main() -> unit {
    q := qalloc(4);
    pz q;
    h q[0];
    cx (q[0], q[1]);
    cx (q[0], q[2]);
    cx (q[0], q[3]);
    results: [4]u1 = mz([4]u1) [q[0], q[1], q[2], q[3]];
    return unit;
}
"#;

/// Generate a program with N qubits creating a GHZ state
fn generate_ghz_program(n: usize) -> String {
    let mut s = String::new();
    s.push_str("pub fn main() -> unit {\n");
    s.push_str(&format!("    q := qalloc({n});\n"));
    s.push_str("    pz q;\n");
    s.push_str("    h q[0];\n");

    // Create entanglement chain
    for i in 1..n {
        s.push_str(&format!("    cx (q[0], q[{i}]);\n"));
    }

    // Measurement
    let indices: Vec<String> = (0..n).map(|i| format!("q[{i}]")).collect();
    s.push_str(&format!(
        "    results: [{n}]u1 = mz([{n}]u1) [{}];\n",
        indices.join(", ")
    ));
    s.push_str("    return unit;\n");
    s.push_str("}\n");
    s
}

/// Generate a program with multiple functions
fn generate_multi_function_program(n_funcs: usize) -> String {
    let mut s = String::new();

    // Helper functions
    for i in 0..n_funcs {
        s.push_str(&format!(
            r#"
fn helper_{i}(x: i32) -> i32 {{
    y := x + {i};
    return y;
}}
"#
        ));
    }

    // Main function that calls helpers
    s.push_str("\npub fn main() -> unit {\n");
    s.push_str("    q := qalloc(2);\n");
    s.push_str("    pz q;\n");
    s.push_str("    h q[0];\n");
    s.push_str("    cx (q[0], q[1]);\n");

    // Call each helper
    for i in 0..n_funcs {
        s.push_str(&format!("    val_{i} := helper_{i}({i});\n"));
    }

    s.push_str("    return unit;\n");
    s.push_str("}\n");
    s
}

// =============================================================================
// Parsing Benchmarks
// =============================================================================

fn bench_parsing(c: &mut Criterion) {
    let mut group = c.benchmark_group("parsing");

    // Small program
    group.throughput(Throughput::Bytes(SMALL_PROGRAM.len() as u64));
    group.bench_function("small_2q", |b| {
        b.iter(|| zlup::parse(black_box(SMALL_PROGRAM)).unwrap())
    });

    // Medium program
    group.throughput(Throughput::Bytes(MEDIUM_PROGRAM.len() as u64));
    group.bench_function("medium_4q", |b| {
        b.iter(|| zlup::parse(black_box(MEDIUM_PROGRAM)).unwrap())
    });

    // Scaling with qubit count
    for n in [8, 16, 32, 64] {
        let program = generate_ghz_program(n);
        group.throughput(Throughput::Bytes(program.len() as u64));
        group.bench_with_input(BenchmarkId::new("ghz", n), &program, |b, prog| {
            b.iter(|| zlup::parse(black_box(prog)).unwrap())
        });
    }

    // Scaling with function count
    for n in [5, 10, 20] {
        let program = generate_multi_function_program(n);
        group.throughput(Throughput::Bytes(program.len() as u64));
        group.bench_with_input(BenchmarkId::new("multi_func", n), &program, |b, prog| {
            b.iter(|| zlup::parse(black_box(prog)).unwrap())
        });
    }

    group.finish();
}

// =============================================================================
// Semantic Analysis Benchmarks
// =============================================================================

fn bench_semantic(c: &mut Criterion) {
    let mut group = c.benchmark_group("semantic");

    // Small program
    let small_ast = zlup::parse(SMALL_PROGRAM).unwrap();
    group.bench_function("small_2q", |b| {
        b.iter(|| {
            let mut analyzer = SemanticAnalyzer::new();
            analyzer.analyze(black_box(&small_ast)).unwrap()
        })
    });

    // Medium program
    let medium_ast = zlup::parse(MEDIUM_PROGRAM).unwrap();
    group.bench_function("medium_4q", |b| {
        b.iter(|| {
            let mut analyzer = SemanticAnalyzer::new();
            analyzer.analyze(black_box(&medium_ast)).unwrap()
        })
    });

    // Scaling with qubit count
    for n in [8, 16, 32, 64] {
        let program = generate_ghz_program(n);
        let ast = zlup::parse(&program).unwrap();
        group.bench_with_input(BenchmarkId::new("ghz", n), &ast, |b, ast| {
            b.iter(|| {
                let mut analyzer = SemanticAnalyzer::new();
                analyzer.analyze(black_box(ast)).unwrap()
            })
        });
    }

    // Scaling with function count
    for n in [5, 10, 20] {
        let program = generate_multi_function_program(n);
        let ast = zlup::parse(&program).unwrap();
        group.bench_with_input(BenchmarkId::new("multi_func", n), &ast, |b, ast| {
            b.iter(|| {
                let mut analyzer = SemanticAnalyzer::new();
                analyzer.analyze(black_box(ast)).unwrap()
            })
        });
    }

    group.finish();
}

// =============================================================================
// SLR Code Generation Benchmarks
// =============================================================================

fn bench_slr_codegen(c: &mut Criterion) {
    let mut group = c.benchmark_group("codegen_slr");

    // Small program
    let small_ast = zlup::parse(SMALL_PROGRAM).unwrap();
    group.bench_function("small_2q", |b| {
        b.iter(|| {
            let mut codegen = SlrCodegen::new();
            codegen.compile(black_box(&small_ast)).unwrap()
        })
    });

    // Medium program
    let medium_ast = zlup::parse(MEDIUM_PROGRAM).unwrap();
    group.bench_function("medium_4q", |b| {
        b.iter(|| {
            let mut codegen = SlrCodegen::new();
            codegen.compile(black_box(&medium_ast)).unwrap()
        })
    });

    // Scaling with qubit count
    for n in [8, 16, 32, 64] {
        let program = generate_ghz_program(n);
        let ast = zlup::parse(&program).unwrap();
        group.bench_with_input(BenchmarkId::new("ghz", n), &ast, |b, ast| {
            b.iter(|| {
                let mut codegen = SlrCodegen::new();
                codegen.compile(black_box(ast)).unwrap()
            })
        });
    }

    group.finish();
}

// =============================================================================
// QASM Code Generation Benchmarks
// =============================================================================

fn bench_qasm_codegen(c: &mut Criterion) {
    let mut group = c.benchmark_group("codegen_qasm");

    // Small program
    let small_ast = zlup::parse(SMALL_PROGRAM).unwrap();
    group.bench_function("small_2q", |b| {
        b.iter(|| {
            let mut codegen = QasmCodegen::new();
            codegen.compile(black_box(&small_ast)).unwrap()
        })
    });

    // Medium program
    let medium_ast = zlup::parse(MEDIUM_PROGRAM).unwrap();
    group.bench_function("medium_4q", |b| {
        b.iter(|| {
            let mut codegen = QasmCodegen::new();
            codegen.compile(black_box(&medium_ast)).unwrap()
        })
    });

    // Scaling with qubit count
    for n in [8, 16, 32, 64] {
        let program = generate_ghz_program(n);
        let ast = zlup::parse(&program).unwrap();
        group.bench_with_input(BenchmarkId::new("ghz", n), &ast, |b, ast| {
            b.iter(|| {
                let mut codegen = QasmCodegen::new();
                codegen.compile(black_box(ast)).unwrap()
            })
        });
    }

    group.finish();
}

// =============================================================================
// End-to-End Pipeline Benchmarks
// =============================================================================

fn bench_full_pipeline(c: &mut Criterion) {
    let mut group = c.benchmark_group("full_pipeline");

    // Parse -> Semantic -> SLR
    for n in [4, 16, 64] {
        let program = generate_ghz_program(n);
        group.bench_with_input(BenchmarkId::new("to_slr", n), &program, |b, prog| {
            b.iter(|| {
                let ast = zlup::parse(black_box(prog)).unwrap();
                let mut analyzer = SemanticAnalyzer::new();
                analyzer.analyze(&ast).unwrap();
                let mut codegen = SlrCodegen::new();
                codegen.compile(&ast).unwrap()
            })
        });
    }

    // Parse -> Semantic -> QASM
    for n in [4, 16, 64] {
        let program = generate_ghz_program(n);
        group.bench_with_input(BenchmarkId::new("to_qasm", n), &program, |b, prog| {
            b.iter(|| {
                let ast = zlup::parse(black_box(prog)).unwrap();
                let mut analyzer = SemanticAnalyzer::new();
                analyzer.analyze(&ast).unwrap();
                let mut codegen = QasmCodegen::new();
                codegen.compile(&ast).unwrap()
            })
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_parsing,
    bench_semantic,
    bench_slr_codegen,
    bench_qasm_codegen,
    bench_full_pipeline,
);
criterion_main!(benches);
