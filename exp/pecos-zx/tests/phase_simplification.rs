// Copyright 2026 The PECOS Developers
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy at https://www.apache.org/licenses/LICENSE-2.0

use pecos_core::{Angle64, radians};
use pecos_quantum::DagCircuit;
use pecos_zx::{GraphLike, ZxGraph, convert::dag_to_zx_circuit, simplify};
use quizx::tensor::ToTensor;

type Simplifier = (&'static str, fn(&mut ZxGraph));
const SIMPLIFIERS: &[Simplifier] = &[
    ("full", |g| {
        simplify::full_simp(g);
    }),
    ("clifford", |g| {
        simplify::clifford_simp(g);
    }),
    ("interior_clifford", |g| {
        simplify::interior_clifford_simp(g);
    }),
    ("id", |g| {
        simplify::id_simp(g);
    }),
    ("local_comp", |g| {
        simplify::local_comp_simp(g);
    }),
    ("spider", |g| {
        simplify::spider_simp(g);
    }),
    ("pivot", |g| {
        simplify::pivot_simp(g);
    }),
    ("gen_pivot", |g| {
        simplify::gen_pivot_simp(g);
    }),
    ("scalar", |g| {
        simplify::scalar_simp(g);
    }),
    ("flow", |g| {
        simplify::flow_simp(g);
    }),
    ("fuse_gadgets", |g| {
        simplify::fuse_gadgets(g);
    }),
    ("local_gslc", |g| {
        let vs = boundary_spiders(g);
        for v in vs {
            if g.contains_vertex(v) {
                simplify::local_gslc_simp(g, [v]);
            }
        }
    }),
    ("local_ap", |g| {
        let vs = boundary_spiders(g);
        for v in vs {
            if g.contains_vertex(v) {
                simplify::local_ap_simp(g, [v]);
            }
        }
    }),
];

fn boundary_spiders(g: &ZxGraph) -> Vec<usize> {
    g.inputs().iter().flat_map(|&v| g.neighbors(v)).collect()
}

fn check_simplifiers(dag: &DagCircuit) {
    let circuit = dag_to_zx_circuit(dag).unwrap();
    let expected = circuit.to_tensor64();
    for (name, simplify) in SIMPLIFIERS {
        let mut graph: ZxGraph = circuit.to_graph();
        simplify(&mut graph);
        let actual = graph.to_tensor64();
        assert_eq!(actual.shape(), expected.shape(), "{name}");
        for (a, b) in actual.iter().zip(expected.iter()) {
            assert!((*a - *b).norm() < 1e-8, "{name}: {a} != {b}");
        }
    }
}

#[test]
fn simplifiers_accept_rz_followed_by_z() {
    let mut dag = DagCircuit::new();
    dag.rz(radians!(0.1), &[0]);
    dag.z(&[0]);
    check_simplifiers(&dag);
}

#[test]
fn simplifiers_accept_seeded_full_precision_phases() {
    let mut state = 0x1234_5678_9abc_def0_u64;
    for _ in 0..16 {
        let mut next = || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            Angle64::new(state)
        };
        let mut dag = DagCircuit::new();
        dag.rz(next(), &[0]);
        dag.z(&[0]);
        dag.rx(next(), &[1]);
        dag.h(&[0]);
        dag.cx(&[(0, 1)]);
        // U(0, 0, lambda) is the phase-shaped U supported by ZX export.
        dag.u(Angle64::ZERO, Angle64::ZERO, next(), &[0]);
        dag.t(&[1]);
        dag.rz(next(), &[1]);
        check_simplifiers(&dag);
    }
}
