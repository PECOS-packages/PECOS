// Copyright 2026 The PECOS Developers
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy at https://www.apache.org/licenses/LICENSE-2.0

use pecos_quantum::TickCircuit;
use pecos_simulators::{CircuitExecutor, SparseStab};

#[test]
fn mx_circuit_executor_restores_x_eigenstate() {
    for minus in [false, true] {
        let mut tc = TickCircuit::new();
        tc.tick().h(&[0]);
        if minus {
            tc.tick().z(&[0]);
        }
        tc.tick().mx(&[0]);
        tc.tick().mx(&[0]);
        let results = CircuitExecutor::new(&tc)
            .run(&mut SparseStab::new(1))
            .unwrap();
        assert_eq!(results.len(), 2);
        assert!(results[0].is_deterministic);
        assert!(results[1].is_deterministic);
        assert_eq!(results[0].outcome, minus);
        assert_eq!(results[1].outcome, results[0].outcome);
    }
}
