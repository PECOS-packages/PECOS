#!/bin/bash
set -e

# Determine the root directory of the project
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ "$SCRIPT_DIR" == */scripts ]]; then
    PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
else
    PROJECT_ROOT="$SCRIPT_DIR"
fi

# Change to project root for all operations
cd "$PROJECT_ROOT"

make clean build test

cargo run --bin pecos run examples/phir/bell.phir.json -s 10 -w 2 -p 0.2
cargo run --bin pecos run examples/llvm/bell.ll -s 10 -w 2 -p 0.2
cargo run --bin pecos run examples/phir/bell.phir.json -s 10 -w 1
cargo run --bin pecos run examples/llvm/bell.ll -s 10 -w 1
cargo run --bin pecos run examples/phir/bell.phir.json -s 10 -w 10
cargo run --bin pecos run examples/llvm/bell.ll -s 10 -w 10
cargo run --example replaying_rng --package pecos-core
cargo run --example bell_state_replay --package pecos-qsim
cargo run --example run_noisy_circ
cargo run --example biased_measurement_example
cargo run --example compare_noise_models
cargo run --example run_noisy_circ_with_general
cargo run --example general_noise_test

.venv/bin/python python/pecos-rslib/examples/bell_state_example.py
.venv/bin/python python/pecos-rslib/examples/bell_state_simulator.py
