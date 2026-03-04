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

// Standalone Qulacs benchmark using both the gate-object API and direct csim kernels.
// Compiled and linked against a CMake-built Qulacs library so that build flags
// are entirely under CMake's control (no Rust build.rs involvement).

#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <algorithm>
#include <vector>

#include "cppsim/state.hpp"
#include "cppsim/gate_factory.hpp"
#include "csim/update_ops.hpp"

// ---------------------------------------------------------------------------
// Timing helpers
// ---------------------------------------------------------------------------

static double now_sec() {
    auto tp = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(tp.time_since_epoch()).count();
}

static double median(std::vector<double>& vals) {
    std::sort(vals.begin(), vals.end());
    size_t n = vals.size();
    if (n % 2 == 1) return vals[n / 2];
    return (vals[n / 2 - 1] + vals[n / 2]) / 2.0;
}

// ---------------------------------------------------------------------------
// Circuit: layered H + RZ + CX  (gate-object API)
// ---------------------------------------------------------------------------

static void run_circuit_gate_api(QuantumStateCpu& state, int num_qubits, int num_layers) {
    for (int layer = 0; layer < num_layers; layer++) {
        for (int q = 0; q < num_qubits; q++) {
            auto* g1 = gate::H(q);
            g1->update_quantum_state(&state);
            delete g1;
            // Qulacs uses opposite sign convention for rotations
            auto* g2 = gate::RZ(q, -0.1);
            g2->update_quantum_state(&state);
            delete g2;
        }
        for (int q = 0; q < num_qubits - 1; q++) {
            auto* g = gate::CNOT(q, q + 1);
            g->update_quantum_state(&state);
            delete g;
        }
    }
}

// ---------------------------------------------------------------------------
// Circuit: layered H + RZ + CX  (direct csim kernels)
// ---------------------------------------------------------------------------

static void run_circuit_csim(QuantumStateCpu& state, int num_qubits, int num_layers) {
    CTYPE* data = state.data_c();
    ITYPE dim = state.dim;

    for (int layer = 0; layer < num_layers; layer++) {
        for (int q = 0; q < num_qubits; q++) {
            H_gate((UINT)q, data, dim);
            // Qulacs uses opposite sign convention for rotations
            RZ_gate((UINT)q, -0.1, data, dim);
        }
        for (int q = 0; q < num_qubits - 1; q++) {
            CNOT_gate((UINT)q, (UINT)(q + 1), data, dim);
        }
    }
}

// ---------------------------------------------------------------------------
// Layered circuit benchmarks
// ---------------------------------------------------------------------------

static void bench_circuit(int num_qubits, int num_layers, int reps, const char* tag,
                          void (*fn)(QuantumStateCpu&, int, int)) {
    QuantumStateCpu state(num_qubits);
    std::vector<double> times(reps);

    for (int r = 0; r < reps; r++) {
        state.set_zero_state();
        double t0 = now_sec();
        fn(state, num_qubits, num_layers);
        double t1 = now_sec();
        times[r] = t1 - t0;
    }

    double med = median(times);
    std::printf("circuit  %2dq %2dl  %-10s %12.3f us\n",
                num_qubits, num_layers, tag, med * 1e6);
}

// ---------------------------------------------------------------------------
// Individual gate benchmarks (18 qubits, 100 iterations)
// ---------------------------------------------------------------------------

// -- Gate-object API variants --

static void bench_gate_h_api(int num_qubits, int iters, int reps) {
    QuantumStateCpu state(num_qubits);
    state.set_zero_state();
    std::vector<double> times(reps);

    for (int r = 0; r < reps; r++) {
        double t0 = now_sec();
        for (int i = 0; i < iters; i++)
            for (int q = 0; q < num_qubits; q++) {
                auto* g = gate::H(q);
                g->update_quantum_state(&state);
                delete g;
            }
        double t1 = now_sec();
        times[r] = t1 - t0;
    }
    std::printf("gate     H        %-10s %12.3f us\n", "gate_api", median(times) * 1e6);
}

static void bench_gate_x_api(int num_qubits, int iters, int reps) {
    QuantumStateCpu state(num_qubits);
    state.set_zero_state();
    std::vector<double> times(reps);

    for (int r = 0; r < reps; r++) {
        double t0 = now_sec();
        for (int i = 0; i < iters; i++)
            for (int q = 0; q < num_qubits; q++) {
                auto* g = gate::X(q);
                g->update_quantum_state(&state);
                delete g;
            }
        double t1 = now_sec();
        times[r] = t1 - t0;
    }
    std::printf("gate     X        %-10s %12.3f us\n", "gate_api", median(times) * 1e6);
}

static void bench_gate_cx_api(int num_qubits, int iters, int reps) {
    QuantumStateCpu state(num_qubits);
    state.set_zero_state();
    std::vector<double> times(reps);

    for (int r = 0; r < reps; r++) {
        double t0 = now_sec();
        for (int i = 0; i < iters; i++)
            for (int q = 0; q < num_qubits - 1; q++) {
                auto* g = gate::CNOT(q, q + 1);
                g->update_quantum_state(&state);
                delete g;
            }
        double t1 = now_sec();
        times[r] = t1 - t0;
    }
    std::printf("gate     CX       %-10s %12.3f us\n", "gate_api", median(times) * 1e6);
}

static void bench_gate_rz_api(int num_qubits, int iters, int reps) {
    QuantumStateCpu state(num_qubits);
    state.set_zero_state();
    std::vector<double> times(reps);

    for (int r = 0; r < reps; r++) {
        double t0 = now_sec();
        for (int i = 0; i < iters; i++)
            for (int q = 0; q < num_qubits; q++) {
                auto* g = gate::RZ(q, -0.1);
                g->update_quantum_state(&state);
                delete g;
            }
        double t1 = now_sec();
        times[r] = t1 - t0;
    }
    std::printf("gate     RZ       %-10s %12.3f us\n", "gate_api", median(times) * 1e6);
}

// -- Direct csim kernel variants --

static void bench_gate_h_csim(int num_qubits, int iters, int reps) {
    QuantumStateCpu state(num_qubits);
    state.set_zero_state();
    CTYPE* data = state.data_c();
    ITYPE dim = state.dim;
    std::vector<double> times(reps);

    for (int r = 0; r < reps; r++) {
        double t0 = now_sec();
        for (int i = 0; i < iters; i++)
            for (int q = 0; q < num_qubits; q++)
                H_gate((UINT)q, data, dim);
        double t1 = now_sec();
        times[r] = t1 - t0;
    }
    std::printf("gate     H        %-10s %12.3f us\n", "csim", median(times) * 1e6);
}

static void bench_gate_x_csim(int num_qubits, int iters, int reps) {
    QuantumStateCpu state(num_qubits);
    state.set_zero_state();
    CTYPE* data = state.data_c();
    ITYPE dim = state.dim;
    std::vector<double> times(reps);

    for (int r = 0; r < reps; r++) {
        double t0 = now_sec();
        for (int i = 0; i < iters; i++)
            for (int q = 0; q < num_qubits; q++)
                X_gate((UINT)q, data, dim);
        double t1 = now_sec();
        times[r] = t1 - t0;
    }
    std::printf("gate     X        %-10s %12.3f us\n", "csim", median(times) * 1e6);
}

static void bench_gate_cx_csim(int num_qubits, int iters, int reps) {
    QuantumStateCpu state(num_qubits);
    state.set_zero_state();
    CTYPE* data = state.data_c();
    ITYPE dim = state.dim;
    std::vector<double> times(reps);

    for (int r = 0; r < reps; r++) {
        double t0 = now_sec();
        for (int i = 0; i < iters; i++)
            for (int q = 0; q < num_qubits - 1; q++)
                CNOT_gate((UINT)q, (UINT)(q + 1), data, dim);
        double t1 = now_sec();
        times[r] = t1 - t0;
    }
    std::printf("gate     CX       %-10s %12.3f us\n", "csim", median(times) * 1e6);
}

static void bench_gate_rz_csim(int num_qubits, int iters, int reps) {
    QuantumStateCpu state(num_qubits);
    state.set_zero_state();
    CTYPE* data = state.data_c();
    ITYPE dim = state.dim;
    std::vector<double> times(reps);

    for (int r = 0; r < reps; r++) {
        double t0 = now_sec();
        for (int i = 0; i < iters; i++)
            for (int q = 0; q < num_qubits; q++)
                RZ_gate((UINT)q, -0.1, data, dim);
        double t1 = now_sec();
        times[r] = t1 - t0;
    }
    std::printf("gate     RZ       %-10s %12.3f us\n", "csim", median(times) * 1e6);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main() {
    int reps = 5;

    std::printf("=== Qulacs standalone benchmarks ===\n");
    std::printf("\n-- Layered circuits (median of %d runs) --\n", reps);

    int configs[][2] = {
        {10, 20}, {14, 20}, {18, 20}, {20, 20}, {22, 10}, {24, 5}
    };
    int n_configs = sizeof(configs) / sizeof(configs[0]);

    for (int i = 0; i < n_configs; i++) {
        bench_circuit(configs[i][0], configs[i][1], reps, "gate_api", run_circuit_gate_api);
        bench_circuit(configs[i][0], configs[i][1], reps, "csim", run_circuit_csim);
    }

    std::printf("\n-- Individual gates at 18 qubits, 100 iters (median of %d runs) --\n", reps);

    bench_gate_h_api(18, 100, reps);
    bench_gate_h_csim(18, 100, reps);

    bench_gate_x_api(18, 100, reps);
    bench_gate_x_csim(18, 100, reps);

    bench_gate_cx_api(18, 100, reps);
    bench_gate_cx_csim(18, 100, reps);

    bench_gate_rz_api(18, 100, reps);
    bench_gate_rz_csim(18, 100, reps);

    return 0;
}
