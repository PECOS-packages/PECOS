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

// Standalone Qulacs GPU benchmark using QuantumStateGpu and direct GPU kernels.
// Compiled and linked against a CUDA-enabled CMake-built Qulacs library.

#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <algorithm>
#include <vector>

#include "cppsim/state_gpu.hpp"
#include "gpusim/update_ops_cuda.h"

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
// Circuit: layered H + RZ + CX (direct GPU kernels)
// ---------------------------------------------------------------------------

static void run_circuit_gpu(QuantumStateGpu& state, int num_qubits, int num_layers) {
    void* data = state.data();
    ITYPE dim = state.dim;
    void* stream = state.get_cuda_stream();
    UINT dev = state.device_number;

    for (int layer = 0; layer < num_layers; layer++) {
        for (int q = 0; q < num_qubits; q++) {
            H_gate_host((UINT)q, data, dim, stream, dev);
            // Qulacs uses opposite sign convention for rotations
            RZ_gate_host((UINT)q, -0.1, data, dim, stream, dev);
        }
        for (int q = 0; q < num_qubits - 1; q++) {
            CNOT_gate_host((UINT)q, (UINT)(q + 1), data, dim, stream, dev);
        }
    }
}

// ---------------------------------------------------------------------------
// Layered circuit benchmark
// ---------------------------------------------------------------------------

static void bench_circuit(int num_qubits, int num_layers, int reps) {
    QuantumStateGpu state(num_qubits);
    std::vector<double> times(reps);

    for (int r = 0; r < reps; r++) {
        state.set_zero_state();
        double t0 = now_sec();
        run_circuit_gpu(state, num_qubits, num_layers);
        double t1 = now_sec();
        times[r] = t1 - t0;
    }

    double med = median(times);
    std::printf("circuit  %2dq %2dl  %12.3f us\n",
                num_qubits, num_layers, med * 1e6);
}

// ---------------------------------------------------------------------------
// Individual gate benchmarks
// ---------------------------------------------------------------------------

static void bench_gate_h(int num_qubits, int iters, int reps) {
    QuantumStateGpu state(num_qubits);
    state.set_zero_state();
    void* data = state.data();
    ITYPE dim = state.dim;
    void* stream = state.get_cuda_stream();
    UINT dev = state.device_number;
    std::vector<double> times(reps);

    for (int r = 0; r < reps; r++) {
        double t0 = now_sec();
        for (int i = 0; i < iters; i++)
            for (int q = 0; q < num_qubits; q++)
                H_gate_host((UINT)q, data, dim, stream, dev);
        double t1 = now_sec();
        times[r] = t1 - t0;
    }
    std::printf("gate     H        %12.3f us\n", median(times) * 1e6);
}

static void bench_gate_x(int num_qubits, int iters, int reps) {
    QuantumStateGpu state(num_qubits);
    state.set_zero_state();
    void* data = state.data();
    ITYPE dim = state.dim;
    void* stream = state.get_cuda_stream();
    UINT dev = state.device_number;
    std::vector<double> times(reps);

    for (int r = 0; r < reps; r++) {
        double t0 = now_sec();
        for (int i = 0; i < iters; i++)
            for (int q = 0; q < num_qubits; q++)
                X_gate_host((UINT)q, data, dim, stream, dev);
        double t1 = now_sec();
        times[r] = t1 - t0;
    }
    std::printf("gate     X        %12.3f us\n", median(times) * 1e6);
}

static void bench_gate_cx(int num_qubits, int iters, int reps) {
    QuantumStateGpu state(num_qubits);
    state.set_zero_state();
    void* data = state.data();
    ITYPE dim = state.dim;
    void* stream = state.get_cuda_stream();
    UINT dev = state.device_number;
    std::vector<double> times(reps);

    for (int r = 0; r < reps; r++) {
        double t0 = now_sec();
        for (int i = 0; i < iters; i++)
            for (int q = 0; q < num_qubits - 1; q++)
                CNOT_gate_host((UINT)q, (UINT)(q + 1), data, dim, stream, dev);
        double t1 = now_sec();
        times[r] = t1 - t0;
    }
    std::printf("gate     CX       %12.3f us\n", median(times) * 1e6);
}

static void bench_gate_rz(int num_qubits, int iters, int reps) {
    QuantumStateGpu state(num_qubits);
    state.set_zero_state();
    void* data = state.data();
    ITYPE dim = state.dim;
    void* stream = state.get_cuda_stream();
    UINT dev = state.device_number;
    std::vector<double> times(reps);

    for (int r = 0; r < reps; r++) {
        double t0 = now_sec();
        for (int i = 0; i < iters; i++)
            for (int q = 0; q < num_qubits; q++)
                RZ_gate_host((UINT)q, -0.1, data, dim, stream, dev);
        double t1 = now_sec();
        times[r] = t1 - t0;
    }
    std::printf("gate     RZ       %12.3f us\n", median(times) * 1e6);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main() {
    int reps = 5;

    std::printf("=== Qulacs GPU standalone benchmarks ===\n");
    std::printf("\n-- Layered circuits (median of %d runs) --\n", reps);

    int configs[][2] = {
        {10, 20}, {14, 20}, {18, 20}, {20, 20}, {22, 20}, {24, 10}, {26, 5}
    };
    int n_configs = sizeof(configs) / sizeof(configs[0]);

    for (int i = 0; i < n_configs; i++) {
        bench_circuit(configs[i][0], configs[i][1], reps);
    }

    std::printf("\n-- Individual gates at 18 qubits, 100 iters (median of %d runs) --\n", reps);
    bench_gate_h(18, 100, reps);
    bench_gate_x(18, 100, reps);
    bench_gate_cx(18, 100, reps);
    bench_gate_rz(18, 100, reps);

    return 0;
}
