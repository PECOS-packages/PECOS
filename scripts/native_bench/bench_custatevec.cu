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

// Standalone cuStateVec benchmark using the native C API directly.
// No Rust, no PECOS -- pure CUDA + cuQuantum.

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <vector>

#include <cuda_runtime.h>
#include <custatevec.h>

// ---------------------------------------------------------------------------
// Error checking
// ---------------------------------------------------------------------------

#define CUDA_CHECK(x) do { \
    cudaError_t err = (x); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error %d at %s:%d: %s\n", err, __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

#define CUSV_CHECK(x) do { \
    custatevecStatus_t err = (x); \
    if (err != CUSTATEVEC_STATUS_SUCCESS) { \
        fprintf(stderr, "cuStateVec error %d at %s:%d\n", err, __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

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
// Gate matrices (column-major, complex128)
// ---------------------------------------------------------------------------

struct Complex2 { double re, im; };

static const Complex2 H_MATRIX[4] = {
    {M_SQRT1_2, 0}, {M_SQRT1_2, 0},
    {M_SQRT1_2, 0}, {-M_SQRT1_2, 0}
};

static const Complex2 X_MATRIX[4] = {
    {0, 0}, {1, 0},
    {1, 0}, {0, 0}
};

static const Complex2 CX_MATRIX[16] = {
    {1,0}, {0,0}, {0,0}, {0,0},
    {0,0}, {1,0}, {0,0}, {0,0},
    {0,0}, {0,0}, {0,0}, {1,0},
    {0,0}, {0,0}, {1,0}, {0,0}
};

static void make_rz_matrix(double theta, Complex2 out[4]) {
    double c = cos(theta / 2.0);
    double s = sin(theta / 2.0);
    out[0] = {c, -s};  out[1] = {0, 0};
    out[2] = {0, 0};   out[3] = {c, s};
}

// ---------------------------------------------------------------------------
// Wrapper to apply a 1-qubit gate
// ---------------------------------------------------------------------------

static void apply_1q(custatevecHandle_t handle, void* d_sv, int nqubits,
                     const Complex2* matrix, int target) {
    int32_t tgt = target;
    CUSV_CHECK(custatevecApplyMatrix(
        handle, d_sv, CUDA_C_64F, nqubits,
        matrix, CUDA_C_64F, CUSTATEVEC_MATRIX_LAYOUT_ROW,
        0,          // adjoint
        &tgt, 1,    // targets
        nullptr, nullptr, 0,  // no controls
        CUSTATEVEC_COMPUTE_64F,
        nullptr, 0  // no extra workspace
    ));
}

// ---------------------------------------------------------------------------
// Wrapper to apply CX (controlled-X)
// ---------------------------------------------------------------------------

static void apply_cx(custatevecHandle_t handle, void* d_sv, int nqubits,
                     int control, int target) {
    int32_t tgts[2] = {control, target};
    CUSV_CHECK(custatevecApplyMatrix(
        handle, d_sv, CUDA_C_64F, nqubits,
        CX_MATRIX, CUDA_C_64F, CUSTATEVEC_MATRIX_LAYOUT_ROW,
        0,          // adjoint
        tgts, 2,    // targets (2-qubit gate)
        nullptr, nullptr, 0,  // no controls
        CUSTATEVEC_COMPUTE_64F,
        nullptr, 0  // no extra workspace
    ));
}

// ---------------------------------------------------------------------------
// Initialize state vector to |0...0>
// ---------------------------------------------------------------------------

static void init_zero_state(void* d_sv, int nqubits) {
    size_t num_amps = 1ULL << nqubits;
    CUDA_CHECK(cudaMemset(d_sv, 0, num_amps * sizeof(Complex2)));
    Complex2 one = {1.0, 0.0};
    CUDA_CHECK(cudaMemcpy(d_sv, &one, sizeof(Complex2), cudaMemcpyHostToDevice));
}

// ---------------------------------------------------------------------------
// Circuit: layered H + RZ + CX
// ---------------------------------------------------------------------------

static void run_circuit(custatevecHandle_t handle, void* d_sv,
                        int nqubits, int nlayers) {
    Complex2 rz[4];
    make_rz_matrix(0.1, rz);

    for (int layer = 0; layer < nlayers; layer++) {
        for (int q = 0; q < nqubits; q++) {
            apply_1q(handle, d_sv, nqubits, H_MATRIX, q);
            apply_1q(handle, d_sv, nqubits, rz, q);
        }
        for (int q = 0; q < nqubits - 1; q++) {
            apply_cx(handle, d_sv, nqubits, q, q + 1);
        }
    }
}

// ---------------------------------------------------------------------------
// Circuit benchmark
// ---------------------------------------------------------------------------

static void bench_circuit(custatevecHandle_t handle, int nqubits, int nlayers, int reps) {
    size_t num_amps = 1ULL << nqubits;
    void* d_sv;
    CUDA_CHECK(cudaMalloc(&d_sv, num_amps * sizeof(Complex2)));

    std::vector<double> times(reps);

    for (int r = 0; r < reps; r++) {
        init_zero_state(d_sv, nqubits);
        CUDA_CHECK(cudaDeviceSynchronize());
        double t0 = now_sec();
        run_circuit(handle, d_sv, nqubits, nlayers);
        CUDA_CHECK(cudaDeviceSynchronize());
        double t1 = now_sec();
        times[r] = t1 - t0;
    }

    double med = median(times);
    printf("circuit  %2dq %2dl  %12.3f us\n", nqubits, nlayers, med * 1e6);
    CUDA_CHECK(cudaFree(d_sv));
}

// ---------------------------------------------------------------------------
// Individual gate benchmarks
// ---------------------------------------------------------------------------

static void bench_gate_h(custatevecHandle_t handle, int nqubits, int iters, int reps) {
    size_t num_amps = 1ULL << nqubits;
    void* d_sv;
    CUDA_CHECK(cudaMalloc(&d_sv, num_amps * sizeof(Complex2)));
    init_zero_state(d_sv, nqubits);

    std::vector<double> times(reps);
    for (int r = 0; r < reps; r++) {
        CUDA_CHECK(cudaDeviceSynchronize());
        double t0 = now_sec();
        for (int i = 0; i < iters; i++)
            for (int q = 0; q < nqubits; q++)
                apply_1q(handle, d_sv, nqubits, H_MATRIX, q);
        CUDA_CHECK(cudaDeviceSynchronize());
        double t1 = now_sec();
        times[r] = t1 - t0;
    }
    printf("gate     H        %12.3f us\n", median(times) * 1e6);
    CUDA_CHECK(cudaFree(d_sv));
}

static void bench_gate_x(custatevecHandle_t handle, int nqubits, int iters, int reps) {
    size_t num_amps = 1ULL << nqubits;
    void* d_sv;
    CUDA_CHECK(cudaMalloc(&d_sv, num_amps * sizeof(Complex2)));
    init_zero_state(d_sv, nqubits);

    std::vector<double> times(reps);
    for (int r = 0; r < reps; r++) {
        CUDA_CHECK(cudaDeviceSynchronize());
        double t0 = now_sec();
        for (int i = 0; i < iters; i++)
            for (int q = 0; q < nqubits; q++)
                apply_1q(handle, d_sv, nqubits, X_MATRIX, q);
        CUDA_CHECK(cudaDeviceSynchronize());
        double t1 = now_sec();
        times[r] = t1 - t0;
    }
    printf("gate     X        %12.3f us\n", median(times) * 1e6);
    CUDA_CHECK(cudaFree(d_sv));
}

static void bench_gate_cx(custatevecHandle_t handle, int nqubits, int iters, int reps) {
    size_t num_amps = 1ULL << nqubits;
    void* d_sv;
    CUDA_CHECK(cudaMalloc(&d_sv, num_amps * sizeof(Complex2)));
    init_zero_state(d_sv, nqubits);

    std::vector<double> times(reps);
    for (int r = 0; r < reps; r++) {
        CUDA_CHECK(cudaDeviceSynchronize());
        double t0 = now_sec();
        for (int i = 0; i < iters; i++)
            for (int q = 0; q < nqubits - 1; q++)
                apply_cx(handle, d_sv, nqubits, q, q + 1);
        CUDA_CHECK(cudaDeviceSynchronize());
        double t1 = now_sec();
        times[r] = t1 - t0;
    }
    printf("gate     CX       %12.3f us\n", median(times) * 1e6);
    CUDA_CHECK(cudaFree(d_sv));
}

static void bench_gate_rz(custatevecHandle_t handle, int nqubits, int iters, int reps) {
    size_t num_amps = 1ULL << nqubits;
    void* d_sv;
    CUDA_CHECK(cudaMalloc(&d_sv, num_amps * sizeof(Complex2)));
    init_zero_state(d_sv, nqubits);

    Complex2 rz[4];
    make_rz_matrix(0.1, rz);

    std::vector<double> times(reps);
    for (int r = 0; r < reps; r++) {
        CUDA_CHECK(cudaDeviceSynchronize());
        double t0 = now_sec();
        for (int i = 0; i < iters; i++)
            for (int q = 0; q < nqubits; q++)
                apply_1q(handle, d_sv, nqubits, rz, q);
        CUDA_CHECK(cudaDeviceSynchronize());
        double t1 = now_sec();
        times[r] = t1 - t0;
    }
    printf("gate     RZ       %12.3f us\n", median(times) * 1e6);
    CUDA_CHECK(cudaFree(d_sv));
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main() {
    custatevecHandle_t handle;
    CUSV_CHECK(custatevecCreate(&handle));

    int reps = 5;

    printf("=== cuStateVec standalone benchmarks (f64) ===\n");
    printf("\n-- Layered circuits (median of %d runs) --\n", reps);

    int configs[][2] = {
        {10, 20}, {14, 20}, {18, 20}, {20, 20}, {22, 20}, {24, 10}, {26, 5}
    };
    int n_configs = sizeof(configs) / sizeof(configs[0]);

    for (int i = 0; i < n_configs; i++) {
        bench_circuit(handle, configs[i][0], configs[i][1], reps);
    }

    printf("\n-- Individual gates at 18 qubits, 100 iters (median of %d runs) --\n", reps);
    bench_gate_h(handle, 18, 100, reps);
    bench_gate_x(handle, 18, 100, reps);
    bench_gate_cx(handle, 18, 100, reps);
    bench_gate_rz(handle, 18, 100, reps);

    CUSV_CHECK(custatevecDestroy(handle));
    return 0;
}
