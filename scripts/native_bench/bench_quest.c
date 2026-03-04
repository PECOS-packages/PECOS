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

// Standalone QuEST v4 benchmark using the native C API.
// Compiled and linked against a CMake-built QuEST library so that build flags
// are entirely under CMake's control (no Rust build.rs involvement).

#define _POSIX_C_SOURCE 199309L

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "quest.h"

// ---------------------------------------------------------------------------
// Timing helpers
// ---------------------------------------------------------------------------

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static int cmp_double(const void *a, const void *b) {
    double da = *(const double *)a;
    double db = *(const double *)b;
    return (da > db) - (da < db);
}

static double median(double *vals, int n) {
    qsort(vals, (size_t)n, sizeof(double), cmp_double);
    if (n % 2 == 1) return vals[n / 2];
    return (vals[n / 2 - 1] + vals[n / 2]) / 2.0;
}

// ---------------------------------------------------------------------------
// Circuit: layered H + RZ + CX
// ---------------------------------------------------------------------------

static void run_circuit(Qureg q, int num_qubits, int num_layers) {
    for (int layer = 0; layer < num_layers; layer++) {
        for (int qb = 0; qb < num_qubits; qb++) {
            applyHadamard(q, qb);
            applyRotateZ(q, qb, 0.1);
        }
        for (int qb = 0; qb < num_qubits - 1; qb++) {
            applyControlledPauliX(q, qb, qb + 1);
        }
    }
}

// ---------------------------------------------------------------------------
// Layered circuit benchmark
// ---------------------------------------------------------------------------

static void bench_circuit(int num_qubits, int num_layers, int reps) {
    Qureg q = createQureg(num_qubits);
    double times[reps];

    for (int r = 0; r < reps; r++) {
        initZeroState(q);
        double t0 = now_sec();
        run_circuit(q, num_qubits, num_layers);
        double t1 = now_sec();
        times[r] = t1 - t0;
    }

    double med = median(times, reps);
    printf("circuit  %2dq %2dl  %12.3f us\n", num_qubits, num_layers, med * 1e6);
    destroyQureg(q);
}

// ---------------------------------------------------------------------------
// Individual gate benchmarks (18 qubits, 100 iterations)
// ---------------------------------------------------------------------------

static void bench_gate_h(int num_qubits, int iters, int reps) {
    Qureg q = createQureg(num_qubits);
    initZeroState(q);
    double times[reps];

    for (int r = 0; r < reps; r++) {
        double t0 = now_sec();
        for (int i = 0; i < iters; i++)
            for (int qb = 0; qb < num_qubits; qb++)
                applyHadamard(q, qb);
        double t1 = now_sec();
        times[r] = t1 - t0;
    }

    printf("gate     H        %12.3f us\n", median(times, reps) * 1e6);
    destroyQureg(q);
}

static void bench_gate_x(int num_qubits, int iters, int reps) {
    Qureg q = createQureg(num_qubits);
    initZeroState(q);
    double times[reps];

    for (int r = 0; r < reps; r++) {
        double t0 = now_sec();
        for (int i = 0; i < iters; i++)
            for (int qb = 0; qb < num_qubits; qb++)
                applyPauliX(q, qb);
        double t1 = now_sec();
        times[r] = t1 - t0;
    }

    printf("gate     X        %12.3f us\n", median(times, reps) * 1e6);
    destroyQureg(q);
}

static void bench_gate_cx(int num_qubits, int iters, int reps) {
    Qureg q = createQureg(num_qubits);
    initZeroState(q);
    double times[reps];

    for (int r = 0; r < reps; r++) {
        double t0 = now_sec();
        for (int i = 0; i < iters; i++)
            for (int qb = 0; qb < num_qubits - 1; qb++)
                applyControlledPauliX(q, qb, qb + 1);
        double t1 = now_sec();
        times[r] = t1 - t0;
    }

    printf("gate     CX       %12.3f us\n", median(times, reps) * 1e6);
    destroyQureg(q);
}

static void bench_gate_rz(int num_qubits, int iters, int reps) {
    Qureg q = createQureg(num_qubits);
    initZeroState(q);
    double times[reps];

    for (int r = 0; r < reps; r++) {
        double t0 = now_sec();
        for (int i = 0; i < iters; i++)
            for (int qb = 0; qb < num_qubits; qb++)
                applyRotateZ(q, qb, 0.1);
        double t1 = now_sec();
        times[r] = t1 - t0;
    }

    printf("gate     RZ       %12.3f us\n", median(times, reps) * 1e6);
    destroyQureg(q);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(void) {
    initQuESTEnv();

    int reps = 5;

    printf("=== QuEST v4 standalone benchmarks ===\n");
    printf("\n-- Layered circuits (median of %d runs) --\n", reps);

    int configs[][2] = {
        {10, 20}, {14, 20}, {18, 20}, {20, 20}, {22, 10}, {24, 5}
    };
    int n_configs = sizeof(configs) / sizeof(configs[0]);

    for (int i = 0; i < n_configs; i++) {
        bench_circuit(configs[i][0], configs[i][1], reps);
    }

    printf("\n-- Individual gates at 18 qubits, 100 iters (median of %d runs) --\n", reps);
    bench_gate_h(18, 100, reps);
    bench_gate_x(18, 100, reps);
    bench_gate_cx(18, 100, reps);
    bench_gate_rz(18, 100, reps);

    finalizeQuESTEnv();
    return 0;
}
