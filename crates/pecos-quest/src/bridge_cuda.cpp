//! GPU-specific bridge for PECOS QuEST
//!
//! This file is compiled into a separate shared library (libpecos_quest_cuda.so)
//! that is loaded at runtime via dlopen when GPU acceleration is requested.
//! This allows the main library to work on systems without CUDA installed.
//!
//! Note: This file is intentionally self-contained and does not depend on
//! quest_ffi.h or CXX bridge headers, as it needs to compile independently
//! with nvcc for CUDA support.

#include "quest.h"

#include <cstdint>
#include <stdexcept>
#include <mutex>
#include <atomic>

// GPU environment info structure - must match Rust's CudaEnvInfo in cuda_loader.rs
struct CudaEnvInfo {
    bool is_multithreaded;
    bool is_gpu_accelerated;
    bool is_distributed;
    int32_t rank;
    int32_t num_nodes;
};

// Global singleton QuEST environment management for GPU
// Same pattern as bridge.cpp but for the GPU library
class GpuGlobalQuestEnv {
private:
    static std::mutex init_mutex;
    static std::atomic<bool> is_initialized;
    static std::atomic<int> ref_count;
    static QuESTEnv* global_env_ptr;

    GpuGlobalQuestEnv() = delete;

public:
    static QuESTEnv& getInstance() {
        std::lock_guard<std::mutex> lock(init_mutex);

        if (!is_initialized.load()) {
            // Initialize QuEST environment only once per process
            initQuESTEnv();
            global_env_ptr = new QuESTEnv(getQuESTEnv());
            is_initialized = true;
        }

        return *global_env_ptr;
    }

    static void addRef() {
        std::lock_guard<std::mutex> lock(init_mutex);
        ref_count++;
    }

    static void releaseRef() {
        std::lock_guard<std::mutex> lock(init_mutex);
        ref_count--;
        // Never finalize - let process termination handle it
    }
};

// Static member definitions
std::mutex GpuGlobalQuestEnv::init_mutex;
std::atomic<bool> GpuGlobalQuestEnv::is_initialized(false);
std::atomic<int> GpuGlobalQuestEnv::ref_count(0);
QuESTEnv* GpuGlobalQuestEnv::global_env_ptr = nullptr;

// GPU environment handle
struct GpuQuestEnvHandle {
    QuESTEnv cached_env;

    GpuQuestEnvHandle() {
        cached_env = GpuGlobalQuestEnv::getInstance();
        GpuGlobalQuestEnv::addRef();
    }

    ~GpuQuestEnvHandle() {
        GpuGlobalQuestEnv::releaseRef();
    }

    // Non-copyable
    GpuQuestEnvHandle(const GpuQuestEnvHandle&) = delete;
    GpuQuestEnvHandle& operator=(const GpuQuestEnvHandle&) = delete;

    QuESTEnv& getEnv() { return cached_env; }
};

// GPU Qureg handle
struct GpuQuregHandle {
    Qureg qureg;
    bool owned;

    GpuQuregHandle(int numQubits, bool isDensity) : owned(true) {
        if (isDensity) {
            qureg = createDensityQureg(numQubits);
        } else {
            qureg = createQureg(numQubits);
        }
    }

    ~GpuQuregHandle() {
        if (owned && qureg.cpuAmps != nullptr) {
            destroyQureg(qureg);
        }
    }

    // Non-copyable
    GpuQuregHandle(const GpuQuregHandle&) = delete;
    GpuQuregHandle& operator=(const GpuQuregHandle&) = delete;
};

// Export C functions for dlopen
extern "C" {

// Environment management
void* pecos_quest_cuda_create_env() {
    try {
        return reinterpret_cast<void*>(new GpuQuestEnvHandle());
    } catch (const std::exception& e) {
        return nullptr;
    }
}

void pecos_quest_cuda_destroy_env(void* env) {
    if (env) {
        delete reinterpret_cast<GpuQuestEnvHandle*>(env);
    }
}

CudaEnvInfo pecos_quest_cuda_get_env_info(void* env) {
    auto* handle = reinterpret_cast<GpuQuestEnvHandle*>(env);
    QuESTEnv& questEnv = handle->getEnv();

    CudaEnvInfo info;
    info.is_multithreaded = questEnv.isMultithreaded != 0;
    info.is_gpu_accelerated = questEnv.isGpuAccelerated != 0;
    info.is_distributed = questEnv.isDistributed != 0;
    info.rank = questEnv.rank;
    info.num_nodes = questEnv.numNodes;
    return info;
}

// Qureg management
void* pecos_quest_cuda_create_qureg(void* env, int32_t numQubits) {
    if (numQubits < 1) {
        return nullptr;
    }
    try {
        return reinterpret_cast<void*>(new GpuQuregHandle(numQubits, false));
    } catch (const std::exception& e) {
        return nullptr;
    }
}

void* pecos_quest_cuda_create_density_qureg(void* env, int32_t numQubits) {
    if (numQubits < 1) {
        return nullptr;
    }
    try {
        return reinterpret_cast<void*>(new GpuQuregHandle(numQubits, true));
    } catch (const std::exception& e) {
        return nullptr;
    }
}

void pecos_quest_cuda_destroy_qureg(void* qureg) {
    if (qureg) {
        delete reinterpret_cast<GpuQuregHandle*>(qureg);
    }
}

// State initialization
void pecos_quest_cuda_init_zero_state(void* qureg) {
    auto* handle = reinterpret_cast<GpuQuregHandle*>(qureg);
    initZeroState(handle->qureg);
}

void pecos_quest_cuda_init_plus_state(void* qureg) {
    auto* handle = reinterpret_cast<GpuQuregHandle*>(qureg);
    initPlusState(handle->qureg);
}

void pecos_quest_cuda_init_classical_state(void* qureg, int64_t stateInd) {
    auto* handle = reinterpret_cast<GpuQuregHandle*>(qureg);
    initClassicalState(handle->qureg, stateInd);
}

// Single-qubit gates
void pecos_quest_cuda_apply_pauli_x(void* qureg, int32_t qubit) {
    auto* handle = reinterpret_cast<GpuQuregHandle*>(qureg);
    applyPauliX(handle->qureg, qubit);
}

void pecos_quest_cuda_apply_pauli_y(void* qureg, int32_t qubit) {
    auto* handle = reinterpret_cast<GpuQuregHandle*>(qureg);
    applyPauliY(handle->qureg, qubit);
}

void pecos_quest_cuda_apply_pauli_z(void* qureg, int32_t qubit) {
    auto* handle = reinterpret_cast<GpuQuregHandle*>(qureg);
    applyPauliZ(handle->qureg, qubit);
}

void pecos_quest_cuda_apply_hadamard(void* qureg, int32_t qubit) {
    auto* handle = reinterpret_cast<GpuQuregHandle*>(qureg);
    applyHadamard(handle->qureg, qubit);
}

void pecos_quest_cuda_apply_s_gate(void* qureg, int32_t qubit) {
    auto* handle = reinterpret_cast<GpuQuregHandle*>(qureg);
    applyS(handle->qureg, qubit);
}

void pecos_quest_cuda_apply_t_gate(void* qureg, int32_t qubit) {
    auto* handle = reinterpret_cast<GpuQuregHandle*>(qureg);
    applyT(handle->qureg, qubit);
}

void pecos_quest_cuda_apply_phase_shift(void* qureg, int32_t qubit, double angle) {
    auto* handle = reinterpret_cast<GpuQuregHandle*>(qureg);
    applyPhaseShift(handle->qureg, qubit, angle);
}

// Rotation gates
void pecos_quest_cuda_apply_rotation_x(void* qureg, int32_t qubit, double angle) {
    auto* handle = reinterpret_cast<GpuQuregHandle*>(qureg);
    applyRotateX(handle->qureg, qubit, angle);
}

void pecos_quest_cuda_apply_rotation_y(void* qureg, int32_t qubit, double angle) {
    auto* handle = reinterpret_cast<GpuQuregHandle*>(qureg);
    applyRotateY(handle->qureg, qubit, angle);
}

void pecos_quest_cuda_apply_rotation_z(void* qureg, int32_t qubit, double angle) {
    auto* handle = reinterpret_cast<GpuQuregHandle*>(qureg);
    applyRotateZ(handle->qureg, qubit, angle);
}

// Two-qubit gates
void pecos_quest_cuda_apply_cnot(void* qureg, int32_t control, int32_t target) {
    auto* handle = reinterpret_cast<GpuQuregHandle*>(qureg);
    applyControlledPauliX(handle->qureg, control, target);
}

void pecos_quest_cuda_apply_cz(void* qureg, int32_t control, int32_t target) {
    auto* handle = reinterpret_cast<GpuQuregHandle*>(qureg);
    applyTwoQubitPhaseFlip(handle->qureg, control, target);
}

void pecos_quest_cuda_apply_swap(void* qureg, int32_t qubit1, int32_t qubit2) {
    auto* handle = reinterpret_cast<GpuQuregHandle*>(qureg);
    applySwap(handle->qureg, qubit1, qubit2);
}

void pecos_quest_cuda_apply_controlled_phase_shift(void* qureg, int32_t control, int32_t target, double angle) {
    auto* handle = reinterpret_cast<GpuQuregHandle*>(qureg);
    applyTwoQubitPhaseShift(handle->qureg, control, target, angle);
}

// Measurement
int32_t pecos_quest_cuda_measure(void* qureg, int32_t qubit) {
    auto* handle = reinterpret_cast<GpuQuregHandle*>(qureg);
    return applyQubitMeasurement(handle->qureg, qubit);
}

double pecos_quest_cuda_calc_prob_of_outcome(void* qureg, int32_t qubit, int32_t outcome) {
    auto* handle = reinterpret_cast<GpuQuregHandle*>(qureg);
    return calcProbOfQubitOutcome(handle->qureg, qubit, outcome);
}

double pecos_quest_cuda_apply_forced_measurement(void* qureg, int32_t qubit, int32_t outcome) {
    auto* handle = reinterpret_cast<GpuQuregHandle*>(qureg);
    return applyForcedQubitMeasurement(handle->qureg, qubit, outcome);
}

// Amplitude access
double pecos_quest_cuda_get_real_amp(void* qureg, int64_t index) {
    auto* handle = reinterpret_cast<GpuQuregHandle*>(qureg);
    return real(getQuregAmp(handle->qureg, index));
}

double pecos_quest_cuda_get_imag_amp(void* qureg, int64_t index) {
    auto* handle = reinterpret_cast<GpuQuregHandle*>(qureg);
    return imag(getQuregAmp(handle->qureg, index));
}

double pecos_quest_cuda_get_prob_amp(void* qureg, int64_t index) {
    auto* handle = reinterpret_cast<GpuQuregHandle*>(qureg);
    return calcProbOfBasisState(handle->qureg, index);
}

double pecos_quest_cuda_calc_total_prob(void* qureg) {
    auto* handle = reinterpret_cast<GpuQuregHandle*>(qureg);
    return calcTotalProb(handle->qureg);
}

double pecos_quest_cuda_calc_purity(void* qureg) {
    auto* handle = reinterpret_cast<GpuQuregHandle*>(qureg);
    return calcPurity(handle->qureg);
}

// Info
int64_t pecos_quest_cuda_get_num_amps(void* qureg) {
    auto* handle = reinterpret_cast<GpuQuregHandle*>(qureg);
    return handle->qureg.numAmps;
}

int32_t pecos_quest_cuda_get_num_qubits(void* qureg) {
    auto* handle = reinterpret_cast<GpuQuregHandle*>(qureg);
    return handle->qureg.numQubits;
}

bool pecos_quest_cuda_is_density_matrix(void* qureg) {
    auto* handle = reinterpret_cast<GpuQuregHandle*>(qureg);
    return handle->qureg.isDensityMatrix != 0;
}

} // extern "C"
