//! C++ bridge implementation for QuEST with independent simulator instances
//! Each simulator gets its own independent Qureg, but they share a global QuEST environment
//! since QuEST only supports one environment per process.

#include "quest_ffi.h"
#include "quest.h"
// Note: quest_ffi.h includes the cxx-generated header and rust/cxx.h before <memory>

#include <stdexcept>
#include <vector>
#include <cstring>
#include <mutex>
#include <atomic>

// Global singleton QuEST environment management
// QuEST requires a single global environment, but Quregs are independent

class GlobalQuestEnv {
private:
    static std::mutex init_mutex;
    static std::atomic<bool> is_initialized;
    static std::atomic<int> ref_count;
    static QuESTEnv* global_env_ptr;

    GlobalQuestEnv() = delete;

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
        // This avoids re-initialization issues in tests
    }
};

// Static member definitions
std::mutex GlobalQuestEnv::init_mutex;
std::atomic<bool> GlobalQuestEnv::is_initialized(false);
std::atomic<int> GlobalQuestEnv::ref_count(0);
QuESTEnv* GlobalQuestEnv::global_env_ptr = nullptr;

// Environment handle that each simulator instance gets
// This provides the illusion of independent environments while sharing the global one
struct QuestEnvHandle {
    QuESTEnv cached_env;  // Cache a copy for thread-safe access

    QuestEnvHandle() {
        cached_env = GlobalQuestEnv::getInstance();
        GlobalQuestEnv::addRef();
    }

    ~QuestEnvHandle() {
        GlobalQuestEnv::releaseRef();
    }

    // Make it non-copyable but moveable
    QuestEnvHandle(const QuestEnvHandle&) = delete;
    QuestEnvHandle& operator=(const QuestEnvHandle&) = delete;
    QuestEnvHandle(QuestEnvHandle&& other) noexcept
        : cached_env(other.cached_env) {
        // Transfer ownership
        other.cached_env = {};
    }
    QuestEnvHandle& operator=(QuestEnvHandle&& other) noexcept {
        if (this != &other) {
            GlobalQuestEnv::releaseRef();
            cached_env = other.cached_env;
            other.cached_env = {};
        }
        return *this;
    }

    QuESTEnv& getEnv() { return cached_env; }
};

// Simple handle struct that owns a QuEST Qureg
struct QuregHandle {
    Qureg qureg;
    bool owned;

    QuregHandle(int numQubits, bool isDensity) : owned(true) {
        if (isDensity) {
            qureg = createDensityQureg(numQubits);
            // Initialization will be done from Rust
        } else {
            qureg = createQureg(numQubits);
            // Initialization will be done from Rust
        }
    }

    QuregHandle(const Qureg& q) : qureg(q), owned(false) {}

    ~QuregHandle() {
        if (owned && qureg.cpuAmps != nullptr) {
            destroyQureg(qureg);
        }
    }

    // Make it non-copyable but moveable
    QuregHandle(const QuregHandle&) = delete;
    QuregHandle& operator=(const QuregHandle&) = delete;
    QuregHandle(QuregHandle&&) = default;
    QuregHandle& operator=(QuregHandle&&) = default;
};

// Environment management functions
uint8_t* quest_create_env() {
    try {
        return reinterpret_cast<uint8_t*>(new QuestEnvHandle());
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("Failed to create QuEST environment: ") + e.what());
    }
}

void quest_destroy_env(uint8_t* env) {
    if (env) {
        delete reinterpret_cast<QuestEnvHandle*>(env);
    }
}

QuESTEnvInfo quest_get_env_info(uint8_t* env) {
    auto* handle = reinterpret_cast<QuestEnvHandle*>(env);
    QuESTEnv& questEnv = handle->getEnv();

    QuESTEnvInfo info;
    info.is_multithreaded = questEnv.isMultithreaded != 0;
    info.is_gpu_accelerated = questEnv.isGpuAccelerated != 0;
    info.is_distributed = questEnv.isDistributed != 0;
    info.rank = questEnv.rank;
    info.num_nodes = questEnv.numNodes;
    return info;
}

void quest_sync_env(uint8_t* env) {
    // For thread-safe usage, we avoid global sync operations
    // Each thread works independently
}

// Qureg creation and destruction - each is completely independent
uint8_t* quest_create_qureg(uint8_t* env, int32_t numQubits) {
    if (numQubits < 1) {
        throw std::invalid_argument("Number of qubits must be at least 1");
    }
    try {
        return reinterpret_cast<uint8_t*>(new QuregHandle(numQubits, false));
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("Failed to create qureg: ") + e.what());
    }
}

uint8_t* quest_create_density_qureg(uint8_t* env, int32_t numQubits) {
    if (numQubits < 1) {
        throw std::invalid_argument("Number of qubits must be at least 1");
    }
    try {
        return reinterpret_cast<uint8_t*>(new QuregHandle(numQubits, true));
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("Failed to create density qureg: ") + e.what());
    }
}

void quest_destroy_qureg(uint8_t* qureg) {
    if (qureg) {
        delete reinterpret_cast<QuregHandle*>(qureg);
    }
}

uint8_t* quest_clone_qureg(uint8_t* qureg) {
    auto* handle = reinterpret_cast<QuregHandle*>(qureg);
    try {
        // Note: QuregHandle constructor will lock for creation
        auto* cloned = new QuregHandle(handle->qureg.numQubits, handle->qureg.isDensityMatrix != 0);
        {
            setQuregToClone(cloned->qureg, handle->qureg);
        }
        return reinterpret_cast<uint8_t*>(cloned);
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("Failed to clone qureg: ") + e.what());
    }
}

QuregInfo quest_get_qureg_info(uint8_t* qureg) {
    auto* handle = reinterpret_cast<QuregHandle*>(qureg);
    QuregInfo info;
    info.num_qubits = handle->qureg.numQubits;
    info.num_amps = handle->qureg.numAmps;
    info.is_density_matrix = handle->qureg.isDensityMatrix != 0;
    return info;
}

// State initialization - operates on independent Quregs
void quest_init_zero_state(uint8_t* qureg) {
    auto* handle = reinterpret_cast<QuregHandle*>(qureg);

    // Initialize to |00...0⟩ state
    initZeroState(handle->qureg);
}

void quest_init_plus_state(uint8_t* qureg) {
    auto* handle = reinterpret_cast<QuregHandle*>(qureg);
    initPlusState(handle->qureg);
}

void quest_init_classical_state(uint8_t* qureg, int64_t stateInd) {
    auto* handle = reinterpret_cast<QuregHandle*>(qureg);
    initClassicalState(handle->qureg, stateInd);
}

void quest_init_pure_state(uint8_t* qureg, uint8_t* pureQureg) {
    auto* handle = reinterpret_cast<QuregHandle*>(qureg);
    auto* pureHandle = reinterpret_cast<QuregHandle*>(pureQureg);
    initPureState(handle->qureg, pureHandle->qureg);
}

void quest_init_random_state(uint8_t* qureg, rust::Slice<const uint64_t> seed) {
    auto* handle = reinterpret_cast<QuregHandle*>(qureg);
    // Convert seed to QuEST format
    std::vector<unsigned long> questSeed;
    for (auto s : seed) {
        questSeed.push_back(static_cast<unsigned long>(s));
    }
    // Each qureg gets its own random state, completely independent
    // Note: QuEST v4 doesn't use seed arrays, just call initRandomPureState
    initRandomPureState(handle->qureg);
}

// All quantum operations operate on independent Quregs
void quest_apply_pauli_x(uint8_t* qureg, int32_t qubit) {
    auto* handle = reinterpret_cast<QuregHandle*>(qureg);
    applyPauliX(handle->qureg, qubit);
}

void quest_apply_pauli_y(uint8_t* qureg, int32_t qubit) {
    auto* handle = reinterpret_cast<QuregHandle*>(qureg);
    applyPauliY(handle->qureg, qubit);
}

void quest_apply_pauli_z(uint8_t* qureg, int32_t qubit) {
    auto* handle = reinterpret_cast<QuregHandle*>(qureg);
    applyPauliZ(handle->qureg, qubit);
}

void quest_apply_hadamard(uint8_t* qureg, int32_t qubit) {
    auto* handle = reinterpret_cast<QuregHandle*>(qureg);
    applyHadamard(handle->qureg, qubit);
}

void quest_apply_s_gate(uint8_t* qureg, int32_t qubit) {
    auto* handle = reinterpret_cast<QuregHandle*>(qureg);
    applyS(handle->qureg, qubit);
}

void quest_apply_t_gate(uint8_t* qureg, int32_t qubit) {
    auto* handle = reinterpret_cast<QuregHandle*>(qureg);
    applyT(handle->qureg, qubit);
}

void quest_apply_phase_shift(uint8_t* qureg, int32_t qubit, double angle) {
    auto* handle = reinterpret_cast<QuregHandle*>(qureg);
    applyPhaseShift(handle->qureg, qubit, angle);
}

void quest_apply_rotation_x(uint8_t* qureg, int32_t qubit, double angle) {
    auto* handle = reinterpret_cast<QuregHandle*>(qureg);
    applyRotateX(handle->qureg, qubit, angle);
}

void quest_apply_rotation_y(uint8_t* qureg, int32_t qubit, double angle) {
    auto* handle = reinterpret_cast<QuregHandle*>(qureg);
    applyRotateY(handle->qureg, qubit, angle);
}

void quest_apply_rotation_z(uint8_t* qureg, int32_t qubit, double angle) {
    auto* handle = reinterpret_cast<QuregHandle*>(qureg);
    applyRotateZ(handle->qureg, qubit, angle);
}

void quest_apply_cnot(uint8_t* qureg, int32_t control, int32_t target) {
    auto* handle = reinterpret_cast<QuregHandle*>(qureg);
    applyControlledPauliX(handle->qureg, control, target);
}

void quest_apply_cz(uint8_t* qureg, int32_t control, int32_t target) {
    auto* handle = reinterpret_cast<QuregHandle*>(qureg);
    applyTwoQubitPhaseFlip(handle->qureg, control, target);
}

void quest_apply_swap(uint8_t* qureg, int32_t qubit1, int32_t qubit2) {
    auto* handle = reinterpret_cast<QuregHandle*>(qureg);
    applySwap(handle->qureg, qubit1, qubit2);
}

void quest_apply_controlled_phase_shift(uint8_t* qureg, int32_t control, int32_t target, double angle) {
    auto* handle = reinterpret_cast<QuregHandle*>(qureg);
    applyTwoQubitPhaseShift(handle->qureg, control, target, angle);
}

void quest_apply_multi_controlled_pauli_z(uint8_t* qureg, rust::Slice<const int32_t> controls, int32_t target) {
    auto* handle = reinterpret_cast<QuregHandle*>(qureg);
    std::vector<int> controlVec(controls.data(), controls.data() + controls.size());
    applyMultiControlledPauliZ(handle->qureg, controlVec.data(), controlVec.size(), target);
}

// Measurements - each qureg maintains its own state
int32_t quest_measure(uint8_t* qureg, int32_t qubit) {
    auto* handle = reinterpret_cast<QuregHandle*>(qureg);
    return applyQubitMeasurement(handle->qureg, qubit);
}

int32_t quest_measure_with_stats(uint8_t* qureg, int32_t qubit, double& outcomeProb) {
    auto* handle = reinterpret_cast<QuregHandle*>(qureg);
    return applyQubitMeasurementAndGetProb(handle->qureg, qubit, &outcomeProb);
}

double quest_calc_prob_of_outcome(uint8_t* qureg, int32_t qubit, int32_t outcome) {
    auto* handle = reinterpret_cast<QuregHandle*>(qureg);
    return calcProbOfQubitOutcome(handle->qureg, qubit, outcome);
}

double quest_apply_forced_measurement(uint8_t* qureg, int32_t qubit, int32_t outcome) {
    auto* handle = reinterpret_cast<QuregHandle*>(qureg);
    return applyForcedQubitMeasurement(handle->qureg, qubit, outcome);
}

double quest_calc_total_prob(uint8_t* qureg) {
    auto* handle = reinterpret_cast<QuregHandle*>(qureg);
    return calcTotalProb(handle->qureg);
}

// Amplitude access - read-only operations on independent states
double quest_get_real_amp(uint8_t* qureg, int64_t index) {
    auto* handle = reinterpret_cast<QuregHandle*>(qureg);
    return real(getQuregAmp(handle->qureg, index));
}

double quest_get_imag_amp(uint8_t* qureg, int64_t index) {
    auto* handle = reinterpret_cast<QuregHandle*>(qureg);
    return imag(getQuregAmp(handle->qureg, index));
}

Complex quest_get_complex_amp(uint8_t* qureg, int64_t index) {
    auto* handle = reinterpret_cast<QuregHandle*>(qureg);
    qcomp amp = getQuregAmp(handle->qureg, index);
    Complex result;
    result.real = real(amp);
    result.imag = imag(amp);
    return result;
}

double quest_get_prob_amp(uint8_t* qureg, int64_t index) {
    auto* handle = reinterpret_cast<QuregHandle*>(qureg);
    return calcProbOfBasisState(handle->qureg, index);
}

int64_t quest_get_num_amps(uint8_t* qureg) {
    auto* handle = reinterpret_cast<QuregHandle*>(qureg);
    return handle->qureg.numAmps;
}

int32_t quest_get_num_qubits(uint8_t* qureg) {
    auto* handle = reinterpret_cast<QuregHandle*>(qureg);
    return handle->qureg.numQubits;
}

bool quest_is_density_matrix(uint8_t* qureg) {
    auto* handle = reinterpret_cast<QuregHandle*>(qureg);
    return handle->qureg.isDensityMatrix != 0;
}

// Utility functions for independent quregs
Complex quest_calc_inner_product(uint8_t* qureg1, uint8_t* qureg2) {
    auto* handle1 = reinterpret_cast<QuregHandle*>(qureg1);
    auto* handle2 = reinterpret_cast<QuregHandle*>(qureg2);
    qcomp prod = calcInnerProduct(handle1->qureg, handle2->qureg);
    Complex result;
    result.real = real(prod);
    result.imag = imag(prod);
    return result;
}

double quest_calc_fidelity(uint8_t* qureg1, uint8_t* qureg2) {
    auto* handle1 = reinterpret_cast<QuregHandle*>(qureg1);
    auto* handle2 = reinterpret_cast<QuregHandle*>(qureg2);
    return calcFidelity(handle1->qureg, handle2->qureg);
}

double quest_calc_purity(uint8_t* qureg) {
    auto* handle = reinterpret_cast<QuregHandle*>(qureg);
    return calcPurity(handle->qureg);
}
