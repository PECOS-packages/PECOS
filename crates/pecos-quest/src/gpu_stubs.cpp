// Minimal GPU stub implementations for CPU-only build
// These functions are referenced by QuEST code but not actually used in CPU mode

#include <complex>
#include <vector>
#include <cstddef>
#include <algorithm>

// Forward declare Qureg structure to match QuEST's definition in qureg.h
typedef long long qindex;
typedef std::complex<double> qcomp;

struct Qureg {
    // deployment configuration
    int isMultithreaded;
    int isGpuAccelerated;
    int isDistributed;

    // distributed configuration
    int rank;
    int numNodes;
    int logNumNodes;

    // dimension
    int isDensityMatrix;
    int numQubits;
    qindex numAmps;
    qindex logNumAmps;

    // distributed load
    qindex numAmpsPerNode;
    qindex logNumAmpsPerNode;
    qindex logNumColsPerNode;

    // amplitudes in CPU and GPU memory
    qcomp* cpuAmps;
    qcomp* gpuAmps;

    // communication buffer in CPU and GPU memory
    qcomp* cpuCommBuffer;
    qcomp* gpuCommBuffer;
};

// GPU availability functions - these use C++ linkage to match QuEST's expectations
bool gpu_isGpuCompiled() { return false; }
bool gpu_isGpuAvailable() { return false; }
bool gpu_isDirectGpuCommPossible() { return false; }
bool gpu_isCuQuantumCompiled() { return false; }
bool gpu_areAnyNodesBoundToSameGpu() { return false; }
bool gpu_doesGpuSupportMemPools() { return false; }

size_t gpu_getCurrentAvailableMemoryInBytes() { return 0; }
int gpu_getComputeCapability() { return 0; }

void gpu_bindLocalGPUsToNodes() {}
void gpu_initCuQuantum() {}

// GPU sync and memory functions
void gpu_sync() {}
std::complex<double>* gpu_allocArray(long long size) { return nullptr; }
void gpu_deallocArray(std::complex<double>* ptr) {}

// GPU copy functions - these need C++ linkage for overloading
void gpu_copyGpuToCpu(Qureg qureg) {}
void gpu_copyGpuToCpu(Qureg qureg, std::complex<double>* cpuPtr, std::complex<double>* gpuPtr, long long size) {}
void gpu_copyGpuToCpu(std::complex<double>* gpuPtr, std::complex<double>* cpuPtr, long long size) {}
void gpu_copyCpuToGpu(Qureg qureg) {}
void gpu_copyCpuToGpu(Qureg qureg, std::complex<double>* cpuPtr, std::complex<double>* gpuPtr, long long size) {}
void gpu_copyCpuToGpu(std::complex<double>* cpuPtr, std::complex<double>* gpuPtr, long long size) {}


// Most accelerator functions are now provided by accelerator.cpp
// We only need to stub functions that accelerator.cpp calls but aren't defined

void gpu_statevec_setQuregToSuperposition_sub(std::complex<double> a, Qureg q1,
    std::complex<double> b, Qureg q2, std::complex<double> c, Qureg q3) {}
void gpu_densmatr_mixQureg_subA(double a, Qureg q1, double b, Qureg q2) {}
void gpu_densmatr_mixQureg_subB(double a, Qureg q1, double b, Qureg q2) {}
void gpu_densmatr_mixQureg_subC(double a, Qureg q1, double b) {}
// Note: gpu_statevec_calcTotalProb_sub is defined later with correct return type
void gpu_statevec_initUniformState_sub(Qureg q, std::complex<double> a) {}


// Additional structures needed for templates
struct CompMatr1 {
    qcomp elems[4];
};
struct DiagMatr1 {};

// Template instantiations outside of extern "C"
// These need to exist but won't be called in CPU mode
template<int N>
long long gpu_statevec_packAmpsIntoBuffer(Qureg q, std::vector<int> a, std::vector<int> b) { return 0; }

template<int N>
void gpu_statevec_anyCtrlOneTargDenseMatr_subA(Qureg q, std::vector<int> a,
    std::vector<int> b, int c, CompMatr1 d) {}

template<int N>
void gpu_statevec_anyCtrlOneTargDenseMatr_subB(Qureg q, std::vector<int> a,
    std::vector<int> b, qcomp c, qcomp d) {}

template<int N>
void gpu_statevec_anyCtrlOneTargDiagMatr_sub(Qureg q, std::vector<int> a,
    std::vector<int> b, int c, DiagMatr1 d) {}

template<int N>
void gpu_statevector_anyCtrlAnyTargZOrPhaseGadget_sub(Qureg q, std::vector<int> a,
    std::vector<int> b, std::vector<int> c, std::complex<double> d, std::complex<double> e) {}

template<int N, int M = 0>
void gpu_statevector_anyCtrlPauliTensorOrGadget_subA(Qureg q, std::vector<int> a,
    std::vector<int> b, std::vector<int> c, std::vector<int> d,
    std::vector<int> e, std::complex<double> f, std::complex<double> g) {}

template<int N>
void gpu_statevector_anyCtrlPauliTensorOrGadget_subB(Qureg q, std::vector<int> a,
    std::vector<int> b, std::vector<int> c, std::vector<int> d,
    std::vector<int> e, std::complex<double> f, std::complex<double> g, long long h) {}

template<int N>
double gpu_statevec_calcProbOfMultiQubitOutcome_sub(Qureg q, std::vector<int> a, std::vector<int> b) { return 0.0; }

template<int N>
double gpu_densmatr_calcProbOfMultiQubitOutcome_sub(Qureg q, std::vector<int> a, std::vector<int> b) { return 0.0; }

template<int N>
void gpu_statevec_multiQubitProjector_sub(Qureg q, std::vector<int> a, std::vector<int> b, double c) {}

template<int N>
void gpu_densmatr_multiQubitProjector_sub(Qureg q, std::vector<int> a, std::vector<int> b, double c) {}

// Explicit instantiations for all the template values QuEST uses
// gpu_statevec_packAmpsIntoBuffer
template long long gpu_statevec_packAmpsIntoBuffer<0>(Qureg, std::vector<int>, std::vector<int>);
template long long gpu_statevec_packAmpsIntoBuffer<1>(Qureg, std::vector<int>, std::vector<int>);
template long long gpu_statevec_packAmpsIntoBuffer<2>(Qureg, std::vector<int>, std::vector<int>);
template long long gpu_statevec_packAmpsIntoBuffer<3>(Qureg, std::vector<int>, std::vector<int>);
template long long gpu_statevec_packAmpsIntoBuffer<4>(Qureg, std::vector<int>, std::vector<int>);
template long long gpu_statevec_packAmpsIntoBuffer<5>(Qureg, std::vector<int>, std::vector<int>);
template long long gpu_statevec_packAmpsIntoBuffer<-1>(Qureg, std::vector<int>, std::vector<int>);

// gpu_statevec_anyCtrlOneTargDenseMatr_subA
template void gpu_statevec_anyCtrlOneTargDenseMatr_subA<0>(Qureg, std::vector<int>, std::vector<int>, int, CompMatr1);
template void gpu_statevec_anyCtrlOneTargDenseMatr_subA<1>(Qureg, std::vector<int>, std::vector<int>, int, CompMatr1);
template void gpu_statevec_anyCtrlOneTargDenseMatr_subA<2>(Qureg, std::vector<int>, std::vector<int>, int, CompMatr1);
template void gpu_statevec_anyCtrlOneTargDenseMatr_subA<3>(Qureg, std::vector<int>, std::vector<int>, int, CompMatr1);
template void gpu_statevec_anyCtrlOneTargDenseMatr_subA<4>(Qureg, std::vector<int>, std::vector<int>, int, CompMatr1);
template void gpu_statevec_anyCtrlOneTargDenseMatr_subA<5>(Qureg, std::vector<int>, std::vector<int>, int, CompMatr1);
template void gpu_statevec_anyCtrlOneTargDenseMatr_subA<-1>(Qureg, std::vector<int>, std::vector<int>, int, CompMatr1);

// gpu_statevec_anyCtrlOneTargDenseMatr_subB
template void gpu_statevec_anyCtrlOneTargDenseMatr_subB<0>(Qureg, std::vector<int>, std::vector<int>, qcomp, qcomp);
template void gpu_statevec_anyCtrlOneTargDenseMatr_subB<1>(Qureg, std::vector<int>, std::vector<int>, qcomp, qcomp);
template void gpu_statevec_anyCtrlOneTargDenseMatr_subB<2>(Qureg, std::vector<int>, std::vector<int>, qcomp, qcomp);
template void gpu_statevec_anyCtrlOneTargDenseMatr_subB<3>(Qureg, std::vector<int>, std::vector<int>, qcomp, qcomp);
template void gpu_statevec_anyCtrlOneTargDenseMatr_subB<4>(Qureg, std::vector<int>, std::vector<int>, qcomp, qcomp);
template void gpu_statevec_anyCtrlOneTargDenseMatr_subB<5>(Qureg, std::vector<int>, std::vector<int>, qcomp, qcomp);
template void gpu_statevec_anyCtrlOneTargDenseMatr_subB<-1>(Qureg, std::vector<int>, std::vector<int>, qcomp, qcomp);

// gpu_statevec_anyCtrlOneTargDiagMatr_sub
template void gpu_statevec_anyCtrlOneTargDiagMatr_sub<0>(Qureg, std::vector<int>, std::vector<int>, int, DiagMatr1);
template void gpu_statevec_anyCtrlOneTargDiagMatr_sub<1>(Qureg, std::vector<int>, std::vector<int>, int, DiagMatr1);
template void gpu_statevec_anyCtrlOneTargDiagMatr_sub<2>(Qureg, std::vector<int>, std::vector<int>, int, DiagMatr1);
template void gpu_statevec_anyCtrlOneTargDiagMatr_sub<3>(Qureg, std::vector<int>, std::vector<int>, int, DiagMatr1);
template void gpu_statevec_anyCtrlOneTargDiagMatr_sub<4>(Qureg, std::vector<int>, std::vector<int>, int, DiagMatr1);
template void gpu_statevec_anyCtrlOneTargDiagMatr_sub<5>(Qureg, std::vector<int>, std::vector<int>, int, DiagMatr1);
template void gpu_statevec_anyCtrlOneTargDiagMatr_sub<-1>(Qureg, std::vector<int>, std::vector<int>, int, DiagMatr1);

template void gpu_statevector_anyCtrlAnyTargZOrPhaseGadget_sub<0>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, std::complex<double>, std::complex<double>);
template void gpu_statevector_anyCtrlAnyTargZOrPhaseGadget_sub<1>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, std::complex<double>, std::complex<double>);
template void gpu_statevector_anyCtrlAnyTargZOrPhaseGadget_sub<2>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, std::complex<double>, std::complex<double>);
template void gpu_statevector_anyCtrlAnyTargZOrPhaseGadget_sub<3>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, std::complex<double>, std::complex<double>);
template void gpu_statevector_anyCtrlAnyTargZOrPhaseGadget_sub<4>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, std::complex<double>, std::complex<double>);
template void gpu_statevector_anyCtrlAnyTargZOrPhaseGadget_sub<5>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, std::complex<double>, std::complex<double>);
template void gpu_statevector_anyCtrlAnyTargZOrPhaseGadget_sub<-1>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, std::complex<double>, std::complex<double>);

// Note: Single template parameter versions are removed as they conflict with
// two-parameter versions where M=0 (which are included below)

// Two template parameter versions - need all combinations
template void gpu_statevector_anyCtrlPauliTensorOrGadget_subA<0, 0>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, std::vector<int>, std::vector<int>, std::complex<double>, std::complex<double>);
template void gpu_statevector_anyCtrlPauliTensorOrGadget_subA<0, 1>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, std::vector<int>, std::vector<int>, std::complex<double>, std::complex<double>);
template void gpu_statevector_anyCtrlPauliTensorOrGadget_subA<0, 2>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, std::vector<int>, std::vector<int>, std::complex<double>, std::complex<double>);
template void gpu_statevector_anyCtrlPauliTensorOrGadget_subA<0, 3>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, std::vector<int>, std::vector<int>, std::complex<double>, std::complex<double>);
template void gpu_statevector_anyCtrlPauliTensorOrGadget_subA<0, 4>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, std::vector<int>, std::vector<int>, std::complex<double>, std::complex<double>);
template void gpu_statevector_anyCtrlPauliTensorOrGadget_subA<0, 5>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, std::vector<int>, std::vector<int>, std::complex<double>, std::complex<double>);
template void gpu_statevector_anyCtrlPauliTensorOrGadget_subA<0, -1>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, std::vector<int>, std::vector<int>, std::complex<double>, std::complex<double>);

template void gpu_statevector_anyCtrlPauliTensorOrGadget_subA<1, 0>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, std::vector<int>, std::vector<int>, std::complex<double>, std::complex<double>);
template void gpu_statevector_anyCtrlPauliTensorOrGadget_subA<1, 1>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, std::vector<int>, std::vector<int>, std::complex<double>, std::complex<double>);
template void gpu_statevector_anyCtrlPauliTensorOrGadget_subA<1, 2>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, std::vector<int>, std::vector<int>, std::complex<double>, std::complex<double>);
template void gpu_statevector_anyCtrlPauliTensorOrGadget_subA<1, 3>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, std::vector<int>, std::vector<int>, std::complex<double>, std::complex<double>);
template void gpu_statevector_anyCtrlPauliTensorOrGadget_subA<1, 4>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, std::vector<int>, std::vector<int>, std::complex<double>, std::complex<double>);
template void gpu_statevector_anyCtrlPauliTensorOrGadget_subA<1, 5>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, std::vector<int>, std::vector<int>, std::complex<double>, std::complex<double>);
template void gpu_statevector_anyCtrlPauliTensorOrGadget_subA<1, -1>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, std::vector<int>, std::vector<int>, std::complex<double>, std::complex<double>);

template void gpu_statevector_anyCtrlPauliTensorOrGadget_subA<2, 0>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, std::vector<int>, std::vector<int>, std::complex<double>, std::complex<double>);
template void gpu_statevector_anyCtrlPauliTensorOrGadget_subA<2, 1>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, std::vector<int>, std::vector<int>, std::complex<double>, std::complex<double>);
template void gpu_statevector_anyCtrlPauliTensorOrGadget_subA<2, 2>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, std::vector<int>, std::vector<int>, std::complex<double>, std::complex<double>);
template void gpu_statevector_anyCtrlPauliTensorOrGadget_subA<2, 3>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, std::vector<int>, std::vector<int>, std::complex<double>, std::complex<double>);
template void gpu_statevector_anyCtrlPauliTensorOrGadget_subA<2, 4>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, std::vector<int>, std::vector<int>, std::complex<double>, std::complex<double>);
template void gpu_statevector_anyCtrlPauliTensorOrGadget_subA<2, 5>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, std::vector<int>, std::vector<int>, std::complex<double>, std::complex<double>);
template void gpu_statevector_anyCtrlPauliTensorOrGadget_subA<2, -1>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, std::vector<int>, std::vector<int>, std::complex<double>, std::complex<double>);

template void gpu_statevector_anyCtrlPauliTensorOrGadget_subA<3, 0>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, std::vector<int>, std::vector<int>, std::complex<double>, std::complex<double>);
template void gpu_statevector_anyCtrlPauliTensorOrGadget_subA<3, 1>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, std::vector<int>, std::vector<int>, std::complex<double>, std::complex<double>);
template void gpu_statevector_anyCtrlPauliTensorOrGadget_subA<3, 2>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, std::vector<int>, std::vector<int>, std::complex<double>, std::complex<double>);
template void gpu_statevector_anyCtrlPauliTensorOrGadget_subA<3, 3>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, std::vector<int>, std::vector<int>, std::complex<double>, std::complex<double>);
template void gpu_statevector_anyCtrlPauliTensorOrGadget_subA<3, 4>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, std::vector<int>, std::vector<int>, std::complex<double>, std::complex<double>);
template void gpu_statevector_anyCtrlPauliTensorOrGadget_subA<3, 5>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, std::vector<int>, std::vector<int>, std::complex<double>, std::complex<double>);
template void gpu_statevector_anyCtrlPauliTensorOrGadget_subA<3, -1>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, std::vector<int>, std::vector<int>, std::complex<double>, std::complex<double>);

template void gpu_statevector_anyCtrlPauliTensorOrGadget_subA<4, 0>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, std::vector<int>, std::vector<int>, std::complex<double>, std::complex<double>);
template void gpu_statevector_anyCtrlPauliTensorOrGadget_subA<4, 1>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, std::vector<int>, std::vector<int>, std::complex<double>, std::complex<double>);
template void gpu_statevector_anyCtrlPauliTensorOrGadget_subA<4, 2>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, std::vector<int>, std::vector<int>, std::complex<double>, std::complex<double>);
template void gpu_statevector_anyCtrlPauliTensorOrGadget_subA<4, 3>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, std::vector<int>, std::vector<int>, std::complex<double>, std::complex<double>);
template void gpu_statevector_anyCtrlPauliTensorOrGadget_subA<4, 4>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, std::vector<int>, std::vector<int>, std::complex<double>, std::complex<double>);
template void gpu_statevector_anyCtrlPauliTensorOrGadget_subA<4, 5>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, std::vector<int>, std::vector<int>, std::complex<double>, std::complex<double>);
template void gpu_statevector_anyCtrlPauliTensorOrGadget_subA<4, -1>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, std::vector<int>, std::vector<int>, std::complex<double>, std::complex<double>);

template void gpu_statevector_anyCtrlPauliTensorOrGadget_subA<5, 0>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, std::vector<int>, std::vector<int>, std::complex<double>, std::complex<double>);
template void gpu_statevector_anyCtrlPauliTensorOrGadget_subA<5, 1>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, std::vector<int>, std::vector<int>, std::complex<double>, std::complex<double>);
template void gpu_statevector_anyCtrlPauliTensorOrGadget_subA<5, 2>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, std::vector<int>, std::vector<int>, std::complex<double>, std::complex<double>);
template void gpu_statevector_anyCtrlPauliTensorOrGadget_subA<5, 3>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, std::vector<int>, std::vector<int>, std::complex<double>, std::complex<double>);
template void gpu_statevector_anyCtrlPauliTensorOrGadget_subA<5, 4>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, std::vector<int>, std::vector<int>, std::complex<double>, std::complex<double>);
template void gpu_statevector_anyCtrlPauliTensorOrGadget_subA<5, 5>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, std::vector<int>, std::vector<int>, std::complex<double>, std::complex<double>);
template void gpu_statevector_anyCtrlPauliTensorOrGadget_subA<5, -1>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, std::vector<int>, std::vector<int>, std::complex<double>, std::complex<double>);

template void gpu_statevector_anyCtrlPauliTensorOrGadget_subA<-1, 0>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, std::vector<int>, std::vector<int>, std::complex<double>, std::complex<double>);
template void gpu_statevector_anyCtrlPauliTensorOrGadget_subA<-1, 1>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, std::vector<int>, std::vector<int>, std::complex<double>, std::complex<double>);
template void gpu_statevector_anyCtrlPauliTensorOrGadget_subA<-1, 2>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, std::vector<int>, std::vector<int>, std::complex<double>, std::complex<double>);
template void gpu_statevector_anyCtrlPauliTensorOrGadget_subA<-1, 3>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, std::vector<int>, std::vector<int>, std::complex<double>, std::complex<double>);
template void gpu_statevector_anyCtrlPauliTensorOrGadget_subA<-1, 4>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, std::vector<int>, std::vector<int>, std::complex<double>, std::complex<double>);
template void gpu_statevector_anyCtrlPauliTensorOrGadget_subA<-1, 5>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, std::vector<int>, std::vector<int>, std::complex<double>, std::complex<double>);
template void gpu_statevector_anyCtrlPauliTensorOrGadget_subA<-1, -1>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, std::vector<int>, std::vector<int>, std::complex<double>, std::complex<double>);

template void gpu_statevector_anyCtrlPauliTensorOrGadget_subB<0>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, std::vector<int>, std::vector<int>, std::complex<double>, std::complex<double>, long long);
template void gpu_statevector_anyCtrlPauliTensorOrGadget_subB<1>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, std::vector<int>, std::vector<int>, std::complex<double>, std::complex<double>, long long);
template void gpu_statevector_anyCtrlPauliTensorOrGadget_subB<2>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, std::vector<int>, std::vector<int>, std::complex<double>, std::complex<double>, long long);
template void gpu_statevector_anyCtrlPauliTensorOrGadget_subB<3>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, std::vector<int>, std::vector<int>, std::complex<double>, std::complex<double>, long long);
template void gpu_statevector_anyCtrlPauliTensorOrGadget_subB<4>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, std::vector<int>, std::vector<int>, std::complex<double>, std::complex<double>, long long);
template void gpu_statevector_anyCtrlPauliTensorOrGadget_subB<5>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, std::vector<int>, std::vector<int>, std::complex<double>, std::complex<double>, long long);
template void gpu_statevector_anyCtrlPauliTensorOrGadget_subB<-1>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, std::vector<int>, std::vector<int>, std::complex<double>, std::complex<double>, long long);

template double gpu_statevec_calcProbOfMultiQubitOutcome_sub<0>(Qureg, std::vector<int>, std::vector<int>);
template double gpu_statevec_calcProbOfMultiQubitOutcome_sub<1>(Qureg, std::vector<int>, std::vector<int>);
template double gpu_statevec_calcProbOfMultiQubitOutcome_sub<2>(Qureg, std::vector<int>, std::vector<int>);
template double gpu_statevec_calcProbOfMultiQubitOutcome_sub<3>(Qureg, std::vector<int>, std::vector<int>);
template double gpu_statevec_calcProbOfMultiQubitOutcome_sub<4>(Qureg, std::vector<int>, std::vector<int>);
template double gpu_statevec_calcProbOfMultiQubitOutcome_sub<5>(Qureg, std::vector<int>, std::vector<int>);
template double gpu_statevec_calcProbOfMultiQubitOutcome_sub<-1>(Qureg, std::vector<int>, std::vector<int>);

template double gpu_densmatr_calcProbOfMultiQubitOutcome_sub<0>(Qureg, std::vector<int>, std::vector<int>);
template double gpu_densmatr_calcProbOfMultiQubitOutcome_sub<1>(Qureg, std::vector<int>, std::vector<int>);
template double gpu_densmatr_calcProbOfMultiQubitOutcome_sub<2>(Qureg, std::vector<int>, std::vector<int>);
template double gpu_densmatr_calcProbOfMultiQubitOutcome_sub<3>(Qureg, std::vector<int>, std::vector<int>);
template double gpu_densmatr_calcProbOfMultiQubitOutcome_sub<4>(Qureg, std::vector<int>, std::vector<int>);
template double gpu_densmatr_calcProbOfMultiQubitOutcome_sub<5>(Qureg, std::vector<int>, std::vector<int>);
template double gpu_densmatr_calcProbOfMultiQubitOutcome_sub<-1>(Qureg, std::vector<int>, std::vector<int>);

template void gpu_statevec_multiQubitProjector_sub<0>(Qureg, std::vector<int>, std::vector<int>, double);
template void gpu_statevec_multiQubitProjector_sub<1>(Qureg, std::vector<int>, std::vector<int>, double);
template void gpu_statevec_multiQubitProjector_sub<2>(Qureg, std::vector<int>, std::vector<int>, double);
template void gpu_statevec_multiQubitProjector_sub<3>(Qureg, std::vector<int>, std::vector<int>, double);
template void gpu_statevec_multiQubitProjector_sub<4>(Qureg, std::vector<int>, std::vector<int>, double);
template void gpu_statevec_multiQubitProjector_sub<5>(Qureg, std::vector<int>, std::vector<int>, double);
template void gpu_statevec_multiQubitProjector_sub<-1>(Qureg, std::vector<int>, std::vector<int>, double);

template void gpu_densmatr_multiQubitProjector_sub<0>(Qureg, std::vector<int>, std::vector<int>, double);
template void gpu_densmatr_multiQubitProjector_sub<1>(Qureg, std::vector<int>, std::vector<int>, double);
template void gpu_densmatr_multiQubitProjector_sub<2>(Qureg, std::vector<int>, std::vector<int>, double);
template void gpu_densmatr_multiQubitProjector_sub<3>(Qureg, std::vector<int>, std::vector<int>, double);
template void gpu_densmatr_multiQubitProjector_sub<4>(Qureg, std::vector<int>, std::vector<int>, double);
template void gpu_densmatr_multiQubitProjector_sub<5>(Qureg, std::vector<int>, std::vector<int>, double);
template void gpu_densmatr_multiQubitProjector_sub<-1>(Qureg, std::vector<int>, std::vector<int>, double);

// Additional GPU stubs needed for finalizeQuESTEnv
void gpu_clearCache() {
    // No-op for CPU-only builds
}

void gpu_finalizeCuQuantum() {
    // No-op for CPU-only builds
}

// Additional GPU info functions
int gpu_getNumberOfLocalGpus() { return 0; }
size_t gpu_getTotalMemoryInBytes() { return 0; }
size_t gpu_getCacheMemoryInBytes() { return 0; }

// Additional matrix structures
struct CompMatr {
    int isDensityMatrix;
    int numQubits;
    qindex numAmps;
    qcomp* real;
    qcomp* imag;
};

struct DiagMatr {
    int numQubits;
    qindex numAmps;
    qcomp* elems;
};

struct SuperOp {
    int numQubits;
    qindex numAmps;
    qcomp* real;
    qcomp* imag;
};

struct FullStateDiagMatr {
    int numQubits;
    qindex numAmps;
    qcomp* elems;
};

struct PauliStrSum {
    int numQubits;
    int numTerms;
    double* coeffs;
    int* pauliCodes;
};

struct CompMatr2 {
    qcomp elems[16];  // 4x4 matrix
};

struct DiagMatr2 {
    qcomp elems[4];   // 2 qubit diagonal
};

// GPU copy functions for matrices
void gpu_copyGpuToCpu(CompMatr m) {}
void gpu_copyGpuToCpu(SuperOp m) {}
void gpu_copyCpuToGpu(CompMatr m) {}
void gpu_copyCpuToGpu(DiagMatr m) {}
void gpu_copyCpuToGpu(FullStateDiagMatr m) {}

// GPU accelerator stub functions
std::complex<double> gpu_statevec_getAmp_sub(Qureg q, long long idx) { return 0.0; }
void gpu_densmatr_setAmpsToPauliStrSum_sub(Qureg q, PauliStrSum p) {}
void gpu_fullstatediagmatr_setElemsToPauliStrSum(FullStateDiagMatr m, PauliStrSum p) {}
long long gpu_statevec_packPairSummedAmpsIntoBuffer(Qureg q, int a, int b, int c, int d) { return 0; }

// Decoherence functions
void gpu_densmatr_oneQubitDephasing_subA(Qureg q, int target, double dephase) {}
void gpu_densmatr_oneQubitDephasing_subB(Qureg q, int target, double dephase) {}
void gpu_densmatr_twoQubitDephasing_subA(Qureg q, int q1, int q2, double dephase) {}
void gpu_densmatr_twoQubitDephasing_subB(Qureg q, int q1, int q2, double dephase) {}
void gpu_densmatr_oneQubitDepolarising_subA(Qureg q, int target, double depolProb) {}
void gpu_densmatr_oneQubitDepolarising_subB(Qureg q, int target, double depolProb) {}
void gpu_densmatr_twoQubitDepolarising_subA(Qureg q, int q1, int q2, double depolProb) {}
void gpu_densmatr_twoQubitDepolarising_subB(Qureg q, int q1, int q2, double depolProb) {}
void gpu_densmatr_twoQubitDepolarising_subC(Qureg q, int q1, int q2, double depolProb) {}
void gpu_densmatr_twoQubitDepolarising_subD(Qureg q, int q1, int q2, double depolProb) {}
void gpu_densmatr_twoQubitDepolarising_subE(Qureg q, int q1, int q2, double depolProb) {}
void gpu_densmatr_twoQubitDepolarising_subF(Qureg q, int q1, int q2, double depolProb) {}
void gpu_densmatr_oneQubitPauliChannel_subA(Qureg q, int target, double px, double py, double pz, double pi) {}
void gpu_densmatr_oneQubitPauliChannel_subB(Qureg q, int target, double px, double py, double pz, double pi) {}
void gpu_densmatr_oneQubitDamping_subA(Qureg q, int target, double damping) {}
void gpu_densmatr_oneQubitDamping_subB(Qureg q, int target, double damping) {}
void gpu_densmatr_oneQubitDamping_subC(Qureg q, int target, double damping) {}
void gpu_densmatr_oneQubitDamping_subD(Qureg q, int target, double damping) {}

// Calculation functions - note return types
double gpu_statevec_calcTotalProb_sub(Qureg q) { return 1.0; }
double gpu_densmatr_calcTotalProb_sub(Qureg q) { return 1.0; }
std::complex<double> gpu_statevec_calcInnerProduct_sub(Qureg q1, Qureg q2) { return 0.0; }
double gpu_densmatr_calcHilbertSchmidtDistance_sub(Qureg q1, Qureg q2) { return 0.0; }
// Note: Function names use calcExpec (not calcExpec) to match QuEST v4.1.0
double gpu_statevec_calcExpecAnyTargZ_sub(Qureg q, std::vector<int> targets) { return 0.0; }
std::complex<double> gpu_densmatr_calcExpecAnyTargZ_sub(Qureg q, std::vector<int> targets) { return 0.0; }
std::complex<double> gpu_statevec_calcExpecPauliStr_subA(Qureg q, std::vector<int> a, std::vector<int> b, std::vector<int> c) { return 0.0; }
std::complex<double> gpu_statevec_calcExpecPauliStr_subB(Qureg q, std::vector<int> a, std::vector<int> b, std::vector<int> c) { return 0.0; }
std::complex<double> gpu_densmatr_calcExpecPauliStr_sub(Qureg q, std::vector<int> a, std::vector<int> b, std::vector<int> c) { return 0.0; }

// Init functions
void gpu_statevec_initDebugState_sub(Qureg q) {}
void gpu_statevec_initUnnormalisedUniformlyRandomPureStateAmps_sub(Qureg q) {}

// Template stubs for SWAP operations
template<int N>
void gpu_statevec_anyCtrlSwap_subA(Qureg q, std::vector<int> ctrls, std::vector<int> ctrlVals, int q1, int q2) {}

template<int N>
void gpu_statevec_anyCtrlSwap_subB(Qureg q, std::vector<int> ctrls, std::vector<int> ctrlVals) {}

template<int N>
void gpu_statevec_anyCtrlSwap_subC(Qureg q, std::vector<int> ctrls, std::vector<int> ctrlVals, int q1, int q2) {}

// Template stubs for two-target dense matrix operations
template<int N>
void gpu_statevec_anyCtrlTwoTargDenseMatr_sub(Qureg q, std::vector<int> ctrls, std::vector<int> ctrlVals, int t1, int t2, CompMatr2 m) {}

// Template stubs for any-target dense matrix operations
template<int NumCtrls, int NumTargs, bool ApplyConj>
void gpu_statevec_anyCtrlAnyTargDenseMatr_sub(Qureg q, std::vector<int> ctrls, std::vector<int> ctrlVals, std::vector<int> targets, CompMatr m) {}

// Template stubs for two-target diagonal matrix operations
template<int N>
void gpu_statevec_anyCtrlTwoTargDiagMatr_sub(Qureg q, std::vector<int> ctrls, std::vector<int> ctrlVals, int t1, int t2, DiagMatr2 m) {}

// Template stubs for any-target diagonal matrix operations
template<int NumCtrls, int NumTargs, bool ApplyConj, bool HasPower>
void gpu_statevec_anyCtrlAnyTargDiagMatr_sub(Qureg q, std::vector<int> ctrls, std::vector<int> ctrlVals, std::vector<int> targets, DiagMatr m, std::complex<double> globalPhase) {}

// Template stubs for all-target diagonal matrix operations
template<bool HasPower>
void gpu_statevec_allTargDiagMatr_sub(Qureg q, FullStateDiagMatr m, std::complex<double> globalPhase) {}

template<bool HasPower, bool MultiplyOnly>
void gpu_densmatr_allTargDiagMatr_sub(Qureg q, FullStateDiagMatr m, std::complex<double> globalPhase) {}

// Template stubs for partial trace operations
template<int N>
void gpu_densmatr_partialTrace_sub(Qureg traceOut, Qureg traceIn, std::vector<int> targets, std::vector<int> controls) {}

// Template stubs for probability calculations
template<int N>
void gpu_statevec_calcProbsOfAllMultiQubitOutcomes_sub(double* probs, Qureg q, std::vector<int> qubits) {}

template<int N>
void gpu_densmatr_calcProbsOfAllMultiQubitOutcomes_sub(double* probs, Qureg q, std::vector<int> qubits) {}

// Template stubs for fidelity calculations
template<bool Conj>
std::complex<double> gpu_densmatr_calcFidelityWithPureState_sub(Qureg densMatr, Qureg pureState) {
    return std::complex<double>(0.0, 0.0);
}

// Template stubs for expectation value calculations
template<bool HasPower, bool UseRealPow>
std::complex<double> gpu_statevec_calcExpecFullStateDiagMatr_sub(Qureg q, FullStateDiagMatr m, std::complex<double> globalPhase) {
    return std::complex<double>(0.0, 0.0);
}

template<bool HasPower, bool UseRealPow>
std::complex<double> gpu_densmatr_calcExpecFullStateDiagMatr_sub(Qureg q, FullStateDiagMatr m, std::complex<double> globalPhase) {
    return std::complex<double>(0.0, 0.0);
}

// Explicit template instantiations for SWAP operations
template void gpu_statevec_anyCtrlSwap_subA<0>(Qureg, std::vector<int>, std::vector<int>, int, int);
template void gpu_statevec_anyCtrlSwap_subA<1>(Qureg, std::vector<int>, std::vector<int>, int, int);
template void gpu_statevec_anyCtrlSwap_subA<2>(Qureg, std::vector<int>, std::vector<int>, int, int);
template void gpu_statevec_anyCtrlSwap_subA<3>(Qureg, std::vector<int>, std::vector<int>, int, int);
template void gpu_statevec_anyCtrlSwap_subA<4>(Qureg, std::vector<int>, std::vector<int>, int, int);
template void gpu_statevec_anyCtrlSwap_subA<5>(Qureg, std::vector<int>, std::vector<int>, int, int);
template void gpu_statevec_anyCtrlSwap_subA<-1>(Qureg, std::vector<int>, std::vector<int>, int, int);

template void gpu_statevec_anyCtrlSwap_subB<0>(Qureg, std::vector<int>, std::vector<int>);
template void gpu_statevec_anyCtrlSwap_subB<1>(Qureg, std::vector<int>, std::vector<int>);
template void gpu_statevec_anyCtrlSwap_subB<2>(Qureg, std::vector<int>, std::vector<int>);
template void gpu_statevec_anyCtrlSwap_subB<3>(Qureg, std::vector<int>, std::vector<int>);
template void gpu_statevec_anyCtrlSwap_subB<4>(Qureg, std::vector<int>, std::vector<int>);
template void gpu_statevec_anyCtrlSwap_subB<5>(Qureg, std::vector<int>, std::vector<int>);
template void gpu_statevec_anyCtrlSwap_subB<-1>(Qureg, std::vector<int>, std::vector<int>);

template void gpu_statevec_anyCtrlSwap_subC<0>(Qureg, std::vector<int>, std::vector<int>, int, int);
template void gpu_statevec_anyCtrlSwap_subC<1>(Qureg, std::vector<int>, std::vector<int>, int, int);
template void gpu_statevec_anyCtrlSwap_subC<2>(Qureg, std::vector<int>, std::vector<int>, int, int);
template void gpu_statevec_anyCtrlSwap_subC<3>(Qureg, std::vector<int>, std::vector<int>, int, int);
template void gpu_statevec_anyCtrlSwap_subC<4>(Qureg, std::vector<int>, std::vector<int>, int, int);
template void gpu_statevec_anyCtrlSwap_subC<5>(Qureg, std::vector<int>, std::vector<int>, int, int);
template void gpu_statevec_anyCtrlSwap_subC<-1>(Qureg, std::vector<int>, std::vector<int>, int, int);

// Explicit template instantiations for two-target dense matrix operations
template void gpu_statevec_anyCtrlTwoTargDenseMatr_sub<0>(Qureg, std::vector<int>, std::vector<int>, int, int, CompMatr2);
template void gpu_statevec_anyCtrlTwoTargDenseMatr_sub<1>(Qureg, std::vector<int>, std::vector<int>, int, int, CompMatr2);
template void gpu_statevec_anyCtrlTwoTargDenseMatr_sub<2>(Qureg, std::vector<int>, std::vector<int>, int, int, CompMatr2);
template void gpu_statevec_anyCtrlTwoTargDenseMatr_sub<3>(Qureg, std::vector<int>, std::vector<int>, int, int, CompMatr2);
template void gpu_statevec_anyCtrlTwoTargDenseMatr_sub<4>(Qureg, std::vector<int>, std::vector<int>, int, int, CompMatr2);
template void gpu_statevec_anyCtrlTwoTargDenseMatr_sub<5>(Qureg, std::vector<int>, std::vector<int>, int, int, CompMatr2);
template void gpu_statevec_anyCtrlTwoTargDenseMatr_sub<-1>(Qureg, std::vector<int>, std::vector<int>, int, int, CompMatr2);

// Explicit template instantiations for two-target diagonal matrix operations
template void gpu_statevec_anyCtrlTwoTargDiagMatr_sub<0>(Qureg, std::vector<int>, std::vector<int>, int, int, DiagMatr2);
template void gpu_statevec_anyCtrlTwoTargDiagMatr_sub<1>(Qureg, std::vector<int>, std::vector<int>, int, int, DiagMatr2);
template void gpu_statevec_anyCtrlTwoTargDiagMatr_sub<2>(Qureg, std::vector<int>, std::vector<int>, int, int, DiagMatr2);
template void gpu_statevec_anyCtrlTwoTargDiagMatr_sub<3>(Qureg, std::vector<int>, std::vector<int>, int, int, DiagMatr2);
template void gpu_statevec_anyCtrlTwoTargDiagMatr_sub<4>(Qureg, std::vector<int>, std::vector<int>, int, int, DiagMatr2);
template void gpu_statevec_anyCtrlTwoTargDiagMatr_sub<5>(Qureg, std::vector<int>, std::vector<int>, int, int, DiagMatr2);
template void gpu_statevec_anyCtrlTwoTargDiagMatr_sub<-1>(Qureg, std::vector<int>, std::vector<int>, int, int, DiagMatr2);
// Explicit template instantiations for any-target dense matrix operations
// gpu_statevec_anyCtrlAnyTargDenseMatr_sub<N1, N2, N3>
template void gpu_statevec_anyCtrlAnyTargDenseMatr_sub<0, 0, false>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, CompMatr);
template void gpu_statevec_anyCtrlAnyTargDenseMatr_sub<0, 0, true>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, CompMatr);
template void gpu_statevec_anyCtrlAnyTargDenseMatr_sub<0, 1, false>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, CompMatr);
template void gpu_statevec_anyCtrlAnyTargDenseMatr_sub<0, 1, true>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, CompMatr);
template void gpu_statevec_anyCtrlAnyTargDenseMatr_sub<0, 2, false>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, CompMatr);
template void gpu_statevec_anyCtrlAnyTargDenseMatr_sub<0, 2, true>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, CompMatr);
template void gpu_statevec_anyCtrlAnyTargDenseMatr_sub<0, 3, false>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, CompMatr);
template void gpu_statevec_anyCtrlAnyTargDenseMatr_sub<0, 3, true>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, CompMatr);
template void gpu_statevec_anyCtrlAnyTargDenseMatr_sub<0, 4, false>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, CompMatr);
template void gpu_statevec_anyCtrlAnyTargDenseMatr_sub<0, 4, true>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, CompMatr);
template void gpu_statevec_anyCtrlAnyTargDenseMatr_sub<0, 5, false>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, CompMatr);
template void gpu_statevec_anyCtrlAnyTargDenseMatr_sub<0, 5, true>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, CompMatr);
template void gpu_statevec_anyCtrlAnyTargDenseMatr_sub<0, -1, false>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, CompMatr);
template void gpu_statevec_anyCtrlAnyTargDenseMatr_sub<0, -1, true>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, CompMatr);
template void gpu_statevec_anyCtrlAnyTargDenseMatr_sub<1, 0, false>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, CompMatr);
template void gpu_statevec_anyCtrlAnyTargDenseMatr_sub<1, 0, true>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, CompMatr);
template void gpu_statevec_anyCtrlAnyTargDenseMatr_sub<1, 1, false>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, CompMatr);
template void gpu_statevec_anyCtrlAnyTargDenseMatr_sub<1, 1, true>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, CompMatr);
template void gpu_statevec_anyCtrlAnyTargDenseMatr_sub<1, 2, false>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, CompMatr);
template void gpu_statevec_anyCtrlAnyTargDenseMatr_sub<1, 2, true>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, CompMatr);
template void gpu_statevec_anyCtrlAnyTargDenseMatr_sub<1, 3, false>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, CompMatr);
template void gpu_statevec_anyCtrlAnyTargDenseMatr_sub<1, 3, true>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, CompMatr);
template void gpu_statevec_anyCtrlAnyTargDenseMatr_sub<1, 4, false>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, CompMatr);
template void gpu_statevec_anyCtrlAnyTargDenseMatr_sub<1, 4, true>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, CompMatr);
template void gpu_statevec_anyCtrlAnyTargDenseMatr_sub<1, 5, false>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, CompMatr);
template void gpu_statevec_anyCtrlAnyTargDenseMatr_sub<1, 5, true>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, CompMatr);
template void gpu_statevec_anyCtrlAnyTargDenseMatr_sub<1, -1, false>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, CompMatr);
template void gpu_statevec_anyCtrlAnyTargDenseMatr_sub<1, -1, true>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, CompMatr);
template void gpu_statevec_anyCtrlAnyTargDenseMatr_sub<2, 0, false>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, CompMatr);
template void gpu_statevec_anyCtrlAnyTargDenseMatr_sub<2, 0, true>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, CompMatr);
template void gpu_statevec_anyCtrlAnyTargDenseMatr_sub<2, 1, false>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, CompMatr);
template void gpu_statevec_anyCtrlAnyTargDenseMatr_sub<2, 1, true>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, CompMatr);
template void gpu_statevec_anyCtrlAnyTargDenseMatr_sub<2, 2, false>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, CompMatr);
template void gpu_statevec_anyCtrlAnyTargDenseMatr_sub<2, 2, true>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, CompMatr);
template void gpu_statevec_anyCtrlAnyTargDenseMatr_sub<2, 3, false>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, CompMatr);
template void gpu_statevec_anyCtrlAnyTargDenseMatr_sub<2, 3, true>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, CompMatr);
template void gpu_statevec_anyCtrlAnyTargDenseMatr_sub<2, 4, false>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, CompMatr);
template void gpu_statevec_anyCtrlAnyTargDenseMatr_sub<2, 4, true>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, CompMatr);
template void gpu_statevec_anyCtrlAnyTargDenseMatr_sub<2, 5, false>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, CompMatr);
template void gpu_statevec_anyCtrlAnyTargDenseMatr_sub<2, 5, true>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, CompMatr);
template void gpu_statevec_anyCtrlAnyTargDenseMatr_sub<2, -1, false>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, CompMatr);
template void gpu_statevec_anyCtrlAnyTargDenseMatr_sub<2, -1, true>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, CompMatr);
template void gpu_statevec_anyCtrlAnyTargDenseMatr_sub<3, 0, false>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, CompMatr);
template void gpu_statevec_anyCtrlAnyTargDenseMatr_sub<3, 0, true>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, CompMatr);
template void gpu_statevec_anyCtrlAnyTargDenseMatr_sub<3, 1, false>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, CompMatr);
template void gpu_statevec_anyCtrlAnyTargDenseMatr_sub<3, 1, true>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, CompMatr);
template void gpu_statevec_anyCtrlAnyTargDenseMatr_sub<3, 2, false>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, CompMatr);
template void gpu_statevec_anyCtrlAnyTargDenseMatr_sub<3, 2, true>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, CompMatr);
template void gpu_statevec_anyCtrlAnyTargDenseMatr_sub<3, 3, false>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, CompMatr);
template void gpu_statevec_anyCtrlAnyTargDenseMatr_sub<3, 3, true>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, CompMatr);
template void gpu_statevec_anyCtrlAnyTargDenseMatr_sub<3, 4, false>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, CompMatr);
template void gpu_statevec_anyCtrlAnyTargDenseMatr_sub<3, 4, true>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, CompMatr);
template void gpu_statevec_anyCtrlAnyTargDenseMatr_sub<3, 5, false>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, CompMatr);
template void gpu_statevec_anyCtrlAnyTargDenseMatr_sub<3, 5, true>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, CompMatr);
template void gpu_statevec_anyCtrlAnyTargDenseMatr_sub<3, -1, false>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, CompMatr);
template void gpu_statevec_anyCtrlAnyTargDenseMatr_sub<3, -1, true>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, CompMatr);
template void gpu_statevec_anyCtrlAnyTargDenseMatr_sub<4, 0, false>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, CompMatr);
template void gpu_statevec_anyCtrlAnyTargDenseMatr_sub<4, 0, true>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, CompMatr);
template void gpu_statevec_anyCtrlAnyTargDenseMatr_sub<4, 1, false>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, CompMatr);
template void gpu_statevec_anyCtrlAnyTargDenseMatr_sub<4, 1, true>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, CompMatr);
template void gpu_statevec_anyCtrlAnyTargDenseMatr_sub<4, 2, false>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, CompMatr);
template void gpu_statevec_anyCtrlAnyTargDenseMatr_sub<4, 2, true>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, CompMatr);
template void gpu_statevec_anyCtrlAnyTargDenseMatr_sub<4, 3, false>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, CompMatr);
template void gpu_statevec_anyCtrlAnyTargDenseMatr_sub<4, 3, true>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, CompMatr);
template void gpu_statevec_anyCtrlAnyTargDenseMatr_sub<4, 4, false>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, CompMatr);
template void gpu_statevec_anyCtrlAnyTargDenseMatr_sub<4, 4, true>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, CompMatr);
template void gpu_statevec_anyCtrlAnyTargDenseMatr_sub<4, 5, false>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, CompMatr);
template void gpu_statevec_anyCtrlAnyTargDenseMatr_sub<4, 5, true>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, CompMatr);
template void gpu_statevec_anyCtrlAnyTargDenseMatr_sub<4, -1, false>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, CompMatr);
template void gpu_statevec_anyCtrlAnyTargDenseMatr_sub<4, -1, true>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, CompMatr);
template void gpu_statevec_anyCtrlAnyTargDenseMatr_sub<5, 0, false>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, CompMatr);
template void gpu_statevec_anyCtrlAnyTargDenseMatr_sub<5, 0, true>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, CompMatr);
template void gpu_statevec_anyCtrlAnyTargDenseMatr_sub<5, 1, false>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, CompMatr);
template void gpu_statevec_anyCtrlAnyTargDenseMatr_sub<5, 1, true>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, CompMatr);
template void gpu_statevec_anyCtrlAnyTargDenseMatr_sub<5, 2, false>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, CompMatr);
template void gpu_statevec_anyCtrlAnyTargDenseMatr_sub<5, 2, true>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, CompMatr);
template void gpu_statevec_anyCtrlAnyTargDenseMatr_sub<5, 3, false>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, CompMatr);
template void gpu_statevec_anyCtrlAnyTargDenseMatr_sub<5, 3, true>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, CompMatr);
template void gpu_statevec_anyCtrlAnyTargDenseMatr_sub<5, 4, false>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, CompMatr);
template void gpu_statevec_anyCtrlAnyTargDenseMatr_sub<5, 4, true>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, CompMatr);
template void gpu_statevec_anyCtrlAnyTargDenseMatr_sub<5, 5, false>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, CompMatr);
template void gpu_statevec_anyCtrlAnyTargDenseMatr_sub<5, 5, true>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, CompMatr);
template void gpu_statevec_anyCtrlAnyTargDenseMatr_sub<5, -1, false>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, CompMatr);
template void gpu_statevec_anyCtrlAnyTargDenseMatr_sub<5, -1, true>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, CompMatr);
template void gpu_statevec_anyCtrlAnyTargDenseMatr_sub<-1, 0, false>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, CompMatr);
template void gpu_statevec_anyCtrlAnyTargDenseMatr_sub<-1, 0, true>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, CompMatr);
template void gpu_statevec_anyCtrlAnyTargDenseMatr_sub<-1, 1, false>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, CompMatr);
template void gpu_statevec_anyCtrlAnyTargDenseMatr_sub<-1, 1, true>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, CompMatr);
template void gpu_statevec_anyCtrlAnyTargDenseMatr_sub<-1, 2, false>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, CompMatr);
template void gpu_statevec_anyCtrlAnyTargDenseMatr_sub<-1, 2, true>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, CompMatr);
template void gpu_statevec_anyCtrlAnyTargDenseMatr_sub<-1, 3, false>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, CompMatr);
template void gpu_statevec_anyCtrlAnyTargDenseMatr_sub<-1, 3, true>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, CompMatr);
template void gpu_statevec_anyCtrlAnyTargDenseMatr_sub<-1, 4, false>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, CompMatr);
template void gpu_statevec_anyCtrlAnyTargDenseMatr_sub<-1, 4, true>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, CompMatr);
template void gpu_statevec_anyCtrlAnyTargDenseMatr_sub<-1, 5, false>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, CompMatr);
template void gpu_statevec_anyCtrlAnyTargDenseMatr_sub<-1, 5, true>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, CompMatr);
template void gpu_statevec_anyCtrlAnyTargDenseMatr_sub<-1, -1, false>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, CompMatr);
template void gpu_statevec_anyCtrlAnyTargDenseMatr_sub<-1, -1, true>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, CompMatr);

// Explicit template instantiations for any-target diagonal matrix operations
// gpu_statevec_anyCtrlAnyTargDiagMatr_sub<N1, N2, N3, N4>
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<0, 0, false, false>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<0, 0, false, true>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<0, 0, true, false>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<0, 0, true, true>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<0, 1, false, false>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<0, 1, false, true>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<0, 1, true, false>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<0, 1, true, true>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<0, 2, false, false>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<0, 2, false, true>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<0, 2, true, false>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<0, 2, true, true>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<0, 3, false, false>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<0, 3, false, true>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<0, 3, true, false>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<0, 3, true, true>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<0, 4, false, false>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<0, 4, false, true>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<0, 4, true, false>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<0, 4, true, true>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<0, 5, false, false>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<0, 5, false, true>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<0, 5, true, false>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<0, 5, true, true>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<0, -1, false, false>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<0, -1, false, true>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<0, -1, true, false>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<0, -1, true, true>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<1, 0, false, false>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<1, 0, false, true>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<1, 0, true, false>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<1, 0, true, true>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<1, 1, false, false>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<1, 1, false, true>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<1, 1, true, false>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<1, 1, true, true>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<1, 2, false, false>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<1, 2, false, true>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<1, 2, true, false>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<1, 2, true, true>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<1, 3, false, false>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<1, 3, false, true>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<1, 3, true, false>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<1, 3, true, true>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<1, 4, false, false>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<1, 4, false, true>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<1, 4, true, false>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<1, 4, true, true>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<1, 5, false, false>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<1, 5, false, true>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<1, 5, true, false>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<1, 5, true, true>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<1, -1, false, false>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<1, -1, false, true>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<1, -1, true, false>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<1, -1, true, true>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<2, 0, false, false>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<2, 0, false, true>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<2, 0, true, false>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<2, 0, true, true>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<2, 1, false, false>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<2, 1, false, true>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<2, 1, true, false>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<2, 1, true, true>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<2, 2, false, false>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<2, 2, false, true>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<2, 2, true, false>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<2, 2, true, true>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<2, 3, false, false>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<2, 3, false, true>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<2, 3, true, false>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<2, 3, true, true>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<2, 4, false, false>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<2, 4, false, true>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<2, 4, true, false>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<2, 4, true, true>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<2, 5, false, false>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<2, 5, false, true>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<2, 5, true, false>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<2, 5, true, true>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<2, -1, false, false>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<2, -1, false, true>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<2, -1, true, false>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<2, -1, true, true>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<3, 0, false, false>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<3, 0, false, true>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<3, 0, true, false>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<3, 0, true, true>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<3, 1, false, false>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<3, 1, false, true>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<3, 1, true, false>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<3, 1, true, true>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<3, 2, false, false>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<3, 2, false, true>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<3, 2, true, false>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<3, 2, true, true>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<3, 3, false, false>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<3, 3, false, true>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<3, 3, true, false>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<3, 3, true, true>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<3, 4, false, false>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<3, 4, false, true>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<3, 4, true, false>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<3, 4, true, true>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<3, 5, false, false>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<3, 5, false, true>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<3, 5, true, false>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<3, 5, true, true>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<3, -1, false, false>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<3, -1, false, true>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<3, -1, true, false>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<3, -1, true, true>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<4, 0, false, false>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<4, 0, false, true>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<4, 0, true, false>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<4, 0, true, true>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<4, 1, false, false>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<4, 1, false, true>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<4, 1, true, false>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<4, 1, true, true>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<4, 2, false, false>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<4, 2, false, true>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<4, 2, true, false>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<4, 2, true, true>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<4, 3, false, false>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<4, 3, false, true>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<4, 3, true, false>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<4, 3, true, true>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<4, 4, false, false>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<4, 4, false, true>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<4, 4, true, false>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<4, 4, true, true>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<4, 5, false, false>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<4, 5, false, true>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<4, 5, true, false>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<4, 5, true, true>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<4, -1, false, false>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<4, -1, false, true>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<4, -1, true, false>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<4, -1, true, true>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<5, 0, false, false>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<5, 0, false, true>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<5, 0, true, false>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<5, 0, true, true>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<5, 1, false, false>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<5, 1, false, true>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<5, 1, true, false>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<5, 1, true, true>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<5, 2, false, false>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<5, 2, false, true>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<5, 2, true, false>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<5, 2, true, true>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<5, 3, false, false>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<5, 3, false, true>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<5, 3, true, false>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<5, 3, true, true>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<5, 4, false, false>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<5, 4, false, true>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<5, 4, true, false>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<5, 4, true, true>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<5, 5, false, false>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<5, 5, false, true>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<5, 5, true, false>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<5, 5, true, true>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<5, -1, false, false>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<5, -1, false, true>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<5, -1, true, false>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<5, -1, true, true>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<-1, 0, false, false>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<-1, 0, false, true>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<-1, 0, true, false>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<-1, 0, true, true>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<-1, 1, false, false>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<-1, 1, false, true>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<-1, 1, true, false>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<-1, 1, true, true>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<-1, 2, false, false>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<-1, 2, false, true>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<-1, 2, true, false>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<-1, 2, true, true>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<-1, 3, false, false>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<-1, 3, false, true>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<-1, 3, true, false>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<-1, 3, true, true>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<-1, 4, false, false>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<-1, 4, false, true>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<-1, 4, true, false>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<-1, 4, true, true>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<-1, 5, false, false>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<-1, 5, false, true>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<-1, 5, true, false>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<-1, 5, true, true>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<-1, -1, false, false>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<-1, -1, false, true>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<-1, -1, true, false>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);
template void gpu_statevec_anyCtrlAnyTargDiagMatr_sub<-1, -1, true, true>(Qureg, std::vector<int>, std::vector<int>, std::vector<int>, DiagMatr, std::complex<double>);

// Explicit template instantiations for all-target diagonal matrix operations
template void gpu_statevec_allTargDiagMatr_sub<false>(Qureg, FullStateDiagMatr, std::complex<double>);
template void gpu_statevec_allTargDiagMatr_sub<true>(Qureg, FullStateDiagMatr, std::complex<double>);

template void gpu_densmatr_allTargDiagMatr_sub<false, false>(Qureg, FullStateDiagMatr, std::complex<double>);
template void gpu_densmatr_allTargDiagMatr_sub<false, true>(Qureg, FullStateDiagMatr, std::complex<double>);
template void gpu_densmatr_allTargDiagMatr_sub<true, false>(Qureg, FullStateDiagMatr, std::complex<double>);
template void gpu_densmatr_allTargDiagMatr_sub<true, true>(Qureg, FullStateDiagMatr, std::complex<double>);

// Explicit template instantiations for partial trace operations
template void gpu_densmatr_partialTrace_sub<0>(Qureg, Qureg, std::vector<int>, std::vector<int>);
template void gpu_densmatr_partialTrace_sub<1>(Qureg, Qureg, std::vector<int>, std::vector<int>);
template void gpu_densmatr_partialTrace_sub<2>(Qureg, Qureg, std::vector<int>, std::vector<int>);
template void gpu_densmatr_partialTrace_sub<3>(Qureg, Qureg, std::vector<int>, std::vector<int>);
template void gpu_densmatr_partialTrace_sub<4>(Qureg, Qureg, std::vector<int>, std::vector<int>);
template void gpu_densmatr_partialTrace_sub<5>(Qureg, Qureg, std::vector<int>, std::vector<int>);
template void gpu_densmatr_partialTrace_sub<-1>(Qureg, Qureg, std::vector<int>, std::vector<int>);

// Explicit template instantiations for probability calculations
template void gpu_statevec_calcProbsOfAllMultiQubitOutcomes_sub<0>(double*, Qureg, std::vector<int>);
template void gpu_statevec_calcProbsOfAllMultiQubitOutcomes_sub<1>(double*, Qureg, std::vector<int>);
template void gpu_statevec_calcProbsOfAllMultiQubitOutcomes_sub<2>(double*, Qureg, std::vector<int>);
template void gpu_statevec_calcProbsOfAllMultiQubitOutcomes_sub<3>(double*, Qureg, std::vector<int>);
template void gpu_statevec_calcProbsOfAllMultiQubitOutcomes_sub<4>(double*, Qureg, std::vector<int>);
template void gpu_statevec_calcProbsOfAllMultiQubitOutcomes_sub<5>(double*, Qureg, std::vector<int>);
template void gpu_statevec_calcProbsOfAllMultiQubitOutcomes_sub<-1>(double*, Qureg, std::vector<int>);

template void gpu_densmatr_calcProbsOfAllMultiQubitOutcomes_sub<0>(double*, Qureg, std::vector<int>);
template void gpu_densmatr_calcProbsOfAllMultiQubitOutcomes_sub<1>(double*, Qureg, std::vector<int>);
template void gpu_densmatr_calcProbsOfAllMultiQubitOutcomes_sub<2>(double*, Qureg, std::vector<int>);
template void gpu_densmatr_calcProbsOfAllMultiQubitOutcomes_sub<3>(double*, Qureg, std::vector<int>);
template void gpu_densmatr_calcProbsOfAllMultiQubitOutcomes_sub<4>(double*, Qureg, std::vector<int>);
template void gpu_densmatr_calcProbsOfAllMultiQubitOutcomes_sub<5>(double*, Qureg, std::vector<int>);
template void gpu_densmatr_calcProbsOfAllMultiQubitOutcomes_sub<-1>(double*, Qureg, std::vector<int>);

// Explicit template instantiations for fidelity calculations
template std::complex<double> gpu_densmatr_calcFidelityWithPureState_sub<false>(Qureg, Qureg);
template std::complex<double> gpu_densmatr_calcFidelityWithPureState_sub<true>(Qureg, Qureg);

// Explicit template instantiations for expectation value calculations
template std::complex<double> gpu_statevec_calcExpecFullStateDiagMatr_sub<false, false>(Qureg, FullStateDiagMatr, std::complex<double>);
template std::complex<double> gpu_statevec_calcExpecFullStateDiagMatr_sub<false, true>(Qureg, FullStateDiagMatr, std::complex<double>);
template std::complex<double> gpu_statevec_calcExpecFullStateDiagMatr_sub<true, false>(Qureg, FullStateDiagMatr, std::complex<double>);
template std::complex<double> gpu_statevec_calcExpecFullStateDiagMatr_sub<true, true>(Qureg, FullStateDiagMatr, std::complex<double>);

template std::complex<double> gpu_densmatr_calcExpecFullStateDiagMatr_sub<false, false>(Qureg, FullStateDiagMatr, std::complex<double>);
template std::complex<double> gpu_densmatr_calcExpecFullStateDiagMatr_sub<false, true>(Qureg, FullStateDiagMatr, std::complex<double>);
template std::complex<double> gpu_densmatr_calcExpecFullStateDiagMatr_sub<true, false>(Qureg, FullStateDiagMatr, std::complex<double>);
template std::complex<double> gpu_densmatr_calcExpecFullStateDiagMatr_sub<true, true>(Qureg, FullStateDiagMatr, std::complex<double>);
