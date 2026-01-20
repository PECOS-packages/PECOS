/*
 * Wrapper header for bindgen
 *
 * This header includes the cuQuantum headers we want to generate bindings for.
 */

/* CUDA runtime API for memory management */
#include <cuda_runtime_api.h>

/* CUDA complex types */
#include <cuComplex.h>

/* cuStateVec - state vector simulation */
#include <custatevec.h>

/* cuStabilizer - stabilizer (Clifford) circuit simulation */
#include <custabilizer.h>

/* cuTensorNet - tensor network contraction */
#include <cutensornet.h>

/* cuDensityMat - density matrix simulation */
#include <cudensitymat.h>
