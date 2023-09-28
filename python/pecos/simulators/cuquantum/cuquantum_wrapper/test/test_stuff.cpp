// Copyright 2022 The PECOS developers
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
// the License.You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

// Initial author: Tyson Lawrence

#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <chrono>
#include <thread>

#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cuComplex.h>        // cuDoubleComplex
#include <custatevec.h>       // custatevecApplyMatrix
#include <stdio.h>            // printf
#include <stdlib.h>           // EXIT_FAILURE

#include "cuda_helper.hpp"         // HANDLE_ERROR, HANDLE_CUDA_ERROR

#include "custatevec_workspace.hpp"
#include "gate_matrices.hpp"         
#include "gate.hpp"         
#include "state_vector.hpp"         
#include "utils.hpp"
#include "quantum_volume.hpp"         
#include "version.h"         

using CD = std::complex<double>;

int main(int argc, char *argv[])
{
    QuantumVolume qv(2);
    qv.print();
}
