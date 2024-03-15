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

#ifndef CUSTATEVEC_WORKSPACE_HPP
#define CUSTATEVEC_WORKSPACE_HPP

#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <custatevec.h>       // custatevecApplyMatrix

/*
 *
 * Helper class for managing custatevector workspaces
 *
 */
class CuStatevecWorkspace
{
    // Only the following classes should access the private
    // members of this class
    friend class StateVector;
    friend class Gate;

    // Members
    custatevecHandle_t handle;
    void* extra;
    size_t extra_sz;

public:

    // Constructor
    CuStatevecWorkspace();

    // Destructor
    ~CuStatevecWorkspace();

    // Return the current size of the external (extra) workspace
    size_t get_extra_sz();

};

#endif // CUSTATEVEC_WORKSPACE_HPP
