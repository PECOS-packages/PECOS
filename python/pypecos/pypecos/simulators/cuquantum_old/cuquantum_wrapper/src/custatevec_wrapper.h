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

#ifndef CUSTATEVEC_WRAPPER_H
#define CUSTATEVEC_WRAPPER_H

/*
 *
 * C API
 *
 */
#ifdef __cplusplus
extern "C" {
#endif

#include "version.h"

int run_quantum_volume(int num_qubits, int num_shots, double results[], int len_results);

int run_quantum_volume_2(int num_qubits, double angles[][8][3], int targets[], double results[], int len_results);

#ifdef __cplusplus
}
#endif

#endif // CUSTATEVEC_WRAPPER_H
