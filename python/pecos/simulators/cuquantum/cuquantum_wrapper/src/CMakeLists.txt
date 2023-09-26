# Copyright 2022 The PECOS developers
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License.You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

# Initial author: Tyson Lawrence

# Configure the version file
configure_file(
  ${PROJECT_SOURCE_DIR}/src/version.h.in
  ${PROJECT_SOURCE_DIR}/src/version.h
  )
configure_file(
  ${PROJECT_SOURCE_DIR}/src/version.c.in
  ${PROJECT_SOURCE_DIR}/src/version.c
  )


set(LIB_INCLUDE_DIRS 
  ${PROJECT_SOURCE_DIR}/src
  ${PROJECT_SOURCE_DIR}/extern/eigen
  ${PROJECT_SOURCE_DIR}/extern/eigen/unsupported
  ${CUSTATEVEC_ROOT}/include
  /opt/nvidia/cuquantum/include
  /usr/include/eigen3
  /usr/include/eigen3/Eigen
  /usr/local/cuda/lib64
  )

set(LIB_SOURCES
  version.c
  custatevec_wrapper.cpp

  custatevec_workspace.cpp
  utils.cpp
  state_vector.cpp
  gate.cpp
  gate_matrices.cpp
  quantum_volume.cpp

  custom_kernels.cu
  )

# TODO FIXME Not all compiler options all work with nvcc
# set(_COMPILE_FLAGS 
#   $<$<BOOL:${USE_DEV_FLAGS}>:${DEV_FLAGS}>
#   $<$<BOOL:${USE_WARNING_FLAGS}>:${WARNING_FLAGS}>
#   $<$<BOOL:${USE_ERROR_FLAGS}>:${ERROR_FLAGS}>
#   )

add_library(custatevec_wrapper SHARED ${LIB_SOURCES})
target_include_directories(custatevec_wrapper PUBLIC ${LIB_INCLUDE_DIRS})
target_link_directories(custatevec_wrapper PUBLIC ${CUSTATEVEC_ROOT}/lib)
target_link_directories(custatevec_wrapper PUBLIC /usr/local/cuda/lib64)
target_link_libraries(custatevec_wrapper PUBLIC custatevec cudart cublas cublasLt)
# target_compile_options(custatevec_wrapper PUBLIC ${_COMPILE_FLAGS})
set_target_properties(custatevec_wrapper PROPERTIES
    SOVERSION  ${PROJECT_VERSION_MAJOR}
    VERSION    ${PROJECT_VERSION}
)

#
# Create the Python bindings
#
set(PYBINDING_SOURCES python_bindings.cpp ${LIB_SOURCES})
pybind11_add_module(cuquantum_wrapper ${PYBINDING_SOURCES})
target_include_directories(cuquantum_wrapper PUBLIC ${LIB_INCLUDE_DIRS})
target_include_directories(cuquantum_wrapper PUBLIC /usr/include/eigen3/)
target_include_directories(cuquantum_wrapper PUBLIC /usr/include/eigen3/Eigen)
target_include_directories(cuquantum_wrapper PUBLIC /usr/include/eigen3/Eigen/src/KLUSupport)
target_link_directories(cuquantum_wrapper PUBLIC ${CUSTATEVEC_ROOT}/lib)
target_link_libraries(cuquantum_wrapper PUBLIC custatevec cudart cublas cublasLt)
set_property(TARGET cuquantum_wrapper PROPERTY CUDA_ARCHITECTURES 80)
