# Copyright 2022 The PECOS Developers
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License.You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

# Initial author: Tyson Lawrence

# cmake_minimum_required(VERSION 3.12...3.17)
cmake_minimum_required(VERSION 3.13.0 FATAL_ERROR)
if(CMAKE_VERSION VERSION_GREATER 3.18)
  cmake_policy(SET CMP0104 OLD)
  set(CUDA_ARCHITECTURES FALSE)
endif()

project(cuquantum_wrapper
  VERSION 0.0.22
  DESCRIPTION "Abstraction layer (wrapper) for cuQuantum"
  LANGUAGES C CXX CUDA)

#
# Options
#
option(DEBUG_BUILD "Build with debugging" OFF)
option(BUILD_TESTS "Build tests" ON)
option(BUILD_EXAMPLES "Build examples" ON)
option(USE_WARNING_FLAGS "Enable warnings" OFF)
option(USE_ERROR_FLAGS "Enable errors" OFF)
option(USE_DEV_FLAGS "Use some flags to disable some warnings during development" OFF)

#
# CUDA
#
find_package(CUDA 10.1 REQUIRED)

if (NOT CUSTATEVEC_ROOT)
  set(CUSTATEVEC_ROOT ${CUDA_TOOLKIT_ROOT_DIR})
endif()

#
# Print build config
#
message(STATUS "  ")
message(STATUS "         Project: " ${PROJECT_NAME})
message(STATUS "Building version: " ${PROJECT_VERSION})
message(STATUS "     System Name: " ${CMAKE_SYSTEM_NAME})
message(STATUS "System Processor: " ${CMAKE_SYSTEM_PROCESSOR})
message(STATUS "  ")

#
# Put all build targets in the same directory for simplicity
#
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)

#
# Set global language standards project wide
#
set(CMAKE_C_STANDARD            11)
set(CMAKE_C_STANDARD_REQUIRED   ON)
set(CMAKE_C_EXTENSIONS          OFF)

set(CMAKE_CXX_STANDARD          17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS        OFF)

# Global CUDA CXX flags/options
set(CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)


#
# Create lists of possible build flags to use in the project. These
# are not necessarily used
#
# TODO FIXME Not all compiler options all work with nvcc
# set(WARNING_FLAGS -Wall -pedantic)
# set(ERROR_FLAGS -Werror -pedantic-errors)
# set(DEV_FLAGS -g
#               -Wno-unused-variable
#               -Wno-unused-but-set-variable
#               -Wno-unused-function
#               -Wno-attributes)

# if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
#   set(CLANG_FLAGS -Wno-unused-private-field
#                   -Wno-unknown-warning-option)
# endif()

add_subdirectory(extern/pybind11)

#
# Main source directories in project
#
add_subdirectory(src)

if (BUILD_EXAMPLES)
  add_subdirectory(examples)
endif ()

#
# Testing
#
if (BUILD_TESTS)
  enable_testing()
  add_subdirectory(test)
endif ()
