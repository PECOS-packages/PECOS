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

#
# Set include directories
#
set(TEST_INCLUDE_DIRS
  ${PROJECT_SOURCE_DIR}/extern/catch
  ${PROJECT_SOURCE_DIR}/extern/eigen
  ${PROJECT_SOURCE_DIR}/extern/eigen/unsupported
  ${PROJECT_SOURCE_DIR}/src
  ${CUSTATEVEC_ROOT}/include
  /opt/nvidia/cuquantum/include
  /usr/include/eigen3
  /usr/include/eigen3/Eigen
  )

#
# Set some shortcuts to some directories we are interested in
#
set(SRC_DIR ${PROJECT_SOURCE_DIR}/src)
set(TEST_DIR ${PROJECT_SOURCE_DIR}/test)
set(CATCH_HEADER_DIR ${PROJECT_SOURCE_DIR}/extern/catch)

#
# Create main catch object file
#
if (NOT TARGET catch_main)
  add_library(catch_main OBJECT ${TEST_DIR}/catch_main.cpp ${CATCH_HEADER_DIR}/catch.hpp )
  target_include_directories(catch_main PUBLIC ${CATCH_HEADER_DIR})
endif()

#
# Function to simplify adding a Catch based unit test. The test name
# should correspond the main test source file. EX: test_foo target
# corresponds to test_foo.cpp
#
function(add_catch_test TEST_NAME TEST_SOURCES)
  # Get the main test source file and append it to the list of test sources
  set(TEST_MAIN "${TEST_DIR}/${TEST_NAME}.cpp")
  set(TEST_SOURCES ${TEST_MAIN} ${TEST_SOURCES})
  # Create the test program
  add_executable(${TEST_NAME} ${TEST_SOURCES} $<TARGET_OBJECTS:catch_main>)
  target_include_directories(${TEST_NAME} PRIVATE ${TEST_INCLUDE_DIRS})
  # Add the test to CTest
  add_test(NAME ${TEST_NAME} COMMAND $<TARGET_FILE:${TEST_NAME}>)
  target_include_directories(${TEST_NAME} PRIVATE ${TEST_INCLUDE_DIRS})
  target_link_directories(${TEST_NAME} PRIVATE ${CUSTATEVEC_ROOT}/lib64)
  target_link_libraries(${TEST_NAME} PRIVATE custatevec cudart cublas cublasLt)
endfunction()

#
# The unit tests
#
set(TEST_SOURCES
  ${SRC_DIR}/gate_matrices.cpp
  )
add_catch_test(test_gate_matrices "${TEST_SOURCES}")

set(TEST_SOURCES
  ${SRC_DIR}/custatevec_workspace.cpp
  ${SRC_DIR}/utils.cpp
  ${SRC_DIR}/state_vector.cpp
  ${SRC_DIR}/custom_kernels.cu
  )
add_catch_test(test_state_vector "${TEST_SOURCES}")

set(TEST_SOURCES
  ${SRC_DIR}/custatevec_workspace.cpp
  ${SRC_DIR}/utils.cpp
  ${SRC_DIR}/gate_matrices.cpp
  ${SRC_DIR}/state_vector.cpp
  ${SRC_DIR}/custom_kernels.cu
  ${SRC_DIR}/gate.cpp
  )
add_catch_test(test_gate "${TEST_SOURCES}")


#
# Generic test / scratch pad
#
set(TEST_SOURCES
  test_stuff.cpp
  )
add_executable(test_stuff "${TEST_SOURCES}")
target_link_libraries(test_stuff PRIVATE custatevec_wrapper)
