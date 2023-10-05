# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License.You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.
#
# Initial author: Tyson Lawrence
#
# Set include directories
#
set(EX_INCLUDE_DIRS
  ${PROJECT_SOURCE_DIR}/extern/argparse
  )
#
# Helper function to simplify adding an example. The example name
# should correspond the example file name.
# e.g. example1 goes with example1.cpp
#
function(add_example EXAMPLE_NAME)
  # Get the main test source file and append it to the list of extra sources
  set(EXAMPLE_SOURCE ${EXAMPLE_NAME}.cpp)
  # Create the example program
  add_executable(${EXAMPLE_NAME} ${EXAMPLE_SOURCE})
  target_include_directories(${EXAMPLE_NAME} PRIVATE ${EX_INCLUDE_DIRS})
  # Link the the wrapper library
  target_include_directories(${EXAMPLE_NAME} PRIVATE /usr/include/eigen3/)
  target_include_directories(${EXAMPLE_NAME} PRIVATE /usr/include/eigen3/Eigen/)
  target_include_directories(${EXAMPLE_NAME} PRIVATE /usr/include/eigen3/Eigen/src)
  target_link_libraries(${EXAMPLE_NAME} PRIVATE custatevec_wrapper)
endfunction()

#
# The examples
#
add_example(bell_state)
add_example(quantum_volume)
