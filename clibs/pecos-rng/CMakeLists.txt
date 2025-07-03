# CMakeLists.txt
cmake_minimum_required(VERSION 3.15...3.27)
project(pecos_rng LANGUAGES C CXX)
set(CMAKE_OSX_ARCHITECTURES "x86_64;arm64")


if (CMAKE_VERSION VERSION_LESS 3.18)
  set(DEV_MODULE Development)
else()
  set(DEV_MODULE Development.Module)
endif()

# Find Python (for Python 3.13)
find_package(Python REQUIRED COMPONENTS Interpreter ${DEV_MODULE})

# Detect the installed nanobind package and import it into CMake
# This executes a Python command to get the nanobind CMake directory
execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE nanobind_ROOT
)

set(CMAKE_PREFIX_PATH "${nanobind_ROOT}" ${CMAKE_PREFIX_PATH})

# This sets up nanobind
find_package(nanobind CONFIG REQUIRED)

# Add your C source file as a library
add_library(rng_pcg STATIC src/rng_pcg.c) # Compile test.c as a static library
nanobind_add_module(pecos_rng src/wrapper.cpp)
target_link_libraries(pecos_rng PRIVATE rng_pcg)
# Install the module so it can be found by pip install
install(TARGETS pecos_rng DESTINATION "pecos_pcg")
