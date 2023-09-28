# cuQuantum Wrapper

## Overview

**cuquantum_wrapper** is an small library that is intended to serve as an abstraction layer around Nvidia's **cuQuantum** state vector and tensor libraries.  At present, only the state vector library is supported. 

The library uses the **Eigen** linear algebra library.

## Use

1. Clone the repository
2. Setup project
   * `$ cd cuquantum_wrapper`
   * `$ source setup_build.sh` or `$ mkdir build && cd build && cmake ..`
3. Build 
   * `$ make`
4. Run test
   * `$ make test` or `$ ctest --verbose`. The latter provides more verbose output.
   