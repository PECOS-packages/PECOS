# pecos-qulacs

Qulacs quantum backend for PECOS.

## Purpose

Wraps the Qulacs C++ state vector simulator for use as a PECOS quantum engine. Provides high-performance quantum circuit simulation.

## Key Types

- `QulacsStateVec` - State vector simulator using Qulacs backend

## Features

- Full Clifford gate set
- Arbitrary rotation gates (Rx, Ry, Rz, etc.)
- GPU acceleration (optional)
- Implements `QuantumSimulator`, `CliffordGateable`, `ArbitraryRotationGateable` traits

## Acknowledgements

This crate wraps [Qulacs](https://github.com/qulacs/qulacs), a high-performance quantum circuit simulator developed by the Qulacs team at Osaka University and QunaSys.

**Paper:**
- Suzuki, Y., Kawase, Y., Masumura, Y., Hiraga, Y., Nakadai, M., Chen, J., Narasimhan, K., Okada, M., Sugiyama, K., Tan, Y.-Y., Takeshita, T., Yamashita, T., Yoshida, K., Shibasaki, Y., & Yamamoto, N. (2021). "Qulacs: a fast and versatile quantum circuit simulator for research purpose." Quantum, 5, 559. [arXiv:2011.13524](https://arxiv.org/abs/2011.13524)
