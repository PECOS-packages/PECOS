![PECOS Logo](assets/images/pecos_logo.svg)

# Introduction

**PECOS** (Performance Estimator of Codes On Surfaces) is a library/framework dedicated to the study, development, and
evaluation of quantum error-correction protocols. It also offers tools for the study and evaluation of hybrid
quantum/classical compute execution models.

## Features

- **Quantum Error-Correction Tools**: Advanced tools for studying quantum error-correction protocols and error models.
- **Hybrid Quantum/Classical Execution**: Evaluate advanced hybrid compute models, including support for classical
  compute, calls to Wasm VMs, conditional branching, and more.
- **Fast Simulation**: Leverages a fast stabilizer simulation algorithm.
- **Multi-language extensions**: Core functionalities implemented via Rust for performance and safety. Additional
  add-ons and extension support in C/C++ via Cython.
- **QIR Support**: Execute Quantum Intermediate Representation programs (requires LLVM version 14).

## Available Implementations

PECOS is available in multiple languages:

- **Python**: [`quantum-pecos`](https://pypi.org/project/quantum-pecos/) package
- **Rust**: [`pecos`](https://crates.io/crates/pecos) crate and related sub-crates

## Documentation Structure

This documentation is organized to help you get the most out of PECOS:

- **[User Guide](user-guide/getting-started.md)**: Concepts and tutorials for using PECOS
- **API Reference**: Detailed API documentation
    - [Python API](https://quantum-pecos.readthedocs.io/en/latest/)
    - [Rust API](https://docs.rs/pecos/latest/pecos/)
- **[Development](development/DEVELOPMENT.md)**: Contributing to PECOS
- **[Releases](releases/changelog.md)**: Version history and changes

## Project History

Initially conceived and developed in 2014 to verify lattice-surgery procedures presented in [arXiv:1407.5103](https://arxiv.org/abs/1407.5103) and
released publicly in 2018, PECOS provided QEC tools not available at that time. Over the years, it has grown into a
framework for studying general QECCs and hybrid computation.

## Getting Support

If you encounter issues or have questions:

- **GitHub Issues**: Submit bug reports or feature requests on [GitHub](https://github.com/PECOS-packages/PECOS/issues)
- **Discussions**: Participate in discussions on [GitHub Discussions](https://github.com/PECOS-packages/PECOS/discussions)

## Citing PECOS

For publications utilizing PECOS, please cite:

```bibtex
@misc{pecos,
 author={Ciar\'{a}n Ryan-Anderson},
 title={PECOS: Performance Estimator of Codes On Surfaces},
 publisher = {GitHub},
 journal = {GitHub repository},
 howpublished={\url{https://github.com/PECOS-packages/PECOS}},
 URL = {https://github.com/PECOS-packages/PECOS},
 year={2018}
}
```

Or use the [Zenodo DOI](https://zenodo.org/records/13700104) for citing a specific version.
