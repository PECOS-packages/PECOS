![PECOS Logo](assets/images/pecos_logo.svg)

# Introduction

**PECOS** (Performance Estimator of Codes On Surfaces) is a library/framework dedicated to the study, development, and
evaluation of quantum error-correction protocols. It also offers tools for the study and evaluation of hybrid
quantum/classical compute execution models.

## Quick Start

Simulate a distance-3 repetition code with syndrome extraction using [Guppy](https://github.com/CQCL/guppylang), a pythonic quantum programming language:

=== ":fontawesome-brands-python: Python"

    ```bash
    pip install quantum-pecos
    ```

    ```python
    from pecos import Guppy, sim, state_vector, depolarizing_noise
    from guppylang import guppy
    from guppylang.std.quantum import qubit, cx, measure


    @guppy
    def repetition_code() -> None:
        # 3 data qubits encode logical |0⟩ = |000⟩
        d0 = qubit()
        d1 = qubit()
        d2 = qubit()

        # 2 ancillas for syndrome extraction
        s0 = qubit()
        s1 = qubit()

        # Measure parity between adjacent data qubits
        cx(d0, s0)
        cx(d1, s0)
        cx(d1, s1)
        cx(d2, s1)

        # Measure syndromes (first two measurements)
        _ = measure(s0)
        _ = measure(s1)

        # Measure data qubits (required by Guppy)
        _ = measure(d0), measure(d1), measure(d2)


    # Run 10 shots with 10% depolarizing noise
    noise = depolarizing_noise().with_uniform_probability(0.1)
    results = sim(Guppy(repetition_code)).qubits(5).quantum(state_vector()).noise(noise).seed(42).run(10)

    # Extract syndromes from first two measured qubits (s0, s1)
    d = results.to_dict()
    syndrome = [[d["q0"][i], d["q1"][i]] for i in range(10)]
    print(syndrome)
    # [[0, 0], [1, 0], [0, 0], [0, 0], [0, 0], [0, 1], [0, 1], [0, 0], [0, 0], [0, 0]]
    ```

    Non-trivial syndromes like `[1, 0]`, `[0, 1]`, `[1, 1]` indicate detected errors that a decoder would use to identify and correct faults.

=== ":fontawesome-brands-rust: Rust"

    ```toml
    # Cargo.toml
    [dependencies]
    pecos = { version = "0.1", features = ["qasm"] }
    ```

    ```rust
    use pecos::prelude::*;

    fn main() -> Result<(), Box<dyn std::error::Error>> {
        // Define a Bell state circuit
        let circuit = Qasm::from_string(r#"
            OPENQASM 2.0;
            include "qelib1.inc";
            qreg q[2];
            creg c[2];
            h q[0];
            cx q[0], q[1];
            measure q -> c;
        "#);

        // Run 10 shots
        let results = sim(circuit).seed(42).run(10)?;
        println!("{:?}", results);
        // 0 = both |0⟩, 3 = both |1⟩ (always correlated!)
        Ok(())
    }
    ```

For OpenQASM, PHIR, or other formats, see the [User Guide](user-guide/getting-started.md).

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

Initially developed in 2014 to verify lattice-surgery procedures presented in [arXiv:1407.5103](https://arxiv.org/abs/1407.5103) and
released publicly in 2018, PECOS provided QEC tools not available at that time. PECOS developed into a
framework for studying general QECCs and hybrid quantum-classical computation.

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

### Additional Citation Formats

**PhD Thesis** (where PECOS was first described):

```bibtex
@phdthesis{crathesis,
 author={Ciar\'{a}n Ryan-Anderson},
 school = {University of New Mexico},
 title={Quantum Algorithms, Architecture, and Error Correction},
 journal={arXiv:1812.04735},
 URL = {https://digitalrepository.unm.edu/phyc_etds/203},
 year={2018}
}
```

**Zenodo DOI** (for citing a specific version):

```bibtex
@software{pecos_version,
  author       = {Ciar\'{a}n Ryan-Anderson},
  title        = {PECOS-packages/PECOS: [version]},
  month        = [month],
  year         = [year],
  publisher    = {Zenodo},
  version      = {[version]},
  doi          = {10.5281/zenodo.13700104},
  url          = {https://doi.org/10.5281/zenodo.13700104}
}
```

See [Zenodo](https://zenodo.org/records/13700104) for version-specific DOIs.
