# pecos-num

`pecos-num` provides numerical computing support for PECOS quantum error correction simulations.

This crate brings together numerical computing dependencies and implements functionality needed for QEC analysis in PECOS, including threshold fitting, data analysis, and optimization. It provides APIs with similar functionality to scipy and numpy for numerical operations.

## Features

- Root finding algorithms (Brent's method, Newton-Raphson)
- Non-linear curve fitting (Levenberg-Marquardt)
- Polynomial fitting and evaluation
- Built on robust Rust numerical libraries (Peroxide, levenberg-marquardt, nalgebra)

## Usage

This is an **internal crate** used by:
- `pecos` - The main PECOS metacrate (via prelude)
- `pecos-rslib` - Python bindings exposing numerical functions

For direct usage in Rust:

```rust
use pecos_num::prelude::*;

// Root finding with Brent's method
let root = brentq(|x| x * x - 2.0, 0.0, 2.0, None).unwrap();

// Curve fitting
let result = curve_fit(
    |x, params| params[0] * x + params[1],
    xdata.view(),
    ydata.view(),
    p0.view(),
    None
).unwrap();
```
