// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0

//! Re-export Clifford conjugation from pecos-core.

pub use pecos_core::pauli::pauli_bitmask::{
    conjugate_cx, conjugate_cy, conjugate_cz, conjugate_h, conjugate_swap, conjugate_sx,
    conjugate_sxdg, conjugate_sy, conjugate_sydg, conjugate_sz, conjugate_szdg, conjugate_x,
    conjugate_y, conjugate_z,
};
