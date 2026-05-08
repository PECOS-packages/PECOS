// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0

//! Measurement result identity.
//!
//! Each measurement gate (MZ, MX, etc.) produces a `MeasId` — a unique
//! identifier for that measurement's outcome. Assigned once at circuit
//! construction time, carried through all transformations (`TickCircuit` →
//! `DagCircuit` → `InfluenceMap` → DEM). Never reassigned.
//!
//! This follows the MLIR SSA pattern: the value is defined at one point
//! and referenced everywhere. Detectors reference `MeasId` values
//! directly instead of fragile position-dependent offsets.
//!
//! Metadata (qubit, basis, coordinates, labels) lives in a side table,
//! not on the `MeasId` itself. The hot path (DEM builder, sampler,
//! decoder) works with `MeasId` only.

use std::fmt;

/// Unique identity of a measurement result.
///
/// Lightweight (pointer-sized), `Copy`, directly usable as an array index.
/// Analogous to [`QubitId`](crate::QubitId) but for measurement outcomes.
///
/// # Example
///
/// ```
/// use pecos_core::MeasId;
///
/// let m0 = MeasId(0);
/// let m1 = MeasId(1);
/// assert_ne!(m0, m1);
///
/// // Direct array indexing
/// let mut outcomes = vec![false; 10];
/// outcomes[m0.0] = true;
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MeasId(pub usize);

impl MeasId {
    /// The underlying index.
    #[inline]
    #[must_use]
    pub fn index(self) -> usize {
        self.0
    }
}

impl fmt::Display for MeasId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "m{}", self.0)
    }
}

impl From<usize> for MeasId {
    fn from(v: usize) -> Self {
        Self(v)
    }
}

impl From<MeasId> for usize {
    fn from(m: MeasId) -> Self {
        m.0
    }
}
