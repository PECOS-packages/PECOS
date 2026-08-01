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
/// Lightweight (pointer-sized), `Copy`. An identity, **not an ordinal**:
/// externally supplied ids may be sparse and out of order, so the numeric
/// value must never be used to index an array. Boundaries that need a dense
/// ordinal build their own `MeasId -> index` map.
/// Analogous to [`QubitId`](crate::QubitId) but for measurement outcomes.
///
/// # Example
///
/// ```
/// use pecos_core::MeasId;
///
/// use std::collections::BTreeMap;
///
/// let m0 = MeasId::from_raw(0);
/// let m1 = MeasId::from_raw(1);
/// assert_ne!(m0, m1);
///
/// // Resolve through a map, never by the numeric value: externally supplied
/// // ids (e.g. MeasId::from_raw(9000)) are legal and would index out of bounds.
/// let ordinal: BTreeMap<MeasId, usize> = [(m0, 0), (m1, 1)].into();
/// let mut outcomes = vec![false; ordinal.len()];
/// outcomes[ordinal[&m0]] = true;
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MeasId(usize);

impl MeasId {
    /// Construct an id from a raw integer.
    ///
    /// This is the ONLY route from an integer to a `MeasId`, so every use is
    /// auditable: it belongs at boundaries where a numbering genuinely enters
    /// the system -- circuit allocators minting fresh ids, bindings accepting
    /// externally supplied (e.g. Guppy) ids, and test fixtures. Reaching for it
    /// inside consumer logic to convert an ordinal back into an id is the
    /// conflation this type exists to prevent.
    #[inline]
    #[must_use]
    pub const fn from_raw(raw: usize) -> Self {
        Self(raw)
    }

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
