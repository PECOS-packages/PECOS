use super::{Location, LossDemError, Side, circuit_error, invalid};
use crate::fault_tolerance::fault_sampler::{GateLoc, flatten_tick_circuit, propagate_forward};
use pecos_core::{BitSet, Pauli};
use pecos_quantum::{F2Matrix, TickCircuit};
use pecos_simulators::BitmaskPauliProp;
use std::collections::HashMap;

/// Propagate one Pauli at a stored-batch operand, returning flipped emission indices.
/// A before injection starts at the batch's first instance; an after injection starts
/// past its last instance. Operations within a tick are disjoint, so other instances
/// of that batch cannot affect this operand.
///
/// # Errors
/// Rejects unsupported circuits and invalid locations.
pub fn located_pauli_flips(
    circuit: &TickCircuit,
    pauli: Pauli,
    location: Location,
) -> Result<BitSet, LossDemError> {
    if let Some(error) = crate::fault_tolerance::propagator::first_unsupported_tick_gate(circuit) {
        return Err(circuit_error("located Pauli propagation", error));
    }
    let (gates, measurements) = flatten_tick_circuit(circuit);
    located_pauli_flips_prepared(&gates, &measurements, pauli, location)
}

// The caller validates and flattens once. Every instance in a stored batch has
// the same arity, so operand lookup needs no circuit walk or batch scan.
pub(super) fn located_pauli_flips_prepared(
    gates: &[GateLoc],
    measurements: &HashMap<usize, usize>,
    pauli: Pauli,
    location: Location,
) -> Result<BitSet, LossDemError> {
    let key = (location.tick, location.batch);
    let first = gates.partition_point(|g| (g.tick, g.gate_index) < key);
    let end = gates.partition_point(|g| (g.tick, g.gate_index) <= key);
    let instances = &gates[first..end];
    let arity = instances.first().map_or(0, |gate| gate.qubits.len());
    if arity == 0 {
        return Err(invalid(format!(
            "location {location:?} has no operand-bearing batch; choose an existing gate batch"
        )));
    }
    let qubit = instances
        .get(location.operand / arity)
        .and_then(|gate| gate.qubits.get(location.operand % arity))
        .ok_or_else(|| {
            invalid(format!(
                "location {location:?} exceeds {} operands; choose an existing operand",
                instances.len() * arity
            ))
        })?;
    let start = match location.side {
        Side::Before => first,
        Side::After => end,
    };
    let mut prop = BitmaskPauliProp::new();
    match pauli {
        Pauli::I => {}
        Pauli::X => prop.track_x(&[*qubit]),
        Pauli::Y => prop.track_y(&[*qubit]),
        Pauli::Z => prop.track_z(&[*qubit]),
    }
    Ok(propagate_forward(&mut prop, start, gates, measurements)
        .into_iter()
        .collect())
}

/// A reduced GF(2) span with an explicit ambient width, including the zero span.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Span {
    width: usize,
    basis: Vec<BitSet>,
    pivots: Vec<usize>,
}

impl Span {
    /// Row-reduce generators using the shared GF(2) matrix implementation.
    ///
    /// # Errors
    /// Rejects any set bit outside `width`.
    pub fn new(width: usize, vectors: &[BitSet]) -> Result<Self, LossDemError> {
        let mut matrix = F2Matrix::zeros(vectors.len(), width);
        for (row, vector) in vectors.iter().enumerate() {
            let mut dense = vec![0; width];
            for column in vector {
                let bit = dense.get_mut(column).ok_or_else(|| {
                    invalid(format!(
                        "vector bit {column} exceeds ambient width {width}; increase the width or correct the vector"
                    ))
                })?;
                *bit = 1;
            }
            matrix.set_row(row, &dense);
        }
        let (reduced, pivots) = matrix.row_reduce();
        let basis = reduced
            .rows()
            .into_iter()
            .take(pivots.len())
            .map(|row| {
                row.into_iter()
                    .enumerate()
                    .filter_map(|(i, bit)| (bit != 0).then_some(i))
                    .collect()
            })
            .collect();
        Ok(Self {
            width,
            basis,
            pivots,
        })
    }

    /// Ambient vector width, preserved by intersection and quotient.
    #[must_use]
    pub fn width(&self) -> usize {
        self.width
    }
    /// Dimension of this span.
    #[must_use]
    pub fn rank(&self) -> usize {
        self.basis.len()
    }
    /// Canonical reduced generators.
    #[must_use]
    pub fn basis(&self) -> &[BitSet] {
        &self.basis
    }

    fn check_width(&self, other: &Self) -> Result<(), LossDemError> {
        if self.width != other.width {
            return Err(invalid(format!(
                "ambient widths {} and {} differ; use the same width",
                self.width, other.width
            )));
        }
        Ok(())
    }

    /// Intersection from the kernel of the transposed stacked bases.
    ///
    /// # Errors
    /// Rejects different ambient widths.
    pub fn intersection(&self, other: &Self) -> Result<Self, LossDemError> {
        self.check_width(other)?;
        let stacked: Vec<_> = self.basis.iter().chain(&other.basis).collect();
        let mut matrix = F2Matrix::zeros(self.width, stacked.len());
        for row in 0..self.width {
            matrix.set_row(
                row,
                &stacked
                    .iter()
                    .map(|v| u8::from(v.contains(row)))
                    .collect::<Vec<_>>(),
            );
        }
        let vectors = matrix
            .kernel()
            .into_iter()
            .map(|coefficients| {
                let mut vector = BitSet::new();
                for (coefficient, basis) in coefficients.into_iter().zip(&self.basis) {
                    if coefficient != 0 {
                        vector ^= basis;
                    }
                }
                vector
            })
            .collect::<Vec<_>>();
        Self::new(self.width, &vectors)
    }

    /// Reduce this span modulo another basis, keeping the ambient coordinates.
    ///
    /// # Errors
    /// Rejects different ambient widths. The modulus need not be a subspace of self.
    pub fn quotient(&self, modulus: &Self) -> Result<Self, LossDemError> {
        self.check_width(modulus)?;
        let vectors = self
            .basis
            .iter()
            .map(|vector| {
                let mut remainder = vector.clone();
                for (&pivot, row) in modulus.pivots.iter().zip(&modulus.basis) {
                    if remainder.contains(pivot) {
                        remainder ^= row;
                    }
                }
                remainder
            })
            .collect::<Vec<_>>();
        Self::new(self.width, &vectors)
    }
}
