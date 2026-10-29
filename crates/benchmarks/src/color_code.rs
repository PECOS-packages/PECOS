// Copyright 2026 The PECOS Developers
// Licensed under the Apache License, Version 2.0.

//! Shared triangular 6.6.6 color-code geometry for benchmark workloads.

use std::collections::BTreeMap;

/// A weight-four boundary face or weight-six interior face.
#[derive(Debug)]
pub struct ColorFace {
    /// Data-qubit indices in coordinate order.
    pub qubits: Vec<usize>,
    /// Face colour, in `0..3`; incident faces have distinct colours.
    pub colour: u8,
}

/// A triangular patch with one logical qubit and identical X/Z face supports.
#[derive(Debug)]
pub struct ColorCode {
    /// Number of data qubits, indexed from zero.
    pub num_data_qubits: usize,
    /// Faces in row-major coordinate order.
    pub faces: Vec<ColorFace>,
}

/// Construct and validate the triangular hexagonal patch of arXiv:1911.00355,
/// Fig. 1(a), using <https://qecsim.github.io/api/models/color.html> coordinates.
///
/// Inside `0 <= column <= row <= 3(d-1)/2`, sites with `(row+column) % 3 == 2`
/// are face centres; the remaining sites are data qubits. Neighbouring sites
/// form hexagons clipped to four vertices at the three boundaries.
///
/// Validation uses sparse incidence throughout, including a full-row-rank
/// certificate, so it does not allocate a dense parity-check matrix.
///
/// # Panics
/// Panics for distances other than odd `d >= 3`, coordinate overflow, or an
/// invalid patch (counts, weights, incidence, colouring, commutation or rank).
#[must_use]
pub fn color_code(d: usize) -> ColorCode {
    assert!(d >= 3 && d % 2 == 1, "color666 d{d}: odd distance >= 3");
    let bound = i32::try_from(3 * (d - 1) / 2).expect("color666 coordinate bound fits i32");
    let mut qubits = BTreeMap::new();
    let mut centres = Vec::new();
    for row in 0..=bound {
        for col in 0..=row {
            if (row + col) % 3 == 2 {
                centres.push((row, col));
            } else {
                qubits.insert((row, col), qubits.len());
            }
        }
    }
    let n = qubits.len();
    assert_eq!(n, (3 * d * d + 1) / 4, "color666 d{d}: data count");
    assert_eq!(
        centres.len(),
        (3 * d * d - 3) / 8,
        "color666 d{d}: face count"
    );
    for side in 0..3 {
        let count = qubits
            .keys()
            .filter(|&&(r, c)| match side {
                0 => c == 0,
                1 => r == c,
                _ => r == bound,
            })
            .count();
        assert_eq!(count, d, "color666 d{d}: boundary {side} length");
    }
    let mut incidence = vec![Vec::new(); n];
    let mut colours_at_qubit = vec![0u8; n];
    let faces: Vec<_> = centres
        .iter()
        .enumerate()
        .map(|(i, &(r, c))| {
            let colour = u8::try_from(r % 3).expect("face colour in 0..3");
            let mut face = ColorFace {
                qubits: Vec::new(),
                colour,
            };
            for (dr, dc) in [(-1, -1), (-1, 0), (0, -1), (0, 1), (1, 0), (1, 1)] {
                if let Some(&q) = qubits.get(&(r + dr, c + dc)) {
                    face.qubits.push(q);
                    incidence[q].push(i);
                    assert_eq!(
                        colours_at_qubit[q] & (1 << colour),
                        0,
                        "color666 d{d}: adjacent faces share a colour at qubit {q}"
                    );
                    colours_at_qubit[q] |= 1 << colour;
                }
            }
            assert!(
                matches!(face.qubits.len(), 4 | 6),
                "color666 d{d}: face ({r},{c}) weight {}",
                face.qubits.len()
            );
            face
        })
        .collect();
    assert!(
        incidence.iter().all(|faces| (1..=3).contains(&faces.len())),
        "color666 d{d}: qubit incidence"
    );
    assert_eq!(
        incidence.iter().filter(|faces| faces.len() == 1).count(),
        3,
        "color666 d{d}: corner count"
    );
    for (i, face) in faces.iter().enumerate() {
        // Only intersecting faces can contribute to Hx Hz^T. Include the face
        // itself to check its even weight, and count each shared vertex once.
        let mut overlaps = BTreeMap::new();
        for &q in &face.qubits {
            for &j in &incidence[q] {
                *overlaps.entry(j).or_insert(0) ^= 1;
            }
        }
        for (j, parity) in overlaps {
            assert_eq!(parity, 0, "color666 d{d}: Hx Hz^T at ({i},{j})");
        }
    }
    assert_full_rank(&faces, &incidence, d);
    assert_eq!(n - 2 * faces.len(), 1, "color666 d{d}: logical count");
    ColorCode {
        num_data_qubits: n,
        faces,
    }
}

fn assert_full_rank(faces: &[ColorFace], incidence: &[Vec<usize>], d: usize) {
    // A column incident to only one remaining row certifies that row is
    // independent. Peel it, starting at the corners. Peeling every row gives
    // a triangular full-rank minor without Gaussian-elimination fill-in.
    let mut degrees: Vec<_> = incidence.iter().map(Vec::len).collect();
    let mut row_ids: Vec<_> = incidence
        .iter()
        .map(|rows| rows.iter().fold(0, |a, b| a ^ b))
        .collect();
    let mut queue: Vec<_> = degrees
        .iter()
        .enumerate()
        .filter_map(|(q, &degree)| (degree == 1).then_some(q))
        .collect();
    let mut rank = 0;
    while let Some(q) = queue.pop() {
        if degrees[q] != 1 {
            continue;
        }
        let row = row_ids[q];
        rank += 1;
        for &p in &faces[row].qubits {
            degrees[p] -= 1;
            row_ids[p] ^= row;
            if degrees[p] == 1 {
                queue.push(p);
            }
        }
    }
    assert_eq!(rank, faces.len(), "color666 d{d}: full row rank");
}

#[cfg(test)]
mod tests {
    use super::*;
    use pecos_qec::ParityCheckMatrix;

    #[test]
    fn sparse_validation_matches_dense_rank() {
        for d in [3, 5, 7, 9, 11, 15] {
            let code = color_code(d);
            let rows = code
                .faces
                .iter()
                .map(|face| {
                    let mut row = vec![0; code.num_data_qubits];
                    for &q in &face.qubits {
                        row[q] = 1;
                    }
                    row
                })
                .collect();
            let h = ParityCheckMatrix::from_dense(rows).expect("binary face incidence");
            assert_eq!(code.num_data_qubits - 2 * h.rank(), 1);
        }
    }
}
