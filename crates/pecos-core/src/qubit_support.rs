// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under
// the License.

pub(crate) fn overlapping_qubits(
    lhs: impl IntoIterator<Item = usize>,
    rhs: impl IntoIterator<Item = usize>,
) -> Vec<usize> {
    let mut lhs_qubits: Vec<usize> = lhs.into_iter().collect();
    lhs_qubits.sort_unstable();
    lhs_qubits.dedup();

    let mut rhs_qubits: Vec<usize> = rhs.into_iter().collect();
    rhs_qubits.sort_unstable();
    rhs_qubits.dedup();

    let mut overlap = Vec::new();
    let mut lhs_idx = 0;
    let mut rhs_idx = 0;
    while lhs_idx < lhs_qubits.len() && rhs_idx < rhs_qubits.len() {
        match lhs_qubits[lhs_idx].cmp(&rhs_qubits[rhs_idx]) {
            std::cmp::Ordering::Less => lhs_idx += 1,
            std::cmp::Ordering::Greater => rhs_idx += 1,
            std::cmp::Ordering::Equal => {
                overlap.push(lhs_qubits[lhs_idx]);
                lhs_idx += 1;
                rhs_idx += 1;
            }
        }
    }
    overlap
}

/// Returns the sorted, distinct qubit indices that occur more than once.
pub fn duplicate_qubits(qubits: impl IntoIterator<Item = usize>) -> Vec<usize> {
    // Most gate validations have at most two operands; avoid a scratch allocation.
    let mut qubits = qubits.into_iter();
    let mut qubits: Vec<usize> = if qubits.size_hint().1.is_none_or(|upper| upper > 2) {
        // Preserve collection specialization, including owned-vector reuse.
        qubits.collect()
    } else {
        let Some(first) = qubits.next() else {
            return Vec::new();
        };
        let Some(second) = qubits.next() else {
            return Vec::new();
        };
        let Some(third) = qubits.next() else {
            return if first == second {
                vec![first]
            } else {
                Vec::new()
            };
        };
        [first, second, third].into_iter().chain(qubits).collect()
    };
    qubits.sort_unstable();

    let mut duplicates = Vec::new();
    for window in qubits.windows(2) {
        if window[0] == window[1] && duplicates.last() != Some(&window[0]) {
            duplicates.push(window[0]);
        }
    }
    duplicates
}

pub(crate) fn assert_distinct_qubits(context: &str, qubits: impl IntoIterator<Item = usize>) {
    let duplicates = duplicate_qubits(qubits);
    assert!(
        duplicates.is_empty(),
        "{context} requires distinct qubits; duplicated qubits: {duplicates:?}"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn overlapping_qubits_returns_sorted_deduplicated_intersection() {
        assert_eq!(overlapping_qubits([3, 1, 1, 2], [2, 2, 3, 4]), vec![2, 3]);
    }

    #[test]
    fn duplicate_qubits_returns_sorted_deduplicated_repeats() {
        assert_eq!(duplicate_qubits([5, 1, 5, 2, 2, 2]), vec![2, 5]);
    }

    #[test]
    fn duplicate_qubits_matches_frequency_oracle_for_short_inputs() {
        for len in 0..=7 {
            for mut encoded in 0..4_usize.pow(len) {
                let mut counts = [0; 4];
                let input: Vec<_> = (0..len)
                    .map(|_| {
                        let qubit = encoded % 4;
                        encoded /= 4;
                        counts[qubit] += 1;
                        qubit
                    })
                    .collect();
                let expected: Vec<_> = (0..4).filter(|&q| counts[q] > 1).collect();
                assert_eq!(
                    duplicate_qubits(input.iter().copied().filter(|_| true)),
                    expected
                );
                assert_eq!(duplicate_qubits(input), expected);
            }
        }
    }

    #[test]
    fn duplicate_qubits_stops_at_first_exhaustion() {
        let mut calls = 0;
        let input = std::iter::from_fn(|| {
            calls += 1;
            match calls {
                1 => Some(usize::MAX),
                2 => None,
                _ => panic!("iterator polled after exhaustion"),
            }
        });
        assert_eq!(duplicate_qubits(input), [] as [usize; 0]);
        assert_eq!(calls, 2);
        assert_eq!(duplicate_qubits([usize::MAX, usize::MAX]), [usize::MAX]);
    }

    #[test]
    #[should_panic(expected = "CX requires distinct qubits; duplicated qubits: [0, 2]")]
    fn assert_distinct_qubits_reports_context_and_duplicates() {
        assert_distinct_qubits("CX", [2, 0, 1, 2, 0]);
    }
}
