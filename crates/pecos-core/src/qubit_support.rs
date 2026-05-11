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

pub(crate) fn duplicate_qubits(qubits: impl IntoIterator<Item = usize>) -> Vec<usize> {
    let mut qubits: Vec<usize> = qubits.into_iter().collect();
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
    #[should_panic(expected = "CX requires distinct qubits; duplicated qubits: [0, 2]")]
    fn assert_distinct_qubits_reports_context_and_duplicates() {
        assert_distinct_qubits("CX", [2, 0, 1, 2, 0]);
    }
}
