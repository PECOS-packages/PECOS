// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file
// except in compliance with the License. You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the
// License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
// express or implied. See the License for the specific language governing permissions and
// limitations under the License.

//! Fixed-width binary masks and Gaussian elimination.

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct Mask(pub(crate) Vec<u64>);

impl Mask {
    pub(crate) fn zero(width: usize) -> Self {
        Self(vec![0; width.div_ceil(64)])
    }

    pub(crate) fn get(&self, bit: usize) -> bool {
        self.0[bit / 64] & (1 << (bit % 64)) != 0
    }

    pub(crate) fn toggle(&mut self, bit: usize) {
        self.0[bit / 64] ^= 1 << (bit % 64);
    }

    pub(crate) fn xor(&mut self, other: &Self) {
        for (a, b) in self.0.iter_mut().zip(&other.0) {
            *a ^= b;
        }
    }

    pub(crate) fn dot(&self, other: &Self) -> bool {
        self.0
            .iter()
            .zip(&other.0)
            .fold(0, |p, (a, b)| p ^ (a & b).count_ones())
            & 1
            != 0
    }

    pub(crate) fn is_zero(&self) -> bool {
        self.0.iter().all(|&word| word == 0)
    }

    pub(crate) fn ones(&self) -> impl Iterator<Item = usize> + '_ {
        self.0.iter().enumerate().flat_map(|(i, &word)| {
            let mut remaining = word;
            std::iter::from_fn(move || {
                if remaining == 0 {
                    return None;
                }
                let bit = remaining.trailing_zeros() as usize;
                remaining &= remaining - 1;
                Some(64 * i + bit)
            })
        })
    }

    pub(crate) fn remove(&mut self, bit: usize, width: usize) {
        for j in bit..width - 1 {
            if self.get(j) != self.get(j + 1) {
                self.toggle(j);
            }
        }
        if self.get(width - 1) {
            self.toggle(width - 1);
        }
    }
}

/// Solve a full-column-rank binary system, returning `None` iff inconsistent.
pub(crate) fn solve(rows: &[Mask], rhs: &[bool], width: usize, capacity: usize) -> Option<Mask> {
    let mut pivots: Vec<Option<(Mask, bool)>> = vec![None; width];
    for (row, &value) in rows.iter().zip(rhs) {
        let mut row = row.clone();
        let mut value = value;
        loop {
            let Some(j) = row.ones().next() else {
                if value {
                    return None;
                }
                break;
            };
            if let Some((pivot, bit)) = &pivots[j] {
                row.xor(pivot);
                value ^= bit;
            } else {
                pivots[j] = Some((row, value));
                break;
            }
        }
    }
    let mut answer = Mask::zero(capacity);
    for (j, pivot) in pivots.iter().enumerate().rev() {
        let (row, value) = pivot.as_ref().expect("support has full column rank");
        if row.dot(&answer) ^ value {
            answer.toggle(j);
        }
    }
    Some(answer)
}

pub(crate) fn rank(rows: &[Mask], width: usize) -> usize {
    let mut pivots: Vec<Option<Mask>> = vec![None; width];
    let mut rank = 0;
    for row in rows {
        let mut row = row.clone();
        loop {
            let Some(j) = row.ones().next() else { break };
            if let Some(pivot) = &pivots[j] {
                row.xor(pivot);
            } else {
                pivots[j] = Some(row);
                rank += 1;
                break;
            }
        }
    }
    rank
}
