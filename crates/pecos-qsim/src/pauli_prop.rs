// Copyright 2024 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
// in compliance with the License.You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under
// the License.
#![allow(unused_variables)]

use super::clifford_simulator::CliffordSimulator;
use crate::quantum_simulator::QuantumSimulator;
use core::marker::PhantomData;
use pecos_core::{IndexableElement, Set, VecSet};

// TODO: Allow for the use of sets of elements of types other than usize

#[expect(clippy::module_name_repetitions)]
pub type StdPauliProp = PauliProp<VecSet<usize>, usize>;

#[derive(Clone, Debug)]
pub struct PauliProp<T, E>
where
    T: for<'a> Set<'a, Element = E>,
    E: IndexableElement,
{
    num_qubits: usize,
    xs: T,
    zs: T,
    _marker: PhantomData<E>,
}

impl<T, E> QuantumSimulator for PauliProp<T, E>
where
    E: IndexableElement,
    T: for<'a> Set<'a, Element = E>,
{
    #[inline]
    fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    #[inline]
    fn reset(&mut self) -> &mut Self {
        todo!()
    }
}

impl<T, E> PauliProp<T, E>
where
    T: for<'a> Set<'a, Element = E>,
    E: IndexableElement,
{
    #[inline]
    pub fn insert_x(&mut self, item: E) {
        self.xs.insert(item);
    }

    #[inline]
    pub fn insert_z(&mut self, item: E) {
        self.zs.insert(item);
    }

    #[inline]
    pub fn insert_y(&mut self, item: E) {
        self.insert_x(item);
        self.insert_z(item);
    }

    #[inline]
    #[expect(clippy::similar_names)]
    pub fn h_v2(&mut self, q: E) {
        let in_xs = self.xs.contains(&q);
        let in_zs = self.zs.contains(&q);

        if in_xs && !in_zs {
            self.xs.remove(&q);
            self.zs.insert(q);
        } else if !in_xs && in_zs {
            self.zs.remove(&q);
            self.xs.insert(q);
        } else if in_xs && in_zs {
            return;
        }
    }
}

impl<T, E> CliffordSimulator<E> for PauliProp<T, E>
where
    T: for<'a> Set<'a, Element = E>,
    E: IndexableElement,
{
    #[inline]
    fn px(&mut self, q: E) -> (bool, bool) {
        todo!()
    }

    #[inline]
    fn pnx(&mut self, q: E) -> (bool, bool) {
        todo!()
    }

    #[inline]
    fn py(&mut self, q: E) -> (bool, bool) {
        todo!()
    }

    #[inline]
    fn pny(&mut self, q: E) -> (bool, bool) {
        todo!()
    }

    #[inline]
    fn pz(&mut self, q: E) -> (bool, bool) {
        self.xs.remove(&q);
        self.zs.remove(&q);
        (true, true) // TODO: fix...
    }

    #[inline]
    fn pnz(&mut self, q: E) -> (bool, bool) {
        todo!()
    }

    #[inline]
    fn mx(&mut self, q: E) -> (bool, bool) {
        todo!()
    }

    #[inline]
    fn mnx(&mut self, q: E) -> (bool, bool) {
        todo!()
    }

    #[inline]
    fn my(&mut self, q: E) -> (bool, bool) {
        todo!()
    }

    #[inline]
    fn mny(&mut self, q: E) -> (bool, bool) {
        todo!()
    }

    /// Output true if there is an X on the qubit.
    #[inline]
    fn mz(&mut self, q: E) -> (bool, bool) {
        (self.xs.contains(&q), true) // TODO: Is deterministic a useful value... maybe, maybe not... fix
    }

    #[inline]
    fn mnz(&mut self, q: E) -> (bool, bool) {
        todo!()
    }

    #[inline]
    fn identity(&mut self, _q: E) {}

    #[inline]
    fn x(&mut self, q: E) {
        todo!()
    }

    #[inline]
    fn y(&mut self, q: E) {
        todo!()
    }

    #[inline]
    fn z(&mut self, q: E) {
        todo!()
    }

    /// X -> Y, Z -> Z, Y -> -X
    #[inline]
    fn sz(&mut self, q: E) {
        if self.xs.contains(&q) {
            self.zs.symmetric_difference_item_update(&q);
        }
    }

    /// X -> Z, Z -> X, Y -> -Y
    #[inline]
    #[expect(clippy::similar_names)]
    fn h(&mut self, q: E) {
        let in_xs = self.xs.contains(&q);
        let in_zs = self.zs.contains(&q);

        if in_xs && in_zs {
        } else if in_xs {
            self.xs.remove(&q);
            self.zs.insert(q);
        } else if in_zs {
            self.zs.remove(&q);
            self.xs.insert(q);
        }
    }

    /// XI -> XX, ZI -> ZI, IX -> IX, IZ -> ZZ
    #[inline]
    fn cx(&mut self, q1: E, q2: E) {
        if self.xs.contains(&q1) {
            self.xs.symmetric_difference_item_update(&q2);
        }
        if self.zs.contains(&q2) {
            self.zs.symmetric_difference_item_update(&q1);
        }
    }
}
