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

use core::fmt::Debug;
use core::marker::PhantomData;
use pecos_core::{IndexableElement, Set};

#[derive(Clone, Debug)]
pub struct Gens<T, E>
where
    T: for<'a> Set<'a, Element = E>,
    E: IndexableElement,
{
    num_qubits: usize,
    pub col_x: Vec<T>,
    pub col_z: Vec<T>,
    pub row_x: Vec<T>,
    pub row_z: Vec<T>,
    pub sign: T,
    pub signs_minus: T,
    pub signs_i: T,
    _marker: PhantomData<E>,
}

impl<T, E> Gens<T, E>
where
    T: for<'a> Set<'a, Element = E>,
    E: IndexableElement,
{
    #[must_use]
    #[inline]
    pub fn new(num_qubits: usize) -> Gens<T, E> {
        Self {
            num_qubits,
            col_x: vec![T::new(); num_qubits],
            col_z: vec![T::new(); num_qubits],
            row_x: vec![T::new(); num_qubits],
            row_z: vec![T::new(); num_qubits],
            sign: T::new(),
            signs_minus: T::new(),
            signs_i: T::new(),
            _marker: PhantomData,
        }
    }

    pub fn get_num_qubits(&self) -> usize {
        self.num_qubits
    }

    #[inline]
    pub fn clear(&mut self) {
        self.col_x.clear();
        self.col_z.clear();
        self.row_x.clear();
        self.row_z.clear();
        self.sign.clear();
        self.signs_minus.clear();
        self.signs_i.clear();
    }

    #[inline]
    pub fn init_all_z(&mut self) {
        self.clear();
        // TODO: Change these to not create a new Vec... instead populate them...
        self.col_x = vec![T::new(); self.get_num_qubits()];
        self.col_z = new_index_set::<T, E>(self.get_num_qubits());
        self.row_x = vec![T::new(); self.get_num_qubits()];
        self.row_z = new_index_set::<T, E>(self.get_num_qubits());
    }

    #[inline]
    pub fn init_all_x(&mut self) {
        // TODO: Change these to not create a new Vec... instead populate them...
        self.clear();
        self.col_x = new_index_set::<T, E>(self.get_num_qubits());
        self.col_z = vec![T::new(); self.get_num_qubits()];
        self.row_x = new_index_set::<T, E>(self.get_num_qubits());
        self.row_z = vec![T::new(); self.get_num_qubits()];
    }
}

#[inline]
fn new_index_set<T, E>(num_qubits: usize) -> Vec<T>
where
    T: for<'a> Set<'a, Element = E>,
    E: IndexableElement,
{
    let mut sets = Vec::with_capacity(num_qubits);
    for i in 0..num_qubits {
        let mut set = T::new();
        set.insert(E::from_usize(i));
        sets.push(set);
    }
    sets
}
