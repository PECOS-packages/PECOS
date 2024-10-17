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
use core::ops::BitAnd;
use pecos_core::{IndexableElement, Set};
use std::collections::HashSet;

#[derive(Clone, Debug)]
pub struct Paulis<T, E>
where
    T: for<'a> Set<'a, Element = E>,
    E: IndexableElement,
{
    pub(crate) xs: T,
    pub(crate) zs: T,
    pub(crate) sign: bool,
    _marker: PhantomData<E>,
}

impl<T, E> Paulis<T, E>
where
    T: for<'a> Set<'a, Element = E>,
    E: IndexableElement,
{
    #[must_use]
    #[inline]
    pub fn new() -> Paulis<T, E> {
        Self {
            xs: T::new(), // TODO: Add .with_capacity to sets.
            zs: T::new(),
            sign: false,
            _marker: PhantomData,
        }
    }

    /// # Errors
    /// If the indices of xs or zs is greater than the number of qubits, an error will be generated.
    #[expect(clippy::similar_names)]
    #[inline]
    pub fn to_string_repr(&self, size: usize) -> Result<String, String> {
        let mut result = String::new();

        // Add the sign to the result
        if self.sign {
            result.push('-');
        } else {
            result.push('+');
        }

        // Check if any element in xs or zs is out of bounds
        for &element in self.xs.iter().chain(self.zs.iter()) {
            let index = element.to_usize();
            if index >= size {
                return Err(format!("Index {index} is out of bounds for size {size}"));
            }
        }

        // Iterate over the range and determine the character for each index
        for i in 0..size {
            let index = E::from_usize(i);

            let in_xs = self.xs.contains(&index);
            let in_zs = self.zs.contains(&index);

            let char = match (in_xs, in_zs) {
                (false, false) => 'I',
                (true, false) => 'X',
                (false, true) => 'Z',
                (true, true) => 'Y',
            };
            result.push(char);
        }

        Ok(result)
    }

    #[inline]
    pub fn insert_xz(&mut self, xs: Vec<E>, zs: Vec<E>) {
        // Insert values into xs and zs sets
        for x in xs {
            self.xs.insert(x);
        }
        for z in zs {
            self.zs.insert(z);
        }
    }

    /// # Errors
    /// If Paulis overlap, errors will be generated
    #[inline]
    pub fn insert_xyz(&mut self, xs: Vec<E>, ys: Vec<E>, zs: Vec<E>) -> Result<(), String> {
        // Convert vectors to sets for easy intersection check
        let xs_set: HashSet<_> = xs.iter().collect();
        let ys_set: HashSet<_> = ys.iter().collect();
        let zs_set: HashSet<_> = zs.iter().collect();

        // Check for common indices
        if !xs_set.is_disjoint(&ys_set)
            || !xs_set.is_disjoint(&zs_set)
            || !ys_set.is_disjoint(&zs_set)
        {
            return Err("xs, ys, and zs should not share common indices.".to_owned());
        }

        // Insert values into respective sets
        for x in xs {
            self.xs.insert(x);
        }
        for y in ys {
            self.xs.insert(y);
            self.zs.insert(y);
        }
        for z in zs {
            self.zs.insert(z);
        }

        Ok(())
    }
}

impl<T, E> Default for Paulis<T, E>
where
    T: for<'a> Set<'a, Element = E>,
    E: IndexableElement,
    for<'a> &'a T: BitAnd<&'a T, Output = T>,
{
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pecos_core::VecSet;

    #[test]
    fn test_to_string_repr() {
        let mut paulis = Paulis::<VecSet<u32>, u32>::new();
        paulis.xs.insert(1);
        paulis.zs.insert(2);
        paulis.xs.insert(3);
        paulis.zs.insert(3);
        paulis.sign = true;

        let result = paulis.to_string_repr(5).unwrap();
        assert_eq!(result, "-IXZYI");
    }

    #[test]
    fn test_to_string_repr_out_of_bounds() {
        let mut paulis = Paulis::<VecSet<u32>, u32>::new();
        paulis.xs.insert(6); // Out of bounds for size 5

        let result = paulis.to_string_repr(5);
        result.unwrap_err();
    }

    #[test]
    fn test_insert_xz() {
        let mut paulis = Paulis::<VecSet<u32>, u32>::new();
        let xs = vec![1, 3];
        let zs = vec![3, 4];

        paulis.insert_xz(xs, zs);

        let result = paulis.to_string_repr(5).unwrap();
        assert_eq!(result, "+IXIYZ");
    }

    #[test]
    fn test_insert_xyz() {
        let mut paulis = Paulis::<VecSet<u32>, u32>::new();
        let xs = vec![1, 3];
        let ys = vec![5];
        let zs = vec![2, 4];

        paulis.insert_xyz(xs, ys, zs).unwrap();

        let result = paulis.to_string_repr(6).unwrap();
        assert_eq!(result, "+IXZXZY");
    }

    #[test]
    fn test_insert_xyz_with_conflict() {
        let mut paulis = Paulis::<VecSet<u32>, u32>::new();
        let xs = vec![1, 3];
        let ys = vec![3, 5];
        let zs = vec![2, 4];

        let result = paulis.insert_xyz(xs, ys, zs);
        assert!(result.is_err());
    }
}
