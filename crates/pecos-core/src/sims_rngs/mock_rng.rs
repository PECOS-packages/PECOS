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

use core::fmt;
use rand::RngCore;

pub struct MockRng {
    pub values: Vec<bool>,
    pub index: usize,
}

impl MockRng {
    #[inline]
    #[must_use]
    pub fn new(values: Vec<bool>) -> Self {
        Self { values, index: 0 }
    }

    // pub fn gen_bool(&mut self, _p: f64) -> bool {
    //     println!("MockRng state before generating value: index: {}, values: {:?}", self.index, self.values);
    //     let result = if self.index < self.values.len() {
    //         let value = self.values[self.index];
    //         println!("MockRng generated value: {}, at index: {}", value, self.index);
    //         self.index = (self.index + 1) % self.values.len(); // Ensure cyclic behavior
    //         println!("RRR {}", value);
    //         value
    //     } else {
    //         println!("MockRng index out of bounds, returning default value: false");
    //         false
    //     };
    //     println!(".,.,.,. {}", result);
    //     false
    // }
    #[inline]
    #[allow(unused)]
    pub fn gen_bool(&mut self, p: f64) -> bool {
        false
    }

    #[inline]
    pub fn give_false(&mut self) -> bool {
        false
    }
}

impl RngCore for MockRng {
    #[inline]
    fn next_u32(&mut self) -> u32 {
        u32::from(self.gen_bool(0.5))
    }

    #[inline]
    fn next_u64(&mut self) -> u64 {
        u64::from(self.gen_bool(0.5))
    }

    #[inline]
    #[allow(clippy::cast_possible_truncation, clippy::as_conversions)]
    fn fill_bytes(&mut self, dest: &mut [u8]) {
        for byte in dest.iter_mut() {
            *byte = self.next_u32() as u8;
        }
    }

    #[inline]
    fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), rand::Error> {
        self.fill_bytes(dest);
        Ok(())
    }
}

// Implement Debug for MockRng
impl fmt::Debug for MockRng {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        return f
            .debug_struct("MockRng")
            .field("values", &self.values)
            .field("index", &self.index)
            .finish();
    }
}
