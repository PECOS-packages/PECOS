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

pub trait MeasValue: From<u8> + Into<u8> {
    // Abstracting measurements a bit in case simulating multi-level measurements such as for leakage or qudits.
    fn from_u8(value: u8) -> Self
    where
        Self: Sized;
    fn to_u8(&self) -> u8;
}

#[expect(dead_code)]
pub struct Measurement<T: MeasValue> {
    is_deterministic: bool,
    value: T,
}

impl<T: MeasValue> Measurement<T> {
    #[expect(dead_code)]
    #[inline]
    fn new(is_deterministic: bool, value: T) -> Self {
        Self {
            is_deterministic,
            value,
        }
    }
}

#[expect(clippy::exhaustive_enums)] // A bit will only measure 0 or 1.
#[repr(u8)]
#[derive(Clone, Debug)]
pub enum MeasBitValue {
    Zero = 0,
    One = 1,
}

impl MeasBitValue {
    #[inline]
    #[expect(clippy::single_call_fn)]
    fn from_bool(value: bool) -> Self
    where
        Self: Sized,
    {
        if value {
            Self::One
        } else {
            Self::Zero
        }
    }

    #[inline]
    fn to_bool(&self) -> bool {
        self.to_u8() != 0
    }
}

impl From<bool> for MeasBitValue {
    #[inline]
    fn from(value: bool) -> Self {
        Self::from_bool(value)
    }
}

impl From<MeasBitValue> for bool {
    #[inline]
    fn from(val: MeasBitValue) -> Self {
        val.to_bool()
    }
}

impl From<u8> for MeasBitValue {
    #[inline]
    fn from(value: u8) -> Self {
        Self::from_u8(value)
    }
}

impl From<MeasBitValue> for u8 {
    #[inline]
    fn from(val: MeasBitValue) -> Self {
        val.to_u8()
    }
}

impl MeasValue for MeasBitValue {
    #[inline]
    fn from_u8(value: u8) -> Self
    where
        Self: Sized,
    {
        match value {
            0 => Self::Zero,
            1 => Self::One,
            _ => panic!("Invalid value for MeasBitValue: {value}"),
        }
    }

    #[inline]
    #[expect(clippy::as_conversions)]
    fn to_u8(&self) -> u8 {
        self.clone() as u8
    }
}
