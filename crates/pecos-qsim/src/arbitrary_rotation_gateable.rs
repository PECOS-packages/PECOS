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

use crate::CliffordGateable;
use pecos_core::IndexableElement;

pub trait ArbitraryRotationGateable<T: IndexableElement>: CliffordGateable<T> {
    fn rx(&mut self, theta: f64, q: T) -> &mut Self;
    fn ry(&mut self, theta: f64, q: T) -> &mut Self;
    fn rz(&mut self, theta: f64, q: T) -> &mut Self;

    fn rxx(&mut self, theta: f64, q1: T, q2: T) -> &mut Self;
    fn ryy(&mut self, theta: f64, q1: T, q2: T) -> &mut Self;
    fn rzz(&mut self, theta: f64, q1: T, q2: T) -> &mut Self;
}
