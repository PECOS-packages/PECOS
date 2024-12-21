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
use std::f64::consts::PI;

pub trait ArbitraryRotationGateable<T: IndexableElement>: CliffordGateable<T> {
    fn rx(&mut self, theta: f64, q: T) -> &mut Self;
    #[inline]
    fn ry(&mut self, theta: f64, q: T) -> &mut Self {
        self.sz(q).rx(theta, q).szdg(q)
    }
    fn rz(&mut self, theta: f64, q: T) -> &mut Self;

    fn u(&mut self, theta: f64, phi: f64, lambda: f64, q: T) -> &mut Self;

    #[inline]
    fn r1xy(&mut self, theta: f64, phi: f64, q: T) -> &mut Self {
        self.rz(-phi + PI, q).rx(theta, q).rz(phi - PI, q)
    }

    #[inline]
    fn rxx(&mut self, theta: f64, q1: T, q2: T) -> &mut Self {
        self.h(q1).h(q2).rzz(theta, q1, q2).h(q1).h(q2)
    }

    #[inline]
    fn ryy(&mut self, theta: f64, q1: T, q2: T) -> &mut Self {
        self.sx(q1).sx(q2).rzz(theta, q1, q2).sxdg(q1).sxdg(q2)
    }
    fn rzz(&mut self, theta: f64, q1: T, q2: T) -> &mut Self;
}
