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

//! Type-erased heterogeneous storage for signals in a command stream.
//!
//! Uses struct-of-arrays (struct-of-arrays) layout: positions are stored separately from
//! signal data. During dispatch, the hot path scans positions without pulling
//! signal payloads into cache -- more positions per cache line, data only
//! loaded on match.
//!
//! The type registry uses a flat `Vec` rather than a `HashMap` -- with
//! typically 1-3 signal types, linear scan on contiguous memory beats hashing.

use pecos_core::Signal;
use std::any::{Any, TypeId};
use std::fmt;

/// Trait object interface for a typed signal vector.
///
/// Enables heterogeneous storage: `Vec<(TypeId, Box<dyn SignalVec>)>`.
pub(crate) trait SignalVec: Send + Sync {
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
    fn len(&self) -> usize;
    fn clear(&mut self);
    fn clone_box(&self) -> Box<dyn SignalVec>;
    fn fmt_debug(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result;

    /// Get the positions slice directly -- no vtable call per element.
    fn positions(&self) -> &[u32];

    /// Get the signal data at `index` as a type-erased reference.
    fn entry_data(&self, index: usize) -> Option<&dyn Any>;
}

/// Concrete typed storage for signals of type `S`.
///
/// struct-of-arrays layout: positions and data in separate contiguous arrays.
/// Position scanning (the hot path during dispatch) touches only `u32`
/// values without pulling signal payloads into cache.
struct TypedSignalVec<S: Signal> {
    positions: Vec<u32>,
    data: Vec<S>,
}

impl<S: Signal> TypedSignalVec<S> {
    fn new() -> Self {
        Self {
            positions: Vec::new(),
            data: Vec::new(),
        }
    }

    fn push(&mut self, position: u32, signal: S) {
        self.positions.push(position);
        self.data.push(signal);
    }
}

impl<S: Signal> SignalVec for TypedSignalVec<S> {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn len(&self) -> usize {
        self.positions.len()
    }

    fn clear(&mut self) {
        self.positions.clear();
        self.data.clear();
    }

    fn clone_box(&self) -> Box<dyn SignalVec> {
        Box::new(TypedSignalVec {
            positions: self.positions.clone(),
            data: self.data.clone(),
        })
    }

    fn fmt_debug(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}[{}]", S::name(), self.positions.len(),)
    }

    fn positions(&self) -> &[u32] {
        &self.positions
    }

    fn entry_data(&self, index: usize) -> Option<&dyn Any> {
        self.data.get(index).map(|signal| signal as &dyn Any)
    }
}

/// Type-erased storage for heterogeneous signal types.
///
/// Each registered signal type gets its own struct-of-arrays channel (separate position
/// and data arrays). Empty when no signals are present (the common case).
///
/// Uses a flat `Vec` for the type registry since signal type counts are
/// typically very small (1-3). Linear scan on contiguous memory beats
/// hashing for these sizes.
pub struct SignalStore {
    channels: Vec<(TypeId, Box<dyn SignalVec>)>,
    total_count: usize,
}

impl SignalStore {
    /// Create an empty signal store.
    pub(crate) fn new() -> Self {
        Self {
            channels: Vec::new(),
            total_count: 0,
        }
    }

    /// Push a signal at the given command position.
    pub(crate) fn push<S: Signal>(&mut self, position: u32, signal: S) {
        let type_id = TypeId::of::<S>();

        // Linear scan for the channel -- fast for small N
        let vec = if let Some((_, vec)) = self.channels.iter_mut().find(|(id, _)| *id == type_id) {
            vec
        } else {
            self.channels
                .push((type_id, Box::new(TypedSignalVec::<S>::new())));
            &mut self.channels.last_mut().expect("just pushed an element").1
        };

        let typed = vec
            .as_any_mut()
            .downcast_mut::<TypedSignalVec<S>>()
            .expect("TypeId mismatch in SignalStore");

        typed.push(position, signal);
        self.total_count += 1;
    }

    /// Total number of signals across all types.
    #[must_use]
    pub fn total_count(&self) -> usize {
        self.total_count
    }

    /// Check if any signals are present.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.total_count == 0
    }

    /// Number of distinct signal types stored.
    #[must_use]
    pub fn type_count(&self) -> usize {
        self.channels.len()
    }

    /// Iterate over signals of a specific type, yielding `(position, &S)`.
    #[must_use]
    pub fn iter<S: Signal>(&self) -> SignalIter<'_, S> {
        let typed = self
            .channels
            .iter()
            .find(|(id, _)| *id == TypeId::of::<S>())
            .and_then(|(_, vec)| vec.as_any().downcast_ref::<TypedSignalVec<S>>());

        match typed {
            Some(t) => SignalIter {
                positions: &t.positions,
                data: &t.data,
                index: 0,
            },
            None => SignalIter {
                positions: &[],
                data: &[],
                index: 0,
            },
        }
    }

    /// Clear all signals from all types.
    pub(crate) fn clear(&mut self) {
        for (_, vec) in &mut self.channels {
            vec.clear();
        }
        self.total_count = 0;
    }

    /// Number of signal channels (used by runner to size cursor array).
    pub(crate) fn channel_count(&self) -> usize {
        self.channels.len()
    }

    /// Get a channel by index. Returns `(TypeId, &dyn SignalVec)`.
    ///
    /// Used by the runner's cursor-based dispatch to avoid repeated
    /// `TypeId` lookups -- each cursor stores a channel index resolved once.
    pub(crate) fn channel_at(&self, index: usize) -> Option<(TypeId, &dyn SignalVec)> {
        self.channels
            .get(index)
            .map(|(id, vec)| (*id, vec.as_ref()))
    }
}

impl Default for SignalStore {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for SignalStore {
    fn clone(&self) -> Self {
        Self {
            channels: self
                .channels
                .iter()
                .map(|(type_id, vec)| (*type_id, vec.clone_box()))
                .collect(),
            total_count: self.total_count,
        }
    }
}

impl fmt::Debug for SignalStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_empty() {
            return write!(f, "SignalStore(empty)");
        }
        write!(
            f,
            "SignalStore({} signals, {} types: [",
            self.total_count,
            self.channels.len()
        )?;
        for (i, (_, vec)) in self.channels.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            vec.fmt_debug(f)?;
        }
        write!(f, "])")
    }
}

/// Iterator over signals of a specific type.
pub struct SignalIter<'a, S> {
    positions: &'a [u32],
    data: &'a [S],
    index: usize,
}

impl<'a, S> Iterator for SignalIter<'a, S> {
    type Item = (u32, &'a S);

    fn next(&mut self) -> Option<Self::Item> {
        if self.index < self.positions.len() {
            let pos = self.positions[self.index];
            let signal = &self.data[self.index];
            self.index += 1;
            Some((pos, signal))
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.positions.len() - self.index;
        (remaining, Some(remaining))
    }
}

impl<S> ExactSizeIterator for SignalIter<'_, S> {}

#[cfg(test)]
mod tests {
    use super::*;
    use pecos_core::impl_signal;

    #[derive(Copy, Clone, Debug, PartialEq)]
    struct Temperature(pub f64);
    impl_signal!(Temperature);

    #[derive(Copy, Clone, Debug, PartialEq)]
    struct RoundBoundary(pub i64);
    impl_signal!(RoundBoundary);

    #[derive(Clone, Debug, PartialEq)]
    struct CalibrationData {
        pub rates: [f64; 4],
    }
    impl_signal!(CalibrationData);

    #[test]
    fn empty_store() {
        let store = SignalStore::new();
        assert!(store.is_empty());
        assert_eq!(store.total_count(), 0);
        assert_eq!(store.type_count(), 0);
        assert_eq!(store.iter::<Temperature>().len(), 0);
    }

    #[test]
    fn push_and_iterate() {
        let mut store = SignalStore::new();
        store.push(0, Temperature(300.0));
        store.push(5, Temperature(350.0));
        store.push(3, RoundBoundary(1));

        assert_eq!(store.total_count(), 3);
        assert_eq!(store.type_count(), 2);

        let temps: Vec<_> = store.iter::<Temperature>().collect();
        assert_eq!(
            temps,
            vec![(0, &Temperature(300.0)), (5, &Temperature(350.0))]
        );

        let rounds: Vec<_> = store.iter::<RoundBoundary>().collect();
        assert_eq!(rounds, vec![(3, &RoundBoundary(1))]);

        // Unregistered type returns empty
        assert_eq!(store.iter::<CalibrationData>().len(), 0);
    }

    #[test]
    fn clone_store() {
        let mut store = SignalStore::new();
        store.push(0, Temperature(300.0));
        store.push(1, RoundBoundary(42));

        let cloned = store.clone();
        assert_eq!(cloned.total_count(), 2);
        assert_eq!(cloned.type_count(), 2);

        let temps: Vec<_> = cloned.iter::<Temperature>().collect();
        assert_eq!(temps, vec![(0, &Temperature(300.0))]);
    }

    #[test]
    fn clear_store() {
        let mut store = SignalStore::new();
        store.push(0, Temperature(300.0));
        store.push(1, RoundBoundary(42));
        assert_eq!(store.total_count(), 2);

        store.clear();
        assert!(store.is_empty());
        assert_eq!(store.total_count(), 0);
        // Type channels still exist but are empty
        assert_eq!(store.type_count(), 2);
    }

    #[test]
    fn complex_signal() {
        let mut store = SignalStore::new();
        store.push(
            10,
            CalibrationData {
                rates: [0.01, 0.02, 0.03, 0.04],
            },
        );

        let entries: Vec<_> = store.iter::<CalibrationData>().collect();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].0, 10);
        for (actual, expected) in entries[0].1.rates.iter().zip(&[0.01, 0.02, 0.03, 0.04]) {
            assert!((actual - expected).abs() < f64::EPSILON);
        }
    }

    #[test]
    fn debug_format() {
        let store = SignalStore::new();
        assert_eq!(format!("{store:?}"), "SignalStore(empty)");

        let mut store = SignalStore::new();
        store.push(0, Temperature(300.0));
        let debug = format!("{store:?}");
        assert!(debug.contains("1 signals"));
        assert!(debug.contains("Temperature[1]"));
    }

    #[test]
    fn channel_at_access() {
        let mut store = SignalStore::new();
        store.push(0, Temperature(300.0));
        store.push(5, Temperature(350.0));
        store.push(3, RoundBoundary(1));

        assert_eq!(store.channel_count(), 2);

        // Access by index
        let (type_id, channel) = store.channel_at(0).unwrap();
        assert_eq!(type_id, TypeId::of::<Temperature>());
        assert_eq!(channel.len(), 2);
        assert_eq!(channel.positions(), &[0, 5]);
        assert!(
            channel
                .entry_data(0)
                .unwrap()
                .downcast_ref::<Temperature>()
                .is_some()
        );

        let (type_id, channel) = store.channel_at(1).unwrap();
        assert_eq!(type_id, TypeId::of::<RoundBoundary>());
        assert_eq!(channel.len(), 1);
        assert_eq!(channel.positions(), &[3]);

        assert!(store.channel_at(2).is_none());
    }

    #[test]
    fn soa_positions_separate_from_data() {
        // Verify struct-of-arrays layout: positions are contiguous u32s
        let mut store = SignalStore::new();
        store.push(0, Temperature(100.0));
        store.push(5, Temperature(200.0));
        store.push(10, Temperature(300.0));

        let (_, channel) = store.channel_at(0).unwrap();
        let positions = channel.positions();
        assert_eq!(positions, &[0, 5, 10]);

        // Data accessible separately
        let t0 = channel
            .entry_data(0)
            .unwrap()
            .downcast_ref::<Temperature>()
            .unwrap();
        assert!((t0.0 - 100.0).abs() < f64::EPSILON);
        let t2 = channel
            .entry_data(2)
            .unwrap()
            .downcast_ref::<Temperature>()
            .unwrap();
        assert!((t2.0 - 300.0).abs() < f64::EPSILON);
        assert!(channel.entry_data(3).is_none());
    }
}
