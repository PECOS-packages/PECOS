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

//! Resource storage for the Tool architecture.
//!
//! Resources are typed singleton data that systems can access during execution.

use std::any::{Any, TypeId};
use std::collections::HashMap;

/// Marker trait for types that can be stored as resources.
///
/// Resources are singleton data accessible by systems during tool execution.
/// They must be `Send + Sync` for thread-safe access.
pub trait Resource: Send + Sync + 'static {}

// Blanket implementation: any Send + Sync + 'static type is a Resource
impl<T: Send + Sync + 'static> Resource for T {}

/// Type-erased resource storage.
///
/// Stores resources by their `TypeId`, allowing retrieval with type safety.
#[derive(Default)]
pub struct Resources {
    storage: HashMap<TypeId, Box<dyn Any + Send + Sync>>,
}

impl Resources {
    /// Create a new empty resource storage.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert a resource, replacing any existing resource of the same type.
    pub fn insert<R: Resource>(&mut self, resource: R) {
        self.storage.insert(TypeId::of::<R>(), Box::new(resource));
    }

    /// Check if a resource of the given type exists.
    #[must_use]
    pub fn contains<R: Resource>(&self) -> bool {
        self.storage.contains_key(&TypeId::of::<R>())
    }

    /// Get a reference to a resource.
    ///
    /// # Panics
    ///
    /// Panics if the resource does not exist.
    #[must_use]
    pub fn get<R: Resource>(&self) -> &R {
        self.try_get::<R>()
            .unwrap_or_else(|| panic!("Resource {} not found", std::any::type_name::<R>()))
    }

    /// Get a mutable reference to a resource.
    ///
    /// # Panics
    ///
    /// Panics if the resource does not exist.
    #[must_use]
    pub fn get_mut<R: Resource>(&mut self) -> &mut R {
        self.try_get_mut::<R>()
            .unwrap_or_else(|| panic!("Resource {} not found", std::any::type_name::<R>()))
    }

    /// Try to get a reference to a resource.
    #[must_use]
    pub fn try_get<R: Resource>(&self) -> Option<&R> {
        self.storage
            .get(&TypeId::of::<R>())
            .and_then(|boxed| boxed.downcast_ref::<R>())
    }

    /// Try to get a mutable reference to a resource.
    #[must_use]
    pub fn try_get_mut<R: Resource>(&mut self) -> Option<&mut R> {
        self.storage
            .get_mut(&TypeId::of::<R>())
            .and_then(|boxed| boxed.downcast_mut::<R>())
    }

    /// Remove and return a resource.
    ///
    /// # Panics
    ///
    /// Panics if the resource does not exist.
    pub fn remove<R: Resource>(&mut self) -> R {
        self.try_remove::<R>()
            .unwrap_or_else(|| panic!("Resource {} not found", std::any::type_name::<R>()))
    }

    /// Try to remove and return a resource.
    pub fn try_remove<R: Resource>(&mut self) -> Option<R> {
        self.storage
            .remove(&TypeId::of::<R>())
            .and_then(|boxed| boxed.downcast::<R>().ok())
            .map(|boxed| *boxed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_insert_and_get() {
        let mut resources = Resources::new();
        resources.insert(42u32);
        resources.insert("hello".to_string());

        assert_eq!(*resources.get::<u32>(), 42);
        assert_eq!(resources.get::<String>(), "hello");
    }

    #[test]
    fn test_get_mut() {
        let mut resources = Resources::new();
        resources.insert(42u32);

        *resources.get_mut::<u32>() = 100;
        assert_eq!(*resources.get::<u32>(), 100);
    }

    #[test]
    fn test_contains() {
        let mut resources = Resources::new();
        assert!(!resources.contains::<u32>());

        resources.insert(42u32);
        assert!(resources.contains::<u32>());
    }

    #[test]
    fn test_remove() {
        let mut resources = Resources::new();
        resources.insert(42u32);

        let value = resources.remove::<u32>();
        assert_eq!(value, 42);
        assert!(!resources.contains::<u32>());
    }

    #[test]
    fn test_try_get_missing() {
        let resources = Resources::new();
        assert!(resources.try_get::<u32>().is_none());
    }

    #[test]
    #[should_panic(expected = "Resource u32 not found")]
    fn test_get_missing_panics() {
        let resources = Resources::new();
        let _ = resources.get::<u32>();
    }
}
