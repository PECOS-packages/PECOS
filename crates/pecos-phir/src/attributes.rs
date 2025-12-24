/*!
Attributes and metadata system for PHIR

This module provides utilities for attaching interface implementations to operations
and regions through semantic metadata that optimization passes can understand and work with.

Key principle: Keep the core IR simple, but allow rich metadata annotation
for optimization passes and analysis tools.

## Interface Philosophy

PHIR embraces an abstract approach to quantum error correction and other
rapidly evolving quantum computing paradigms. Instead of hard-coding specific
QEC schemes or protocols into the IR, we use attributes to indicate which
interfaces an operation implements, with semantic metadata that can be
interpreted by appropriate passes.

This approach allows:
- Multiple QEC paradigms (surface codes, LDPC, color codes) to coexist
- New schemes to be added without changing the core IR
- Researchers to prototype new ideas with custom attributes
- Progressive optimization from generic to specialized passes

## Flexible Attribute System

The attribute system is intentionally flexible - you can add any string key
with any supported value type. This allows for:

1. **Domain-specific attributes**: Add attributes specific to your use case
2. **Evolving standards**: New attribute conventions can emerge organically
3. **Research flexibility**: Prototype new ideas without core IR changes
4. **Progressive enhancement**: Start simple, add metadata as needed

Example:
```rust
use pecos_phir::attributes::AttributeBuilder;
use pecos_phir::phir::AttributeValue;
use std::collections::BTreeMap;

// Start simple
let attrs = AttributeBuilder::new()
    .with_tag("my_algorithm")
    .build();

// Add domain-specific attributes as needed
let mut schedule_params = BTreeMap::new();
schedule_params.insert("rounds".to_string(), AttributeValue::Int(3));
schedule_params.insert("type".to_string(), AttributeValue::String("xy".to_string()));

let attrs = AttributeBuilder::new()
    .with_tag("syndrome_extraction")
    .with_string("qec.code_type", "surface_code")  // Add as you develop
    .with_int("qec.distance", 7)
    .with_dict("qec.schedule", schedule_params)
    .build();
```

## Attribute Naming Conventions

While the system is flexible, we recommend these conventions:

- Use dots for namespacing: `qec.distance`, `protocol.type`
- Use underscores within names: `syndrome_type`, `error_rate`
- Start general, get specific: `qec` → `qec.code_type` → `qec.surface_code.distance`
- Document your attributes for others to understand and reuse

The core IR doesn't need to understand these attributes - specialized
passes interpret them to apply appropriate optimizations.
*/

use std::collections::BTreeMap;

/// Common attribute keys used throughout PHIR
///
/// Note: This module provides commonly-used attribute keys, but the attribute
/// system is designed to be extensible. You can use any string as an attribute
/// key - these are just conventions for common patterns.
pub mod keys {
    /// Region/operation semantic tags
    pub const SEMANTIC_TAG: &str = "semantic_tag";
    pub const ALGORITHM: &str = "algorithm";
    pub const PATTERN: &str = "pattern";

    /// Interface specifications
    pub const INPUT_INTERFACE: &str = "input_interface";
    pub const OUTPUT_INTERFACE: &str = "output_interface";
    pub const INVARIANTS: &str = "invariants";

    /// Performance hints
    pub const PARALLELIZABLE: &str = "parallelizable";
    pub const ESTIMATED_COST: &str = "estimated_cost";
    pub const RESOURCE_REQUIREMENTS: &str = "resource_requirements";

    /// Verification
    pub const VERIFIED: &str = "verified";
    pub const VERIFICATION_METHOD: &str = "verification_method";
}

/// Common semantic tags for regions and operations
///
/// These are example tags - you can use any string as a semantic tag.
/// The tag system is designed to be extensible for domain-specific needs.
pub mod tags {
    // Algorithm patterns
    pub const QFT: &str = "qft";
    pub const GROVER_ORACLE: &str = "grover_oracle";
    pub const GROVER_DIFFUSION: &str = "grover_diffusion";
    pub const PHASE_ESTIMATION: &str = "phase_estimation";
    pub const AMPLITUDE_AMPLIFICATION: &str = "amplitude_amplification";

    // Circuit patterns
    pub const STATE_PREPARATION: &str = "state_preparation";
    pub const UNCOMPUTE: &str = "uncompute";
    pub const CONTROLLED_UNITARY: &str = "controlled_unitary";
    pub const SWAP_NETWORK: &str = "swap_network";

    // Resource management
    pub const RESOURCE_ALLOCATION: &str = "resource_allocation";
    pub const RESOURCE_CLEANUP: &str = "resource_cleanup";
}

/// Builder for creating attribute sets with common patterns
pub struct AttributeBuilder {
    attrs: BTreeMap<String, crate::phir::AttributeValue>,
}

impl AttributeBuilder {
    #[must_use]
    pub fn new() -> Self {
        Self {
            attrs: BTreeMap::new(),
        }
    }

    /// Tag this region/operation with a semantic meaning
    #[must_use]
    pub fn with_tag(mut self, tag: &str) -> Self {
        self.attrs.insert(
            keys::SEMANTIC_TAG.to_string(),
            crate::phir::AttributeValue::String(tag.to_string()),
        );
        self
    }

    /// Specify the algorithm this implements
    #[must_use]
    pub fn with_algorithm(mut self, algorithm: &str) -> Self {
        self.attrs.insert(
            keys::ALGORITHM.to_string(),
            crate::phir::AttributeValue::String(algorithm.to_string()),
        );
        self
    }

    /// Add interface specification
    #[must_use]
    pub fn with_interface(mut self, inputs: Vec<String>, outputs: Vec<String>) -> Self {
        self.attrs.insert(
            keys::INPUT_INTERFACE.to_string(),
            crate::phir::AttributeValue::Array(
                inputs
                    .into_iter()
                    .map(crate::phir::AttributeValue::String)
                    .collect(),
            ),
        );
        self.attrs.insert(
            keys::OUTPUT_INTERFACE.to_string(),
            crate::phir::AttributeValue::Array(
                outputs
                    .into_iter()
                    .map(crate::phir::AttributeValue::String)
                    .collect(),
            ),
        );
        self
    }

    /// Mark as parallelizable
    #[must_use]
    pub fn parallelizable(mut self) -> Self {
        self.attrs.insert(
            keys::PARALLELIZABLE.to_string(),
            crate::phir::AttributeValue::Bool(true),
        );
        self
    }

    /// Add custom attribute (flexible key-value pair)
    #[must_use]
    pub fn with_attr(mut self, key: &str, value: crate::phir::AttributeValue) -> Self {
        self.attrs.insert(key.to_string(), value);
        self
    }

    /// Add a string attribute
    #[must_use]
    pub fn with_string(mut self, key: &str, value: &str) -> Self {
        self.attrs.insert(
            key.to_string(),
            crate::phir::AttributeValue::String(value.to_string()),
        );
        self
    }

    /// Add an integer attribute
    #[must_use]
    pub fn with_int(mut self, key: &str, value: i64) -> Self {
        self.attrs
            .insert(key.to_string(), crate::phir::AttributeValue::Int(value));
        self
    }

    /// Add a boolean attribute
    #[must_use]
    pub fn with_bool(mut self, key: &str, value: bool) -> Self {
        self.attrs
            .insert(key.to_string(), crate::phir::AttributeValue::Bool(value));
        self
    }

    /// Add a float attribute
    #[must_use]
    pub fn with_float(mut self, key: &str, value: f64) -> Self {
        self.attrs
            .insert(key.to_string(), crate::phir::AttributeValue::Float(value));
        self
    }

    /// Add an array attribute
    #[must_use]
    pub fn with_array(mut self, key: &str, values: Vec<crate::phir::AttributeValue>) -> Self {
        self.attrs
            .insert(key.to_string(), crate::phir::AttributeValue::Array(values));
        self
    }

    /// Add a nested dictionary attribute
    #[must_use]
    pub fn with_dict(
        mut self,
        key: &str,
        dict: BTreeMap<String, crate::phir::AttributeValue>,
    ) -> Self {
        self.attrs
            .insert(key.to_string(), crate::phir::AttributeValue::Dict(dict));
        self
    }

    /// Build the attribute map
    #[must_use]
    pub fn build(self) -> BTreeMap<String, crate::phir::AttributeValue> {
        self.attrs
    }
}

/// Helper functions for working with boxed regions
pub mod helpers {
    use super::keys;
    use std::collections::BTreeMap;

    /// Check if a region/operation has a specific semantic tag
    #[must_use]
    pub fn has_tag(attrs: &BTreeMap<String, crate::phir::AttributeValue>, tag: &str) -> bool {
        attrs
            .get(keys::SEMANTIC_TAG)
            .and_then(|v| match v {
                crate::phir::AttributeValue::String(s) => Some(s.as_str()),
                _ => None,
            })
            .is_some_and(|s| s == tag)
    }

    /// Get the algorithm name if specified
    #[must_use]
    pub fn get_algorithm(attrs: &BTreeMap<String, crate::phir::AttributeValue>) -> Option<String> {
        attrs.get(keys::ALGORITHM).and_then(|v| match v {
            crate::phir::AttributeValue::String(s) => Some(s.clone()),
            _ => None,
        })
    }

    /// Check if marked as parallelizable
    #[must_use]
    pub fn is_parallelizable(attrs: &BTreeMap<String, crate::phir::AttributeValue>) -> bool {
        attrs
            .get(keys::PARALLELIZABLE)
            .and_then(|v| match v {
                crate::phir::AttributeValue::Bool(b) => Some(*b),
                _ => None,
            })
            .unwrap_or(false)
    }

    /// Get any string attribute by key
    #[must_use]
    pub fn get_string_attr(
        attrs: &BTreeMap<String, crate::phir::AttributeValue>,
        key: &str,
    ) -> Option<String> {
        attrs.get(key).and_then(|v| match v {
            crate::phir::AttributeValue::String(s) => Some(s.clone()),
            _ => None,
        })
    }

    /// Get any integer attribute by key
    #[must_use]
    pub fn get_int_attr(
        attrs: &BTreeMap<String, crate::phir::AttributeValue>,
        key: &str,
    ) -> Option<i64> {
        attrs.get(key).and_then(|v| match v {
            crate::phir::AttributeValue::Int(i) => Some(*i),
            _ => None,
        })
    }

    /// Get any boolean attribute by key
    #[must_use]
    pub fn get_bool_attr(
        attrs: &BTreeMap<String, crate::phir::AttributeValue>,
        key: &str,
    ) -> Option<bool> {
        attrs.get(key).and_then(|v| match v {
            crate::phir::AttributeValue::Bool(b) => Some(*b),
            _ => None,
        })
    }

    /// Create attributes from a list of key-value pairs
    #[must_use]
    pub fn attrs_from_pairs(
        pairs: &[(&str, crate::phir::AttributeValue)],
    ) -> BTreeMap<String, crate::phir::AttributeValue> {
        pairs
            .iter()
            .map(|(k, v)| ((*k).to_string(), v.clone()))
            .collect()
    }
}

/// Example: Creating a "boxed" QFT region with metadata
#[must_use]
pub fn example_qft_box() -> BTreeMap<String, crate::phir::AttributeValue> {
    AttributeBuilder::new()
        .with_tag(tags::QFT)
        .with_algorithm("quantum_fourier_transform")
        .with_interface(vec!["qubits[n]".to_string()], vec!["qubits[n]".to_string()])
        .parallelizable()
        .with_attr("reversible", crate::phir::AttributeValue::Bool(true))
        .with_attr("circuit_depth", crate::phir::AttributeValue::Int(100)) // O(n²)
        .build()
}

/// Example: Creating a custom boxed operation with flexible attributes
#[must_use]
pub fn example_custom_box() -> BTreeMap<String, crate::phir::AttributeValue> {
    AttributeBuilder::new()
        .with_tag("custom_protocol")
        .with_interface(vec!["inputs".to_string()], vec!["outputs".to_string()])
        // Flexible attribute system - add any domain-specific attributes
        .with_string("protocol.name", "my_custom_protocol")
        .with_int("protocol.version", 2)
        .with_bool("protocol.verified", true)
        .with_float("protocol.fidelity", 0.999)
        // Nested attributes for complex metadata
        .with_dict("protocol.parameters", {
            let mut params = BTreeMap::new();
            params.insert("rounds".to_string(), crate::phir::AttributeValue::Int(10));
            params.insert(
                "threshold".to_string(),
                crate::phir::AttributeValue::Float(0.95),
            );
            params
        })
        .build()
}

impl Default for AttributeBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_attribute_builder() {
        let attrs = AttributeBuilder::new()
            .with_tag(tags::QFT)
            .parallelizable()
            .build();

        assert!(helpers::has_tag(&attrs, tags::QFT));
        assert!(helpers::is_parallelizable(&attrs));
    }

    #[test]
    fn test_qft_box() {
        let attrs = example_qft_box();
        assert!(helpers::has_tag(&attrs, tags::QFT));
        assert_eq!(
            helpers::get_algorithm(&attrs),
            Some("quantum_fourier_transform".to_string())
        );
    }

    #[test]
    fn test_flexible_attributes() {
        let attrs = AttributeBuilder::new()
            .with_string("custom.key", "custom_value")
            .with_int("custom.counter", 42)
            .with_bool("custom.enabled", true)
            .build();

        assert_eq!(
            helpers::get_string_attr(&attrs, "custom.key"),
            Some("custom_value".to_string())
        );
        assert_eq!(helpers::get_int_attr(&attrs, "custom.counter"), Some(42));
        assert_eq!(helpers::get_bool_attr(&attrs, "custom.enabled"), Some(true));
    }
}
