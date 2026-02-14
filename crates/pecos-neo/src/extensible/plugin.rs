//! Plugin system for modular gate registration.
//!
//! Inspired by Bevy's plugin architecture, this allows composable gate sets:
//! - `GatePlugin` trait for registering gates
//! - `CoreGatesPlugin` with base gates every simulator should support
//! - `StandardDecompositionsPlugin` with common derived gates
//! - Dependency resolution between plugins

use super::{DecompOp, Decomposition, DecompositionRegistry, GateSupportSet, gates};
use std::any::TypeId;
use std::collections::HashSet;

/// Trait for plugins that register gates with the decomposition system.
///
/// Plugins provide a modular way to define gate sets and their decompositions.
/// They can declare dependencies on other plugins to ensure proper initialization order.
///
/// # Example
///
/// ```
/// use pecos_neo::extensible::{GatePlugin, DecompositionRegistry};
///
/// struct MyCustomGates;
///
/// impl GatePlugin for MyCustomGates {
///     fn name(&self) -> &'static str {
///         "my-custom-gates"
///     }
///
///     fn build(&self, registry: &mut DecompositionRegistry) {
///         // Register custom gates here
///     }
/// }
/// ```
pub trait GatePlugin: Send + Sync + 'static {
    /// Human-readable name for this plugin.
    fn name(&self) -> &'static str;

    /// Register gates with the decomposition registry.
    fn build(&self, registry: &mut DecompositionRegistry);

    /// Declare dependencies on other plugins.
    ///
    /// Returns type IDs of plugins that must be loaded before this one.
    /// Default implementation returns no dependencies.
    fn dependencies(&self) -> Vec<TypeId> {
        vec![]
    }
}

/// Core gates plugin - the base set every simulator should support.
///
/// This registers all fundamental gates as native (no decomposition needed):
/// - Identity: I
/// - Paulis: X, Y, Z
/// - Cliffords: H, SX, SY, SZ and their daggers
/// - T gates: T, Tdg
/// - Rotations: RX, RY, RZ
/// - Two-qubit: CX, CY, CZ
pub struct CoreGatesPlugin;

impl GatePlugin for CoreGatesPlugin {
    fn name(&self) -> &'static str {
        "core-gates"
    }

    fn build(&self, registry: &mut DecompositionRegistry) {
        // All core gates are native - they don't decompose further
        // The DecompositionRegistry::new() already registers these,
        // but we make it explicit here for documentation purposes.

        // Single-qubit gates
        for &gate in &[
            gates::I,
            gates::X,
            gates::Y,
            gates::Z,
            gates::H,
            gates::SX,
            gates::SXdg,
            gates::SY,
            gates::SYdg,
            gates::SZ,
            gates::SZdg,
            gates::T,
            gates::Tdg,
            gates::RX,
            gates::RY,
            gates::RZ,
        ] {
            registry.register_native(gate);
        }

        // Two-qubit gates
        for &gate in &[gates::CX, gates::CY, gates::CZ] {
            registry.register_native(gate);
        }
    }
}

/// Standard decompositions plugin - common derived gates.
///
/// This registers gates that have standard decompositions into core gates:
/// - SWAP: 3 CX gates
/// - iSWAP: S, H, CX combination
///
/// Depends on `CoreGatesPlugin`.
pub struct StandardDecompositionsPlugin;

impl GatePlugin for StandardDecompositionsPlugin {
    fn name(&self) -> &'static str {
        "standard-decompositions"
    }

    fn dependencies(&self) -> Vec<TypeId> {
        vec![TypeId::of::<CoreGatesPlugin>()]
    }

    fn build(&self, registry: &mut DecompositionRegistry) {
        // SWAP = CX(0,1); CX(1,0); CX(0,1)
        registry.register(
            gates::SWAP,
            GateSupportSet::from_iter([gates::CX]),
            Decomposition::SwapViaCx,
        );

        // iSWAP decomposition
        registry.register(
            gates::ISWAP,
            GateSupportSet::from_iter([gates::SZ, gates::H, gates::CX]),
            Decomposition::ISwapDecomp,
        );
    }
}

/// Extended decompositions for less common gates.
///
/// This can be extended to include:
/// - Toffoli (CCX)
/// - Fredkin (CSWAP)
/// - Multi-controlled gates
pub struct ExtendedDecompositionsPlugin;

impl GatePlugin for ExtendedDecompositionsPlugin {
    fn name(&self) -> &'static str {
        "extended-decompositions"
    }

    fn dependencies(&self) -> Vec<TypeId> {
        vec![
            TypeId::of::<CoreGatesPlugin>(),
            TypeId::of::<StandardDecompositionsPlugin>(),
        ]
    }

    fn build(&self, registry: &mut DecompositionRegistry) {
        // Toffoli (CCX) decomposition - standard 15-gate decomposition
        // CCX = H(2); CX(1,2); Tdg(2); CX(0,2); T(2); CX(1,2); Tdg(2); CX(0,2);
        //       T(1); T(2); H(2); CX(0,1); T(0); Tdg(1); CX(0,1)
        let ccx_ops = vec![
            DecompOp::gate1(gates::H, 2),
            DecompOp::gate2(gates::CX, 1, 2),
            DecompOp::gate1(gates::Tdg, 2),
            DecompOp::gate2(gates::CX, 0, 2),
            DecompOp::gate1(gates::T, 2),
            DecompOp::gate2(gates::CX, 1, 2),
            DecompOp::gate1(gates::Tdg, 2),
            DecompOp::gate2(gates::CX, 0, 2),
            DecompOp::gate1(gates::T, 1),
            DecompOp::gate1(gates::T, 2),
            DecompOp::gate1(gates::H, 2),
            DecompOp::gate2(gates::CX, 0, 1),
            DecompOp::gate1(gates::T, 0),
            DecompOp::gate1(gates::Tdg, 1),
            DecompOp::gate2(gates::CX, 0, 1),
        ];

        registry.register_dynamic(
            gates::CCX,
            GateSupportSet::from_iter([gates::H, gates::T, gates::Tdg, gates::CX]),
            ccx_ops,
        );
    }
}

/// A plugin with its type ID for dependency tracking.
struct PluginEntry {
    plugin: Box<dyn GatePlugin>,
    type_id: TypeId,
}

/// Builder for loading plugins with dependency resolution.
///
/// Ensures plugins are loaded in the correct order based on their dependencies.
pub struct PluginLoader {
    plugins: Vec<PluginEntry>,
    loaded: HashSet<TypeId>,
}

impl Default for PluginLoader {
    fn default() -> Self {
        Self::new()
    }
}

impl PluginLoader {
    /// Create a new plugin loader.
    #[must_use]
    pub fn new() -> Self {
        Self {
            plugins: Vec::new(),
            loaded: HashSet::new(),
        }
    }

    /// Add a plugin to be loaded.
    ///
    /// Plugins are loaded in dependency order when `build` is called.
    #[must_use]
    pub fn with_plugin<P: GatePlugin>(mut self, plugin: P) -> Self {
        self.plugins.push(PluginEntry {
            type_id: TypeId::of::<P>(),
            plugin: Box::new(plugin),
        });
        self
    }

    /// Build the decomposition registry by loading all plugins.
    ///
    /// Plugins are loaded in topological order based on their dependencies.
    ///
    /// # Errors
    ///
    /// Returns an error if there are missing dependencies or circular dependencies.
    pub fn build(mut self) -> Result<DecompositionRegistry, PluginError> {
        let mut registry = DecompositionRegistry::new();

        // Simple topological sort based on dependencies
        let mut remaining: Vec<_> = self.plugins.drain(..).collect();
        let mut made_progress = true;

        while !remaining.is_empty() && made_progress {
            made_progress = false;

            let mut still_remaining = Vec::new();

            for entry in remaining {
                let deps = entry.plugin.dependencies();
                let deps_satisfied = deps.iter().all(|dep| self.loaded.contains(dep));

                if deps_satisfied {
                    entry.plugin.build(&mut registry);
                    self.loaded.insert(entry.type_id);
                    made_progress = true;
                } else {
                    still_remaining.push(entry);
                }
            }

            remaining = still_remaining;
        }

        if !remaining.is_empty() {
            let names: Vec<_> = remaining.iter().map(|e| e.plugin.name()).collect();
            return Err(PluginError::UnresolvedDependencies(names));
        }

        Ok(registry)
    }

    /// Build with the standard plugin set.
    ///
    /// Includes: `CoreGatesPlugin`, `StandardDecompositionsPlugin`
    #[must_use]
    pub fn with_standard_plugins() -> Self {
        Self::new()
            .with_plugin(CoreGatesPlugin)
            .with_plugin(StandardDecompositionsPlugin)
    }

    /// Build with all available plugins.
    ///
    /// Includes: `CoreGatesPlugin`, `StandardDecompositionsPlugin`, `ExtendedDecompositionsPlugin`
    #[must_use]
    pub fn with_all_plugins() -> Self {
        Self::new()
            .with_plugin(CoreGatesPlugin)
            .with_plugin(StandardDecompositionsPlugin)
            .with_plugin(ExtendedDecompositionsPlugin)
    }
}

/// Error during plugin loading.
#[derive(Clone, Debug)]
pub enum PluginError {
    /// Some plugins have unresolved dependencies.
    UnresolvedDependencies(Vec<&'static str>),
}

impl std::fmt::Display for PluginError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnresolvedDependencies(names) => {
                write!(
                    f,
                    "Plugins with unresolved dependencies: {}",
                    names.join(", ")
                )
            }
        }
    }
}

impl std::error::Error for PluginError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_core_gates_plugin() {
        let mut registry = DecompositionRegistry::new();
        CoreGatesPlugin.build(&mut registry);

        // All core gates should be native
        assert!(registry.is_native(gates::H));
        assert!(registry.is_native(gates::X));
        assert!(registry.is_native(gates::CX));
    }

    #[test]
    fn test_standard_decompositions_plugin() {
        let mut registry = DecompositionRegistry::new();
        CoreGatesPlugin.build(&mut registry);
        StandardDecompositionsPlugin.build(&mut registry);

        // SWAP should have decomposition
        assert!(registry.contains(gates::SWAP));
        assert!(!registry.is_native(gates::SWAP));

        // Can execute SWAP with CX support
        let sim_support = GateSupportSet::from_iter([gates::CX]);
        assert!(registry.can_execute(gates::SWAP, &sim_support));
    }

    #[test]
    fn test_extended_decompositions_plugin() {
        let mut registry = DecompositionRegistry::new();
        CoreGatesPlugin.build(&mut registry);
        StandardDecompositionsPlugin.build(&mut registry);
        ExtendedDecompositionsPlugin.build(&mut registry);

        // CCX should have decomposition
        assert!(registry.contains(gates::CCX));
        assert!(!registry.is_native(gates::CCX));

        // Can execute CCX with required gates
        let sim_support = GateSupportSet::from_iter([gates::H, gates::T, gates::Tdg, gates::CX]);
        assert!(registry.can_execute(gates::CCX, &sim_support));
    }

    #[test]
    fn test_plugin_loader_standard() {
        let registry = PluginLoader::with_standard_plugins().build().unwrap();

        assert!(registry.is_native(gates::H));
        assert!(registry.contains(gates::SWAP));
    }

    #[test]
    fn test_plugin_loader_all() {
        let registry = PluginLoader::with_all_plugins().build().unwrap();

        assert!(registry.is_native(gates::H));
        assert!(registry.contains(gates::SWAP));
        assert!(registry.contains(gates::CCX));
    }

    #[test]
    fn test_plugin_loader_dependency_order() {
        // Add plugins in wrong order - should still work due to dependency resolution
        let registry = PluginLoader::new()
            .with_plugin(StandardDecompositionsPlugin)
            .with_plugin(CoreGatesPlugin)
            .build()
            .unwrap();

        assert!(registry.is_native(gates::H));
        assert!(registry.contains(gates::SWAP));
    }

    #[test]
    fn test_plugin_loader_missing_dependency() {
        // Try to load StandardDecompositionsPlugin without CoreGatesPlugin
        let result = PluginLoader::new()
            .with_plugin(StandardDecompositionsPlugin)
            .build();

        assert!(matches!(
            result,
            Err(PluginError::UnresolvedDependencies(_))
        ));
    }

    #[test]
    fn test_custom_plugin() {
        struct MyPlugin;

        impl GatePlugin for MyPlugin {
            fn name(&self) -> &'static str {
                "my-plugin"
            }

            fn dependencies(&self) -> Vec<TypeId> {
                vec![TypeId::of::<CoreGatesPlugin>()]
            }

            fn build(&self, registry: &mut DecompositionRegistry) {
                // Register a custom decomposition for RZ using RX and H
                // RZ(θ) = H RX(θ) H
                let rz_via_rx = vec![
                    DecompOp::gate1(gates::H, 0),
                    DecompOp::rotation(gates::RX, 0, 0), // angle from input
                    DecompOp::gate1(gates::H, 0),
                ];

                // Note: This would override the native RZ, which is just for testing
                registry.register_dynamic(
                    gates::RZ,
                    GateSupportSet::from_iter([gates::H, gates::RX]),
                    rz_via_rx,
                );
            }
        }

        let registry = PluginLoader::new()
            .with_plugin(CoreGatesPlugin)
            .with_plugin(MyPlugin)
            .build()
            .unwrap();

        // RZ should now have a decomposition (overriding native)
        assert!(!registry.is_native(gates::RZ));
    }

    #[test]
    fn test_plugin_names() {
        assert_eq!(CoreGatesPlugin.name(), "core-gates");
        assert_eq!(
            StandardDecompositionsPlugin.name(),
            "standard-decompositions"
        );
        assert_eq!(
            ExtendedDecompositionsPlugin.name(),
            "extended-decompositions"
        );
    }
}
