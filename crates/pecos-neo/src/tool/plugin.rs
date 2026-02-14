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

//! Plugin system for the Tool architecture.
//!
//! Plugins bundle related resources, systems, and configuration into reusable units.

use super::Tool;

/// A plugin that can configure a [`Tool`].
///
/// Plugins are the primary way to add functionality to a Tool. They can:
/// - Insert resources
/// - Add systems to execution stages
/// - Add other plugins
///
/// # Example
///
/// ```no_run
/// use pecos_neo::tool::{Plugin, Tool, Stage, Resources};
///
/// struct MyPlugin;
///
/// impl Plugin for MyPlugin {
///     fn build(&self, tool: &mut Tool) {
///         tool.insert_resource_mut(42u32);
///         tool.add_system_mut(Stage::Execute, |res: &mut Resources| {
///             *res.get_mut::<u32>() += 1;
///         });
///     }
/// }
/// ```
pub trait Plugin: Send + Sync {
    /// Configure the tool with this plugin's resources and systems.
    fn build(&self, tool: &mut Tool);

    /// Optional: name for debugging/logging.
    fn name(&self) -> &str {
        std::any::type_name::<Self>()
    }
}

/// A group of plugins that can be added together.
///
/// This is useful for bundling related plugins or providing "default" plugin sets.
///
/// # Example
///
/// ```no_run
/// use pecos_neo::tool::{PluginGroup, Tool};
///
/// struct DefaultPlugins;
///
/// impl PluginGroup for DefaultPlugins {
///     fn build(self, tool: &mut Tool) {
///         // Add multiple plugins as a group
///     }
/// }
/// ```
pub trait PluginGroup {
    /// Add all plugins in this group to the tool.
    fn build(self, tool: &mut Tool);
}

/// A tuple of plugins can be added as a group.
macro_rules! impl_plugin_group_tuple {
    ($($name:ident),+) => {
        impl<$($name: PluginGroup),+> PluginGroup for ($($name,)+) {
            #[allow(non_snake_case)]
            fn build(self, tool: &mut Tool) {
                let ($($name,)+) = self;
                $($name.build(tool);)+
            }
        }
    };
}

impl_plugin_group_tuple!(P1, P2);
impl_plugin_group_tuple!(P1, P2, P3);
impl_plugin_group_tuple!(P1, P2, P3, P4);
impl_plugin_group_tuple!(P1, P2, P3, P4, P5);
impl_plugin_group_tuple!(P1, P2, P3, P4, P5, P6);
