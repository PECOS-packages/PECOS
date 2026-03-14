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

//! Core Tool type - the Bevy-inspired foundation.

use super::Stage;
use super::plugin::{Plugin, PluginGroup};
use super::resource::{Resource, Resources};
use super::system::{IntoSystem, Schedule};

/// The core Tool type - a Bevy-inspired application container.
///
/// `Tool` is the foundation for building quantum simulation and validation tools.
/// It manages:
/// - **Resources**: Typed singleton data (circuits, configs, results)
/// - **Systems**: Functions that operate on resources during execution
/// - **Plugins**: Bundles of resources and systems for reusable functionality
///
/// # Example
///
/// ```
/// use pecos_neo::tool::{Tool, Stage, Resources};
///
/// let mut tool = Tool::new()
///     .insert_resource(42u32)
///     .add_system(Stage::Startup, |res: &mut Resources| {
///         let value = res.get::<u32>();
///         println!("Starting with value: {}", value);
///     });
///
/// tool.run();
/// ```
pub struct Tool {
    resources: Resources,
    schedule: Schedule,
    has_run_startup: bool,
}

impl Default for Tool {
    fn default() -> Self {
        Self::new()
    }
}

impl Tool {
    /// Create a new empty Tool.
    #[must_use]
    pub fn new() -> Self {
        Self {
            resources: Resources::new(),
            schedule: Schedule::new(),
            has_run_startup: false,
        }
    }

    // ========================================================================
    // Plugin Management
    // ========================================================================

    /// Add a plugin to the tool.
    ///
    /// Plugins configure the tool with resources and systems.
    #[must_use]
    pub fn add_plugin<P: Plugin + 'static>(mut self, plugin: P) -> Self {
        plugin.build(&mut self);
        self
    }

    /// Add a plugin to the tool (mutable version).
    pub fn add_plugin_mut<P: Plugin + 'static>(&mut self, plugin: P) {
        plugin.build(self);
    }

    /// Add a group of plugins to the tool.
    #[must_use]
    pub fn add_plugins<G: PluginGroup>(mut self, group: G) -> Self {
        group.build(&mut self);
        self
    }

    /// Add a group of plugins to the tool (mutable version).
    pub fn add_plugins_mut<G: PluginGroup>(&mut self, group: G) {
        group.build(self);
    }

    // ========================================================================
    // Resource Management
    // ========================================================================

    /// Insert a resource into the tool.
    ///
    /// If a resource of the same type already exists, it will be replaced.
    #[must_use]
    pub fn insert_resource<R: Resource>(mut self, resource: R) -> Self {
        self.resources.insert(resource);
        self
    }

    /// Insert a resource into the tool (mutable version).
    pub fn insert_resource_mut<R: Resource>(&mut self, resource: R) {
        self.resources.insert(resource);
    }

    /// Check if a resource exists.
    #[must_use]
    pub fn contains_resource<R: Resource>(&self) -> bool {
        self.resources.contains::<R>()
    }

    /// Get a reference to a resource.
    ///
    /// # Panics
    ///
    /// Panics if the resource does not exist.
    #[must_use]
    pub fn resource<R: Resource>(&self) -> &R {
        self.resources.get::<R>()
    }

    /// Get a mutable reference to a resource.
    ///
    /// # Panics
    ///
    /// Panics if the resource does not exist.
    #[must_use]
    pub fn resource_mut<R: Resource>(&mut self) -> &mut R {
        self.resources.get_mut::<R>()
    }

    /// Try to get a reference to a resource.
    #[must_use]
    pub fn try_resource<R: Resource>(&self) -> Option<&R> {
        self.resources.try_get::<R>()
    }

    /// Try to get a mutable reference to a resource.
    #[must_use]
    pub fn try_resource_mut<R: Resource>(&mut self) -> Option<&mut R> {
        self.resources.try_get_mut::<R>()
    }

    /// Remove and return a resource.
    ///
    /// # Panics
    ///
    /// Panics if the resource does not exist.
    pub fn take_resource<R: Resource>(&mut self) -> R {
        self.resources.remove::<R>()
    }

    /// Try to remove and return a resource.
    pub fn try_take_resource<R: Resource>(&mut self) -> Option<R> {
        self.resources.try_remove::<R>()
    }

    // ========================================================================
    // System Management
    // ========================================================================

    /// Add a system to a specific stage.
    ///
    /// Systems are executed in the order they are added within each stage.
    #[must_use]
    pub fn add_system<S: IntoSystem + 'static>(mut self, stage: Stage, system: S) -> Self {
        self.schedule.add_system(stage, system.into_system());
        self
    }

    /// Add a system to a specific stage (mutable version).
    pub fn add_system_mut<S: IntoSystem + 'static>(&mut self, stage: Stage, system: S) {
        self.schedule.add_system(stage, system.into_system());
    }

    // ========================================================================
    // Execution
    // ========================================================================

    /// Run the tool.
    ///
    /// This executes all stages in order:
    /// 1. `Startup` (only on first run)
    /// 2. `PreShot`
    /// 3. `Execute`
    /// 4. `PostShot`
    /// 5. `Finish`
    ///
    /// For multi-shot simulation, this runs a single "iteration".
    /// Use `run_shots()` for multiple iterations.
    pub fn run(&mut self) {
        // Startup runs only once
        if !self.has_run_startup {
            self.schedule.run_stage(Stage::Startup, &mut self.resources);
            self.has_run_startup = true;
        }

        // Main execution stages
        self.schedule.run_stage(Stage::PreShot, &mut self.resources);
        self.schedule.run_stage(Stage::Execute, &mut self.resources);
        self.schedule
            .run_stage(Stage::PostShot, &mut self.resources);
        self.schedule.run_stage(Stage::Finish, &mut self.resources);
    }

    /// Run multiple shots.
    ///
    /// This executes:
    /// 1. `Startup` (once)
    /// 2. For each shot:
    ///    - `PreShot`
    ///    - `Execute`
    ///    - `PostShot`
    /// 3. `Finish` (once)
    pub fn run_shots(&mut self, shots: usize) {
        // Startup runs only once
        if !self.has_run_startup {
            self.schedule.run_stage(Stage::Startup, &mut self.resources);
            self.has_run_startup = true;
        }

        // Shot loop
        for _ in 0..shots {
            self.schedule.run_stage(Stage::PreShot, &mut self.resources);
            self.schedule.run_stage(Stage::Execute, &mut self.resources);
            self.schedule
                .run_stage(Stage::PostShot, &mut self.resources);
        }

        // Finish runs once at end
        self.schedule.run_stage(Stage::Finish, &mut self.resources);
    }

    /// Run the complete shot loop using the schedule directly on provided resources.
    ///
    /// Unlike `run_shots()`, this always runs Startup (no `has_run_startup` guard)
    /// and operates on the provided resources rather than the Tool's own resources.
    /// Used by parallel workers that each have their own Resources.
    pub fn run_shots_on(&self, resources: &mut Resources, shots: usize) {
        self.schedule.run_shots(resources, shots);
    }

    /// Reset the tool for another run.
    ///
    /// This clears the startup flag, allowing `Startup` systems to run again.
    pub fn reset(&mut self) {
        self.has_run_startup = false;
    }

    /// Get direct access to resources (for advanced use).
    #[must_use]
    pub fn resources(&self) -> &Resources {
        &self.resources
    }

    /// Get mutable access to resources (for advanced use).
    #[must_use]
    pub fn resources_mut(&mut self) -> &mut Resources {
        &mut self.resources
    }

    /// Get a reference to the schedule (for advanced use).
    #[must_use]
    pub fn schedule(&self) -> &Schedule {
        &self.schedule
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tool_basic() {
        let mut tool =
            Tool::new()
                .insert_resource(0u32)
                .add_system(Stage::Execute, |res: &mut Resources| {
                    *res.get_mut::<u32>() += 1;
                });

        tool.run();
        assert_eq!(*tool.resource::<u32>(), 1);

        tool.run();
        assert_eq!(*tool.resource::<u32>(), 2);
    }

    #[test]
    fn test_tool_startup_runs_once() {
        let mut tool = Tool::new()
            .insert_resource(Vec::<&str>::new())
            .add_system(Stage::Startup, |res: &mut Resources| {
                res.get_mut::<Vec<&str>>().push("startup");
            })
            .add_system(Stage::Execute, |res: &mut Resources| {
                res.get_mut::<Vec<&str>>().push("execute");
            });

        tool.run();
        tool.run();

        // Startup should only appear once
        assert_eq!(
            *tool.resource::<Vec<&str>>(),
            vec!["startup", "execute", "execute"]
        );
    }

    #[test]
    fn test_tool_run_shots() {
        let mut tool =
            Tool::new()
                .insert_resource(0u32)
                .add_system(Stage::PreShot, |res: &mut Resources| {
                    *res.get_mut::<u32>() += 1;
                });

        tool.run_shots(5);
        assert_eq!(*tool.resource::<u32>(), 5);
    }

    #[test]
    fn test_tool_stage_order() {
        let mut tool = Tool::new()
            .insert_resource(Vec::<&str>::new())
            .add_system(Stage::Startup, |res: &mut Resources| {
                res.get_mut::<Vec<&str>>().push("startup");
            })
            .add_system(Stage::PreShot, |res: &mut Resources| {
                res.get_mut::<Vec<&str>>().push("pre_shot");
            })
            .add_system(Stage::Execute, |res: &mut Resources| {
                res.get_mut::<Vec<&str>>().push("execute");
            })
            .add_system(Stage::PostShot, |res: &mut Resources| {
                res.get_mut::<Vec<&str>>().push("post_shot");
            })
            .add_system(Stage::Finish, |res: &mut Resources| {
                res.get_mut::<Vec<&str>>().push("finish");
            });

        tool.run();

        assert_eq!(
            *tool.resource::<Vec<&str>>(),
            vec!["startup", "pre_shot", "execute", "post_shot", "finish"]
        );
    }

    struct TestPlugin {
        value: u32,
    }

    impl Plugin for TestPlugin {
        fn build(&self, tool: &mut Tool) {
            tool.insert_resource_mut(self.value);
        }
    }

    #[test]
    fn test_tool_plugin() {
        let tool = Tool::new().add_plugin(TestPlugin { value: 42 });

        assert_eq!(*tool.resource::<u32>(), 42);
    }
}
