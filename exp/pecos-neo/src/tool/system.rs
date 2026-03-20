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

//! System scheduling for the Tool architecture.
//!
//! Systems are functions that operate on resources during tool execution.

use super::Stage;
use super::resource::Resources;

/// A system that can be executed during a stage.
///
/// Systems are the primary way to implement tool behavior. They receive
/// mutable access to resources and can read/write any resource they need.
///
/// # Example
///
/// ```no_run
/// use pecos_neo::tool::Resources;
///
/// fn my_system(resources: &mut Resources) {
///     // Access resources and do work
///     // let config = resources.get::<MyConfig>();
///     // let mut results = resources.get_mut::<MyResults>();
/// }
/// ```
pub trait System: Send + Sync {
    /// Execute this system with access to resources.
    fn run(&self, resources: &mut Resources);

    /// Optional: name for debugging/logging.
    fn name(&self) -> &str {
        std::any::type_name::<Self>()
    }
}

/// Wrapper to convert a function into a System.
pub struct FnSystem<F>
where
    F: Fn(&mut Resources) + Send + Sync,
{
    func: F,
    name: &'static str,
}

impl<F> FnSystem<F>
where
    F: Fn(&mut Resources) + Send + Sync,
{
    /// Create a new function-based system.
    pub fn new(func: F, name: &'static str) -> Self {
        Self { func, name }
    }
}

impl<F> System for FnSystem<F>
where
    F: Fn(&mut Resources) + Send + Sync,
{
    fn run(&self, resources: &mut Resources) {
        (self.func)(resources);
    }

    fn name(&self) -> &str {
        self.name
    }
}

/// Convert a function into a boxed System.
///
/// This is the primary way to create systems from closures or functions.
pub fn into_system<F>(func: F) -> Box<dyn System>
where
    F: Fn(&mut Resources) + Send + Sync + 'static,
{
    Box::new(FnSystem::new(func, std::any::type_name::<F>()))
}

/// Schedule of systems organized by execution stage.
#[derive(Default)]
pub struct Schedule {
    startup: Vec<Box<dyn System>>,
    pre_shot: Vec<Box<dyn System>>,
    execute: Vec<Box<dyn System>>,
    post_shot: Vec<Box<dyn System>>,
    finish: Vec<Box<dyn System>>,
}

impl Schedule {
    /// Create a new empty schedule.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a system to a stage.
    pub fn add_system(&mut self, stage: Stage, system: Box<dyn System>) {
        match stage {
            Stage::Startup => self.startup.push(system),
            Stage::PreShot => self.pre_shot.push(system),
            Stage::Execute => self.execute.push(system),
            Stage::PostShot => self.post_shot.push(system),
            Stage::Finish => self.finish.push(system),
        }
    }

    /// Get systems for a stage.
    #[must_use]
    pub fn systems(&self, stage: Stage) -> &[Box<dyn System>] {
        match stage {
            Stage::Startup => &self.startup,
            Stage::PreShot => &self.pre_shot,
            Stage::Execute => &self.execute,
            Stage::PostShot => &self.post_shot,
            Stage::Finish => &self.finish,
        }
    }

    /// Run all systems in a stage.
    pub fn run_stage(&self, stage: Stage, resources: &mut Resources) {
        for system in self.systems(stage) {
            system.run(resources);
        }
    }

    /// Run a complete shot loop: Startup, then per-shot PreShot/Execute/PostShot, then Finish.
    pub fn run_shots(&self, resources: &mut Resources, shots: usize) {
        self.run_stage(Stage::Startup, resources);

        for _ in 0..shots {
            self.run_stage(Stage::PreShot, resources);
            self.run_stage(Stage::Execute, resources);
            self.run_stage(Stage::PostShot, resources);
        }

        self.run_stage(Stage::Finish, resources);
    }
}

/// Trait for types that can be converted into a System.
///
/// This allows both `Box<dyn System>` and function pointers/closures
/// to be used with `add_system`.
pub trait IntoSystem {
    /// Convert into a boxed system.
    fn into_system(self) -> Box<dyn System>;
}

impl IntoSystem for Box<dyn System> {
    fn into_system(self) -> Box<dyn System> {
        self
    }
}

impl<F> IntoSystem for F
where
    F: Fn(&mut Resources) + Send + Sync + 'static,
{
    fn into_system(self) -> Box<dyn System> {
        into_system(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fn_system() {
        let mut resources = Resources::new();
        resources.insert(0u32);

        let system = into_system(|res: &mut Resources| {
            *res.get_mut::<u32>() += 1;
        });

        system.run(&mut resources);
        assert_eq!(*resources.get::<u32>(), 1);

        system.run(&mut resources);
        assert_eq!(*resources.get::<u32>(), 2);
    }

    #[test]
    fn test_schedule() {
        let mut schedule = Schedule::new();
        let mut resources = Resources::new();
        resources.insert(Vec::<&str>::new());

        schedule.add_system(
            Stage::Startup,
            into_system(|res: &mut Resources| {
                res.get_mut::<Vec<&str>>().push("startup");
            }),
        );

        schedule.add_system(
            Stage::Execute,
            into_system(|res: &mut Resources| {
                res.get_mut::<Vec<&str>>().push("execute");
            }),
        );

        schedule.add_system(
            Stage::Finish,
            into_system(|res: &mut Resources| {
                res.get_mut::<Vec<&str>>().push("finish");
            }),
        );

        schedule.run_stage(Stage::Startup, &mut resources);
        schedule.run_stage(Stage::Execute, &mut resources);
        schedule.run_stage(Stage::Finish, &mut resources);

        assert_eq!(
            *resources.get::<Vec<&str>>(),
            vec!["startup", "execute", "finish"]
        );
    }
}
