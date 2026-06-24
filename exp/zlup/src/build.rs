//! Build system for Zlup projects.
//!
//! This module implements the build.zlp execution system, following Zig's philosophy:
//! **the build system IS the language**.
//!
//! ## Overview
//!
//! Like Zig's `build.zig`, Zlup uses `build.zlp` - a Zlup program that runs at
//! compile time to configure the build. No separate DSL, no YAML, no TOML for
//! build logic - just Zlup with comptime.
//!
//! ## Usage
//!
//! ```bash
//! # Build using build.zlp
//! zlup build
//!
//! # Build with options
//! zlup build -Dnoise=true -Doptimize=release
//!
//! # Run tests
//! zlup build test
//! ```
//!
//! ## Status
//!
//! This is Phase 1 of the build system implementation, providing the foundational
//! infrastructure. Full comptime execution of build.zlp requires additional
//! interpreter capabilities.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

// =============================================================================
// Build Configuration
// =============================================================================

/// Target operating system.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Os {
    Linux,
    MacOS,
    Windows,
    FreeBSD,
    Native,
}

impl Default for Os {
    fn default() -> Self {
        #[cfg(target_os = "linux")]
        return Os::Linux;
        #[cfg(target_os = "macos")]
        return Os::MacOS;
        #[cfg(target_os = "windows")]
        return Os::Windows;
        #[cfg(target_os = "freebsd")]
        return Os::FreeBSD;
        #[cfg(not(any(
            target_os = "linux",
            target_os = "macos",
            target_os = "windows",
            target_os = "freebsd"
        )))]
        return Os::Native;
    }
}

/// Target CPU architecture.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Arch {
    X86_64,
    Aarch64,
    Arm,
    Wasm32,
    Native,
}

impl Default for Arch {
    fn default() -> Self {
        #[cfg(target_arch = "x86_64")]
        return Arch::X86_64;
        #[cfg(target_arch = "aarch64")]
        return Arch::Aarch64;
        #[cfg(target_arch = "arm")]
        return Arch::Arm;
        #[cfg(target_arch = "wasm32")]
        return Arch::Wasm32;
        #[cfg(not(any(
            target_arch = "x86_64",
            target_arch = "aarch64",
            target_arch = "arm",
            target_arch = "wasm32"
        )))]
        return Arch::Native;
    }
}

/// Build target specification.
#[derive(Debug, Clone, Default)]
pub struct Target {
    pub os: Os,
    pub arch: Arch,
}

impl Target {
    /// Create a native target (current platform).
    pub fn native() -> Self {
        Self::default()
    }

    /// Create a specific target.
    pub fn new(os: Os, arch: Arch) -> Self {
        Self { os, arch }
    }
}

/// Optimization level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Optimize {
    /// No optimization, fastest compilation
    #[default]
    Debug,
    /// Optimize for speed
    ReleaseFast,
    /// Optimize for size
    ReleaseSmall,
    /// Optimize for safety (bounds checks, etc.)
    ReleaseSafe,
}

// =============================================================================
// Build Artifacts
// =============================================================================

/// Unique identifier for a build step.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StepId(usize);

/// A build step that can be executed.
#[derive(Debug, Clone)]
pub struct Step {
    pub id: StepId,
    pub name: String,
    pub description: String,
    pub dependencies: Vec<StepId>,
}

/// Options for creating an executable.
#[derive(Debug, Clone, Default)]
pub struct ExecutableOptions {
    pub name: String,
    pub root_source: PathBuf,
    pub target: Option<Target>,
    pub optimize: Option<Optimize>,
    /// NASA Power of 10 strict mode
    pub strict: bool,
}

/// Options for creating a library.
#[derive(Debug, Clone, Default)]
pub struct LibraryOptions {
    pub name: String,
    pub root_source: PathBuf,
    pub target: Option<Target>,
    pub optimize: Option<Optimize>,
    pub strict: bool,
}

/// Options for creating a test.
#[derive(Debug, Clone, Default)]
pub struct TestOptions {
    pub root_source: PathBuf,
    pub strict: bool,
}

/// An executable artifact.
#[derive(Debug, Clone)]
pub struct Executable {
    pub name: String,
    pub root_source: PathBuf,
    pub target: Target,
    pub optimize: Optimize,
    pub strict: bool,
    pub defines: BTreeMap<String, String>,
    pub libraries: Vec<String>,
    pub library_paths: Vec<PathBuf>,
    pub step: StepId,
}

/// A library artifact.
#[derive(Debug, Clone)]
pub struct Library {
    pub name: String,
    pub root_source: PathBuf,
    pub target: Target,
    pub optimize: Optimize,
    pub strict: bool,
    pub step: StepId,
}

/// A test artifact.
#[derive(Debug, Clone)]
pub struct Test {
    pub root_source: PathBuf,
    pub strict: bool,
    pub step: StepId,
}

// =============================================================================
// Build Context
// =============================================================================

/// The main build context.
///
/// This is passed to the `build` function in `build.zlp` and provides
/// the API for configuring the build.
#[derive(Debug)]
pub struct Build {
    /// Project root directory
    project_root: PathBuf,
    /// Default target
    default_target: Target,
    /// Default optimization level
    default_optimize: Optimize,
    /// Named build steps
    steps: Vec<Step>,
    /// Executables to build
    executables: Vec<Executable>,
    /// Libraries to build
    libraries: Vec<Library>,
    /// Tests to run
    tests: Vec<Test>,
    /// User-defined options from command line
    options: BTreeMap<String, OptionValue>,
    /// Install directory
    install_prefix: PathBuf,
    /// Next step ID
    next_step_id: usize,
}

/// Value of a user-defined build option.
#[derive(Debug, Clone)]
pub enum OptionValue {
    Bool(bool),
    String(String),
    Int(i64),
}

impl Build {
    /// Create a new build context.
    pub fn new(project_root: impl Into<PathBuf>) -> Self {
        let project_root = project_root.into();
        let install_prefix = project_root.join("zig-out");

        Self {
            project_root,
            default_target: Target::native(),
            default_optimize: Optimize::Debug,
            steps: Vec::new(),
            executables: Vec::new(),
            libraries: Vec::new(),
            tests: Vec::new(),
            options: BTreeMap::new(),
            install_prefix,
            next_step_id: 0,
        }
    }

    /// Get the project root directory.
    pub fn project_root(&self) -> &Path {
        &self.project_root
    }

    /// Set a command-line option.
    pub fn set_option(&mut self, name: impl Into<String>, value: OptionValue) {
        self.options.insert(name.into(), value);
    }

    /// Get a boolean option.
    pub fn option_bool(&self, name: &str) -> Option<bool> {
        match self.options.get(name) {
            Some(OptionValue::Bool(v)) => Some(*v),
            _ => None,
        }
    }

    /// Get a string option.
    pub fn option_string(&self, name: &str) -> Option<&str> {
        match self.options.get(name) {
            Some(OptionValue::String(v)) => Some(v),
            _ => None,
        }
    }

    /// Get an integer option.
    pub fn option_int(&self, name: &str) -> Option<i64> {
        match self.options.get(name) {
            Some(OptionValue::Int(v)) => Some(*v),
            _ => None,
        }
    }

    /// Get the standard target options (from CLI or defaults).
    pub fn standard_target_options(&self) -> Target {
        self.default_target.clone()
    }

    /// Get the standard optimization option (from CLI or defaults).
    pub fn standard_optimize_option(&self) -> Optimize {
        self.default_optimize
    }

    /// Set the default target.
    pub fn set_default_target(&mut self, target: Target) {
        self.default_target = target;
    }

    /// Set the default optimization level.
    pub fn set_default_optimize(&mut self, optimize: Optimize) {
        self.default_optimize = optimize;
    }

    /// Allocate a new step ID.
    fn alloc_step_id(&mut self) -> StepId {
        let id = StepId(self.next_step_id);
        self.next_step_id += 1;
        id
    }

    /// Create a new named build step.
    pub fn step(&mut self, name: impl Into<String>, description: impl Into<String>) -> StepId {
        let id = self.alloc_step_id();
        self.steps.push(Step {
            id,
            name: name.into(),
            description: description.into(),
            dependencies: Vec::new(),
        });
        id
    }

    /// Add a dependency between steps.
    pub fn add_step_dependency(&mut self, step: StepId, depends_on: StepId) {
        if let Some(s) = self.steps.iter_mut().find(|s| s.id == step) {
            s.dependencies.push(depends_on);
        }
    }

    /// Add an executable to build.
    pub fn add_executable(&mut self, options: ExecutableOptions) -> &mut Executable {
        let step = self.alloc_step_id();
        let exe = Executable {
            name: options.name,
            root_source: self.project_root.join(&options.root_source),
            target: options.target.unwrap_or_else(|| self.default_target.clone()),
            optimize: options.optimize.unwrap_or(self.default_optimize),
            strict: options.strict,
            defines: BTreeMap::new(),
            libraries: Vec::new(),
            library_paths: Vec::new(),
            step,
        };
        self.executables.push(exe);
        self.executables.last_mut().unwrap()
    }

    /// Add a library to build.
    pub fn add_library(&mut self, options: LibraryOptions) -> &mut Library {
        let step = self.alloc_step_id();
        let lib = Library {
            name: options.name,
            root_source: self.project_root.join(&options.root_source),
            target: options.target.unwrap_or_else(|| self.default_target.clone()),
            optimize: options.optimize.unwrap_or(self.default_optimize),
            strict: options.strict,
            step,
        };
        self.libraries.push(lib);
        self.libraries.last_mut().unwrap()
    }

    /// Add a test to run.
    pub fn add_test(&mut self, options: TestOptions) -> &mut Test {
        let step = self.alloc_step_id();
        let test = Test {
            root_source: self.project_root.join(&options.root_source),
            strict: options.strict,
            step,
        };
        self.tests.push(test);
        self.tests.last_mut().unwrap()
    }

    /// Mark an artifact for installation.
    pub fn install_artifact(&mut self, _step: StepId) {
        // In the future, this will add the artifact to the install step
    }

    /// Get all executables.
    pub fn executables(&self) -> &[Executable] {
        &self.executables
    }

    /// Get all libraries.
    pub fn libraries(&self) -> &[Library] {
        &self.libraries
    }

    /// Get all tests.
    pub fn tests(&self) -> &[Test] {
        &self.tests
    }

    /// Get all named steps.
    pub fn steps(&self) -> &[Step] {
        &self.steps
    }

    /// Find a step by name.
    pub fn find_step(&self, name: &str) -> Option<&Step> {
        self.steps.iter().find(|s| s.name == name)
    }
}

impl Executable {
    /// Add a compile-time define.
    pub fn add_define(&mut self, name: impl Into<String>, value: impl Into<String>) {
        self.defines.insert(name.into(), value.into());
    }

    /// Link a library.
    pub fn link_library(&mut self, name: impl Into<String>) {
        self.libraries.push(name.into());
    }

    /// Add a library search path.
    pub fn add_library_path(&mut self, path: impl Into<PathBuf>) {
        self.library_paths.push(path.into());
    }
}

// =============================================================================
// Build Runner
// =============================================================================

/// Error that can occur during build.
#[derive(Debug, Clone)]
pub struct BuildError {
    pub message: String,
}

impl std::fmt::Display for BuildError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "build error: {}", self.message)
    }
}

impl std::error::Error for BuildError {}

/// Result type for build operations.
pub type BuildResult<T> = Result<T, BuildError>;

/// Build runner that executes build.zlp.
pub struct BuildRunner {
    build: Build,
}

impl BuildRunner {
    /// Create a new build runner for the given project.
    pub fn new(project_root: impl Into<PathBuf>) -> Self {
        Self {
            build: Build::new(project_root),
        }
    }

    /// Set a command-line option.
    pub fn set_option(&mut self, name: impl Into<String>, value: OptionValue) {
        self.build.set_option(name, value);
    }

    /// Parse command-line options in the format `-Dname=value`.
    pub fn parse_options(&mut self, args: &[String]) -> BuildResult<Vec<String>> {
        let mut remaining = Vec::new();

        for arg in args {
            if let Some(opt) = arg.strip_prefix("-D") {
                if let Some((name, value)) = opt.split_once('=') {
                    // Try to parse as different types
                    if value == "true" {
                        self.build.set_option(name, OptionValue::Bool(true));
                    } else if value == "false" {
                        self.build.set_option(name, OptionValue::Bool(false));
                    } else if let Ok(n) = value.parse::<i64>() {
                        self.build.set_option(name, OptionValue::Int(n));
                    } else {
                        self.build.set_option(name, OptionValue::String(value.to_string()));
                    }
                } else {
                    // -Dflag without value means true
                    self.build.set_option(opt, OptionValue::Bool(true));
                }
            } else if arg == "-Doptimize=release" || arg == "--release" {
                self.build.set_default_optimize(Optimize::ReleaseFast);
            } else if arg == "-Doptimize=debug" || arg == "--debug" {
                self.build.set_default_optimize(Optimize::Debug);
            } else {
                remaining.push(arg.clone());
            }
        }

        Ok(remaining)
    }

    /// Load and parse the build.zlp file.
    pub fn load_build_file(&mut self) -> BuildResult<()> {
        let build_file = self.build.project_root.join("build.zlp");

        if !build_file.exists() {
            return Err(BuildError {
                message: format!("build.zlp not found in {}", self.build.project_root.display()),
            });
        }

        let source = std::fs::read_to_string(&build_file).map_err(|e| BuildError {
            message: format!("failed to read build.zlp: {}", e),
        })?;

        // Parse the build file
        let _ast = crate::parse_file(&source, build_file.to_string_lossy()).map_err(|e| {
            BuildError {
                message: format!("failed to parse build.zlp: {}", e),
            }
        })?;

        // TODO: Execute the build function using the comptime evaluator
        // For now, we just validate that build.zlp parses correctly

        Ok(())
    }

    /// Get the build context.
    pub fn build(&self) -> &Build {
        &self.build
    }

    /// Get mutable access to the build context.
    pub fn build_mut(&mut self) -> &mut Build {
        &mut self.build
    }

    /// Run the default build step.
    pub fn run_default(&self) -> BuildResult<()> {
        // Build all executables
        for exe in &self.build.executables {
            self.build_executable(exe)?;
        }
        Ok(())
    }

    /// Run a named build step.
    pub fn run_step(&self, name: &str) -> BuildResult<()> {
        match name {
            "test" => self.run_tests(),
            _ => {
                if self.build.find_step(name).is_some() {
                    // TODO: Execute custom step
                    Ok(())
                } else {
                    Err(BuildError {
                        message: format!("unknown build step: {}", name),
                    })
                }
            }
        }
    }

    /// Build an executable.
    fn build_executable(&self, exe: &Executable) -> BuildResult<()> {
        println!("Building executable: {}", exe.name);
        println!("  Source: {}", exe.root_source.display());
        println!("  Target: {:?} / {:?}", exe.target.os, exe.target.arch);
        println!("  Optimize: {:?}", exe.optimize);

        if exe.strict {
            println!("  Strict mode: enabled");
        }

        for (name, value) in &exe.defines {
            println!("  Define: {} = {}", name, value);
        }

        for lib in &exe.libraries {
            println!("  Link: {}", lib);
        }

        // TODO: Actually compile the executable
        // This would invoke the parser, semantic analyzer, and code generator

        Ok(())
    }

    /// Run all tests.
    fn run_tests(&self) -> BuildResult<()> {
        use crate::test_runner::{format_results, TestOutcome, TestRunConfig, TestRunner};

        for test in &self.build.tests {
            println!("Running test: {}", test.root_source.display());

            let source = std::fs::read_to_string(&test.root_source).map_err(|e| {
                BuildError { message: format!("failed to read {}: {}", test.root_source.display(), e) }
            })?;

            let program = crate::parser::parse(&source).map_err(|e| {
                BuildError { message: format!("parse error: {}", e) }
            })?;

            let config = TestRunConfig::default();
            let runner = TestRunner::new(config);
            let results = runner.run(&program);
            print!("{}", format_results(&results));

            let has_failures = results.iter().any(|r| matches!(r.outcome, TestOutcome::Fail(_)));
            if has_failures {
                return Err(BuildError { message: "some tests failed".to_string() });
            }
        }
        Ok(())
    }

    /// Print available build steps.
    pub fn print_help(&self) {
        println!("Build steps:");
        println!("  (default)    Build all targets");
        println!("  test         Run all tests");

        for step in &self.build.steps {
            println!("  {:12} {}", step.name, step.description);
        }

        println!();
        println!("Options:");
        println!("  -Dname=value   Set a build option");
        println!("  --release      Build in release mode");
        println!("  --debug        Build in debug mode (default)");
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_target_native() {
        let target = Target::native();
        // Should not panic
        assert!(matches!(
            target.os,
            Os::Linux | Os::MacOS | Os::Windows | Os::FreeBSD | Os::Native
        ));
    }

    #[test]
    fn test_build_context() {
        let mut build = Build::new("/tmp/test-project");

        build.set_option("noise", OptionValue::Bool(true));
        build.set_option("distance", OptionValue::Int(5));

        assert_eq!(build.option_bool("noise"), Some(true));
        assert_eq!(build.option_int("distance"), Some(5));
        assert_eq!(build.option_bool("unknown"), None);
    }

    #[test]
    fn test_add_executable() {
        let mut build = Build::new("/tmp/test-project");

        let exe = build.add_executable(ExecutableOptions {
            name: "test-exe".to_string(),
            root_source: PathBuf::from("src/main.zlp"),
            ..Default::default()
        });

        assert_eq!(exe.name, "test-exe");
        assert!(exe.root_source.ends_with("src/main.zlp"));
    }

    #[test]
    fn test_add_step() {
        let mut build = Build::new("/tmp/test-project");

        let step1 = build.step("compile", "Compile source files");
        let step2 = build.step("link", "Link object files");

        build.add_step_dependency(step2, step1);

        assert_eq!(build.steps().len(), 2);
        assert_eq!(build.find_step("compile").unwrap().name, "compile");
    }

    #[test]
    fn test_parse_options() {
        let mut runner = BuildRunner::new("/tmp/test");

        let args = vec![
            "-Dnoise=true".to_string(),
            "-Ddistance=5".to_string(),
            "-Dname=test".to_string(),
            "-Dflag".to_string(),
            "--release".to_string(),
            "build".to_string(),
        ];

        let remaining = runner.parse_options(&args).unwrap();

        assert_eq!(runner.build().option_bool("noise"), Some(true));
        assert_eq!(runner.build().option_int("distance"), Some(5));
        assert_eq!(runner.build().option_string("name"), Some("test"));
        assert_eq!(runner.build().option_bool("flag"), Some(true));
        assert_eq!(runner.build().default_optimize, Optimize::ReleaseFast);
        assert_eq!(remaining, vec!["build"]);
    }
}
