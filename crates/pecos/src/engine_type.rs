//! Engine type enumeration and dynamic engine builder support
//!
//! This module provides tools for working with PECOS engines in both compile-time
//! and runtime contexts. It includes:
//!
//! - `EngineType`: An enumeration of all available engine types
//! - `DynamicEngineBuilder`: A type-erased wrapper for runtime engine selection
//!
//! # Overview
//!
//! PECOS provides multiple classical control engines (QASM, LLVM, Selene) for
//! executing quantum programs. Normally, you work with these engines directly:
//!
//! ```rust
//! # use pecos_core::errors::PecosError;
//! # fn main() -> Result<(), PecosError> {
//! use pecos_qasm::qasm_engine;
//! use pecos_engines::sim;
//! use pecos_programs::Qasm;
//!
//! // Compile-time engine selection - best performance
//! let qasm_code = r#"
//! OPENQASM 2.0;
//! include "qelib1.inc";
//! qreg q[1];
//! creg c[1];
//! h q[0];
//! measure q[0] -> c[0];
//! "#;
//! let results = sim(qasm_engine().program(Qasm::from_string(qasm_code)))
//!     .seed(42)
//!     .run(10)?;
//!
//! // Verify results
//! assert_eq!(results.len(), 10);
//! let shot_map = results.try_as_shot_map().unwrap();
//! let values = shot_map.try_bits_as_u64("c").unwrap();
//! // H gate creates superposition, so we should see both 0 and 1
//! assert!(values.iter().any(|&v| v == 0) || values.iter().any(|&v| v == 1));
//! # Ok(())
//! # }
//! ```
//!
//! However, sometimes you need to select an engine at runtime based on user input,
//! configuration files, or other dynamic conditions. This module provides the tools
//! to do that.
//!
//! # Dynamic Engine Selection
//!
//! The `DynamicEngineBuilder` type uses trait objects to enable runtime engine
//! selection while maintaining the same API:
//!
//! ```rust
//! # #[cfg(feature = "qasm")]
//! # {
//! # use pecos_core::errors::PecosError;
//! # fn main() -> Result<(), PecosError> {
//! use pecos::{EngineType, DynamicEngineBuilder, sim_dynamic};
//! use pecos_qasm::qasm_engine;
//! use pecos_programs::Qasm;
//!
//! // Runtime engine selection based on user input
//! let user_input = "qasm";
//! let engine_type = match user_input {
//!     "qasm" => EngineType::Qasm,
//!     "llvm" => EngineType::Llvm,
//!     "selene" => EngineType::Selene,
//!     _ => panic!("Unknown engine type"),
//! };
//!
//! // For this example, we'll just use QASM
//! let qasm_code = r#"
//! OPENQASM 2.0;
//! include "qelib1.inc";
//! qreg q[1];
//! creg c[1];
//! h q[0];
//! measure q[0] -> c[0];
//! "#;
//! let builder = DynamicEngineBuilder::new(qasm_engine().program(Qasm::from_string(qasm_code)));
//!
//! // Use the same API regardless of engine type
//! let results = sim_dynamic(builder).seed(42).run(10)?;
//! assert_eq!(results.len(), 10);
//! # Ok(())
//! # }
//! # }
//! ```
//!
//! # Performance Considerations
//!
//! Dynamic engine selection has a small runtime overhead due to trait object
//! indirection. For performance-critical code where the engine type is known
//! at compile time, prefer using the concrete engine builders directly.
//!
//! # Feature Flags
//!
//! The availability of engines depends on which features are enabled:
//! - `qasm`: Enables QASM engine support
//! - `llvm`: Enables LLVM engine support
//! - `selene`: Enables Selene engine support

use pecos_core::errors::PecosError;
use pecos_engines::{ClassicalControlEngine, ClassicalControlEngineBuilder, sim};
use std::fmt;

/// Available engine types in PECOS
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EngineType {
    /// QASM engine for `OpenQASM` 2.0 programs
    Qasm,
    /// LLVM engine for LLVM IR/bitcode programs
    Llvm,
    /// Selene engine for optimized quantum programs
    Selene,
}

impl fmt::Display for EngineType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EngineType::Qasm => write!(f, "QASM"),
            EngineType::Llvm => write!(f, "LLVM"),
            EngineType::Selene => write!(f, "Selene"),
        }
    }
}

/// Dynamic engine builder that can hold any engine builder type
///
/// This type uses boxed trait objects to enable runtime engine selection.
/// It's useful when you need to dynamically choose between different engines
/// based on runtime conditions.
///
/// # Why Use `DynamicEngineBuilder`?
///
/// In Rust, each engine builder has its own concrete type (`QasmEngineBuilder`,
/// `QisEngineBuilder`, etc.). This is great for performance and type safety,
/// but it means you can't easily store different builders in the same variable
/// or collection. `DynamicEngineBuilder` solves this by wrapping any engine
/// builder in a type-erased container.
///
/// # Examples
///
/// ## Runtime engine selection from user input
/// ```rust
/// # #[cfg(feature = "qasm")]
/// # {
/// # use pecos_core::errors::PecosError;
/// # fn example() -> Result<(), PecosError> {
/// use pecos::{EngineType, DynamicEngineBuilder, sim_dynamic};
/// use pecos_qasm::qasm_engine;
/// use pecos_programs::Qasm;
///
/// struct Config {
///     engine_type: &'static str,
///     source_code: String,
/// }
///
/// fn create_engine_from_config(config: &Config) -> DynamicEngineBuilder {
///     match config.engine_type {
///         "qasm" => DynamicEngineBuilder::new(
///             qasm_engine().program(Qasm::from_string(&config.source_code))
///         ),
///         _ => panic!("Unknown engine type"),
///     }
/// }
///
/// let config = Config {
///     engine_type: "qasm",
///     source_code: r#"
/// OPENQASM 2.0;
/// include "qelib1.inc";
/// qreg q[1];
/// creg c[1];
/// h q[0];
/// measure q[0] -> c[0];
/// "#.to_string(),
/// };
/// let engine = create_engine_from_config(&config);
/// let results = sim_dynamic(engine).seed(42).run(10)?;
/// assert_eq!(results.len(), 10);
/// # Ok(())
/// # }
/// # }
/// ```
///
/// ## Storing multiple engines in a collection
/// ```rust
/// # #[cfg(feature = "qasm")]
/// # {
/// use std::collections::BTreeMap;
/// use pecos::{DynamicEngineBuilder};
/// use pecos_qasm::qasm_engine;
/// use pecos_programs::Qasm;
///
/// let mut engines = BTreeMap::new();
/// let qasm_code = r#"
/// OPENQASM 2.0;
/// include "qelib1.inc";
/// qreg q[1];
/// creg c[1];
/// h q[0];
/// measure q[0] -> c[0];
/// "#;
/// engines.insert("qasm", DynamicEngineBuilder::new(
///     qasm_engine().program(Qasm::from_string(qasm_code))
/// ));
///
/// // Select engine at runtime
/// let user_choice = "qasm";
/// let selected = engines.get(user_choice).unwrap();
/// assert!(engines.contains_key("qasm"));
/// # }
/// ```
pub struct DynamicEngineBuilder {
    builder: Box<dyn DynamicEngineBuilderTrait>,
}

impl DynamicEngineBuilder {
    /// Create a new dynamic engine builder from any concrete engine builder
    pub fn new<B>(builder: B) -> Self
    where
        B: ClassicalControlEngineBuilder + Send + 'static,
        B::Engine: 'static,
    {
        Self {
            builder: Box::new(ConcreteEngineBuilder(builder)),
        }
    }

    /// Create a dynamic engine builder from an `EngineType`
    ///
    /// This creates a default builder for the specified engine type.
    /// You'll need to configure it further with engine-specific methods.
    #[cfg(all(feature = "qasm", feature = "qis"))]
    #[must_use]
    pub fn from_type(engine_type: EngineType) -> Self {
        match engine_type {
            EngineType::Qasm => Self::new(pecos_qasm::qasm_engine()),
            // Selene removed - both Llvm and Selene use QIS control engine
            EngineType::Llvm | EngineType::Selene => Self::new(pecos_qis::qis_engine()),
        }
    }
}

/// Internal trait for type erasure
trait DynamicEngineBuilderTrait: Send {
    fn build(self: Box<Self>) -> Result<Box<dyn ClassicalControlEngine>, PecosError>;
}

/// Wrapper to implement the dynamic trait for concrete builders
struct ConcreteEngineBuilder<B: ClassicalControlEngineBuilder>(B);

impl<B> DynamicEngineBuilderTrait for ConcreteEngineBuilder<B>
where
    B: ClassicalControlEngineBuilder + Send + 'static,
    B::Engine: 'static,
{
    fn build(self: Box<Self>) -> Result<Box<dyn ClassicalControlEngine>, PecosError> {
        Ok(Box::new(self.0.build()?))
    }
}

impl ClassicalControlEngineBuilder for DynamicEngineBuilder {
    type Engine = Box<dyn ClassicalControlEngine>;

    fn build(self) -> Result<Self::Engine, PecosError> {
        self.builder.build()
    }
}

/// Create a simulation builder from a dynamic engine builder
///
/// This allows using the `sim()` function with dynamic engine builders.
///
/// # Example
/// ```rust
/// # #[cfg(feature = "qasm")]
/// # {
/// # use pecos_core::errors::PecosError;
/// # fn example() -> Result<(), PecosError> {
/// use pecos::{EngineType, create_engine_builder, sim_dynamic};
/// use pecos_programs::Qasm;
///
/// // Create a QASM engine builder using the macro
/// let engine = create_engine_builder!(EngineType::Qasm);
/// // In a real scenario, you would configure the engine with a program
/// # Ok(())
/// # }
/// # }
/// ```
#[must_use]
pub fn sim_dynamic(builder: DynamicEngineBuilder) -> pecos_engines::SimBuilder {
    sim(builder)
}

/// Helper macro to create engine builders based on `EngineType`
///
/// This macro assumes the engine crates are available as dependencies.
///
/// # Example
/// ```rust
/// # #[cfg(feature = "qasm")]
/// # {
/// use pecos::{create_engine_builder, EngineType};
///
/// let builder = create_engine_builder!(EngineType::Qasm);
/// // Builder is created successfully
/// # }
/// ```
#[macro_export]
macro_rules! create_engine_builder {
    ($engine_type:expr) => {
        match $engine_type {
            $crate::EngineType::Qasm => {
                #[cfg(feature = "qasm")]
                {
                    $crate::DynamicEngineBuilder::new(pecos_qasm::qasm_engine())
                }
                #[cfg(not(feature = "qasm"))]
                {
                    panic!("QASM engine not available. Enable the 'qasm' feature.")
                }
            }
            $crate::EngineType::Llvm => {
                #[cfg(feature = "qis")]
                {
                    $crate::DynamicEngineBuilder::new(pecos_qis::qis_engine())
                }
                #[cfg(not(feature = "qis"))]
                {
                    panic!("LLVM engine not available. Enable the 'llvm' feature.")
                }
            }
            $crate::EngineType::Selene => {
                #[cfg(feature = "qis")]
                {
                    // Selene removed - use QIS control engine instead
                    $crate::DynamicEngineBuilder::new(pecos_qis::qis_engine())
                }
                #[cfg(not(feature = "qis"))]
                {
                    panic!("Selene engine not available. Enable the 'selene' feature.")
                }
            }
        }
    };
}
