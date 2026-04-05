//! Engine type enumeration and dynamic engine builder.
//!
//! Use `EngineType` to enumerate available engines and `DynamicEngineBuilder`
//! for runtime engine selection via trait objects.
//!
//! For a full guide with examples, see `docs/user-guide/engine-selection.md`.

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
