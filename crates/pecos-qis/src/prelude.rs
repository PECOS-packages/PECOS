//! A prelude for pecos-qis users.
//!
//! This prelude re-exports the most commonly used types and traits from pecos-qis.
//!
//! ## Usage
//!
//! ```rust,no_run
//! use pecos_qis::prelude::*;
//! ```

// Core traits
pub use crate::qis_interface::{InterfaceError, ProgramFormat, QisInterface};
pub use crate::runtime::{QisRuntime, RuntimeError};

// Engine and builder
pub use crate::ccengine::QisEngine;
pub use crate::engine_builder::{QisEngineBuilder, qis_engine};

// Program types
pub use crate::program::{
    InterfaceChoice, IntoQisInterface, ProgramType, QisEngineProgram, QisInterfaceBuilder,
};

// Convenience functions
pub use crate::setup_qis_engine_with_runtime;

// Selene implementation (when enabled)
#[cfg(feature = "selene")]
pub use crate::executor::{HeliosSyncHandle, QisHeliosInterface};
#[cfg(feature = "selene")]
pub use crate::selene_builder::{HeliosInterfaceBuilder, helios_interface_builder};
#[cfg(feature = "selene")]
pub use crate::selene_runtime::SeleneRuntime;
#[cfg(feature = "selene")]
pub use crate::selene_runtimes::{
    RuntimeFetchError, find_selene_runtime, selene_runtime_auto, selene_simple_runtime,
    selene_soft_rz_runtime,
};
