// Re-export main engine types and functions
pub use crate::builder::{PhirJsonEngineBuilder, phir_json_engine};
pub use crate::{PhirJsonEngine, setup_phir_json_engine};

// Re-export common shot result types and formatters from pecos-engines
pub use pecos_engines::{
    BitVecDisplayFormat, Shot, ShotMap, ShotMapDisplayExt, ShotMapDisplayOptions, ShotVec,
};
