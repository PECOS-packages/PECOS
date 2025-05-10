pub mod byte_message;
pub mod core;
pub mod engines;
pub mod errors;

// Re-exports for commonly used types
pub use byte_message::{ByteMessage, ByteMessageBuilder, GateType, QuantumGate};
pub use core::record_data::RecordData;
pub use core::result_id::ResultId;
pub use core::shot_results::{ShotResult, ShotResults};
pub use engines::{
    ClassicalEngine, ControlEngine, Engine, EngineStage, EngineSystem,
    hybrid::HybridEngine,
    monte_carlo::MonteCarloEngine,
    noise::{DepolarizingNoiseModel, NoiseModel, PassThroughNoiseModel},
    phir::PHIREngine,
    qir::QirEngine,
    quantum::QuantumEngine,
    quantum_system::QuantumSystem,
};
pub use errors::QueueError;

// Re-export engine setup functions
pub use engines::classical::{setup_phir_engine, setup_qir_engine};
