//! Explicit exposure placement for downstream runtime normalization.
use std::collections::BTreeSet;

use anyhow::{Result, bail, ensure};
use pecos_engines::prelude::*;
use selene_core::error_model::BatchResult;
use selene_core::error_model::interface::ErrorModelInterface;
use selene_core::runtime::BatchOperation;
use selene_core::simulator::SimulatorInterface;

use crate::{GeneralNoiseErrorModel, MAX_PAYLOAD_QUBITS};

/// One measurement-induced exposure. This describes noise, not physical occupancy.
/// Lists must contain distinct in-capacity qubits. Repeated *records* are meaningful.
#[derive(Clone, Debug)]
pub enum CrosstalkExposure {
    /// Apply local crosstalk to these victims in the supplied order.
    Local(Vec<usize>),
    /// Apply global crosstalk to tracked qubits other than these exclusions.
    GlobalExcluding(Vec<usize>),
}
impl CrosstalkExposure {
    pub(crate) fn append(&self, builder: &mut ByteMessageBuilder) {
        match self {
            Self::Local(qubits) => {
                builder.meas_crosstalk_local_payload(qubits);
            }
            Self::GlobalExcluding(qubits) => {
                builder.meas_crosstalk_global_payload(qubits);
            }
        }
    }

    fn qubits(&self) -> &[usize] {
        match self {
            Self::Local(qubits) | Self::GlobalExcluding(qubits) => qubits,
        }
    }
}

/// Insert an exposure before the indexed ordinary operation, or at batch end
/// when `before_operation == operations.len()`. Equal positions preserve list
/// order; positions must be nondecreasing. Placement precedes any deferred idle
/// charged to that ordinary operation. Exposures never advance the idle clock.
#[derive(Clone, Debug)]
pub struct CrosstalkPlacement {
    /// Index into the supplied ordinary Selene batch (not the raw custom-event batch).
    pub before_operation: usize,
    /// Explicit effect to insert; no automatic measurement exposure is added.
    pub exposure: CrosstalkExposure,
}

/// Opt-in bridge for complete runtime batches with explicit crosstalk placement.
///
/// A downstream plugin interprets its runtime events and calls this owner once per
/// original batch, preserving timing, operation order and measurement IDs. An empty
/// exposure list explicitly means no measurement-crosstalk exposure; preparation
/// crosstalk still follows the configured general model. This type deliberately does not
/// implement `ErrorModelInterface`: a wrapper must supply placements rather than
/// accidentally using automatic crosstalk. Parameters use the ordinary
/// `GeneralNoiseConfig`; local groups and custom-event interpretation are absent.
///
/// Admission is transactional. Any execution failure (including a caught panic)
/// permanently poisons the instance. Replace it **and reset the simulator** to
/// recover; starting a new shot cannot clear partial execution. Successful shots
/// follow `shot_start`, zero or more `handle_operations`, then `shot_end`.
pub struct ExplicitCrosstalkErrorModel {
    inner: GeneralNoiseErrorModel,
    active: bool,
    poisoned: bool,
}

impl ExplicitCrosstalkErrorModel {
    /// Construct without starting a shot.
    ///
    /// # Errors
    /// Rejects invalid noise parameters.
    pub fn new(n_qubits: usize, config: &GeneralNoiseConfig) -> Result<Self> {
        Ok(Self::from_inner(GeneralNoiseErrorModel::new(
            n_qubits,
            config,
            &[],
        )?))
    }

    pub(crate) fn from_inner(inner: GeneralNoiseErrorModel) -> Self {
        Self {
            inner,
            active: false,
            poisoned: false,
        }
    }

    /// Begin a shot after the host has reset its simulator.
    ///
    /// # Errors
    /// Rejects an active or poisoned instance.
    pub fn shot_start(&mut self, shot_id: u64, error_seed: u64) -> Result<()> {
        ensure!(
            !self.poisoned,
            "explicit crosstalk instance is poisoned by a failed execution; create a new instance and reset the simulator"
        );
        ensure!(
            !self.active,
            "explicit crosstalk shot is already active; call shot_end before starting another"
        );
        self.inner.shot_start(shot_id, error_seed)?;
        self.active = true;
        Ok(())
    }

    /// Complete a successful shot and clear its noise state.
    ///
    /// # Errors
    /// Rejects an inactive or poisoned instance; reset failures poison the owner.
    pub fn shot_end(&mut self) -> Result<()> {
        self.require_active()?;
        self.poisoned = true;
        self.inner.shot_end()?;
        self.active = false;
        self.poisoned = false;
        Ok(())
    }

    fn require_active(&self) -> Result<()> {
        ensure!(
            !self.poisoned,
            "explicit crosstalk instance is poisoned by a failed execution; create a new instance and reset the simulator"
        );
        ensure!(
            self.active,
            "explicit crosstalk shot is not active; call shot_start first"
        );
        Ok(())
    }

    fn admit(&self, operations: &BatchOperation, placements: &[CrosstalkPlacement]) -> Result<()> {
        let capacity = self.inner.last_operation_end.len();
        let mut previous = 0;
        for (index, placement) in placements.iter().enumerate() {
            let position = placement.before_operation;
            ensure!(
                position <= operations.len(),
                "crosstalk placement {index} is at position {position}, but the batch has {} operations (use {} to place it after the last operation)",
                operations.len(),
                operations.len()
            );
            ensure!(
                position >= previous,
                "crosstalk placement {index} is at position {position} after a placement at position {previous}; sort placements by position"
            );
            previous = position;
            let qubits = placement.exposure.qubits();
            ensure!(
                qubits.len() <= MAX_PAYLOAD_QUBITS,
                "crosstalk placement {index} lists {} qubits, but one crosstalk payload holds at most {MAX_PAYLOAD_QUBITS}",
                qubits.len()
            );
            let mut seen = BTreeSet::new();
            for &qubit in qubits {
                if qubit >= capacity {
                    bail!(
                        "crosstalk placement {index} lists qubit {qubit}, but the simulation has {capacity} qubits"
                    );
                }
                if !seen.insert(qubit) {
                    bail!("crosstalk placement {index} lists qubit {qubit} more than once");
                }
            }
        }
        Ok(())
    }

    /// Execute a complete, normalized runtime batch with explicit placements.
    ///
    /// # Errors
    /// Rejects inactive/poisoned instances, invalid batches or placements before
    /// execution. Model/simulator/output failures poison the instance.
    pub fn handle_operations(
        &mut self,
        operations: BatchOperation,
        placements: &[CrosstalkPlacement],
        simulator: &mut dyn SimulatorInterface,
    ) -> Result<BatchResult> {
        self.require_active()?;
        self.admit(&operations, placements)?;
        let prepared = self
            .inner
            .prepare_operations(operations, Some(placements))?;
        // Set before crossing either execution boundary, so caught panics cannot
        // leave a reusable owner with partial controller or simulator state.
        self.poisoned = true;
        let result = self.inner.execute_batch(prepared, simulator)?;
        self.poisoned = false;
        Ok(result)
    }
}
