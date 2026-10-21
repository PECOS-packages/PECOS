//! Explicit exposure placement for downstream runtime normalization.
use anyhow::{Result, bail, ensure};
use pecos_engines::prelude::*;
use selene_core::error_model::BatchResult;
use selene_core::error_model::interface::ErrorModelInterface;
use selene_core::runtime::BatchOperation;
use selene_core::simulator::SimulatorInterface;

use crate::GeneralNoiseErrorModel;

const MAX_RECORDS: usize = 4096;
const MAX_TARGETS: usize = 16_384;
pub(crate) const MAX_EXCHANGES: usize = 65_536;
pub(crate) const MAX_SIMULATOR_OPERATIONS: usize = 262_144;

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
/// exposure list explicitly means no exposure. This type deliberately does not
/// implement `ErrorModelInterface`: a wrapper must supply placements rather than
/// accidentally using automatic crosstalk. Parameters use the ordinary
/// `GeneralNoiseConfig`; local groups and custom-event interpretation are absent.
///
/// Admission is transactional. Any execution failure (including a caught panic)
/// permanently poisons the instance. Replace it **and reset the simulator** to
/// recover; starting a new shot cannot clear partial execution. Successful shots
/// follow `shot_start`, zero or more `handle_operations`, then `shot_end`.
///
/// Per batch: at most 4096 ordinary operations plus exposures, 16384 supplied
/// targets plus global candidate targets, 65536 controller exchanges, and 262144
/// emitted simulator operations. These are guardrails, not a promise that every
/// valid noise configuration fits them or an allocation bound inside the model.
/// The simulation capacity must be in `1..=16384`.
pub struct ExplicitCrosstalkErrorModel {
    inner: GeneralNoiseErrorModel,
    active_shot: Option<u64>,
    poisoned: bool,
}

impl ExplicitCrosstalkErrorModel {
    /// Construct without starting a shot.
    ///
    /// # Errors
    /// Rejects invalid noise parameters or capacity outside `1..=16384`.
    pub fn new(n_qubits: usize, config: &GeneralNoiseConfig) -> Result<Self> {
        ensure!(
            (1..=MAX_TARGETS).contains(&n_qubits),
            "explicit crosstalk capacity must be in 1..=16384"
        );
        Ok(Self {
            inner: GeneralNoiseErrorModel::new(n_qubits, config, &[])?,
            active_shot: None,
            poisoned: false,
        })
    }

    /// Begin a shot after the host has reset its simulator.
    ///
    /// # Errors
    /// Rejects an active or poisoned instance; model failures poison the owner.
    pub fn shot_start(&mut self, shot_id: u64, error_seed: u64) -> Result<()> {
        ensure!(
            !self.poisoned && self.active_shot.is_none(),
            "explicit crosstalk instance is active or poisoned"
        );
        self.poisoned = true;
        self.inner.shot_start(shot_id, error_seed)?;
        self.active_shot = Some(shot_id);
        self.poisoned = false;
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
        self.active_shot = None;
        self.poisoned = false;
        Ok(())
    }

    fn require_active(&self) -> Result<()> {
        ensure!(
            !self.poisoned && self.active_shot.is_some(),
            "explicit crosstalk instance is inactive or poisoned"
        );
        Ok(())
    }

    fn admit(&self, operations: &BatchOperation, placements: &[CrosstalkPlacement]) -> Result<()> {
        ensure!(
            operations.len() <= MAX_RECORDS && placements.len() <= MAX_RECORDS - operations.len(),
            "explicit crosstalk record limit exceeded"
        );
        let capacity = self.inner.last_operation_end.len();
        let mut targets = 0usize;
        let mut previous = 0;
        for placement in placements {
            ensure!(
                placement.before_operation >= previous
                    && placement.before_operation <= operations.len(),
                "explicit crosstalk positions must be ordered and within the batch"
            );
            previous = placement.before_operation;
            let (qubits, candidates) = match &placement.exposure {
                CrosstalkExposure::Local(qubits) => (qubits, 0),
                CrosstalkExposure::GlobalExcluding(qubits) => (qubits, capacity),
            };
            ensure!(
                qubits.len() <= MAX_TARGETS - targets,
                "explicit crosstalk target limit exceeded"
            );
            targets += qubits.len();
            ensure!(
                candidates <= MAX_TARGETS - targets,
                "explicit crosstalk target limit exceeded"
            );
            targets += candidates;
            let mut seen = std::collections::BTreeSet::new();
            for &qubit in qubits {
                if qubit >= capacity || !seen.insert(qubit) {
                    bail!("explicit crosstalk targets must be distinct and within capacity");
                }
            }
        }
        Ok(())
    }

    /// Execute a complete, normalized runtime batch with explicit placements.
    ///
    /// # Errors
    /// Rejects inactive/poisoned instances, invalid batches, placements or bounds
    /// before execution. Model/simulator/output failures poison the instance.
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
        let result = self.inner.execute_batch(prepared, simulator, true)?;
        self.poisoned = false;
        Ok(result)
    }
}
