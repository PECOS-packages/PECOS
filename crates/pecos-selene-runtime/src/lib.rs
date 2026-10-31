//! Lazy virtual-Z scheduling through Selene's standard `RuntimeInterface`.
//!
//! Inspired by Quantinuum's Apache-2.0 `SoftRZ` runtime (Selene revision
//! 458d640dc4cf84a7a7988ea65ce5a9f7bcb8c511). No simulator or noise model lives
//! here. Rust callers and the separately exported Selene plugin share this code.

use anyhow::{Result, bail, ensure};
use clap::Parser;
use selene_core::runtime::{
    BatchOperation, Operation, RuntimeInterface, interface::RuntimeInterfaceFactory,
};
use selene_core::time::Instant;
use selene_core::utils::MetricValue;
use std::collections::{BTreeMap, VecDeque};
use std::sync::Arc;

/// Gate durations are nanoseconds. Optional result limits bound retained futures.
#[derive(Clone, Debug, Parser)]
pub struct RuntimeConfig {
    #[arg(long, default_value_t = 0)]
    pub duration_ns_rxy: u64,
    #[arg(long, default_value_t = 0)]
    pub duration_ns_rzz: u64,
    #[arg(long, default_value_t = 0)]
    pub duration_ns_measure: u64,
    #[arg(long, default_value_t = 0)]
    pub duration_ns_measure_leaked: u64,
    #[arg(long, default_value_t = 0)]
    pub duration_ns_reset: u64,
    #[arg(long, default_value_t = 1)]
    pub max_batch_size: usize,
    /// At this limit ready work must be drained before more batches are added.
    #[arg(long, default_value_t = 4096)]
    pub max_queued_batches: usize,
    /// Optional retained-future limit; omitted for hosts that do not retire eager futures.
    #[arg(long)]
    pub max_live_results: Option<usize>,
}
impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            duration_ns_rxy: 0,
            duration_ns_rzz: 0,
            duration_ns_measure: 0,
            duration_ns_measure_leaked: 0,
            duration_ns_reset: 0,
            max_batch_size: 1,
            max_queued_batches: 4096,
            max_live_results: None,
        }
    }
}
#[derive(Clone, Default)]
struct Qubit {
    allocated: bool,
    phase: f64,
    available: u64,
}
struct Future {
    leaked: bool,
    refs: u64,
    emitted: bool,
    value: Option<u64>,
}

/// Reusable Rust implementation of Selene's existing runtime contract.
///
/// A force seals and releases a queue prefix through the requested measurement.
/// Unforced work remains queued. Callers must drain ready batches after calls,
/// deliver each emitted measurement, and drain shutdown work before starting the next shot.
pub struct PecosRuntime {
    config: RuntimeConfig,
    qubits: Vec<Qubit>,
    queue: VecDeque<BatchOperation>,
    ready: usize,
    results: BTreeMap<u64, Future>,
    next_result: u64,
    tail: u64,
    emitted_end: u64,
    closed: bool,
    ending: bool,
}
// Only the physical operations emitted by this scheduler reach this helper.
// Their one or two qubit IDs fit on the stack; conflict checks need no hash set.
fn operation_qubits(op: &Operation) -> impl Iterator<Item = u64> + Clone + use<> {
    let ids = match *op {
        Operation::RXYGate { qubit_id, .. }
        | Operation::Reset { qubit_id }
        | Operation::Measure { qubit_id, .. }
        | Operation::MeasureLeaked { qubit_id, .. } => [Some(qubit_id), None],
        Operation::RZZGate {
            qubit_id_1,
            qubit_id_2,
            ..
        } => [Some(qubit_id_1), Some(qubit_id_2)],
        _ => unreachable!("unsupported internal scheduler operation"),
    };
    ids.into_iter().flatten()
}

impl PecosRuntime {
    /// Construct a runtime. Gate durations may be zero; storage limits must be positive.
    /// # Errors
    /// Rejects zero limits and unrepresentable qubit capacity.
    pub fn new(n_qubits: u64, start: Instant, config: RuntimeConfig) -> Result<Self> {
        ensure!(
            config.max_batch_size > 0
                && config.max_queued_batches > 0
                && config.max_live_results.is_none_or(|limit| limit > 0),
            "runtime limits must be positive"
        );
        let origin = start.into();
        Ok(Self {
            config,
            qubits: vec![Qubit::default(); usize::try_from(n_qubits)?],
            queue: VecDeque::new(),
            ready: 0,
            results: BTreeMap::new(),
            next_result: 0,
            tail: origin,
            emitted_end: origin,
            closed: false,
            ending: false,
        })
    }
    fn live(&self) -> Result<()> {
        ensure!(!self.closed, "runtime has exited");
        Ok(())
    }
    fn qubit(&self, id: u64) -> Result<&Qubit> {
        self.live()?;
        ensure!(
            !self.ending,
            "start a new shot before submitting operations"
        );
        self.qubits
            .get(usize::try_from(id)?)
            .filter(|q| q.allocated)
            .ok_or_else(|| anyhow::anyhow!("qubit {id} is not allocated"))
    }
    fn future(&self, id: u64) -> Result<&Future> {
        self.live()?;
        self.results
            .get(&id)
            .filter(|r| r.refs > 0)
            .ok_or_else(|| anyhow::anyhow!("unknown or retired result {id}"))
    }
    fn push(&mut self, op: Operation, duration: u64) -> Result<()> {
        self.live()?;
        let qubits = operation_qubits(&op);
        let mut earliest = self.emitted_end;
        for q in qubits.clone() {
            earliest = earliest.max(self.qubit(q)?.available);
        }
        let mut candidate = None;
        // Singleton batches cannot accept another operation, so skip the search.
        if self.config.max_batch_size > 1 {
            // Sealed batches cannot accept work submitted after a force or barrier.
            for (i, batch) in self.queue.iter().enumerate().skip(self.ready).rev() {
                if batch
                    .iter_ops()
                    .any(|other| operation_qubits(other).any(|q| qubits.clone().any(|id| id == q)))
                {
                    break;
                }
                let source = batch.runtime_source().expect("runtime batch");
                let start: u64 = source.start().into();
                if start < earliest {
                    break;
                }
                if batch.len() < self.config.max_batch_size
                    && batch
                        .iter_ops()
                        .all(|other| std::mem::discriminant(other) == std::mem::discriminant(&op))
                {
                    candidate = Some(i);
                }
                // Measurement results are delivered in source order, including when
                // a host maps multiple native futures to the same program slot.
                if matches!(
                    op,
                    Operation::Measure { .. } | Operation::MeasureLeaked { .. }
                ) && batch.iter_ops().any(|other| {
                    matches!(
                        other,
                        Operation::Measure { .. } | Operation::MeasureLeaked { .. }
                    )
                }) {
                    break;
                }
            }
        }
        if let Some(i) = candidate {
            self.queue[i].add_operation(op);
        } else {
            ensure!(
                self.queue.len() < self.config.max_queued_batches,
                "runtime queue full; drain ready operations before submitting more"
            );
            let start = self.tail.max(earliest);
            let end = start
                .checked_add(duration)
                .ok_or_else(|| anyhow::anyhow!("runtime time overflow"))?;
            self.queue.push_back(BatchOperation::runtime(
                vec![op],
                start.into(),
                duration.into(),
            ));
            self.tail = end;
        }
        if self.queue.len() == self.config.max_queued_batches {
            self.ready = self.queue.len();
        }
        Ok(())
    }
    fn measure_impl(&mut self, q: u64, leaked: bool) -> Result<u64> {
        self.qubit(q)?;
        ensure!(
            self.config
                .max_live_results
                .is_none_or(|limit| self.results.len() < limit),
            "live result limit exceeded; retire unused futures"
        );
        let id = self.next_result;
        let next = id
            .checked_add(1)
            .ok_or_else(|| anyhow::anyhow!("result ID overflow"))?;
        let (op, duration) = if leaked {
            (
                Operation::MeasureLeaked {
                    qubit_id: q,
                    result_id: id,
                },
                self.config.duration_ns_measure_leaked,
            )
        } else {
            (
                Operation::Measure {
                    qubit_id: q,
                    result_id: id,
                },
                self.config.duration_ns_measure,
            )
        };
        self.push(op, duration)?;
        self.results.insert(
            id,
            Future {
                leaked,
                refs: 1,
                emitted: false,
                value: None,
            },
        );
        self.next_result = next;
        Ok(id)
    }
    fn set_result(&mut self, id: u64, value: u64) -> Result<()> {
        self.live()?;
        let r = self
            .results
            .get_mut(&id)
            .ok_or_else(|| anyhow::anyhow!("unknown result {id}"))?;
        ensure!(
            r.emitted && r.value.is_none(),
            "result {id} was not emitted or was already delivered"
        );
        ensure!(
            value <= if r.leaked { 2 } else { 1 },
            "invalid measurement outcome {value}"
        );
        r.value = Some(value);
        if r.refs == 0 {
            self.results.remove(&id);
        }
        Ok(())
    }
}

impl RuntimeInterface for PecosRuntime {
    fn exit(&mut self) -> Result<()> {
        self.live()?;
        self.queue.clear();
        self.results.clear();
        self.qubits.clear();
        self.closed = true;
        Ok(())
    }
    fn get_next_operations(&mut self) -> Result<Option<BatchOperation>> {
        self.live()?;
        if self.ready == 0 {
            return Ok(None);
        }
        let batch = self.queue.pop_front().expect("ready prefix exists");
        self.ready -= 1;
        self.emitted_end = batch.runtime_source().expect("runtime batch").end().into();
        for op in batch.iter_ops() {
            if let Operation::Measure { result_id, .. }
            | Operation::MeasureLeaked { result_id, .. } = op
            {
                self.results
                    .get_mut(result_id)
                    .expect("queued future exists")
                    .emitted = true;
            }
        }
        Ok(Some(batch))
    }
    fn shot_start(&mut self, _shot_id: u64, _seed: u64) -> Result<()> {
        self.live()?;
        ensure!(
            self.queue.is_empty() && self.results.values().all(|r| r.value.is_some()),
            "drain the previous shot and deliver its measurement feedback first"
        );
        ensure!(
            self.ending || (self.results.is_empty() && !self.qubits.iter().any(|q| q.allocated)),
            "finish the previous shot before starting another"
        );
        self.results.clear();
        self.ending = false;
        self.next_result = 0;
        // Runtime timestamps are continuous across shots in both hosts. Keep
        // barrier-only advances too, even when no later gate was emitted.
        self.tail = self
            .qubits
            .iter()
            .fold(self.tail, |t, q| t.max(q.available));
        self.emitted_end = self.tail;
        self.ready = 0;
        self.qubits.fill(Qubit::default());
        Ok(())
    }
    fn shot_end(&mut self) -> Result<()> {
        self.live()?;
        ensure!(!self.ending, "shot already ended");
        // Selene drains AFTER this callback. Preserve futures for that feedback;
        // PECOS may already have drained. Both hosts reset on the next shot.
        self.ready = self.queue.len();
        self.ending = true;
        Ok(())
    }
    fn qalloc(&mut self) -> Result<u64> {
        self.live()?;
        ensure!(!self.ending, "start a new shot before allocating");
        if let Some((i, q)) = self
            .qubits
            .iter_mut()
            .enumerate()
            .find(|(_, q)| !q.allocated)
        {
            q.allocated = true;
            q.phase = 0.0;
            Ok(u64::try_from(i)?)
        } else {
            Ok(u64::MAX)
        }
    }
    fn qfree(&mut self, id: u64) -> Result<()> {
        self.local_barrier(&[id], 0)?;
        self.qubits[usize::try_from(id)?].allocated = false;
        Ok(())
    }
    fn rxy_gate(&mut self, id: u64, theta: f64, phi: f64) -> Result<()> {
        ensure!(theta.is_finite() && phi.is_finite(), "non-finite rotation");
        let phase = self.qubit(id)?.phase;
        let phi = phi - phase;
        ensure!(phi.is_finite(), "rotation phase overflow");
        self.push(
            Operation::RXYGate {
                qubit_id: id,
                theta,
                phi,
            },
            self.config.duration_ns_rxy,
        )
    }
    fn rz_gate(&mut self, id: u64, theta: f64) -> Result<()> {
        let phase = self.qubit(id)?.phase + theta;
        ensure!(phase.is_finite(), "non-finite rotation phase");
        self.qubits[usize::try_from(id)?].phase = phase;
        Ok(())
    }
    fn rzz_gate(&mut self, a: u64, b: u64, theta: f64) -> Result<()> {
        ensure!(a != b && theta.is_finite(), "invalid RZZ operands");
        self.push(
            Operation::RZZGate {
                qubit_id_1: a,
                qubit_id_2: b,
                theta,
            },
            self.config.duration_ns_rzz,
        )
    }
    fn reset(&mut self, id: u64) -> Result<()> {
        self.push(
            Operation::Reset { qubit_id: id },
            self.config.duration_ns_reset,
        )
    }
    fn measure(&mut self, id: u64) -> Result<u64> {
        self.measure_impl(id, false)
    }
    fn measure_leaked(&mut self, id: u64) -> Result<u64> {
        self.measure_impl(id, true)
    }
    fn force_result(&mut self, id: u64) -> Result<()> {
        if self.future(id)?.value.is_some() {
            return Ok(());
        }
        if let Some(i) = self.queue.iter().position(|batch| batch.iter_ops().any(|op| {
            matches!(op, Operation::Measure { result_id, .. } | Operation::MeasureLeaked { result_id, .. } if *result_id == id)
        })) { self.ready = self.ready.max(i + 1); }
        // Already emitted futures await their host's feedback; never re-emit.
        Ok(())
    }
    fn get_bool_result(&mut self, id: u64) -> Result<Option<bool>> {
        match self.future(id)?.value {
            Some(2) => bail!("leakage outcome requires an integer result read"),
            value => Ok(value.map(|v| v == 1)),
        }
    }
    fn get_u64_result(&mut self, id: u64) -> Result<Option<u64>> {
        Ok(self.future(id)?.value)
    }
    fn set_bool_result(&mut self, id: u64, result: bool) -> Result<()> {
        self.set_result(id, u64::from(result))
    }
    fn set_u64_result(&mut self, id: u64, result: u64) -> Result<()> {
        self.set_result(id, result)
    }
    fn increment_future_refcount(&mut self, id: u64) -> Result<()> {
        let refs = self
            .future(id)?
            .refs
            .checked_add(1)
            .ok_or_else(|| anyhow::anyhow!("reference count overflow"))?;
        self.results.get_mut(&id).expect("validated future").refs = refs;
        Ok(())
    }
    fn decrement_future_refcount(&mut self, id: u64) -> Result<()> {
        let refs = self.future(id)?.refs - 1;
        let r = self.results.get_mut(&id).expect("validated future");
        r.refs = refs;
        // Keep a tombstone until feedback arrives, even if discarded before emission.
        if refs == 0 && r.value.is_some() {
            self.results.remove(&id);
        }
        Ok(())
    }
    fn local_barrier(&mut self, ids: &[u64], sleep_ns: u64) -> Result<()> {
        self.live()?;
        for id in ids {
            self.qubit(*id)?;
        }
        let mut updates = Vec::new();
        let mut ready = self.ready;
        for id in ids {
            let mut end = self.emitted_end.max(self.qubit(*id)?.available);
            if let Some((i, batch)) = self.queue.iter().enumerate().rev().find(|(_, b)| {
                b.iter_ops()
                    .any(|op| operation_qubits(op).any(|q| q == *id))
            }) {
                ready = ready.max(i + 1);
                end = end.max(batch.runtime_source().expect("runtime batch").end().into());
            }
            updates.push((
                *id,
                end.checked_add(sleep_ns)
                    .ok_or_else(|| anyhow::anyhow!("barrier time overflow"))?,
            ));
        }
        self.ready = ready;
        for (id, time) in updates {
            self.qubits[usize::try_from(id)?].available = time;
        }
        Ok(())
    }
    fn global_barrier(&mut self, sleep_ns: u64) -> Result<()> {
        self.live()?;
        let end = self
            .qubits
            .iter()
            .filter(|q| q.allocated)
            .fold(self.tail, |t, q| t.max(q.available));
        let end = end
            .checked_add(sleep_ns)
            .ok_or_else(|| anyhow::anyhow!("barrier time overflow"))?;
        self.ready = self.queue.len();
        self.tail = end;
        for q in &mut self.qubits {
            if q.allocated {
                q.available = end;
            }
        }
        Ok(())
    }
    fn simulate_delay(&mut self, delay_ns: u64) -> Result<()> {
        self.global_barrier(delay_ns)
    }
    fn get_metric(&mut self, _nth_metric: u8) -> Result<Option<(String, MetricValue)>> {
        self.live()?;
        Ok(None)
    }
}

/// Factory shared by the Rust API and standard Selene plugin descriptor.
#[derive(Default)]
pub struct RuntimeFactory;
impl RuntimeInterfaceFactory for RuntimeFactory {
    type Interface = PecosRuntime;
    fn init(
        self: Arc<Self>,
        n_qubits: u64,
        start: Instant,
        args: &[impl AsRef<str>],
    ) -> Result<Box<Self::Interface>> {
        let args: Vec<_> = args.iter().map(AsRef::as_ref).collect();
        Ok(Box::new(PecosRuntime::new(
            n_qubits,
            start,
            RuntimeConfig::try_parse_from(args)?,
        )?))
    }
}
