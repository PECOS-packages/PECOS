//! Test-only Rust vertical slice. No library export, Selene transport or wire codec.
//! One original `GeneralNoiseModel` sampling phase and one completion lifecycle.
use pecos_core::rng::rng_manageable::derive_seed;
use pecos_core::{RngManageable, errors::PecosError};
use pecos_engines::noise::{GeneralNoiseModel, GeneralNoiseModelBuilder};
use pecos_engines::quantum::{QuantumEngine, StateVecEngine};
use pecos_engines::{ByteMessage, ControlEngine, Engine, EngineStage, Gate, GateType};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};

const META: u32 = 4101;
const FLIP: u32 = 4102;
const PHASE: u32 = 4103;
const MAX_RECORDS: usize = 128;
static NEXT_RUN: AtomicU64 = AtomicU64::new(1);

fn error(message: &str) -> PecosError {
    PecosError::Input(message.into())
}

// In-process envelope only. Discriminants deliberately are not MessageType.
#[derive(Clone)]
enum Record {
    Gate(Box<Gate>),
    Event(u32, usize),
    Unknown,
}
#[derive(Clone)]
struct Envelope {
    version: u8,
    records: Vec<Record>,
}
enum RequestedModel {
    General(Box<GeneralNoiseModelBuilder>),
    PassThrough,
}

fn general(builder: GeneralNoiseModelBuilder) -> RequestedModel {
    RequestedModel::General(Box::new(builder))
}
fn gate_record(gate: Gate) -> Record {
    Record::Gate(Box::new(gate))
}

// Implementation-owned admission: callers cannot supply a capability boolean
// or an arbitrary NoiseModel implementation to a prepared frame.
#[derive(Clone)]
struct Factory {
    builder: GeneralNoiseModelBuilder,
    plan: Envelope,
}
impl Factory {
    fn compile(request: RequestedModel, plan: Envelope) -> Result<Self, PecosError> {
        let RequestedModel::General(builder) = request else {
            return Err(error("model has no implemented resumable profile"));
        };
        if plan.version != 2 || plan.records.len() > MAX_RECORDS {
            return Err(error("unsupported or oversized envelope"));
        }
        builder.validate_configuration().map_err(error)?;
        let physical_profile = builder.simple_probabilities().is_some();
        for record in &plan.records {
            match record {
                Record::Gate(gate) => {
                    gate.validate().map_err(|err| error(&err))?;
                    if !matches!(
                        gate.gate_type,
                        GateType::PZ
                            | GateType::X
                            | GateType::Z
                            | GateType::H
                            | GateType::MZ
                            | GateType::MeasureLeaked
                            | GateType::Idle
                            | GateType::MeasCrosstalkLocalPayload
                    ) || gate.qubits.len() != 1
                        || gate.qubits[0].0 >= 2
                        || gate.params.len() != usize::from(gate.gate_type == GateType::Idle)
                        || gate.params.iter().any(|p| !p.is_finite())
                        || (gate.gate_type == GateType::Idle && gate.params[0] < 0.0)
                        || !gate.angles.is_empty()
                        || gate.channel.is_some()
                        || !gate.meas_ids.is_empty()
                    {
                        return Err(error("unsupported gate in frame"));
                    }
                }
                Record::Event(META, target) if *target < 2 => {}
                Record::Event(FLIP | PHASE, target) if *target < 2 && physical_profile => {}
                _ => return Err(error("unsupported mandatory record or physical profile")),
            }
        }
        Ok(Self {
            builder: *builder,
            plan,
        })
    }

    fn spawn(&self, seed: u64) -> Runner {
        let mut model = self.builder.clone().build();
        model.set_seed(derive_seed(seed, "noise_model"));
        let mut sim = StateVecEngine::new(2);
        QuantumEngine::set_seed(&mut sim, derive_seed(seed, "quantum_engine"));
        Runner {
            factory: self.clone(),
            model,
            sim: Box::new(sim),
            state: State::Ready,
            id: NEXT_RUN.fetch_add(1, Ordering::Relaxed),
            generation: 0,
            cursor: 0,
            queue: VecDeque::new(),
            raw: Vec::new(),
            trace: Vec::new(),
            completions: 0,
        }
    }
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
struct Token {
    run: u64,
    generation: u64,
    position: usize,
}
#[derive(Debug)]
enum Step {
    Yield(Token),
    Complete(Vec<u32>),
}
enum Planned {
    Quantum(ByteMessage),
    Event(u32, usize),
}
enum State {
    Ready,
    Active,
    Yielded(Token, u32, usize),
    Complete,
    Failed,
}

// Deliberately not Clone: workers originate from Factory, not live continuations.
struct Runner {
    factory: Factory,
    model: GeneralNoiseModel,
    sim: Box<dyn Engine<Input = ByteMessage, Output = ByteMessage>>,
    state: State,
    id: u64,
    generation: u64,
    cursor: usize,
    queue: VecDeque<Planned>,
    raw: Vec<usize>,
    trace: Vec<Gate>,
    completions: usize,
}

fn message(gates: &[Gate]) -> ByteMessage {
    let mut b = ByteMessage::quantum_operations_builder();
    for gate in gates {
        b.add_gate_command(gate);
    }
    b.build()
}

impl Runner {
    fn abort(&mut self) {
        self.state = State::Failed;
        self.queue.clear();
        self.raw.clear();
    }

    fn reset(&mut self, seed: u64) -> Result<(), PecosError> {
        self.abort();
        // Do not clear Failed until BOTH resets succeed.
        self.model.reset()?;
        self.sim.reset()?;
        self.model.set_seed(derive_seed(seed, "noise_model"));
        let mut sim = StateVecEngine::new(2);
        QuantumEngine::set_seed(&mut sim, derive_seed(seed, "quantum_engine"));
        self.sim = Box::new(sim);
        self.generation += 1;
        self.cursor = 0;
        self.trace.clear();
        self.completions = 0;
        self.state = State::Ready;
        Ok(())
    }

    fn execute(&mut self, input: ByteMessage) -> Result<ByteMessage, PecosError> {
        self.trace.extend(input.quantum_ops()?);
        self.sim.process(input)
    }

    fn poll(&mut self) -> Result<Step, PecosError> {
        let result = self.poll_inner();
        if result.is_err() {
            self.abort();
        }
        result
    }

    fn poll_inner(&mut self) -> Result<Step, PecosError> {
        if matches!(self.state, State::Ready) {
            // All start-phase noise is sampled before ANY simulator operation,
            // preserving the existing model's temporal/RNG semantics. Calling
            // the start-phase primitive per gate records expansion boundaries;
            // it does not call start/continue/Complete on segmented inputs.
            for record in &self.factory.plan.records {
                match record {
                    Record::Gate(gate) => {
                        let noisy = self
                            .model
                            .apply_noise_on_start(&message(std::slice::from_ref(gate)))
                            .map_err(|e| error(&e))?;
                        self.queue.push_back(Planned::Quantum(noisy));
                    }
                    Record::Event(tag, target) => {
                        self.queue.push_back(Planned::Event(*tag, *target));
                    }
                    Record::Unknown => return Err(error("invalid prepared plan")),
                }
            }
            self.state = State::Active;
        }
        if !matches!(self.state, State::Active) {
            return Err(error("frame is not active"));
        }
        while let Some(operation) = self.queue.pop_front() {
            self.cursor += 1;
            match operation {
                Planned::Quantum(commands) => {
                    let result = self.execute(commands)?;
                    self.raw
                        .extend(result.outcomes()?.into_iter().map(|x| x as usize));
                }
                Planned::Event(tag, target) => {
                    let token = Token {
                        run: self.id,
                        generation: self.generation,
                        position: self.cursor,
                    };
                    self.state = State::Yielded(token, tag, target);
                    return Ok(Step::Yield(token));
                }
            }
        }
        // Deliver the original whole-input outcome vector once. Extra model
        // continuations remain in this same frame and are never event yields.
        let mut b = ByteMessage::outcomes_builder();
        b.add_outcomes(&self.raw);
        self.raw.clear();
        let mut reply = b.build();
        loop {
            match self.model.continue_processing(reply)? {
                EngineStage::NeedsProcessing(commands) => {
                    reply = self.execute(commands)?;
                }
                EngineStage::Complete(outcomes) => {
                    self.completions += 1;
                    self.state = State::Complete;
                    return Ok(Step::Complete(outcomes.outcomes()?));
                }
            }
        }
    }

    fn resume(&mut self, token: Token) -> Result<(), PecosError> {
        let result = self.resume_inner(token);
        if result.is_err() {
            self.abort();
        }
        result
    }

    fn resume_inner(&mut self, token: Token) -> Result<(), PecosError> {
        let State::Yielded(expected, tag, target) = self.state else {
            return Err(error("no outstanding yield"));
        };
        if token != expected {
            return Err(error("stale or foreign continuation"));
        }
        // Only factory-validated unconditional Paulis, never arbitrary handlers.
        // The admitted physical profile cannot leak or produce crosstalk work.
        let effect = match tag {
            FLIP => Some(Gate::x(&[target])),
            PHASE => Some(Gate::z(&[target])),
            META => None,
            _ => return Err(error("unknown prepared event")),
        };
        if let Some(effect) = effect {
            self.execute(message(&[effect]))?;
        }
        self.state = State::Active;
        Ok(())
    }
}

fn finish(runner: &mut Runner) -> Result<Vec<u32>, PecosError> {
    loop {
        match runner.poll()? {
            Step::Yield(token) => runner.resume(token)?,
            Step::Complete(outcomes) => return Ok(outcomes),
        }
    }
}

fn envelope(gates: &[Gate], annotations: bool) -> Envelope {
    let mut records = Vec::new();
    for gate in gates {
        if annotations {
            records.push(Record::Event(META, 0));
        }
        records.push(gate_record(gate.clone()));
    }
    if annotations {
        records.push(Record::Event(META, 0));
    }
    Envelope {
        version: 2,
        records,
    }
}

fn factory(builder: GeneralNoiseModelBuilder, plan: Envelope) -> Factory {
    Factory::compile(general(builder), plan).unwrap()
}

fn tail(model: &GeneralNoiseModel) -> [u64; 8] {
    let mut rng = model.rng().clone();
    std::array::from_fn(|_| rng.next_u64())
}

#[test]
fn metadata_preserves_legacy_commands_outcomes_leakage_idle_and_rng() {
    use std::collections::BTreeMap;
    let mut b = ByteMessage::quantum_operations_builder();
    b.measure_leakages(&[0])
        .pz(&[0])
        .idle(1.0, &[0])
        .h(&[0])
        .mz(&[0])
        .x(&[0])
        .mz(&[0]);
    let commands = b.build();
    let gates = commands.quantum_ops().unwrap();
    let builder = GeneralNoiseModel::builder()
        .with_p1(0.375)
        .with_p_meas_0(0.625)
        .with_p_idle_sin_squared(
            std::f64::consts::FRAC_PI_2,
            &BTreeMap::from([("X".to_owned(), 1.0)]),
        );
    for seed in 0..64 {
        let mut with = factory(builder.clone(), envelope(&gates, true)).spawn(seed);
        let mut without = factory(builder.clone(), envelope(&gates, false)).spawn(seed);
        let mut baseline = builder.clone().build();
        baseline.set_seed(derive_seed(seed, "noise_model"));
        baseline.mark_as_leaked(0);
        with.model.mark_as_leaked(0);
        without.model.mark_as_leaked(0);
        let mut sim = StateVecEngine::new(2);
        QuantumEngine::set_seed(&mut sim, derive_seed(seed, "quantum_engine"));
        let mut expected_trace = Vec::new();
        let mut stage = baseline.start(commands.clone()).unwrap();
        let expected = loop {
            match stage {
                EngineStage::NeedsProcessing(cmd) => {
                    expected_trace.extend(cmd.quantum_ops().unwrap());
                    stage = baseline
                        .continue_processing(sim.process(cmd).unwrap())
                        .unwrap();
                }
                EngineStage::Complete(result) => break result.outcomes().unwrap(),
            }
        };
        assert_eq!(finish(&mut with).unwrap(), expected);
        assert_eq!(finish(&mut without).unwrap(), expected);
        assert_eq!(with.trace, expected_trace);
        assert_eq!(without.trace, expected_trace);
        assert_eq!(tail(&with.model), tail(&baseline));
        assert_eq!(tail(&without.model), tail(&baseline));
        assert_eq!(with.completions, 1);
        assert_eq!(without.completions, 1);
        assert!(with.poll().is_err());
        assert_eq!(with.completions, 1);
    }
}

#[test]
fn physical_effects_follow_gate_and_measurement_boundaries_without_renoising() {
    for (tag, before, after, expected) in [
        (
            PHASE,
            vec![Gate::h(&[0])],
            vec![Gate::h(&[0]), Gate::mz(&[0])],
            vec![1],
        ),
        (FLIP, vec![Gate::mz(&[0])], vec![Gate::mz(&[0])], vec![0, 1]),
    ] {
        let mut plan = envelope(&before, false);
        plan.records.push(Record::Event(tag, 0));
        plan.records.extend(after.into_iter().map(gate_record));
        let mut run = factory(GeneralNoiseModel::builder(), plan).spawn(4);
        assert_eq!(finish(&mut run).unwrap(), expected);
        assert_eq!(run.completions, 1);
    }
    // With p1=1, ordinary X would receive a fault. An event X must be the
    // sole X between the two measurements and consume no gate-noise draws.
    let mut plan = envelope(&[Gate::mz(&[0])], false);
    plan.records.push(Record::Event(FLIP, 0));
    plan.records.push(gate_record(Gate::mz(&[0])));
    let mut run = factory(GeneralNoiseModel::builder().with_p1(1.0), plan).spawn(9);
    let mut reference = factory(
        GeneralNoiseModel::builder().with_p1(1.0),
        envelope(&[Gate::mz(&[0]), Gate::mz(&[0])], false),
    )
    .spawn(9);
    finish(&mut reference).unwrap();
    assert_eq!(finish(&mut run).unwrap(), [0, 1]);
    assert_eq!(
        run.trace,
        vec![Gate::mz(&[0]), Gate::x(&[0]), Gate::mz(&[0])]
    );
    assert_eq!(tail(&run.model), tail(&reference.model));
}

#[test]
fn admission_rejects_unsupported_models_envelopes_and_physical_profiles() {
    let plan = envelope(&[Gate::x(&[0])], false);
    assert!(Factory::compile(RequestedModel::PassThrough, plan.clone()).is_err());
    for bad in [
        Record::Unknown,
        Record::Event(9999, 0),
        Record::Event(FLIP, 2),
        gate_record(Gate::idle(-1.0, vec![pecos_core::QubitId(0)])),
    ] {
        let mut invalid = plan.clone();
        invalid.records.push(bad); // Valid X prefix must never run.
        assert!(Factory::compile(general(GeneralNoiseModel::builder()), invalid).is_err());
    }
    let mut invalid = plan.clone();
    invalid.version = 99;
    assert!(Factory::compile(general(GeneralNoiseModel::builder()), invalid).is_err());
    let mut too_large = plan.clone();
    too_large.records = vec![Record::Event(META, 0); MAX_RECORDS + 1];
    assert!(Factory::compile(general(GeneralNoiseModel::builder()), too_large).is_err());
    let mut physical = plan;
    physical.records.push(Record::Event(FLIP, 0));
    assert!(
        Factory::compile(
            general(GeneralNoiseModel::builder().with_prep_leak_ratio(0.5)),
            physical
        )
        .is_err()
    );
}

#[test]
fn admission_rejects_invalid_configuration_before_model_construction() {
    let builder = GeneralNoiseModel::builder()
        .with_p1(0.75)
        .with_p1_scale(2.0);
    assert!(builder.validate_configuration().is_err());
    assert!(Factory::compile(general(builder), envelope(&[Gate::x(&[0])], true)).is_err());
}

#[test]
fn admission_rejects_measurement_ids_that_positional_transport_cannot_preserve() {
    for mut gate in [Gate::x(&[0]), Gate::mz(&[0])] {
        gate.meas_ids.push(pecos_core::MeasId::from_raw(17));
        let plan = envelope(&[Gate::x(&[0]), gate], false);
        assert!(Factory::compile(general(GeneralNoiseModel::builder()), plan).is_err());
    }
}

#[test]
fn stale_tokens_poison_until_reset_and_workers_are_independent() {
    let f = factory(
        GeneralNoiseModel::builder(),
        envelope(&[Gate::x(&[0]), Gate::mz(&[0])], true),
    );
    let mut first = f.spawn(3);
    let mut second = f.spawn(3);
    let Step::Yield(token) = first.poll().unwrap() else {
        panic!("yield expected");
    };
    let Step::Yield(_) = second.poll().unwrap() else {
        panic!("yield expected");
    };
    assert!(second.resume(token).is_err());
    assert!(second.poll().is_err());
    second.reset(3).unwrap();
    assert_eq!(finish(&mut second).unwrap(), [1]);
    first.resume(token).unwrap();
    assert!(first.resume(token).is_err());
    first.reset(3).unwrap();
    assert_eq!(finish(&mut first).unwrap(), [1]);
    let noisy_factory = factory(
        GeneralNoiseModel::builder()
            .with_p1(0.375)
            .with_p_meas_0(0.25),
        envelope(
            &[Gate::h(&[0]), Gate::mz(&[0]), Gate::x(&[0]), Gate::mz(&[0])],
            true,
        ),
    );
    let workers: Vec<_> = (0..4)
        .map(|seed| {
            let f = noisy_factory.clone();
            std::thread::spawn(move || {
                let mut run = f.spawn(seed);
                (
                    seed,
                    finish(&mut run).unwrap(),
                    run.trace.clone(),
                    tail(&run.model),
                    run.completions,
                )
            })
        })
        .collect();
    for worker in workers {
        let (seed, result, trace, rng, completions) = worker.join().unwrap();
        let mut reference = noisy_factory.spawn(seed);
        assert_eq!(result, finish(&mut reference).unwrap());
        assert_eq!(trace, reference.trace);
        assert_eq!(rng, tail(&reference.model));
        assert_eq!(completions, 1);
    }
}

#[derive(Clone)]
struct FailingSimulator;
impl Engine for FailingSimulator {
    type Input = ByteMessage;
    type Output = ByteMessage;
    fn process(&mut self, _: ByteMessage) -> Result<ByteMessage, PecosError> {
        Err(error("synthetic simulator failure"))
    }
    fn reset(&mut self) -> Result<(), PecosError> {
        Err(error("synthetic reset failure"))
    }
}

#[test]
fn simulator_failure_and_partial_reset_cannot_resume_abandoned_execution() {
    let mut plan = envelope(&[Gate::mz(&[0]), Gate::x(&[0]), Gate::mz(&[0])], false);
    plan.records.insert(1, Record::Event(META, 0));
    let f = factory(GeneralNoiseModel::builder(), plan);
    let mut run = f.spawn(0);
    let Step::Yield(token) = run.poll().unwrap() else {
        panic!("yield expected");
    };
    assert_eq!(run.raw, [0]);
    run.resume(token).unwrap();
    run.sim = Box::new(FailingSimulator);
    assert!(run.poll().is_err());
    assert_eq!(run.completions, 0);
    assert!(run.queue.is_empty() && run.raw.is_empty());
    assert!(run.reset(0).is_err());
    assert!(run.poll().is_err());
    run.sim = Box::new(StateVecEngine::new(2));
    run.reset(0).unwrap();
    assert_eq!(finish(&mut run).unwrap(), [0, 1]);
}

#[test]
fn completion_crosstalk_runs_once_after_all_yields() {
    use std::collections::BTreeMap;
    let builder = GeneralNoiseModel::builder()
        .with_p_meas_crosstalk_local(1.0)
        .with_p_meas_crosstalk_model(&BTreeMap::from([
            ("0->1".to_owned(), 1.0),
            ("1->0".to_owned(), 1.0),
        ]));
    let gates = [
        Gate::pz(&[0]),
        Gate::pz(&[1]),
        Gate::mz(&[0]),
        Gate::meas_crosstalk_local_payload(&[1]),
    ];
    let mut annotated = factory(builder.clone(), envelope(&gates, true)).spawn(8);
    let mut plain = factory(builder, envelope(&gates, false)).spawn(8);
    let mut yields = 0;
    loop {
        match annotated.poll().unwrap() {
            Step::Yield(token) => {
                yields += 1;
                assert_eq!(annotated.completions, 0);
                assert!(!annotated.trace.contains(&Gate::x(&[1])));
                annotated.resume(token).unwrap();
            }
            Step::Complete(outcomes) => {
                assert_eq!(outcomes, [0]);
                break;
            }
        }
    }
    assert_eq!(yields, gates.len() + 1);
    assert_eq!(finish(&mut plain).unwrap(), [0]);
    assert_eq!(annotated.trace, plain.trace);
    assert_eq!(tail(&annotated.model), tail(&plain.model));
    assert_eq!(
        annotated
            .trace
            .iter()
            .filter(|g| **g == Gate::x(&[1]))
            .count(),
        1
    );
    assert_eq!(annotated.completions, 1);
    assert!(annotated.poll().is_err());
    annotated.reset(8).unwrap();
    assert_eq!(finish(&mut annotated).unwrap(), [0]);
    assert_eq!(annotated.trace, plain.trace);
}

#[test]
fn metadata_around_physical_yields_preserves_seeded_execution_and_reset() {
    let builder = GeneralNoiseModel::builder()
        .with_p1(0.375)
        .with_p_meas_0(0.25);
    let plan = Envelope {
        version: 2,
        records: vec![
            gate_record(Gate::h(&[0])),
            Record::Event(PHASE, 0),
            gate_record(Gate::h(&[0])),
            gate_record(Gate::mz(&[0])),
            Record::Event(FLIP, 0),
            gate_record(Gate::mz(&[0])),
        ],
    };
    let mut annotated = plan.clone();
    annotated.records = plan
        .records
        .iter()
        .flat_map(|r| [Record::Event(META, 0), r.clone()])
        .collect();
    let plain_factory = factory(builder.clone(), plan);
    let annotated_factory = factory(builder, annotated);
    for seed in 0..64 {
        let mut plain = plain_factory.spawn(seed);
        let mut with = annotated_factory.spawn(seed);
        let expected = finish(&mut plain).unwrap();
        assert_eq!(finish(&mut with).unwrap(), expected);
        assert_eq!(with.trace, plain.trace);
        assert_eq!(tail(&with.model), tail(&plain.model));
        with.reset(seed).unwrap();
        assert_eq!(finish(&mut with).unwrap(), expected);
        assert_eq!(with.trace, plain.trace);
        assert_eq!(tail(&with.model), tail(&plain.model));
    }
}
