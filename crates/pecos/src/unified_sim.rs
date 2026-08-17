//! Unified simulation API with automatic engine selection
//!
//! Convenience wrapper around the lower-level `sim_builder`
//! from pecos-engines, adding automatic engine selection based on program type.

use pecos_core::errors::PecosError;
use pecos_engines::sampling::monte_carlo;
use pecos_engines::{
    ClassicalControlEngineBuilder, MonteCarloBuilder, MonteCarloEngine, SimBuilder, sim_builder,
};
use pecos_programs::Program;
use pecos_qasm::qasm_engine;
#[cfg(feature = "qis")]
use pecos_qis::{IntoQisInterface, qis_engine};

/// Set up a QIS engine with Selene runtime and Helios interface for the given program.
#[cfg(feature = "qis")]
fn build_qis_engine<P: IntoQisInterface + 'static>(
    program: P,
) -> Result<pecos_qis::QisEngineBuilder, PecosError> {
    let selene_runtime = crate::selene_simple_runtime()
        .map_err(|e| PecosError::Generic(format!("Failed to load Selene runtime: {e}")))?;
    let helios_builder = crate::helios_interface_builder();
    qis_engine()
        .runtime(selene_runtime)
        .interface(helios_builder)
        .try_program(program)
        .map_err(|e| PecosError::Generic(format!("Failed to load program: {e}")))
}

/// Which simulation stack executes the program.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SimStack {
    /// The engine/`EngineSystem` stack in `pecos-engines` (current default).
    #[default]
    Engines,
    /// The data-oriented `pecos-neo` stack (experimental).
    ///
    /// Requires building pecos with the `neo` cargo feature. Routes QASM and
    /// HUGR programs with the default quantum backend. HUGR runs through the
    /// PHIR engine, so its results use the same named-register contract as the
    /// engines/QASM path with no Selene/LLVM dependency -- but only for the
    /// PHIR converter's STRAIGHT-LINE subset; HUGR with classical control flow
    /// (loops, conditionals) is rejected (use `SimStack::Engines` for those).
    /// (Note: the engines stack runs HUGR through QIS/Selene, a different and
    /// broader HUGR engine -- a consideration for the eventual default flip.)
    /// The translated noise surface is the depolarizing family
    /// (`PassThroughNoise`, `DepolarizingNoise`, `BiasedDepolarizingNoise`,
    /// and their builders) and the `GeneralNoiseModel` simple-probability
    /// subset, including angle-dependent two-qubit scaling and the
    /// gate-removing spontaneous-emission ratios (with the default uniform
    /// emission distribution). Other noise configurations (leakage, idle,
    /// crosstalk, custom emission distributions, ...), explicit
    /// `.classical()`, and explicit `.quantum()` are not yet translated and
    /// are rejected with an error at `run()`.
    Neo,
}

/// Extension trait for `SimBuilder` to add program-based methods
pub trait SimBuilderExt {
    /// Set the program and automatically select an appropriate engine
    ///
    /// This method inspects the program type and selects:
    /// - QASM programs → QASM engine
    /// - QIS programs → QIS control engine (Selene Helios interface)
    /// - HUGR programs → QIS control engine (Selene Helios interface)
    /// - WASM/WAT programs → Error (not yet supported)
    /// - PHIR JSON programs → Error (not yet supported)
    ///
    /// The engine can be overridden by calling `.classical()` after this method.
    fn program<P: Into<Program>>(self, program: P) -> ProgrammedSimBuilder;
}

impl SimBuilderExt for SimBuilder {
    fn program<P: Into<Program>>(self, program: P) -> ProgrammedSimBuilder {
        ProgrammedSimBuilder {
            base_builder: self,
            program: program.into(),
            override_classical: false,
            stack: SimStack::default(),
            routed: RoutedConfig::default(),
        }
    }
}

/// Config recorded at the facade for routing to the neo stack.
///
/// The engines `SimBuilder` keeps its own copy via the delegating setters;
/// this records what the neo translation needs (values it can map, flags
/// for config it cannot yet map and must reject).
#[derive(Default)]
struct RoutedConfig {
    seed: Option<u64>,
    workers: Option<usize>,
    auto_workers: bool,
    qubits: Option<usize>,
    /// Monte Carlo shot count, set via `.shots(n)` and consumed by the argless
    /// `.run()`. `None` until configured -- `.run()` fails fast rather than
    /// defaulting silently.
    shots: Option<usize>,
    /// The noise config as passed, for translation to the neo stack.
    /// Type-erased because `.noise()` is generic; the neo route downcasts
    /// against the known engines noise types.
    noise: Option<Box<dyn std::any::Any + Send>>,
    quantum_set: bool,
}

/// A simulation builder that has a program set and can auto-select engines
pub struct ProgrammedSimBuilder {
    base_builder: SimBuilder,
    program: Program,
    override_classical: bool,
    stack: SimStack,
    routed: RoutedConfig,
}

impl ProgrammedSimBuilder {
    /// Auto-select the classical engine based on program type, returning a configured `SimBuilder`.
    fn configure_engine(self) -> Result<SimBuilder, PecosError> {
        if self.override_classical {
            return Ok(self.base_builder);
        }

        match self.program {
            Program::Qasm(qasm) => Ok(self.base_builder.classical(qasm_engine().program(qasm))),
            Program::Qis(qis) => {
                #[cfg(feature = "qis")]
                {
                    let engine_builder = build_qis_engine(qis)?;
                    Ok(self.base_builder.classical(engine_builder))
                }
                #[cfg(not(feature = "qis"))]
                {
                    let _ = qis;
                    Err(PecosError::Generic(
                        "QIS programs require Selene and LLVM support. Please rebuild with --features selene,llvm".to_string()
                    ))
                }
            }
            Program::Hugr(hugr) => {
                #[cfg(feature = "qis")]
                {
                    let engine_builder = build_qis_engine(hugr)?;
                    Ok(self.base_builder.classical(engine_builder))
                }
                #[cfg(not(feature = "qis"))]
                {
                    let _ = hugr;
                    Err(PecosError::Generic(
                        "HUGR programs require Selene and LLVM support. Please rebuild with --features selene,llvm".to_string()
                    ))
                }
            }
            Program::Wasm(_) => Err(PecosError::Input(
                "WASM programs are not yet supported in unified simulation".to_string(),
            )),
            Program::Wat(_) => Err(PecosError::Input(
                "WAT programs are not yet supported in unified simulation".to_string(),
            )),
            Program::PhirJson(_) => Err(PecosError::Input(
                "PHIR JSON programs are not yet supported in unified simulation".to_string(),
            )),
            Program::SeleneInterface(_) => Err(PecosError::Input(
                "SeleneInterface programs are not yet supported in unified simulation".to_string(),
            )),
        }
    }

    /// Select which simulation stack executes the program.
    ///
    /// Defaults to [`SimStack::Engines`]. [`SimStack::Neo`] is experimental
    /// and requires the `neo` cargo feature; see [`SimStack`] for the
    /// configuration it can route so far. The result type and contract are
    /// identical on both stacks.
    #[must_use]
    pub fn stack(mut self, stack: SimStack) -> Self {
        self.stack = stack;
        self
    }

    /// Build the simulation with automatic engine selection
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The program type is not yet supported (WASM, WAT, PHIR JSON, `SeleneInterface`)
    /// - Engine building fails
    /// - The neo stack is selected (it has no `MonteCarloEngine`; use
    ///   [`run()`](Self::run) directly)
    pub fn build(self) -> Result<MonteCarloEngine, PecosError> {
        if self.stack == SimStack::Neo {
            return Err(PecosError::Input(
                "The neo stack does not expose a MonteCarloEngine; call .shots(n).run() directly."
                    .to_string(),
            ));
        }
        self.configure_engine()?.build()
    }

    /// Set the number of Monte Carlo shots to run.
    ///
    /// Shorthand for [`sampling(monte_carlo(shots))`](Self::sampling). Shots are
    /// a builder concern, not a `run()` argument: configure the count here, then
    /// call the argless [`run()`](Self::run). Both stacks (and the neo
    /// `sim_neo()` builder) share this `.shots(n).run()` shape.
    #[must_use]
    pub fn shots(self, shots: usize) -> Self {
        self.sampling(monte_carlo(shots))
    }

    /// Set the Monte Carlo sampling strategy (shot count plus optional worker
    /// parallelism), e.g. `.sampling(monte_carlo(1000).workers(8))`.
    ///
    /// [`monte_carlo()`](pecos_engines::sampling::monte_carlo) is the shared
    /// cross-stack run-spec, so the SAME spelling works on both the engines and
    /// neo stacks. The shot count is required; worker settings are applied only
    /// when explicitly configured on the spec, so this never silently overrides
    /// a separate [`workers()`](Self::workers) call unless the spec sets workers
    /// too. (Richer rare-event strategies -- importance sampling, subset
    /// simulation -- are neo-only and configured via `sim_neo()` directly.)
    #[must_use]
    pub fn sampling(mut self, sampling: impl Into<MonteCarloBuilder>) -> Self {
        let mc = sampling.into();
        self.routed.shots = Some(mc.shots());
        // Worker settings are mutually exclusive and last-writer-wins (see
        // `workers`/`auto_workers`): apply only what the spec sets, clearing the
        // other, so the neo route can't end up with both flags live.
        if mc.auto_workers_requested() {
            self.routed.auto_workers = true;
            self.routed.workers = None;
            self.base_builder = self.base_builder.auto_workers();
        } else if let Some(workers) = mc.worker_count() {
            self.routed.workers = Some(workers);
            self.routed.auto_workers = false;
            self.base_builder = self.base_builder.workers(workers);
        }
        self
    }

    /// Build and run the simulation with automatic engine selection.
    ///
    /// The shot count must be configured first via [`shots()`](Self::shots);
    /// `run()` takes no argument and fails fast if no count was set, rather than
    /// defaulting silently.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - No shot count was configured via [`shots()`](Self::shots)
    /// - The program type is not yet supported (WASM, WAT, PHIR JSON, `SeleneInterface`)
    /// - Engine building or running fails
    /// - The neo stack is selected with configuration it cannot route yet
    pub fn run(self) -> Result<pecos_engines::shot_results::ShotVec, PecosError> {
        let shots = self.routed.shots.ok_or_else(|| {
            PecosError::Input(
                "No shot count configured; set one with .shots(n) before .run(). \
                 Example: sim(program).shots(1000).run()."
                    .to_string(),
            )
        })?;
        match self.stack {
            SimStack::Engines => self.configure_engine()?.run(shots),
            SimStack::Neo => self.run_neo(shots),
        }
    }

    /// Run the program on the pecos-neo stack.
    #[cfg(feature = "neo")]
    fn run_neo(self, shots: usize) -> Result<pecos_engines::shot_results::ShotVec, PecosError> {
        use pecos_neo::tool::{monte_carlo, sim_neo, sim_neo_builder};

        if self.override_classical {
            return Err(PecosError::Input(
                "Explicit .classical() engine builders are not yet routed to the neo stack; \
                 remove .classical() or use .stack(SimStack::Engines)."
                    .to_string(),
            ));
        }
        let neo_noise = match &self.routed.noise {
            None => None,
            Some(noise) => map_noise_to_neo(noise.as_ref())?,
        };
        if self.routed.quantum_set {
            return Err(PecosError::Input(
                "Explicit quantum backends are not yet routed to the neo stack (it uses the \
                 default sparse stabilizer); remove .quantum() or use .stack(SimStack::Engines)."
                    .to_string(),
            ));
        }

        let mut sampler = monte_carlo(shots);
        if let Some(workers) = self.routed.workers {
            sampler = sampler.workers(workers);
        }
        if self.routed.auto_workers {
            sampler = sampler.auto_workers();
        }

        // QASM auto-selects the QASM engine. HUGR is routed through the PHIR
        // engine (HUGR -> PHIR), which emits the program's NAMED classical
        // register (e.g. "c") -- matching the engines/QASM result contract --
        // and needs no Selene/LLVM. (neo's own `hugr_engine` would instead emit
        // per-qubit `q0`/`q1` and a `measurements` array, which is not
        // drop-in compatible; the named-register PHIR path is, so it is the one
        // routed here.) The PHIR converter is STRAIGHT-LINE only: HUGR with
        // classical control flow is rejected by `from_hugr_bytes` below (and
        // any residual empty-result shape is caught by the contract guard after
        // `run`).
        let configured = match self.program {
            Program::Qasm(qasm) => sim_neo(qasm).auto(),
            Program::Hugr(hugr) => {
                let phir_engine = pecos_phir::phir_engine()
                    .from_hugr_bytes(&hugr.hugr)
                    .map_err(|e| {
                        PecosError::Generic(format!("Failed to load HUGR program: {e}"))
                    })?;
                sim_neo_builder().with_engine(phir_engine).auto()
            }
            _ => {
                return Err(PecosError::Input(
                    "Only QASM and HUGR programs are routed to the neo stack so far; \
                     use .stack(SimStack::Engines) for other program types."
                        .to_string(),
                ));
            }
        };

        let mut builder = configured.sampling(sampler);
        if let Some(seed) = self.routed.seed {
            builder = builder.seed(seed);
        }
        if let Some(qubits) = self.routed.qubits {
            builder = builder.qubits(qubits);
        }
        if let Some(noise) = neo_noise {
            builder = builder.noise(noise);
        }

        let results = builder.run();
        let shot_vec = results.shots.ok_or_else(|| {
            PecosError::Generic(
                "The neo stack produced no register results for a classical-engine program; \
                 this is a bug in the neo routing."
                    .to_string(),
            )
        })?;

        // Result-contract guard. A HUGR shape the straight-line PHIR converter
        // cannot represent can yield shots with NO register data instead of a
        // clean load error (e.g. an op silently skipped during conversion).
        // Surface that as an error rather than returning empty results that
        // look like a successful run. (QASM always carries its cregs, so this
        // never trips there.)
        if !shot_vec.shots.is_empty() && shot_vec.shots.iter().all(|shot| shot.data.is_empty()) {
            return Err(PecosError::Input(
                "The neo stack produced empty results (no register data) for this program. \
                 If it is a HUGR program, it likely uses features the straight-line PHIR \
                 route does not support; use .stack(SimStack::Engines)."
                    .to_string(),
            ));
        }
        Ok(shot_vec)
    }

    /// Stub when pecos is built without the `neo` feature.
    // `self` is required for signature parity with the feature-enabled variant:
    // the shared call site invokes `self.run_neo(shots)` under both cfgs.
    #[cfg(not(feature = "neo"))]
    #[expect(clippy::unused_self)]
    fn run_neo(self, _shots: usize) -> Result<pecos_engines::shot_results::ShotVec, PecosError> {
        Err(PecosError::Input(
            "pecos was built without the 'neo' cargo feature; rebuild with features = [\"neo\"] \
             to route sim() to the neo stack."
                .to_string(),
        ))
    }
}

/// Translate an engines noise config into the neo stack's noise model.
///
/// Gate and prep conventions are identical on both stacks (uniform X/Y/Z
/// at p1, uniform 15 two-qubit Paulis at p2, X after prep for `p_prep`)
/// and probabilities map one-to-one. Measurement noise differs BY MODEL
/// on the engines side and the mapping preserves each model's physics:
///
/// - The depolarizing family injects a physical X into the state before
///   each measurement (the error persists and propagates — a qubit
///   measured twice without a reset flips at `2p(1-p)` the second time),
///   mapped to neo's `MeasurementStateFlipChannel` via
///   `with_p_meas_state_flip`.
/// - `GeneralNoiseModel` flips only the classical record (the
///   post-measurement state is untouched), mapped to neo's
///   record-flipping `MeasurementChannel` via `with_p_meas`.
///
/// `GeneralNoiseModel` beyond the simple probability subset is NOT
/// mapped: its full configuration (leakage, idle, crosstalk, emission
/// models) is not readable from the built model; configure `sim_neo()`
/// directly with neo's `GeneralNoiseModelBuilder` for those.
///
/// Returns `Ok(None)` for pass-through (no noise).
#[cfg(feature = "neo")]
fn map_noise_to_neo(
    noise: &(dyn std::any::Any + Send),
) -> Result<Option<pecos_neo::noise::GeneralNoiseModelBuilder>, PecosError> {
    use pecos_engines::noise::{
        BiasedDepolarizingNoiseModelBuilder, DepolarizingNoiseModelBuilder,
        PassThroughNoiseModelBuilder,
    };
    use pecos_engines::{BiasedDepolarizingNoise, DepolarizingNoise, PassThroughNoise};
    use pecos_neo::noise::{AngleScaling, GeneralNoiseModelBuilder};

    let uniform = |p_prep: f64, p_meas: f64, p1: f64, p2: f64| {
        GeneralNoiseModelBuilder::new()
            .with_p_prep(p_prep)
            .with_p_meas_state_flip(p_meas)
            .with_p1(p1)
            .with_p2(p2)
    };

    // The biased-depolarizing family applies its measurement bias to the
    // RECORDED outcome AFTER readout (`apply_bias_to_measurement`), never to
    // the state -- the opposite of the plain depolarizing family, which injects
    // a physical X BEFORE measurement. So its measurement maps to neo's
    // record-flipping channel (`with_p_meas`), which also carries the
    // asymmetric `p_meas_0` (0->1) / `p_meas_1` (1->0) bias one-to-one. Gate
    // and prep noise are ordinary uniform depolarizing.
    let biased = |p_prep: f64, p_meas_0: f64, p_meas_1: f64, p1: f64, p2: f64| {
        GeneralNoiseModelBuilder::new()
            .with_p_prep(p_prep)
            .with_p_meas(p_meas_0, p_meas_1)
            .with_p1(p1)
            .with_p2(p2)
    };

    if noise.downcast_ref::<PassThroughNoise>().is_some()
        || noise
            .downcast_ref::<PassThroughNoiseModelBuilder>()
            .is_some()
    {
        return Ok(None);
    }
    if let Some(depolarizing) = noise.downcast_ref::<DepolarizingNoise>() {
        let p = depolarizing.p;
        return Ok(Some(uniform(p, p, p, p)));
    }
    if let Some(builder) = noise.downcast_ref::<DepolarizingNoiseModelBuilder>() {
        // Resolve the configured probabilities via the built model; this
        // enforces the same all-probabilities-set requirement the engines
        // path would.
        let (p_prep, p_meas, p1, p2) = builder.clone().build().probabilities();
        return Ok(Some(uniform(p_prep, p_meas, p1, p2)));
    }
    if let Some(biased_noise) = noise.downcast_ref::<BiasedDepolarizingNoise>() {
        // `BiasedDepolarizingNoise { p }` builds `new_uniform(p)`: every rate is
        // `p`, with symmetric measurement bias.
        let p = biased_noise.p;
        return Ok(Some(biased(p, p, p, p, p)));
    }
    if let Some(builder) = noise.downcast_ref::<BiasedDepolarizingNoiseModelBuilder>() {
        let (p_prep, p_meas_0, p_meas_1, p1, p2) = builder.clone().build().probabilities();
        return Ok(Some(biased(p_prep, p_meas_0, p_meas_1, p1, p2)));
    }
    if let Some(builder) = noise.downcast_ref::<pecos_engines::noise::GeneralNoiseModelBuilder>() {
        // The stored p1/p2 are already in standard depolarizing convention
        // (the with_average_* setters convert on the way in), so they map
        // one-to-one onto neo's builder. Angle-dependent two-qubit scaling and
        // the spontaneous-emission ratios, if present, are translated below;
        // everything else outside the simple Pauli subset is still rejected.
        let Some((p_prep, p_meas_0, p_meas_1, p1, p2, angle, p1_emission, p2_emission)) =
            builder.pauli_with_angle_scaling()
        else {
            return Err(PecosError::Input(
                "This GeneralNoiseModel configuration uses features beyond the simple \
                 probability subset (leakage, seepage, idle, crosstalk, scales, custom \
                 emission distributions, or noiseless gates), which are not yet mapped to \
                 the neo stack. Use .stack(SimStack::Engines) or configure sim_neo() \
                 directly with a neo noise model."
                    .to_string(),
            ));
        };
        // Emission is gate-removing in both stacks with the default uniform
        // emission distribution, so carrying the resolved ratios reproduces it
        // exactly. The ratios are set unconditionally (with the engines-resolved
        // values, defaults included) so neo cannot silently fall back to its own
        // default emission fraction. (Locked by the facade emission differential
        // test in `neo_emission_test.rs`.)
        let mut neo_builder = GeneralNoiseModelBuilder::new()
            .with_p_prep(p_prep)
            .with_p_meas(p_meas_0, p_meas_1)
            .with_p1(p1)
            .with_p2(p2)
            .with_p1_emission_ratio(p1_emission)
            .with_p2_emission_ratio(p2_emission);
        if let Some((a, b, c, d, power)) = angle {
            // Engines' angle-dependent two-qubit error rate is
            //   p2 * (a*|theta/pi|^power + b)  for theta < 0
            //   p2 * (c*|theta/pi|^power + d)  for theta > 0
            // (GeneralNoiseModel::p2_angle_error_rate). neo's AngleScaling
            // evaluates offset + linear*|theta/pi| + scale*|theta/pi|^power per
            // sign, so the engines coefficients map to scale (the power term)
            // and the engines offsets map to offset, with the linear terms
            // zero. This reproduces engines exactly, including the zero-angle
            // (b+d)/2 average. (NOT AngleScaling::from_general_params, which is
            // a different symmetric offset/linear/scale parameterization.)
            //
            // Both stacks read the gate angle as the SIGNED principal value
            // (-pi, pi] -- neo via `to_radians_signed`, engines likewise after
            // its noise call site was aligned with its gate unitaries -- so the
            // sign-dependent coefficients agree cross-stack at every angle
            // (locked by `gnm_angle_scaling_negative_matches`).
            neo_builder = neo_builder
                .with_p2_angle_scaling(AngleScaling::asymmetric(b, 0.0, a, d, 0.0, c, power));
        }
        return Ok(Some(neo_builder));
    }

    Err(PecosError::Input(
        "This noise type is not yet mapped to the neo stack (mapped so far: PassThroughNoise, \
         DepolarizingNoise, DepolarizingNoiseModelBuilder, BiasedDepolarizingNoise, \
         BiasedDepolarizingNoiseModelBuilder, GeneralNoiseModelBuilder's simple \
         probability subset). Remove .noise(), use .stack(SimStack::Engines), or configure \
         sim_neo() directly with a neo noise model."
            .to_string(),
    ))
}

impl ProgrammedSimBuilder {
    /// Override the classical engine selection
    ///
    /// This allows you to specify a different engine than the auto-selected one.
    #[must_use]
    pub fn classical<B: ClassicalControlEngineBuilder + Send + 'static>(
        mut self,
        engine_builder: B,
    ) -> Self
    where
        B::Engine: 'static,
    {
        self.base_builder = self.base_builder.classical(engine_builder);
        self.override_classical = true;
        self
    }

    /// Set the random seed (delegates to base builder)
    #[must_use]
    pub fn seed(mut self, seed: u64) -> Self {
        self.routed.seed = Some(seed);
        self.base_builder = self.base_builder.seed(seed);
        self
    }

    /// Set the number of worker threads (delegates to base builder)
    #[must_use]
    pub fn workers(mut self, workers: usize) -> Self {
        // Last-writer-wins, matching the engines `SimBuilder`: an explicit
        // count clears a prior `.auto_workers()` so the two never both apply on
        // the neo route (where they are resolved separately).
        self.routed.workers = Some(workers);
        self.routed.auto_workers = false;
        self.base_builder = self.base_builder.workers(workers);
        self
    }

    /// Use automatic worker count (delegates to base builder)
    #[must_use]
    pub fn auto_workers(mut self) -> Self {
        self.routed.auto_workers = true;
        self.routed.workers = None;
        self.base_builder = self.base_builder.auto_workers();
        self
    }

    /// Enable verbose output (delegates to base builder)
    #[must_use]
    pub fn verbose(mut self, verbose: bool) -> Self {
        self.base_builder = self.base_builder.verbose(verbose);
        self
    }

    /// Set the noise model (delegates to base builder)
    #[must_use]
    pub fn noise<N>(mut self, noise_builder: N) -> Self
    where
        N: pecos_engines::noise::IntoNoiseModel + Clone + Send + 'static,
    {
        self.routed.noise = Some(Box::new(noise_builder.clone()));
        self.base_builder = self.base_builder.noise(noise_builder);
        self
    }

    /// Set the quantum engine (delegates to base builder)
    #[must_use]
    pub fn quantum<Q>(mut self, quantum_builder: Q) -> Self
    where
        Q: pecos_engines::quantum_engine_builder::IntoQuantumEngineBuilder + 'static,
        Q::Builder: Send + 'static,
    {
        self.routed.quantum_set = true;
        self.base_builder = self.base_builder.quantum(quantum_builder);
        self
    }

    /// Set the number of qubits (delegates to base builder)
    #[must_use]
    pub fn qubits(mut self, num_qubits: usize) -> Self {
        self.routed.qubits = Some(num_qubits);
        self.base_builder = self.base_builder.qubits(num_qubits);
        self
    }
}

/// Create a simulation builder with a program and automatic engine selection
///
/// Primary API for quantum simulations in PECOS.
/// Automatically selects the appropriate classical engine based on the program type.
///
/// # Automatic Engine Selection
///
/// - QASM programs → QASM engine
/// - QIS programs → QIS control engine (Selene Helios interface)
/// - HUGR programs → QIS control engine (Selene Helios interface)
/// - Other formats → Error (not yet supported)
///
/// # Examples
///
/// ```rust,no_run
/// use pecos::sim;
/// use pecos_programs::Qasm;
/// use pecos_engines::{sparse_stab, DepolarizingNoise};
///
/// // Automatic engine selection based on program type
/// let qasm_prog = Qasm::from_string("OPENQASM 2.0; qreg q[1]; h q[0];");
/// let results = sim(qasm_prog)
///     .quantum(sparse_stab())
///     .noise(DepolarizingNoise { p: 0.01 })
///     .seed(42)
///     .shots(100)
///     .run()?;
/// # Ok::<(), pecos_core::errors::PecosError>(())
/// ```
pub fn sim<P: Into<Program>>(program: P) -> ProgrammedSimBuilder {
    ProgrammedSimBuilder {
        base_builder: sim_builder(),
        program: program.into(),
        override_classical: false,
        stack: SimStack::default(),
        routed: RoutedConfig::default(),
    }
}
