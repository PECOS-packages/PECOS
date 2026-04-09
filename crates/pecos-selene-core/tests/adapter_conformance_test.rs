//! Verify the `SeleneAdapter` works with Selene's own conformance test suite.
//!
//! Tests both rotation-capable (`StateVec`) and Clifford-only (Stabilizer) paths.

use anyhow::{Result, anyhow};
use pecos_core::{Angle64, QubitId};
use pecos_selene_core::{SeleneAdapter, SeleneSimBehavior, to_usize};
use pecos_simulators::{ArbitraryRotationGateable, CliffordGateable, Stabilizer, StateVec};
use selene_core::simulator::conformance_testing::run_basic_tests;
use selene_core::simulator::interface::SimulatorInterfaceFactory;
use std::sync::Arc;

// ============================================================================
// StateVec behavior (rotation-capable)
// ============================================================================

struct StateVecBehavior {
    sim: StateVec,
}

impl SeleneSimBehavior for StateVecBehavior {
    type Sim = StateVec;

    fn create_sim(&self, num_qubits: usize, seed: u64) -> StateVec {
        StateVec::with_seed(num_qubits, seed)
    }

    fn sim_mut(&mut self) -> &mut StateVec {
        &mut self.sim
    }

    fn apply_rxy(&mut self, qubit: QubitId, theta: f64, phi: f64) -> Result<()> {
        self.sim
            .rz(Angle64::from_radians(-phi), &[qubit])
            .rx(Angle64::from_radians(theta), &[qubit])
            .rz(Angle64::from_radians(phi), &[qubit]);
        Ok(())
    }

    fn apply_rz(&mut self, qubit: QubitId, theta: f64) -> Result<()> {
        self.sim.rz(Angle64::from_radians(theta), &[qubit]);
        Ok(())
    }

    fn apply_rzz(&mut self, q1: QubitId, q2: QubitId, theta: f64) -> Result<()> {
        self.sim.rzz(Angle64::from_radians(theta), &[(q1, q2)]);
        Ok(())
    }

    fn reset_qubit(&mut self, qubit: QubitId) -> Result<()> {
        let results = self.sim.mz(&[qubit]);
        if results[0].outcome {
            self.sim.x(&[qubit]);
        }
        Ok(())
    }
}

struct StateVecAdapterFactory;

impl SimulatorInterfaceFactory for StateVecAdapterFactory {
    type Interface = SeleneAdapter<StateVecBehavior>;

    fn init(
        self: Arc<Self>,
        n_qubits: u64,
        args: &[impl AsRef<str>],
    ) -> Result<Box<Self::Interface>> {
        let args: Vec<String> = args.iter().map(|s| s.as_ref().to_string()).collect();
        if args.len() > 1 {
            anyhow::bail!("Expected no arguments, got {:?}", &args[1..]);
        }

        Ok(Box::new(SeleneAdapter {
            behavior: StateVecBehavior {
                sim: StateVec::with_seed(to_usize(n_qubits), 0),
            },
            num_qubits: n_qubits,
        }))
    }
}

#[test]
fn selene_conformance_statevec() {
    let factory = Arc::new(StateVecAdapterFactory);
    let args: Vec<String> = vec![String::new()];
    run_basic_tests(factory, args);
}

// ============================================================================
// Stabilizer behavior (Clifford-only, angle approximation)
// ============================================================================

/// Angle approximation result for Clifford rotations (multiples of pi/2).
enum ApproxAngle {
    Zero,
    FracPi2,
    Pi,
    Frac3Pi2,
    NotClifford,
}

fn approximate_angle(theta: f64, threshold: f64) -> ApproxAngle {
    let quadrant_float = theta * 2.0 / std::f64::consts::PI;
    let quadrant_rounded = quadrant_float.round();
    let within_threshold = (quadrant_float - quadrant_rounded).abs() < threshold;

    let quadrant_mod4 = quadrant_rounded.rem_euclid(4.0);
    let quadrant = if (quadrant_mod4 - 0.0).abs() < 0.5 {
        0
    } else if (quadrant_mod4 - 1.0).abs() < 0.5 {
        1
    } else if (quadrant_mod4 - 2.0).abs() < 0.5 {
        2
    } else {
        3
    };

    match (within_threshold, quadrant) {
        (true, 0) => ApproxAngle::Zero,
        (true, 1) => ApproxAngle::FracPi2,
        (true, 2) => ApproxAngle::Pi,
        (true, 3) => ApproxAngle::Frac3Pi2,
        _ => ApproxAngle::NotClifford,
    }
}

struct StabilizerBehavior {
    sim: Stabilizer,
    angle_threshold: f64,
}

impl StabilizerBehavior {
    fn apply_rz_clifford(&mut self, qubit: QubitId, theta: f64) -> Result<()> {
        match approximate_angle(theta, self.angle_threshold) {
            ApproxAngle::Zero => {}
            ApproxAngle::FracPi2 => {
                self.sim.sz(&[qubit]);
            }
            ApproxAngle::Pi => {
                self.sim.z(&[qubit]);
            }
            ApproxAngle::Frac3Pi2 => {
                self.sim.szdg(&[qubit]);
            }
            ApproxAngle::NotClifford => {
                return Err(anyhow!("RZ({theta}) is not a Clifford rotation"));
            }
        }
        Ok(())
    }

    fn apply_rx_clifford(&mut self, qubit: QubitId, theta: f64) -> Result<()> {
        match approximate_angle(theta, self.angle_threshold) {
            ApproxAngle::Zero => {}
            ApproxAngle::FracPi2 => {
                self.sim.sx(&[qubit]);
            }
            ApproxAngle::Pi => {
                self.sim.x(&[qubit]);
            }
            ApproxAngle::Frac3Pi2 => {
                self.sim.sxdg(&[qubit]);
            }
            ApproxAngle::NotClifford => {
                return Err(anyhow!("RX({theta}) is not a Clifford rotation"));
            }
        }
        Ok(())
    }
}

impl SeleneSimBehavior for StabilizerBehavior {
    type Sim = Stabilizer;

    fn create_sim(&self, num_qubits: usize, seed: u64) -> Stabilizer {
        Stabilizer::with_seed(num_qubits, seed)
    }

    fn sim_mut(&mut self) -> &mut Stabilizer {
        &mut self.sim
    }

    fn apply_rxy(&mut self, qubit: QubitId, theta: f64, phi: f64) -> Result<()> {
        // RXY(theta, phi) = Rz(phi) * Rx(theta) * Rz(-phi)
        // Applied left-to-right in code: Rz(-phi) first, then Rx(theta), then Rz(phi)
        self.apply_rz_clifford(qubit, -phi)?;
        self.apply_rx_clifford(qubit, theta)?;
        self.apply_rz_clifford(qubit, phi)?;
        Ok(())
    }

    fn apply_rz(&mut self, qubit: QubitId, theta: f64) -> Result<()> {
        self.apply_rz_clifford(qubit, theta)
    }

    fn apply_rzz(&mut self, q1: QubitId, q2: QubitId, theta: f64) -> Result<()> {
        match approximate_angle(theta, self.angle_threshold) {
            ApproxAngle::Zero => {}
            ApproxAngle::FracPi2 => {
                self.sim.szz(&[(q1, q2)]);
            }
            ApproxAngle::Pi => {
                self.sim.z(&[q1]).z(&[q2]);
            }
            ApproxAngle::Frac3Pi2 => {
                self.sim.szzdg(&[(q1, q2)]);
            }
            ApproxAngle::NotClifford => {
                return Err(anyhow!("RZZ({theta}) is not a Clifford rotation"));
            }
        }
        Ok(())
    }

    fn reset_qubit(&mut self, qubit: QubitId) -> Result<()> {
        self.sim.mpz(&[qubit]);
        Ok(())
    }

    fn postselect(&mut self, qubit: QubitId, target: bool) -> Result<()> {
        let results = self.sim.mz(&[qubit]);
        if results[0].outcome != target {
            if results[0].is_deterministic {
                return Err(anyhow!(
                    "Postselect failed: outcome was deterministically {} but target was {target}",
                    results[0].outcome
                ));
            }
            return Err(anyhow!(
                "Postselect failed: measurement collapsed to {} but target was {target}",
                results[0].outcome
            ));
        }
        Ok(())
    }
}

struct StabilizerAdapterFactory;

impl SimulatorInterfaceFactory for StabilizerAdapterFactory {
    type Interface = SeleneAdapter<StabilizerBehavior>;

    fn init(
        self: Arc<Self>,
        n_qubits: u64,
        args: &[impl AsRef<str>],
    ) -> Result<Box<Self::Interface>> {
        // Parse --angle-threshold from args
        use clap::Parser;

        #[derive(Parser)]
        struct Params {
            #[arg(long)]
            angle_threshold: f64,
        }

        let args: Vec<String> = args.iter().map(|s| s.as_ref().to_string()).collect();
        let params = Params::try_parse_from(&args)
            .map_err(|e| anyhow!("Error parsing stabilizer args: {e}"))?;

        Ok(Box::new(SeleneAdapter {
            behavior: StabilizerBehavior {
                sim: Stabilizer::with_seed(to_usize(n_qubits), 0),
                angle_threshold: params.angle_threshold,
            },
            num_qubits: n_qubits,
        }))
    }
}

#[test]
fn selene_conformance_stabilizer() {
    let factory = Arc::new(StabilizerAdapterFactory);
    let args = vec![String::new(), "--angle-threshold=0.001".to_string()];
    run_basic_tests(factory, args);
}
