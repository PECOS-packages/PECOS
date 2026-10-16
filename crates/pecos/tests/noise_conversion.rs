#![cfg(feature = "runtime")]

#[test]
fn sim_reports_recorded_noise_setter_error() {
    use pecos::{SimStack, sim};
    use pecos_core::errors::PecosError;
    use pecos_engines::noise::GeneralNoiseModelBuilder;
    use pecos_programs::Qasm;

    let error = sim(Qasm::from_string("OPENQASM 2.0; qreg q[1];"))
        .stack(SimStack::Engines)
        .noise(GeneralNoiseModelBuilder::new().with_p1(1.5))
        .build()
        .err()
        .expect("invalid noise must fail to build");
    assert!(matches!(error, PecosError::Input(message)
        if message == "Probability must be between 0 and 1, got 1.5"));
}

#[cfg(feature = "neo")]
#[test]
fn neo_reports_invalid_depolarizing_builder() {
    let error = pecos::sim(pecos_programs::Qasm::from_string(
        "OPENQASM 2.0; qreg q[1];",
    ))
    .stack(pecos::SimStack::Neo)
    .noise(pecos_engines::noise::DepolarizingNoiseModelBuilder::new().with_uniform_probability(1.5))
    .shots(1)
    .run()
    .expect_err("invalid noise must fail");
    assert!(
        matches!(error, pecos_core::errors::PecosError::Input(message)
        if message == "Probability must be between 0.0 and 1.0, got 1.5; supply a finite probability in 0.0..=1.0")
    );
}

#[cfg(feature = "neo")]
#[test]
fn neo_reports_incomplete_depolarizing_builder() {
    let error = pecos::sim(pecos_programs::Qasm::from_string(
        "OPENQASM 2.0; qreg q[1];",
    ))
    .stack(pecos::SimStack::Neo)
    .noise(pecos_engines::noise::DepolarizingNoiseModelBuilder::new())
    .shots(1)
    .run()
    .expect_err("invalid noise must fail");
    assert!(
        matches!(error, pecos_core::errors::PecosError::Input(message)
        if message == "Preparation probability must be set")
    );
}

#[cfg(feature = "neo")]
#[test]
fn neo_reports_invalid_biased_builder() {
    let error = pecos::sim(pecos_programs::Qasm::from_string(
        "OPENQASM 2.0; qreg q[1];",
    ))
    .stack(pecos::SimStack::Neo)
    .noise(
        pecos_engines::noise::BiasedDepolarizingNoiseModelBuilder::new()
            .with_uniform_probability(-0.1),
    )
    .shots(1)
    .run()
    .expect_err("invalid noise must fail");
    assert!(
        matches!(error, pecos_core::errors::PecosError::Input(message)
        if message == "Probability must be between 0.0 and 1.0, got -0.1; supply a finite probability in 0.0..=1.0")
    );
}

#[cfg(feature = "neo")]
#[test]
fn neo_reports_incomplete_biased_builder() {
    let error = pecos::sim(pecos_programs::Qasm::from_string(
        "OPENQASM 2.0; qreg q[1];",
    ))
    .stack(pecos::SimStack::Neo)
    .noise(pecos_engines::noise::BiasedDepolarizingNoiseModelBuilder::new())
    .shots(1)
    .run()
    .expect_err("invalid noise must fail");
    assert!(
        matches!(error, pecos_core::errors::PecosError::Input(message)
        if message == "Preparation probability must be set")
    );
}

#[cfg(feature = "neo")]
#[test]
fn neo_reports_invalid_depolarizing() {
    let error = pecos::sim(pecos_programs::Qasm::from_string(
        "OPENQASM 2.0; qreg q[1];",
    ))
    .stack(pecos::SimStack::Neo)
    .noise(pecos_engines::DepolarizingNoise { p: 1.5 })
    .shots(1)
    .run()
    .expect_err("invalid noise must fail");
    assert!(
        matches!(error, pecos_core::errors::PecosError::Input(message)
        if message == "Probability must be between 0.0 and 1.0, got 1.5; supply a finite probability in 0.0..=1.0")
    );
}

#[cfg(feature = "neo")]
#[test]
fn neo_reports_invalid_biased() {
    let error = pecos::sim(pecos_programs::Qasm::from_string(
        "OPENQASM 2.0; qreg q[1];",
    ))
    .stack(pecos::SimStack::Neo)
    .noise(pecos_engines::BiasedDepolarizingNoise { p: -0.1 })
    .shots(1)
    .run()
    .expect_err("invalid noise must fail");
    assert!(
        matches!(error, pecos_core::errors::PecosError::Input(message)
        if message == "Probability must be between 0.0 and 1.0, got -0.1; supply a finite probability in 0.0..=1.0")
    );
}

#[cfg(feature = "neo")]
#[test]
fn neo_reports_recorded_general_error() {
    let error = pecos::sim(pecos_programs::Qasm::from_string(
        "OPENQASM 2.0; qreg q[1];",
    ))
    .stack(pecos::SimStack::Neo)
    .noise(pecos_engines::noise::GeneralNoiseModelBuilder::new().with_p1(1.5))
    .shots(1)
    .run()
    .expect_err("invalid noise must fail");
    assert!(
        matches!(error, pecos_core::errors::PecosError::Input(message)
        if message == "Probability must be between 0 and 1, got 1.5")
    );
}
