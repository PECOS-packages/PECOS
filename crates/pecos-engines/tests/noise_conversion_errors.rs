use pecos_core::errors::PecosError;
use pecos_engines::noise::{
    BiasedDepolarizingNoiseModel, BiasedDepolarizingNoiseModelBuilder, DepolarizingNoiseModel,
    DepolarizingNoiseModelBuilder,
};

fn assert_input<T>(result: Result<T, PecosError>, expected: &str) {
    assert!(matches!(result, Err(PecosError::Input(message)) if message == expected));
}

#[test]
fn fallible_constructors_check_every_probability() {
    for invalid in [-0.1, 1.5, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let expected = &format!(
            "Probability must be between 0.0 and 1.0, got {invalid}; supply a finite probability in 0.0..=1.0"
        );
        assert_input(DepolarizingNoiseModel::try_new_uniform(invalid), expected);
        assert_input(
            BiasedDepolarizingNoiseModel::try_new_uniform(invalid),
            expected,
        );
        for index in 0..4 {
            let mut values = [0.0; 4];
            values[index] = invalid;
            let [prep, meas, p1, p2] = values;
            assert_input(
                DepolarizingNoiseModel::try_new(prep, meas, p1, p2),
                expected,
            );
        }
        for index in 0..5 {
            let mut values = [0.0; 5];
            values[index] = invalid;
            let [prep, meas0, meas1, p1, p2] = values;
            assert_input(
                BiasedDepolarizingNoiseModel::try_new(prep, meas0, meas1, p1, p2),
                expected,
            );
        }
    }
    for valid in [0.0, 1.0] {
        assert!(DepolarizingNoiseModel::try_new_uniform(valid).is_ok());
        assert!(BiasedDepolarizingNoiseModel::try_new_uniform(valid).is_ok());
    }
}

#[test]
fn depolarizing_builder_reports_each_missing_field() {
    let builder = DepolarizingNoiseModelBuilder::new()
        .with_p_meas(0.0)
        .with_p1(0.0)
        .with_p2(0.0);
    assert_input(
        builder.validate_configuration(),
        "Preparation probability must be set",
    );
    assert_input(builder.try_build(), "Preparation probability must be set");
    let builder = DepolarizingNoiseModelBuilder::new()
        .with_p_prep(0.0)
        .with_p1(0.0)
        .with_p2(0.0);
    assert_input(
        builder.validate_configuration(),
        "Measurement probability must be set",
    );
    assert_input(builder.try_build(), "Measurement probability must be set");
    let builder = DepolarizingNoiseModelBuilder::new()
        .with_p_prep(0.0)
        .with_p_meas(0.0)
        .with_p2(0.0);
    assert_input(
        builder.validate_configuration(),
        "Single-qubit probability must be set",
    );
    assert_input(builder.try_build(), "Single-qubit probability must be set");
    let builder = DepolarizingNoiseModelBuilder::new()
        .with_p_prep(0.0)
        .with_p_meas(0.0)
        .with_p1(0.0);
    assert_input(
        builder.validate_configuration(),
        "Two-qubit probability must be set",
    );
    assert_input(builder.try_build(), "Two-qubit probability must be set");
}

#[test]
#[should_panic(expected = "Probability must be between 0.0 and 1.0")]
fn depolarizing_new_keeps_panic_message() {
    let _ = DepolarizingNoiseModel::new(1.5, 0.0, 0.0, 0.0);
}

#[test]
#[should_panic(expected = "Probability must be between 0.0 and 1.0")]
fn depolarizing_uniform_keeps_panic_message() {
    let _ = DepolarizingNoiseModel::new_uniform(-0.1);
}

#[test]
#[should_panic(expected = "Probability must be between 0.0 and 1.0")]
fn depolarizing_build_invalid_keeps_panic_message() {
    let _ = DepolarizingNoiseModelBuilder::new()
        .with_uniform_probability(f64::NAN)
        .build();
}

#[test]
#[should_panic(expected = "Preparation probability must be set")]
fn depolarizing_build_unset_keeps_panic_message() {
    let _ = DepolarizingNoiseModelBuilder::new().build();
}

#[test]
fn biased_builder_reports_each_missing_field() {
    let builder = BiasedDepolarizingNoiseModelBuilder::new()
        .with_p_meas_0(0.0)
        .with_p_meas_1(0.0)
        .with_p1(0.0)
        .with_p2(0.0);
    assert_input(
        builder.validate_configuration(),
        "Preparation probability must be set",
    );
    assert_input(builder.try_build(), "Preparation probability must be set");
    let builder = BiasedDepolarizingNoiseModelBuilder::new()
        .with_p_prep(0.0)
        .with_p_meas_1(0.0)
        .with_p1(0.0)
        .with_p2(0.0);
    assert_input(
        builder.validate_configuration(),
        "Measurement 0->1 flip probability must be set",
    );
    assert_input(
        builder.try_build(),
        "Measurement 0->1 flip probability must be set",
    );
    let builder = BiasedDepolarizingNoiseModelBuilder::new()
        .with_p_prep(0.0)
        .with_p_meas_0(0.0)
        .with_p1(0.0)
        .with_p2(0.0);
    assert_input(
        builder.validate_configuration(),
        "Measurement 1->0 flip probability must be set",
    );
    assert_input(
        builder.try_build(),
        "Measurement 1->0 flip probability must be set",
    );
    let builder = BiasedDepolarizingNoiseModelBuilder::new()
        .with_p_prep(0.0)
        .with_p_meas_0(0.0)
        .with_p_meas_1(0.0)
        .with_p2(0.0);
    assert_input(
        builder.validate_configuration(),
        "Single-qubit probability must be set",
    );
    assert_input(builder.try_build(), "Single-qubit probability must be set");
    let builder = BiasedDepolarizingNoiseModelBuilder::new()
        .with_p_prep(0.0)
        .with_p_meas_0(0.0)
        .with_p_meas_1(0.0)
        .with_p1(0.0);
    assert_input(
        builder.validate_configuration(),
        "Two-qubit probability must be set",
    );
    assert_input(builder.try_build(), "Two-qubit probability must be set");
}

#[test]
#[should_panic(expected = "Probability must be between 0.0 and 1.0")]
fn biased_new_keeps_panic_message() {
    let _ = BiasedDepolarizingNoiseModel::new(1.5, 0.0, 0.0, 0.0, 0.0);
}

#[test]
#[should_panic(expected = "Probability must be between 0.0 and 1.0")]
fn biased_uniform_keeps_panic_message() {
    let _ = BiasedDepolarizingNoiseModel::new_uniform(-0.1);
}

#[test]
#[should_panic(expected = "Probability must be between 0.0 and 1.0")]
fn biased_build_invalid_keeps_panic_message() {
    let _ = BiasedDepolarizingNoiseModelBuilder::new()
        .with_uniform_probability(f64::NAN)
        .build();
}

#[test]
#[should_panic(expected = "Preparation probability must be set")]
fn biased_build_unset_keeps_panic_message() {
    let _ = BiasedDepolarizingNoiseModelBuilder::new().build();
}
