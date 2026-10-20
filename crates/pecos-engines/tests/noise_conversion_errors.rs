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
            "uniform depolarizing probability must be between 0.0 and 1.0, got {invalid}; supply a finite probability in 0.0..=1.0"
        );
        assert_input(DepolarizingNoiseModel::try_new_uniform(invalid), expected);
        assert_input(
            BiasedDepolarizingNoiseModel::try_new_uniform(invalid),
            &expected.replace("uniform depolarizing", "uniform biased depolarizing"),
        );
        for (index, name) in ["p_prep", "p_meas", "p1", "p2"].into_iter().enumerate() {
            let mut values = [0.0; 4];
            values[index] = invalid;
            let [prep, meas, p1, p2] = values;
            assert_input(
                DepolarizingNoiseModel::try_new(prep, meas, p1, p2),
                &expected.replace("uniform depolarizing probability", name),
            );
        }
        for (index, name) in ["p_prep", "p_meas_0", "p_meas_1", "p1", "p2"]
            .into_iter()
            .enumerate()
        {
            let mut values = [0.0; 5];
            values[index] = invalid;
            let [prep, meas0, meas1, p1, p2] = values;
            assert_input(
                BiasedDepolarizingNoiseModel::try_new(prep, meas0, meas1, p1, p2),
                &expected.replace("uniform depolarizing probability", name),
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
#[should_panic(expected = "p_prep must be between 0.0 and 1.0")]
fn depolarizing_new_keeps_panic_message() {
    let _ = DepolarizingNoiseModel::new(1.5, 0.0, 0.0, 0.0);
}

#[test]
#[should_panic(expected = "uniform depolarizing probability must be between 0.0 and 1.0")]
fn depolarizing_uniform_keeps_panic_message() {
    let _ = DepolarizingNoiseModel::new_uniform(-0.1);
}

#[test]
#[should_panic(expected = "uniform depolarizing probability must be between 0.0 and 1.0")]
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
#[should_panic(expected = "p_prep must be between 0.0 and 1.0")]
fn biased_new_keeps_panic_message() {
    let _ = BiasedDepolarizingNoiseModel::new(1.5, 0.0, 0.0, 0.0, 0.0);
}

#[test]
#[should_panic(expected = "uniform biased depolarizing probability must be between 0.0 and 1.0")]
fn biased_uniform_keeps_panic_message() {
    let _ = BiasedDepolarizingNoiseModel::new_uniform(-0.1);
}

#[test]
#[should_panic(expected = "uniform biased depolarizing probability must be between 0.0 and 1.0")]
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

#[test]
fn probability_errors_name_each_parameter() {
    let names = ["p_prep", "p_meas", "p1", "p2"];
    for (index, name) in names.into_iter().enumerate() {
        let mut p = [0.0; 4];
        p[index] = f64::NAN;
        let error = pecos_engines::noise::DepolarizingNoiseModel::try_new(p[0], p[1], p[2], p[3])
            .err()
            .unwrap()
            .to_string();
        assert!(error.contains(name) && error.contains("NaN"), "{error}");
    }
    let names = ["p_prep", "p_meas_0", "p_meas_1", "p1", "p2"];
    for (index, name) in names.into_iter().enumerate() {
        let mut p = [0.0; 5];
        p[index] = 1.5;
        let error = pecos_engines::noise::BiasedDepolarizingNoiseModel::try_new(
            p[0], p[1], p[2], p[3], p[4],
        )
        .err()
        .unwrap()
        .to_string();
        assert!(error.contains(name) && error.contains("1.5"), "{error}");
    }
}

#[test]
fn uniform_builders_preserve_the_callers_parameter_name() {
    for invalid in [-0.1, 1.5, f64::NAN, f64::INFINITY] {
        let expected = format!(
            "uniform depolarizing probability must be between 0.0 and 1.0, got {invalid}; supply a finite probability in 0.0..=1.0"
        );
        let builder = DepolarizingNoiseModelBuilder::new().with_uniform_probability(invalid);
        assert_input(builder.validate_configuration(), &expected);
        assert_input(builder.clone().try_build(), &expected);
        // A later setter must not erase an invalid earlier configuration.
        assert_input(builder.with_uniform_probability(0.1).try_build(), &expected);
        let expected = expected.replace("uniform depolarizing", "uniform biased depolarizing");
        let builder = BiasedDepolarizingNoiseModelBuilder::new().with_uniform_probability(invalid);
        assert_input(builder.validate_configuration(), &expected);
        assert_input(builder.clone().try_build(), &expected);
        assert_input(builder.with_uniform_probability(0.1).try_build(), &expected);
    }
}
