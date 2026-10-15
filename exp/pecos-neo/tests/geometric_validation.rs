use pecos_neo::noise::composite::batch::{GeometricSampler, geometric};

#[test]
fn sampler_invalid_probabilities_return_errors() {
    for probability in [-0.1, 1.1, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(GeometricSampler::new(probability).is_err());
    }
}

#[test]
fn geometric_invalid_probabilities_return_errors() {
    for probability in [-0.1, 1.1, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(geometric(probability).is_err());
    }
}

#[test]
fn checked_sampler_invalid_probabilities_return_errors() {
    for probability in [-0.1, 1.1, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(GeometricSampler::new_checked(probability).is_err());
    }
}
