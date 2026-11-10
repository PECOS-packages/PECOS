// Copyright 2026 The PECOS Developers
// Licensed under the Apache License, Version 2.0

//! Shared Python DEM noise signature and argument assembly.

macro_rules! dem_noise_methods {
    ($class:ty, $(#[$attr:meta])* $method:ident, [$($receiver:tt)*], [$($signature:tt)*],
     $noise:ident, $result:ty, $body:block, {$($methods:tt)*}) => {
        #[pyo3::pymethods]
        impl $class {
            $(#[$attr])*
            #[pyo3(signature = ($($signature)* p1=0.001, p2=0.01, p_meas=0.001, p_prep=0.001, p_idle=None, t1=None, t2=None, idle_rz=None, p_idle_linear_rate=None, p_idle_quadratic_rate=None, p_idle_x_linear_rate=None, p_idle_y_linear_rate=None, p_idle_z_linear_rate=None, p_idle_x_quadratic_rate=None, p_idle_y_quadratic_rate=None, p_idle_z_quadratic_rate=None, p_idle_quadratic_sine_rate=None, p_idle_x_quadratic_sine_rate=None, p_idle_y_quadratic_sine_rate=None, p_idle_z_quadratic_sine_rate=None, p1_weights=None, p2_weights=None, p2_replacement_approximation=None, p_meas_crosstalk_local=None, p_meas_crosstalk_global=None, p_meas_crosstalk_model=None, measurement_crosstalk_dem_mode=None, p2_gate_rates=None, p1_gate_rates=None, p_idle_linear=None, p_idle_linear_model=None, p_idle_sin_squared=None, p_idle_sin_squared_model=None, p_idle_coherent=None, p_idle_coherent_model=None))]
            fn $method(
                $($receiver)*,
                p1: f64,
                p2: f64,
                p_meas: f64,
                p_prep: f64,
                p_idle: Option<f64>,
                t1: Option<f64>,
                t2: Option<f64>,
                idle_rz: Option<f64>,
                p_idle_linear_rate: Option<f64>,
                p_idle_quadratic_rate: Option<f64>,
                p_idle_x_linear_rate: Option<f64>,
                p_idle_y_linear_rate: Option<f64>,
                p_idle_z_linear_rate: Option<f64>,
                p_idle_x_quadratic_rate: Option<f64>,
                p_idle_y_quadratic_rate: Option<f64>,
                p_idle_z_quadratic_rate: Option<f64>,
                p_idle_quadratic_sine_rate: Option<f64>,
                p_idle_x_quadratic_sine_rate: Option<f64>,
                p_idle_y_quadratic_sine_rate: Option<f64>,
                p_idle_z_quadratic_sine_rate: Option<f64>,
                p1_weights: Option<BTreeMap<String, f64>>,
                p2_weights: Option<BTreeMap<String, f64>>,
                p2_replacement_approximation: Option<String>,
                p_meas_crosstalk_local: Option<f64>,
                p_meas_crosstalk_global: Option<f64>,
                p_meas_crosstalk_model: Option<BTreeMap<String, f64>>,
                measurement_crosstalk_dem_mode: Option<String>,
                p2_gate_rates: Option<BTreeMap<String, f64>>,
                p1_gate_rates: Option<BTreeMap<String, f64>>,
                p_idle_linear: Option<f64>,
                p_idle_linear_model: Option<BTreeMap<String, f64>>,
                p_idle_sin_squared: Option<f64>,
                p_idle_sin_squared_model: Option<BTreeMap<String, f64>>,
                p_idle_coherent: Option<f64>,
                p_idle_coherent_model: Option<BTreeMap<String, f64>>,
            ) -> pyo3::PyResult<$result> {
                let $noise = pecos_qec::NoiseConfig::try_from(pecos_qec::NoiseOptions {
                    p1,
                    p2,
                    p_meas,
                    p_prep,
                    p_idle,
                    t1,
                    t2,
                    idle_rz,
                    p_idle_linear_rate,
                    p_idle_quadratic_rate,
                    p_idle_x_linear_rate,
                    p_idle_y_linear_rate,
                    p_idle_z_linear_rate,
                    p_idle_x_quadratic_rate,
                    p_idle_y_quadratic_rate,
                    p_idle_z_quadratic_rate,
                    p_idle_quadratic_sine_rate,
                    p_idle_x_quadratic_sine_rate,
                    p_idle_y_quadratic_sine_rate,
                    p_idle_z_quadratic_sine_rate,
                    p1_weights,
                    p2_weights,
                    p2_replacement_approximation,
                    p_meas_crosstalk_local,
                    p_meas_crosstalk_global,
                    p_meas_crosstalk_model,
                    measurement_crosstalk_dem_mode,
                    p2_gate_rates,
                    p1_gate_rates,
                    p_idle_linear,
                    p_idle_linear_model,
                    p_idle_sin_squared,
                    p_idle_sin_squared_model,
                    p_idle_coherent,
                    p_idle_coherent_model,
                })
                .map_err(pyo3::exceptions::PyValueError::new_err)?;
                $body
            }
            $($methods)*
        }
    };
}

pub(crate) use dem_noise_methods;
