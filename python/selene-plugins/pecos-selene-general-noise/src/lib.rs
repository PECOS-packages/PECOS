//! Selene error-model plugin exporting PECOS general noise.
//!
//! The model lives in `pecos-selene-general-noise-model`; this crate only
//! provides the FFI factory, so other plugins can reuse the model without
//! linking a second set of Selene entry points.

use std::sync::Arc;

use anyhow::{Context, Result, bail};
use pecos_selene_general_noise_model::GeneralNoiseErrorModel;
use selene_core::error_model::interface::ErrorModelInterfaceFactory;
use selene_core::export_error_model_plugin;

#[derive(Default)]
struct GeneralNoiseFactory;

impl ErrorModelInterfaceFactory for GeneralNoiseFactory {
    type Interface = GeneralNoiseErrorModel;

    fn init(
        self: Arc<Self>,
        n_qubits: u64,
        error_model_args: &[impl AsRef<str>],
    ) -> Result<Box<Self::Interface>> {
        if error_model_args.len() != 2 {
            bail!(
                "PECOS general noise expects one JSON configuration argument, got {}",
                error_model_args.len().saturating_sub(1)
            );
        }
        Ok(Box::new(GeneralNoiseErrorModel::from_json(
            usize::try_from(n_qubits).context("qubit count does not fit usize")?,
            error_model_args[1].as_ref(),
        )?))
    }
}

export_error_model_plugin!(crate::GeneralNoiseFactory);
