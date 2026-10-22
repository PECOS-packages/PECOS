// Copyright 2026 The PECOS Developers
// Licensed under the Apache License, Version 2.0

//! LLVM-independent analysis of QIS text and PECOS runtime helper ABIs.
//!
//! `pecos-phir::qis_parser` lowers instructions into PHIR and collects blocks
//! across definitions into one region, losing function boundaries. It also
//! retains only selected declarations and does not preserve global strings or
//! arbitrary call sites, so it cannot supply this lossless text reader.

mod helpers;
mod reader;
mod result_tags;

pub use helpers::{HelperNormalizationError, PECOS_HELPER_ABIS, normalize_pecos_helper_symbols};
pub use result_tags::{
    ResultTagCertificationError, ResultTagMeasurements, extract_result_tag_measurements,
};
