//! Compiler-owned control provenance for a deliberately restricted LLVM subset.
//! This certifies source control execution, not physical fault independence or
//! attribution of individual deferred runtime operations to source branches.
mod compiler;
mod state;
pub(crate) use compiler::instrument;
pub(crate) use state::ControlState;
pub use state::{ControlEvent, ControlLoop, ControlSnapshot};
use std::collections::BTreeSet;

pub(crate) const EVENT_KEY: &str = "__pecos_control_v1__";
pub(crate) const MAX_EVENTS: usize = 4096;

#[derive(Clone, Debug, serde::Serialize)]
pub struct ControlSite {
    pub site_id: usize,
    pub block: usize,
    pub kind: String,
    pub scalar_type: String,
    /// Static possible input sites, not a claim of dynamic causal dependence.
    pub input_dependencies: BTreeSet<usize>,
}
#[derive(Clone, Debug, serde::Serialize)]
pub struct ControlManifest {
    pub module_sha256: String,
    pub instrumenter: String,
    pub sites: Vec<ControlSite>,
    pub successors: Vec<Vec<usize>>,
    pub loop_members: Vec<BTreeSet<usize>>,
    pub loop_headers: Vec<usize>,
}
