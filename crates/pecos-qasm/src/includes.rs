/// Embedded include files for QASM parser
///
/// This module provides the standard include files as embedded strings
/// so they can be used even when the filesystem paths are not accessible
///
/// The qelib1.inc file content
pub const QELIB1_INC: &str = include_str!("../includes/qelib1.inc");

/// The pecos.inc file content
pub const PECOS_INC: &str = include_str!("../includes/pecos.inc");

/// Get all standard virtual includes
#[must_use]
pub fn get_standard_includes() -> Vec<(&'static str, &'static str)> {
    vec![("qelib1.inc", QELIB1_INC), ("pecos.inc", PECOS_INC)]
}
