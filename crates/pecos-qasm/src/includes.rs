/// Embedded include files for QASM parser
/// This module provides the standard include files as embedded strings
/// so they can be used even when the filesystem paths are not accessible

/// The qelib1.inc file content
pub const QELIB1_INC: &str = include_str!("../includes/qelib1.inc");

/// The pecos.inc file content  
pub const PECOS_INC: &str = include_str!("../includes/pecos.inc");

/// Get all standard virtual includes
pub fn get_standard_includes() -> Vec<(String, String)> {
    vec![
        ("qelib1.inc".to_string(), QELIB1_INC.to_string()),
        ("pecos.inc".to_string(), PECOS_INC.to_string()),
    ]
}