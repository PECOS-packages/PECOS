#[path = "features/includes.rs"]
pub mod includes;

#[path = "features/comments.rs"]
pub mod comments;

#[path = "features/parameters.rs"]
pub mod parameters;

#[path = "features/feature_flags.rs"]
pub mod feature_flags;

// Single test files
#[path = "features/custom_include_paths_test.rs"]
pub mod custom_include_paths_test;

#[path = "features/debug_includes.rs"]
pub mod debug_includes;

#[path = "features/virtual_includes_test.rs"]
pub mod virtual_includes_test;

#[path = "features/empty_param_list_test.rs"]
pub mod empty_param_list_test;

#[path = "features/qasm_feature_showcase_test.rs"]
pub mod qasm_feature_showcase_test;

#[path = "features/scientific_notation_test.rs"]
pub mod scientific_notation_test;

#[path = "features/binary_ops_test.rs"]
pub mod binary_ops_test;

#[path = "features/power_operator_test.rs"]
pub mod power_operator_test;

#[path = "features/comparison_operators_debug_test.rs"]
pub mod comparison_operators_debug_test;
