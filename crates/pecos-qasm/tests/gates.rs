#[path = "gates/native_gates.rs"]
pub mod native_gates;

#[path = "gates/standard_gates.rs"]
pub mod standard_gates;

#[path = "gates/custom_gates.rs"]
pub mod custom_gates;

#[path = "gates/controlled_gates.rs"]
pub mod controlled_gates;

#[path = "gates/rotation_gates.rs"]
pub mod rotation_gates;

#[path = "gates/special_gates.rs"]
pub mod special_gates;

#[path = "gates/expansion.rs"]
mod expansion;

#[path = "gates/identity_and_zero_angle.rs"]
mod identity_and_zero_angle;

// Single test files
#[path = "gates/gate_definition_syntax_test.rs"]
pub mod gate_definition_syntax_test;

#[path = "gates/gate_body_content_test.rs"]
pub mod gate_body_content_test;

#[path = "gates/register_gate_expansion_test.rs"]
pub mod register_gate_expansion_test;

#[path = "gates/simple_gate_test.rs"]
pub mod simple_gate_test;

#[path = "gates/mixed_gates_test.rs"]
pub mod mixed_gates_test;

#[path = "gates/extended_gates_test.rs"]
pub mod extended_gates_test;

#[path = "gates/opaque_gate_test.rs"]
pub mod opaque_gate_test;

#[path = "gates/gate_composition_test.rs"]
pub mod gate_composition_test;
