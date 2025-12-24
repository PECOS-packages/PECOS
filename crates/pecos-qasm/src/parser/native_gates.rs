use pecos_core::gate_type::GateType as CoreGateType;

/// Check if a gate name corresponds to a native PECOS gate
/// Note: Only uppercase names are considered native gates
#[must_use]
pub fn parse_native_gate(name: &str) -> Option<CoreGateType> {
    match name {
        "I" => Some(CoreGateType::I),
        "X" => Some(CoreGateType::X),
        "Y" => Some(CoreGateType::Y),
        "Z" => Some(CoreGateType::Z),
        "H" => Some(CoreGateType::H),
        "CX" => Some(CoreGateType::CX),
        "SZZ" => Some(CoreGateType::SZZ),
        "SZZDG" => Some(CoreGateType::SZZdg),
        "RZ" => Some(CoreGateType::RZ),
        "RX" => Some(CoreGateType::RX),
        "RY" => Some(CoreGateType::RY),
        "RZZ" => Some(CoreGateType::RZZ),
        "R1XY" => Some(CoreGateType::R1XY),
        "U" => Some(CoreGateType::U),
        _ => None,
    }
}

/// Check if a name is a native operation (including special ops)
#[must_use]
pub fn is_native_operation(name: &str) -> bool {
    parse_native_gate(name).is_some()
        || matches!(
            name.to_lowercase().as_str(),
            "barrier" | "reset" | "measure" | "opaque"
        )
}

/// Check if the gate name requires uppercase (native gates should be uppercase)
#[must_use]
pub fn requires_uppercase(name: &str) -> bool {
    parse_native_gate(name).is_some()
}

/// Get the canonical (uppercase) name for a native gate
#[must_use]
pub fn canonical_gate_name(name: &str) -> String {
    if requires_uppercase(name) {
        name.to_uppercase()
    } else {
        name.to_string()
    }
}
