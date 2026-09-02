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
        "S" | "SZ" => Some(CoreGateType::SZ),
        "SDG" | "SZDG" | "SZdg" => Some(CoreGateType::SZdg),
        "T" => Some(CoreGateType::T),
        "TDG" | "Tdg" => Some(CoreGateType::Tdg),
        "SX" => Some(CoreGateType::SX),
        "SXDG" | "SXdg" => Some(CoreGateType::SXdg),
        "SY" => Some(CoreGateType::SY),
        "SYDG" | "SYdg" => Some(CoreGateType::SYdg),
        "CX" => Some(CoreGateType::CX),
        "CY" => Some(CoreGateType::CY),
        "CZ" => Some(CoreGateType::CZ),
        "CH" => Some(CoreGateType::CH),
        "SWAP" => Some(CoreGateType::SWAP),
        "SXX" => Some(CoreGateType::SXX),
        "SXXDG" | "SXXdg" => Some(CoreGateType::SXXdg),
        "SYY" => Some(CoreGateType::SYY),
        "SYYDG" | "SYYdg" => Some(CoreGateType::SYYdg),
        "SZZ" => Some(CoreGateType::SZZ),
        "SZZDG" | "SZZdg" => Some(CoreGateType::SZZdg),
        "RZ" => Some(CoreGateType::RZ),
        "RX" => Some(CoreGateType::RX),
        "RY" => Some(CoreGateType::RY),
        "RZZ" => Some(CoreGateType::RZZ),
        "RXY1Q" | "R1XY" => Some(CoreGateType::RXY1Q),
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

#[cfg(test)]
mod tests {
    use super::*;

    const fn is_qasm_native_gate(gate: CoreGateType) -> bool {
        match gate {
            CoreGateType::I
            | CoreGateType::X
            | CoreGateType::Y
            | CoreGateType::Z
            | CoreGateType::SX
            | CoreGateType::SXdg
            | CoreGateType::SY
            | CoreGateType::SYdg
            | CoreGateType::SZ
            | CoreGateType::SZdg
            | CoreGateType::H
            | CoreGateType::RX
            | CoreGateType::RY
            | CoreGateType::RZ
            | CoreGateType::T
            | CoreGateType::Tdg
            | CoreGateType::U
            | CoreGateType::RXY1Q
            | CoreGateType::CX
            | CoreGateType::CY
            | CoreGateType::CZ
            | CoreGateType::CH
            | CoreGateType::SXX
            | CoreGateType::SXXdg
            | CoreGateType::SYY
            | CoreGateType::SYYdg
            | CoreGateType::SZZ
            | CoreGateType::SZZdg
            | CoreGateType::SWAP
            | CoreGateType::RZZ => true,
            CoreGateType::F
            | CoreGateType::Fdg
            | CoreGateType::RXX
            | CoreGateType::RYY
            | CoreGateType::RXXRYYRZZ
            | CoreGateType::U2q
            | CoreGateType::CCX
            | CoreGateType::MX
            | CoreGateType::MZ
            | CoreGateType::MeasureLeaked
            | CoreGateType::MeasureFree
            | CoreGateType::MPZ
            | CoreGateType::PX
            | CoreGateType::PZ
            | CoreGateType::QAlloc
            | CoreGateType::QFree
            | CoreGateType::Idle
            | CoreGateType::TrackedPauliMeta
            | CoreGateType::MeasCrosstalkGlobalPayload
            | CoreGateType::MeasCrosstalkLocalPayload
            | CoreGateType::Channel
            | CoreGateType::Custom => false,
        }
    }

    #[test]
    fn canonical_names_round_trip_for_every_qasm_native_gate() {
        for gate_id in u8::MIN..=u8::MAX {
            let Ok(gate) = CoreGateType::try_from(gate_id) else {
                continue;
            };
            if is_qasm_native_gate(gate) {
                assert_eq!(parse_native_gate(&gate.to_string()), Some(gate));
            }
        }
    }

    #[test]
    fn legacy_uppercase_szz_dagger_name_remains_supported() {
        assert_eq!(parse_native_gate("SZZDG"), Some(CoreGateType::SZZdg));
    }
}
