use pecos_core::errors::PecosError;
use pest::iterators::Pair;

use crate::parser::{Program, QASMParser, Rule};

/// Parse a register declaration
///
/// # Errors
///
/// Returns an error if the register syntax is invalid
pub fn parse_register(pair: Pair<Rule>, program: &mut Program) -> Result<(), PecosError> {
    let inner = pair
        .into_inner()
        .next()
        .ok_or_else(|| QASMParser::error("Empty register declaration"))?;

    match inner.as_rule() {
        Rule::qreg => {
            let indexed_id = inner.into_inner().next().ok_or_else(|| {
                QASMParser::error("Missing indexed identifier in qreg declaration")
            })?;
            let (name, size) = parse_indexed_id(&indexed_id)?;

            // Assign global qubit IDs
            let mut qubit_ids = Vec::new();
            for i in 0..size {
                let global_id = program.total_qubits;
                qubit_ids.push(global_id);
                program.qubit_map.insert(global_id, (name.clone(), i));
                program.total_qubits += 1;
            }

            program.quantum_registers.insert(name, qubit_ids);
        }
        Rule::creg => {
            let indexed_id = inner.into_inner().next().ok_or_else(|| {
                QASMParser::error("Missing indexed identifier in creg declaration")
            })?;
            let (name, size) = parse_indexed_id(&indexed_id)?;
            program.classical_registers.insert(name, size);
        }
        _ => {
            return Err(QASMParser::error(format!(
                "Unexpected register type: {:?}",
                inner.as_rule()
            )));
        }
    }

    Ok(())
}

// Consolidated method for parsing indexed identifiers
/// Parse an indexed identifier (e.g., `q[0]`)
///
/// # Errors
///
/// Returns an error if the indexed identifier syntax is invalid
pub fn parse_indexed_id(pair: &Pair<Rule>) -> Result<(String, usize), PecosError> {
    let content = pair.as_str();

    if let Some(bracket_pos) = content.find('[') {
        let name = content[0..bracket_pos].to_string();
        let size_str = &content[bracket_pos + 1..content.len() - 1];
        let size = size_str
            .parse::<usize>()
            .map_err(|e| PecosError::CompileInvalidRegisterSize(e.to_string()))?;
        Ok((name, size))
    } else {
        Err(PecosError::ParseInvalidExpression(format!(
            "Invalid indexed identifier: {content}"
        )))
    }
}
