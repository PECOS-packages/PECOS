//! The program-wide declaration namespace specified by PHIR-JSON v0.1.

use pecos_core::errors::PecosError;
use std::collections::{BTreeMap, btree_map::Entry};
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DeclarationKind {
    Quantum,
    Classical,
}

impl fmt::Display for DeclarationKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Quantum => "quantum",
            Self::Classical => "classical",
        })
    }
}

#[derive(Debug, Clone, Default)]
pub(crate) struct Declarations(BTreeMap<String, DeclarationKind>);

impl Declarations {
    pub(crate) fn ensure_available(
        &self,
        name: &str,
        kind: DeclarationKind,
    ) -> Result<(), PecosError> {
        if let Some(&existing) = self.0.get(name) {
            return Err(Self::redeclaration_error(name, existing, kind));
        }
        Ok(())
    }

    fn redeclaration_error(
        name: &str,
        existing: DeclarationKind,
        kind: DeclarationKind,
    ) -> PecosError {
        PecosError::Input(format!(
            "Variable '{name}' is already declared as {existing}; cannot redeclare as {kind}"
        ))
    }

    pub(crate) fn register(&mut self, name: &str, kind: DeclarationKind) -> Result<(), PecosError> {
        match self.0.entry(name.to_string()) {
            Entry::Vacant(entry) => {
                entry.insert(kind);
                Ok(())
            }
            Entry::Occupied(entry) => Err(Self::redeclaration_error(name, *entry.get(), kind)),
        }
    }
}

impl Declarations {
    pub(crate) fn validate_operations(
        &mut self,
        ops: &[super::ast::Operation],
    ) -> Result<(), PecosError> {
        use super::ast::Operation;
        for op in ops {
            match op {
                Operation::VariableDefinition { data, variable, .. } => {
                    if let Some(kind) = declaration_kind(data) {
                        self.register(variable, kind)?;
                    }
                }
                Operation::Block {
                    ops,
                    true_branch,
                    false_branch,
                    ..
                } => {
                    self.validate_operations(ops)?;
                    self.validate_operations(true_branch.as_deref().unwrap_or_default())?;
                    self.validate_operations(false_branch.as_deref().unwrap_or_default())?;
                }
                _ => {}
            }
        }
        Ok(())
    }
}

pub(crate) fn validate_operations(ops: &[super::ast::Operation]) -> Result<(), PecosError> {
    Declarations::default().validate_operations(ops)
}

pub(crate) fn validate_json_operations(ops: &[serde_json::Value]) -> Result<(), PecosError> {
    fn visit(ops: &[serde_json::Value], declarations: &mut Declarations) -> Result<(), PecosError> {
        for op in ops {
            if let Some(kind) = op
                .get("data")
                .and_then(serde_json::Value::as_str)
                .and_then(declaration_kind)
                && let Some(name) = op.get("variable").and_then(serde_json::Value::as_str)
            {
                declarations.register(name, kind)?;
            }
            if op.get("block").is_some() {
                for field in ["ops", "true_branch", "false_branch"] {
                    if let Some(nested) = op.get(field).and_then(serde_json::Value::as_array) {
                        visit(nested, declarations)?;
                    }
                }
            }
        }
        Ok(())
    }
    visit(ops, &mut Declarations::default())
}

fn declaration_kind(data: &str) -> Option<DeclarationKind> {
    match data {
        "qvar_define" => Some(DeclarationKind::Quantum),
        "cvar_define" => Some(DeclarationKind::Classical),
        _ => None,
    }
}
