//! Static, program-wide `cvar_export` selection. Values are resolved only at shot end.
use super::ast::{ArgItem, Operation};
use super::declarations::{DeclarationKind, declaration_kind};
use pecos_core::errors::PecosError;
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};

pub(crate) type ExportMap = BTreeMap<String, String>;

/// Decode the directive without losing malformed field values in serde errors.
pub(crate) fn decode(op: &Value) -> Result<(Vec<String>, Option<Vec<String>>), PecosError> {
    fn strings(op: &Value, field: &str) -> Result<Vec<String>, PecosError> {
        let value = op.get(field).unwrap_or(&Value::Null);
        let error = || {
            PecosError::Input(format!(
                "cvar_export: invalid {field} {value}; supply an array of strings"
            ))
        };
        value
            .as_array()
            .ok_or_else(error)?
            .iter()
            .map(|entry| entry.as_str().map(str::to_owned).ok_or_else(error))
            .collect()
    }
    let variables = strings(op, "variables")?;
    let to = op.get("to").map(|_| strings(op, "to")).transpose()?;
    check_length(&variables, to.as_deref())?;
    Ok((variables, to))
}

fn check_length(variables: &[String], to: Option<&[String]>) -> Result<(), PecosError> {
    if let Some(to) = to
        && to.len() != variables.len()
    {
        return Err(PecosError::Input(format!(
            "cvar_export: invalid to {}; supply one output name for each of {} variables",
            serde_json::json!(to),
            variables.len()
        )));
    }
    Ok(())
}

#[derive(Default)]
struct Collector {
    declarations: BTreeMap<String, DeclarationKind>,
    results: BTreeSet<String>,
    pairs: Vec<(String, String)>,
    present: bool,
}

impl Collector {
    fn export(&mut self, variables: &[String], to: Option<&[String]>) -> Result<(), PecosError> {
        check_length(variables, to)?;
        self.present = true;
        self.pairs.extend(
            to.unwrap_or(variables)
                .iter()
                .cloned()
                .zip(variables.iter().cloned()),
        );
        Ok(())
    }

    fn result(&mut self, returns: &[ArgItem]) {
        for reference in returns {
            match reference {
                ArgItem::Simple(name) | ArgItem::Indexed((name, _)) => {
                    self.results.insert(name.clone());
                }
                _ => {}
            }
        }
    }

    fn finish(self) -> Result<Option<ExportMap>, PecosError> {
        if !self.present {
            return Ok(None);
        }
        let mut outputs = BTreeMap::new();
        for (output, source) in self.pairs {
            let problem = if self.declarations.get(&source) != Some(&DeclarationKind::Classical) {
                Some(format!(
                    "unknown classical variable {source:?}; declare it with cvar_define"
                ))
            } else if output.starts_with("_width_") {
                Some(format!(
                    "reserved output {output:?}; choose a name without the _width_ prefix"
                ))
            } else if output != source && self.declarations.contains_key(&output) {
                Some(format!(
                    "output {output:?} conflicts with a declared variable; choose a different output name"
                ))
            } else if self.results.contains(&output) {
                Some(format!(
                    "output {output:?} conflicts with a Result destination; choose a different output name"
                ))
            } else if outputs
                .get(&output)
                .is_some_and(|previous| previous != &source)
            {
                Some(format!(
                    "output {output:?} has different sources; choose a distinct output name for {source:?}"
                ))
            } else {
                None
            };
            if let Some(problem) = problem {
                return Err(PecosError::Input(format!("cvar_export: {problem}")));
            }
            outputs.insert(output, source);
        }
        Ok(Some(outputs))
    }
}

/// Visit all operations, including unexecuted branches, in source order.
pub(crate) fn all_operations(ops: &[Operation]) -> Vec<&Operation> {
    fn visit<'a>(ops: &'a [Operation], all: &mut Vec<&'a Operation>) {
        for op in ops {
            all.push(op);
            if let Operation::Block {
                ops,
                true_branch,
                false_branch,
                ..
            } = op
            {
                visit(ops, all);
                visit(true_branch.as_deref().unwrap_or_default(), all);
                visit(false_branch.as_deref().unwrap_or_default(), all);
            }
        }
    }
    let mut all = Vec::new();
    visit(ops, &mut all);
    all
}

pub(crate) fn collect(ops: &[Operation]) -> Result<Option<ExportMap>, PecosError> {
    let mut collector = Collector::default();
    for op in all_operations(ops) {
        match op {
            Operation::VariableDefinition { data, variable, .. } => {
                if let Some(kind) = declaration_kind(data) {
                    collector.declarations.insert(variable.clone(), kind);
                }
            }
            Operation::DataExport {
                data,
                variables,
                to,
            } if data == "cvar_export" => {
                collector.export(variables, to.as_deref())?;
            }
            Operation::ClassicalOp { cop, returns, .. } if cop == "Result" => {
                collector.result(returns);
            }
            _ => {}
        }
    }
    collector.finish()
}

pub(crate) fn collect_json(ops: &[Value]) -> Result<Option<ExportMap>, PecosError> {
    fn visit(ops: &[Value], collector: &mut Collector) -> Result<(), PecosError> {
        for op in ops {
            if let Some(data) = op.get("data").and_then(Value::as_str) {
                if let Some(kind) = declaration_kind(data) {
                    let name = super::declarations::declaration_name(data, op.get("variable"))?;
                    collector.declarations.insert(name.to_owned(), kind);
                } else if data == "cvar_export" {
                    let (variables, to) = decode(op)?;
                    collector.export(&variables, to.as_deref())?;
                }
            }
            if op.get("cop").and_then(Value::as_str) == Some("Result") {
                collector.result(&super::classical_validation::result_field(op, "returns")?);
            }
            if op.get("block").is_some() {
                for field in ["ops", "true_branch", "false_branch"] {
                    if let Some(nested) = op.get(field).and_then(Value::as_array) {
                        visit(nested, collector)?;
                    }
                }
            }
        }
        Ok(())
    }
    let mut collector = Collector::default();
    visit(ops, &mut collector)?;
    collector.finish()
}
