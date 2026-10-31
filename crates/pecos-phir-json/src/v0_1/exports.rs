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
            let value = if value.is_null() {
                "<missing or null>".to_owned()
            } else {
                value.to_string()
            };
            PecosError::Input(format!(
                "cvar_export: invalid {field} {value}; supply an array of strings"
            ))
        };
        value
            .as_array()
            .ok_or_else(error)?
            .iter()
            .map(|entry| {
                if entry.is_null() {
                    return Err(PecosError::Input(format!(
                        "cvar_export: invalid {field} entry <missing or null>; supply an array of strings"
                    )));
                }
                entry.as_str().map(str::to_owned).ok_or_else(error)
            })
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
    nested_declaration: Option<(String, String)>,
}

impl Collector {
    fn declare(&mut self, data: &str, name: &str, nested: bool) {
        if let Some(kind) = declaration_kind(data) {
            self.declarations.insert(name.to_owned(), kind);
            if nested && self.nested_declaration.is_none() {
                self.nested_declaration = Some((data.to_owned(), name.to_owned()));
            }
        }
    }

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
            if let Some((data, name)) = self.nested_declaration {
                return Err(PecosError::Input(format!(
                    "Nested declaration {data} {name:?}: move the declaration to the top level; declarations inside blocks are only supported in programs that use cvar_export"
                )));
            }
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
            } else if output != source && self.results.contains(&output) {
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

/// Visit operations by reference, retaining whether each one is inside a block.
fn walk_operations<'a>(ops: impl IntoIterator<Item = &'a Operation>) -> Vec<(&'a Operation, bool)> {
    fn visit<'a>(op: &'a Operation, nested: bool, all: &mut Vec<(&'a Operation, bool)>) {
        all.push((op, nested));
        if let Operation::Block {
            ops,
            true_branch,
            false_branch,
            ..
        } = op
        {
            for child in ops
                .iter()
                .chain(true_branch.as_deref().unwrap_or_default())
                .chain(false_branch.as_deref().unwrap_or_default())
            {
                visit(child, true, all);
            }
        }
    }
    let mut all = Vec::new();
    for op in ops {
        visit(op, false, &mut all);
    }
    all
}

/// Exports require every declaration to have storage before any branch executes.
/// Otherwise only top-level declarations are initialized at load time.
pub(crate) fn declarations<'a>(
    ops: impl IntoIterator<Item = &'a Operation>,
    has_exports: bool,
) -> Vec<&'a Operation> {
    let operations = if has_exports {
        walk_operations(ops).into_iter().map(|(op, _)| op).collect()
    } else {
        ops.into_iter().collect::<Vec<_>>()
    };
    operations
        .into_iter()
        .filter(|op| matches!(op, Operation::VariableDefinition { .. }))
        .collect()
}

pub(crate) fn collect<'a>(
    ops: impl IntoIterator<Item = &'a Operation>,
) -> Result<Option<ExportMap>, PecosError> {
    let mut collector = Collector::default();
    for (op, nested) in walk_operations(ops) {
        match op {
            Operation::VariableDefinition { data, variable, .. } => {
                collector.declare(data, variable, nested);
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
    fn visit(ops: &[Value], nested: bool, collector: &mut Collector) -> Result<(), PecosError> {
        for op in ops {
            if let Some(data) = op.get("data").and_then(Value::as_str) {
                if declaration_kind(data).is_some() {
                    let name = super::declarations::declaration_name(data, op.get("variable"))?;
                    collector.declare(data, name, nested);
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
                    if let Some(children) = op.get(field).and_then(Value::as_array) {
                        visit(children, true, collector)?;
                    }
                }
            }
        }
        Ok(())
    }
    let mut collector = Collector::default();
    super::declarations::validate_json_operations(ops)?;
    visit(ops, false, &mut collector)?;
    collector.finish()
}

/// Allocate every Result destination before execution, resolving chains of copies.
/// Declarations supply the type; inferred storage includes every copy contribution.
pub(crate) fn prepare_results<'a>(
    ops: impl IntoIterator<Item = &'a Operation>,
    environment: &mut super::environment::Environment,
) -> Result<(), PecosError> {
    use super::environment::{Environment, ResultReference};
    let ops = walk_operations(ops);
    let mut types = Environment::new();
    let mut pairs = Vec::new();
    let mut declared = BTreeSet::new();
    for (op, _) in &ops {
        match op {
            Operation::VariableDefinition {
                data,
                data_type,
                variable,
                size,
            } => {
                let size = super::ast::declaration_size(data, data_type, variable, *size)?;
                if data == "cvar_define" {
                    declared.insert(variable.clone());
                    types.add_classical_variable(variable, data_type.parse()?, size)?;
                } else if data == "qvar_define" {
                    types.add_quantum_register(variable, size)?;
                }
            }
            Operation::ClassicalOp {
                cop, args, returns, ..
            } if cop == "Result" => {
                super::operations::validate_result_counts(args.len(), returns.len())?;
                for (source, destination) in args.iter().zip(returns) {
                    pairs.push((
                        ResultReference::from_arg(source, "args")?,
                        ResultReference::from_arg(destination, "returns")?,
                    ));
                }
            }
            Operation::QuantumOp {
                qop, args, returns, ..
            } if qop == "Measure" => {
                super::classical_validation::validate_measure_args(args, returns)?;
            }
            _ => {}
        }
    }
    infer_result_types(&pairs, &declared, &mut types)?;
    for (source, destination) in pairs {
        types.validate_result_index(&source)?;
        types.validate_result_index(&destination)?;
        let info = types.classical_info(&destination.name)?;
        if !declared.contains(&destination.name) {
            environment.ensure_classical_variable(
                &destination.name,
                info.data_type.clone(),
                info.size,
            )?;
        }
        environment.result_destinations.insert(destination.name);
    }
    Ok(())
}

/// Propagate widths and whole-source signedness through the Result graph to a fixed point.
/// Indexed destinations retain the legacy unsigned 64-bit storage rule.
fn infer_result_types(
    pairs: &[(
        super::environment::ResultReference,
        super::environment::ResultReference,
    )],
    declared: &BTreeSet<String>,
    types: &mut super::environment::Environment,
) -> Result<(), PecosError> {
    use super::environment::DataType;
    let mut widths = BTreeMap::new();
    // Signedness is a bit set: 1 for unsigned sources, 2 for signed sources.
    let mut signs = BTreeMap::new();
    let indexed: BTreeSet<_> = pairs
        .iter()
        .filter(|(_, dest)| dest.index.is_some())
        .map(|(_, dest)| dest.name.clone())
        .collect();
    let whole: BTreeSet<_> = pairs
        .iter()
        .filter(|(source, _)| source.index.is_none())
        .map(|(_, dest)| dest.name.clone())
        .collect();
    for name in declared {
        let info = types.classical_info(name)?;
        widths.insert(name.clone(), info.size);
        signs.insert(
            name.clone(),
            if info.data_type.is_signed() { 2u8 } else { 1u8 },
        );
    }
    for (source, dest) in pairs {
        // Existing quantum names must fail before any inferred storage is allocated.
        types.validate_result_destination_name(&dest.name)?;
        if declared.contains(&dest.name) {
            continue;
        }
        if let Some(index) = dest.index {
            // Validate against the backing width before allocating storage.
            if index >= 64 {
                types.validate_result_index(dest)?;
            }
            let size = (index as usize + 1).max(32);
            let width = widths.entry(dest.name.clone()).or_insert(0);
            *width = (*width).max(size);
        }
        if source.index.is_some() {
            widths.entry(dest.name.clone()).or_insert(1);
        }
        if indexed.contains(&dest.name) || !whole.contains(&dest.name) {
            signs.insert(dest.name.clone(), 1);
        }
    }
    propagate_result_types(pairs, declared, &indexed, &mut widths, &mut signs);
    for (_, dest) in pairs {
        if signs.get(&dest.name).copied().unwrap_or(0) == 0 {
            signs.insert(dest.name.clone(), 1);
        }
    }
    propagate_result_types(pairs, declared, &indexed, &mut widths, &mut signs);
    for (source, _) in pairs {
        if !widths.contains_key(&source.name) {
            types.classical_info(&source.name)?;
        }
    }
    for (_, dest) in pairs {
        if declared.contains(&dest.name) {
            continue;
        }
        let mut sources = BTreeMap::new();
        for (other, target) in pairs {
            if target.name == dest.name && other.index.is_none() {
                let sign = signs.get(&other.name).copied().unwrap_or(1);
                sources.insert(other.name.clone(), sign);
            }
        }
        let signed = sources.iter().find(|(_, sign)| **sign & 2 != 0);
        let unsigned = sources
            .iter()
            .find(|(name, sign)| **sign & 1 != 0 && signed.is_none_or(|(other, _)| *name != other));
        if let (Some((a, _)), Some((b, _))) = (signed, unsigned) {
            return Err(PecosError::Input(format!(
                "Result: destination {:?} has conflicting signedness from sources {a:?} and {b:?}; declare the destination or use whole-register sources with the same signedness",
                dest.name
            )));
        }
        let dtype = if !indexed.contains(&dest.name) && signed.is_some() {
            DataType::I64
        } else {
            DataType::U64
        };
        types.ensure_classical_variable(&dest.name, dtype, widths[&dest.name])?;
    }
    Ok(())
}

fn propagate_result_types(
    pairs: &[(
        super::environment::ResultReference,
        super::environment::ResultReference,
    )],
    declared: &BTreeSet<String>,
    indexed: &BTreeSet<String>,
    widths: &mut BTreeMap<String, usize>,
    signs: &mut BTreeMap<String, u8>,
) {
    loop {
        let mut changed = false;
        for (source, dest) in pairs {
            if declared.contains(&dest.name) {
                continue;
            }
            if let Some(size) = widths.get(&source.name).copied() {
                let size = if source.index.is_some() { 1 } else { size };
                let width = widths.entry(dest.name.clone()).or_insert(0);
                changed |= *width < size;
                *width = (*width).max(size);
            }
            if source.index.is_none() && !indexed.contains(&dest.name) {
                let sign = signs.get(&source.name).copied().unwrap_or(0);
                let previous = signs.entry(dest.name.clone()).or_insert(0);
                changed |= *previous | sign != *previous;
                *previous |= sign;
            }
        }
        if !changed {
            break;
        }
    }
}
