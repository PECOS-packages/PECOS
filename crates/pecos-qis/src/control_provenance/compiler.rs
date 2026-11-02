use super::{ControlManifest, ControlSite};
use crate::qis_interface::{InterfaceError, ProgramFormat};
use regex::Regex;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write;

const HOOK: &str = "pecos_runtime_control_v1";
const NAME: &str = r"%[-a-zA-Z$._0-9]+";
const VALUE: &str = r"(?:%[-a-zA-Z$._0-9]+|-?[0-9]+|true|false)";
fn error(reason: impl std::fmt::Display) -> InterfaceError {
    InterfaceError::InvalidFormat(format!("compiler control provenance unsupported: {reason}"))
}
fn re(pattern: &str) -> Regex {
    Regex::new(pattern).expect("constant instrumentation grammar")
}
#[derive(Default)]
struct Block {
    label: String,
    lines: Vec<String>,
    successors: Vec<String>,
}
struct Scalar {
    operands: Vec<String>,
    input: Option<usize>,
}
fn site(sites: &mut Vec<ControlSite>, block: usize, kind: &str, scalar_type: &str) -> usize {
    let id = sites.len();
    sites.push(ControlSite {
        site_id: id,
        block,
        kind: kind.into(),
        scalar_type: scalar_type.into(),
        input_dependencies: BTreeSet::new(),
    });
    id
}
fn hook(id: usize, value: &str) -> String {
    format!("  call void @{HOOK}(i64 {id}, i64 {value})\n")
}
fn values(text: &str) -> Vec<String> {
    re(NAME)
        .find_iter(text)
        .map(|m| m.as_str().to_string())
        .collect()
}
fn dependencies(
    name: &str,
    scalars: &BTreeMap<String, Scalar>,
    seen: &mut BTreeSet<String>,
    out: &mut BTreeSet<usize>,
) -> Result<(), InterfaceError> {
    let mut pending = vec![name];
    while let Some(name) = pending.pop() {
        if !name.starts_with('%') || !seen.insert(name.into()) {
            continue;
        }
        let scalar = scalars
            .get(name)
            .ok_or_else(|| error(format!("unresolved or resource-valued scalar {name}")))?;
        if let Some(input) = scalar.input {
            out.insert(input);
        }
        pending.extend(scalar.operands.iter().map(String::as_str));
    }
    Ok(())
}

/// Instrument a canonical, bounded single-entry LLVM module. No input hooks,
/// arbitrary callees, pointer arithmetic/loads, mutable globals, EH or indirect
/// control flow can enter this certified subset. LLVM verifies before and after.
pub(crate) fn instrument(source: &str) -> Result<(String, ControlManifest), InterfaceError> {
    if source.len() > 4 * 1024 * 1024 {
        return Err(error("module exceeds 4 MiB"));
    }
    let canonical =
        crate::executor::canonical_qis_ir(source.as_bytes(), ProgramFormat::LlvmIrText)?;
    if canonical.contains(HOOK) || canonical.contains("pecos.ctrl.") {
        return Err(error("preexisting protected instrumentation"));
    }
    let canonical = canonical
        .lines()
        .filter(|line| !line.starts_with("; ModuleID") && !line.starts_with("source_filename ="))
        .collect::<Vec<_>>()
        .join("\n");
    let header = re(&format!(
        r"^define i64 @qmain\(i64 ({NAME})\)(?: #[0-9]+)? \{{$"
    ));
    let label = re(r"^([-a-zA-Z$._0-9]+):(?:\s*;.*)?$");
    let global = re(
        r#"^@([-a-zA-Z$._0-9]+) = private (?:unnamed_addr )?constant \[([0-9]+) x i8\] c"(?:[^"\\]|\\[0-9A-Fa-f]{2})*"(?:, align [0-9]+)?$"#,
    );
    let decl = re(r"^declare (void|i1|i64) @([-a-zA-Z$._0-9]+)\(([^()]*)\)$");
    let mut globals = BTreeMap::<String, usize>::new();
    let mut prefix = String::new();
    let mut suffix = String::new();
    let mut blocks = Vec::<Block>::new();
    let mut argument = None;
    let mut inside = false;
    let mut finished = false;
    for raw in canonical.lines() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with(';') {
            continue;
        }
        if inside {
            if line == "}" {
                inside = false;
                finished = true;
                continue;
            }
            if let Some(c) = label.captures(line) {
                blocks.push(Block {
                    label: c[1].into(),
                    ..Block::default()
                });
            } else {
                let block = blocks
                    .last_mut()
                    .ok_or_else(|| error("entry block must be named"))?;
                block.lines.push(line.into());
            }
        } else if let Some(c) = header.captures(line) {
            if argument.is_some() {
                return Err(error("multiple entry definitions"));
            }
            argument = Some(c[1].to_string());
            inside = true;
            prefix.push_str(line);
            prefix.push('\n');
        } else if !finished {
            if let Some(c) = global.captures(line) {
                globals.insert(
                    format!("@{}", &c[1]),
                    c[2].parse().map_err(|_| error("global extent"))?,
                );
            } else if let Some(c) = decl.captures(line) {
                let (ret, roles) = call_shape(&c[2])?;
                let parameters: Vec<_> = c[3]
                    .split(',')
                    .map(str::trim)
                    .filter(|s| !s.is_empty())
                    .collect();
                if &c[1] != ret || parameters != roles.iter().map(|(ty, _)| *ty).collect::<Vec<_>>()
                {
                    return Err(error("callee declaration ABI"));
                }
            } else if !(line.starts_with("target triple = ")
                || line.starts_with("target datalayout = "))
            {
                return Err(error(format!("module entity {line}")));
            }
            prefix.push_str(line);
            prefix.push('\n');
        } else if re(r#"^attributes #[0-9]+ = \{ "EntryPoint" \}$"#).is_match(line) {
            suffix.push_str(line);
            suffix.push('\n');
        } else {
            return Err(error(format!("module entity {line}")));
        }
    }
    if inside || !finished || blocks.is_empty() || blocks.len() > 256 {
        return Err(error("single bounded qmain definition required"));
    }
    let argument = argument.ok_or_else(|| error("missing qmain"))?;
    let names: BTreeMap<_, _> = blocks
        .iter()
        .enumerate()
        .map(|(i, b)| (b.label.clone(), i))
        .collect();
    if names.len() != blocks.len() {
        return Err(error("duplicate block"));
    }
    let branch = re(&format!(
        r"^br i1 ({VALUE}), label %([-a-zA-Z$._0-9]+), label %([-a-zA-Z$._0-9]+)$"
    ));
    let jump = re(r"^br label %([-a-zA-Z$._0-9]+)$");
    for block in &mut blocks {
        let last = block.lines.last().ok_or_else(|| error("empty block"))?;
        block.successors = if let Some(c) = branch.captures(last) {
            vec![c[2].into(), c[3].into()]
        } else if let Some(c) = jump.captures(last) {
            vec![c[1].into()]
        } else if last == "ret i64 0" {
            vec![]
        } else {
            return Err(error("only br and ret i64 0 terminators are supported"));
        };
    }
    let successors = blocks
        .iter()
        .map(|block| {
            block
                .successors
                .iter()
                .map(|name| {
                    names
                        .get(name)
                        .copied()
                        .ok_or_else(|| error("missing successor"))
                })
                .collect::<Result<Vec<_>, _>>()
        })
        .collect::<Result<Vec<_>, _>>()?;
    let (loop_headers, loop_members) = loops(&successors)?;
    // Conservative PHI selector dependencies: every conditional on a path to
    // an incoming edge may select the value, including prior loop iterations.
    let mut predecessors = vec![Vec::new(); blocks.len()];
    for (from, edges) in successors.iter().enumerate() {
        for &to in edges {
            predecessors[to].push(from);
        }
    }
    let phi_selectors: Vec<Vec<String>> = (0..blocks.len())
        .map(|block| {
            let mut pending = predecessors[block].clone();
            let mut seen = BTreeSet::new();
            let mut predicates = Vec::new();
            while let Some(previous) = pending.pop() {
                if seen.insert(previous) {
                    if let Some(c) = branch.captures(blocks[previous].lines.last().unwrap()) {
                        predicates.push(c[1].to_string());
                    }
                    pending.extend(&predecessors[previous]);
                }
            }
            predicates
        })
        .collect();
    let mut sites = Vec::new();
    let mut scalars = BTreeMap::new();
    let mut resources = BTreeMap::<String, &'static str>::new();
    let mut cells = BTreeSet::<String>::new();
    let mut body = String::new();
    let mut dependency_checks = Vec::<(usize, Vec<String>)>::new();
    let assignment = re(&format!(r"^({NAME}) = (.*)$"));
    let call = re(r"^call (void|i1|i64) @([-a-zA-Z$._0-9]+)\((.*)\)$");
    let binary = re(&format!(
        r"^(?:add|sub|mul|and|or|xor) i(?:1|8|64) ({VALUE}), ({VALUE})$"
    ));
    let compare = re(&format!(
        r"^icmp (?:eq|ne|ugt|uge|ult|ule|sgt|sge|slt|sle) i(?:1|8|64) ({VALUE}), ({VALUE})$"
    ));
    let cast = re(&format!(
        r"^(?:zext|trunc) i(?:1|8|64) ({VALUE}) to i(?:1|8|64)$"
    ));
    let phi = re(&format!(
        r"^phi i(?:1|8|64) (\[ {VALUE}, %[-a-zA-Z$._0-9]+ \](?:, \[ {VALUE}, %[-a-zA-Z$._0-9]+ \])*)$"
    ));
    let phi_value = re(&format!(r"\[ ({VALUE}), %[-a-zA-Z$._0-9]+ \]"));
    let select = re(&format!(
        r"^select i1 ({VALUE}), i(?:1|8|64) ({VALUE}), i(?:1|8|64) ({VALUE})$"
    ));
    let store = re(&format!(
        r"^store i8 ({VALUE}), ptr ({NAME})(?:, align [0-9]+)?$"
    ));
    for (block_id, block) in blocks.iter().enumerate() {
        writeln!(body, "{}:", block.label).expect("String write");
        let block_site = site(&mut sites, block_id, "block", "u64");
        let mut entry_emitted = false;
        for (index, line) in block.lines.iter().enumerate() {
            let (lhs, instruction) = assignment
                .captures(line)
                .map_or((None, line.as_str()), |c| {
                    (Some(c.get(1).unwrap().as_str()), c.get(2).unwrap().as_str())
                });
            if !instruction.starts_with("phi ") && !entry_emitted {
                body.push_str(&hook(block_site, "0"));
                entry_emitted = true;
                if block_id == 0 {
                    let id = site(&mut sites, 0, "shot", "u64");
                    scalars.insert(
                        argument.clone(),
                        Scalar {
                            operands: vec![],
                            input: Some(id),
                        },
                    );
                    body.push_str(&hook(id, &argument));
                }
            }
            let mut after = String::new();
            if let Some(c) = call.captures(instruction) {
                let name = &c[2];
                let (ret, roles) = call_shape(name)?;
                if &c[1] != ret {
                    return Err(error("callee return ABI"));
                }
                let args: Vec<_> = c[3]
                    .split(',')
                    .map(str::trim)
                    .filter(|s| !s.is_empty())
                    .collect();
                if args.len() != roles.len() {
                    return Err(error("callee argument count"));
                }
                for (arg, (ty, role)) in args.iter().zip(roles.iter()) {
                    let operand = arg
                        .strip_prefix(&format!("{ty} "))
                        .ok_or_else(|| error("callee argument type"))?;
                    match *role {
                        "qubit" | "future" => {
                            if resources.get(operand) != Some(role) {
                                return Err(error("unresolved native resource"));
                            }
                        }
                        "global" => {
                            if !globals.contains_key(operand) {
                                return Err(error("expected constant byte global"));
                            }
                        }
                        "cell" => {
                            if !cells.contains(operand) {
                                return Err(error("expected local syndrome cell"));
                            }
                        }
                        "float" => {
                            if operand.parse::<f64>().is_err()
                                && !re(r"^0x[0-9A-Fa-f]{16}$").is_match(operand)
                            {
                                return Err(error("gate angles must be constants"));
                            }
                        }
                        "scalar" => {
                            if !re(&format!("^{VALUE}$")).is_match(operand) {
                                return Err(error("invalid scalar operand"));
                            }
                            dependency_checks.push((block_site, vec![operand.into()]));
                        }
                        _ => unreachable!(),
                    }
                }
                // Native byte reads must stay within statically known storage.
                let constant_length = |position: usize| -> Result<usize, InterfaceError> {
                    args[position]
                        .strip_prefix("i64 ")
                        .and_then(|v| v.parse().ok())
                        .ok_or_else(|| error("buffer lengths must be nonnegative constants"))
                };
                let check_global = |pointer: usize, length: usize| -> Result<(), InterfaceError> {
                    let extent = globals[args[pointer].strip_prefix("ptr ").unwrap()];
                    let length = constant_length(length)?;
                    if length == 0 || length > extent {
                        return Err(error("constant buffer extent"));
                    }
                    Ok(())
                };
                if matches!(name, "pecos_runtime_read_bool_v1" | "print_bool_selene") {
                    check_global(0, 1)?;
                }
                if name == "pecos_runtime_decode_observed_v1" {
                    check_global(4, 5)?;
                    let cell = args[2].strip_prefix("ptr ").unwrap();
                    let initialized = index
                        .checked_sub(1)
                        .and_then(|i| store.captures(&block.lines[i]))
                        .is_some_and(|c| &c[2] == cell);
                    if constant_length(3)? != 1 || !initialized {
                        return Err(error(
                            "decode requires one byte and immediately preceding same-cell store",
                        ));
                    }
                    let id = site(&mut sites, block_id, "decode", "u64");
                    body.push_str(&hook(id, "0"));
                }
                if ret == "void" {
                    if lhs.is_some() {
                        return Err(error("void assignment"));
                    }
                } else {
                    let lhs = lhs.ok_or_else(|| error("named call result required"))?;
                    if name == "___qalloc" {
                        resources.insert(lhs.into(), "qubit");
                    } else if name == "___lazy_measure" {
                        resources.insert(lhs.into(), "future");
                    } else {
                        let kind = if name == "pecos_runtime_decode_observed_v1" {
                            "decoder_frame"
                        } else {
                            "read_bool"
                        };
                        let id = site(
                            &mut sites,
                            block_id,
                            kind,
                            if ret == "i1" { "bool" } else { "u64" },
                        );
                        scalars.insert(
                            lhs.into(),
                            Scalar {
                                operands: vec![],
                                input: Some(id),
                            },
                        );
                        if ret == "i1" {
                            writeln!(after, "  %pecos.ctrl.{id} = zext i1 {lhs} to i64")
                                .expect("String write");
                            after.push_str(&hook(id, &format!("%pecos.ctrl.{id}")));
                        } else {
                            after.push_str(&hook(id, lhs));
                        }
                    }
                }
            } else if let Some(c) = branch.captures(instruction) {
                if index + 1 != block.lines.len() {
                    return Err(error("nonterminal branch"));
                }
                let id = site(&mut sites, block_id, "branch", "bool");
                dependency_checks.push((id, vec![c[1].into()]));
                writeln!(body, "  %pecos.ctrl.{id} = zext i1 {} to i64", &c[1])
                    .expect("String write");
                body.push_str(&hook(id, &format!("%pecos.ctrl.{id}")));
            } else if jump.is_match(instruction) || instruction == "ret i64 0" {
                if index + 1 != block.lines.len() {
                    return Err(error("nonterminal control instruction"));
                }
            } else if re(r"^alloca i8(?:, align [0-9]+)?$").is_match(instruction) {
                if block_id != 0 {
                    return Err(error("syndrome alloca must be in entry block"));
                }
                cells.insert(lhs.ok_or_else(|| error("unnamed alloca"))?.into());
            } else if let Some(c) = store.captures(instruction) {
                if !cells.contains(&c[2]) {
                    return Err(error("nonlocal store"));
                }
                dependency_checks.push((block_site, vec![c[1].into()]));
            } else {
                let operands = if let Some(c) = phi.captures(instruction) {
                    if entry_emitted {
                        return Err(error("nonleading phi"));
                    }
                    let mut operands = phi_selectors[block_id].clone();
                    operands.extend(phi_value.captures_iter(&c[1]).map(|v| v[1].to_string()));
                    operands
                } else if binary.is_match(instruction)
                    || compare.is_match(instruction)
                    || cast.is_match(instruction)
                    || select.is_match(instruction)
                {
                    values(instruction)
                } else {
                    return Err(error(format!("instruction {instruction}")));
                };
                scalars.insert(
                    lhs.ok_or_else(|| error("unnamed scalar"))?.into(),
                    Scalar {
                        operands,
                        input: None,
                    },
                );
            }
            body.push_str("  ");
            body.push_str(line);
            body.push('\n');
            body.push_str(&after);
        }
    }
    if scalars.len() > 4096 {
        return Err(error("scalar table limit exceeded"));
    }
    for scalar in scalars.keys() {
        dependencies(scalar, &scalars, &mut BTreeSet::new(), &mut BTreeSet::new())?;
    }
    let mut dependency_entries = 0;
    for (id, operands) in dependency_checks {
        let previous = sites[id].input_dependencies.len();
        for operand in operands {
            dependencies(
                &operand,
                &scalars,
                &mut BTreeSet::new(),
                &mut sites[id].input_dependencies,
            )?;
        }
        dependency_entries += sites[id].input_dependencies.len() - previous;
        if dependency_entries > 65536 {
            return Err(error("dependency table limit exceeded"));
        }
    }
    if sites.len() > 4096 || !sites.iter().any(|site| site.kind == "decode") {
        return Err(error("bounded observed decode sites required"));
    }
    let manifest = ControlManifest {
        module_sha256: Sha256::digest(canonical.as_bytes()).iter().fold(
            String::with_capacity(64),
            |mut text, byte| {
                write!(text, "{byte:02x}").expect("String write");
                text
            },
        ),
        instrumenter: "pecos-qis-control-v1".into(),
        sites,
        successors,
        loop_members,
        loop_headers,
    };
    // Move the generated declaration before the function header.
    let header_pos = prefix
        .find("define i64 @qmain")
        .ok_or_else(|| error("missing header"))?;
    let identity = format!("{}:{}", manifest.instrumenter, manifest.module_sha256);
    prefix.insert_str(header_pos, &format!(
        "@pecos.ctrl.identity = private constant [{} x i8] c\"{identity}\"\ndeclare void @{HOOK}(i64, i64)\n", identity.len()
    ));
    let transformed = format!("{prefix}{body}}}\n{suffix}");
    let verified =
        crate::executor::canonical_qis_ir(transformed.as_bytes(), ProgramFormat::LlvmIrText)?;
    let verified = verified
        .lines()
        .filter(|line| !line.starts_with("; ModuleID") && !line.starts_with("source_filename ="))
        .collect::<Vec<_>>()
        .join("\n");
    Ok((verified, manifest))
}

type Roles = Vec<(&'static str, &'static str)>;
fn call_shape(name: &str) -> Result<(&'static str, Roles), InterfaceError> {
    Ok(match name {
        "___qalloc" => ("i64", vec![]),
        "___qfree" | "___reset" => ("void", vec![("i64", "qubit")]),
        "___rxy" => (
            "void",
            vec![("i64", "qubit"), ("double", "float"), ("double", "float")],
        ),
        "___rz" => ("void", vec![("i64", "qubit"), ("double", "float")]),
        "___lazy_measure" => ("i64", vec![("i64", "qubit")]),
        "___read_future_bool" => ("i1", vec![("i64", "future")]),
        "___dec_future_refcount" | "___inc_future_refcount" => ("void", vec![("i64", "future")]),
        "print_bool_selene" => (
            "void",
            vec![("ptr", "global"), ("i64", "scalar"), ("i1", "scalar")],
        ),
        "pecos_runtime_read_bool_v1" => (
            "i1",
            vec![("ptr", "global"), ("i64", "scalar"), ("i64", "future")],
        ),
        "pecos_runtime_decode_observed_v1" => (
            "i64",
            vec![
                ("i64", "scalar"),
                ("i64", "scalar"),
                ("ptr", "cell"),
                ("i64", "scalar"),
                ("ptr", "global"),
                ("i64", "scalar"),
            ],
        ),
        _ => return Err(error(format!("callee {name}"))),
    })
}

fn loops(successors: &[Vec<usize>]) -> Result<(Vec<usize>, Vec<BTreeSet<usize>>), InterfaceError> {
    let n = successors.len();
    let mut predecessors = vec![BTreeSet::new(); n];
    for (i, edges) in successors.iter().enumerate() {
        for &j in edges {
            predecessors[j].insert(i);
        }
    }
    if !predecessors[0].is_empty() {
        return Err(error("entry has incoming edges"));
    }
    let mut reached = BTreeSet::new();
    let mut pending = vec![0];
    while let Some(i) = pending.pop() {
        if reached.insert(i) {
            pending.extend(&successors[i]);
        }
    }
    if reached.len() != n {
        return Err(error("unreachable blocks"));
    }
    let all: BTreeSet<_> = (0..n).collect();
    let mut dom = vec![all.clone(); n];
    dom[0] = BTreeSet::from([0]);
    loop {
        let previous = dom.clone();
        for i in 1..n {
            let mut set = all.clone();
            for &p in &predecessors[i] {
                set = set.intersection(&previous[p]).copied().collect();
            }
            set.insert(i);
            dom[i] = set;
        }
        if dom == previous {
            break;
        }
    }
    let mut natural = BTreeMap::<usize, BTreeSet<usize>>::new();
    let mut acyclic = successors.to_vec();
    for (tail, edges) in successors.iter().enumerate() {
        for &header in edges {
            if dom[tail].contains(&header) {
                acyclic[tail].retain(|&node| node != header);
                let members = natural
                    .entry(header)
                    .or_insert_with(|| BTreeSet::from([header]));
                let mut stack = vec![tail];
                while let Some(node) = stack.pop() {
                    if members.insert(node) {
                        stack.extend(&predecessors[node]);
                    }
                }
            }
        }
    }
    // Removing natural backedges must leave a DAG; otherwise control is irreducible.
    let mut indegree = vec![0; n];
    for edges in &acyclic {
        for &j in edges {
            indegree[j] += 1;
        }
    }
    let mut ready: Vec<_> = (0..n).filter(|&i| indegree[i] == 0).collect();
    let mut count = 0;
    while let Some(i) = ready.pop() {
        count += 1;
        for &j in &acyclic[i] {
            indegree[j] -= 1;
            if indegree[j] == 0 {
                ready.push(j);
            }
        }
    }
    if count != n {
        return Err(error("irreducible loop"));
    }
    for (&header, members) in &natural {
        if members
            .iter()
            .any(|&node| node != header && predecessors[node].iter().any(|p| !members.contains(p)))
        {
            return Err(error("loop has a side entrance"));
        }
    }
    for a in natural.values() {
        for b in natural.values() {
            if !a.is_disjoint(b) && !a.is_subset(b) && !b.is_subset(a) {
                return Err(error("overlapping nonnested loops"));
            }
        }
    }
    Ok(natural.into_iter().unzip())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn source() -> String {
        let source = include_str!("../../tests/fixtures/runtime_decode_native.ll");
        let source = source.replace(
            "@label =",
            "@ctx = private constant [2 x i8] c\"{}\"\n@label =",
        );
        source.replace("declare i64 @pecos_runtime_decode(i64, i64, ptr, i64)",
            "declare i64 @pecos_runtime_decode_observed_v1(i64, i64, ptr, i64, ptr, i64)")
            .replace("@pecos_runtime_decode(i64 7, i64 %instance, ptr %data, i64 1)",
            "@pecos_runtime_decode_observed_v1(i64 7, i64 %instance, ptr %data, i64 1, ptr @ctx, i64 2)")
    }
    #[test]
    fn control_compiler_generates_complete_sites_and_stable_module_identity() {
        let (ir, manifest) = instrument(&source()).unwrap();
        assert!(ir.contains(HOOK));
        assert_eq!(ir, instrument(&source()).unwrap().0);
        assert_eq!(
            manifest.module_sha256,
            instrument(&source()).unwrap().1.module_sha256
        );
        assert_eq!(manifest.loop_headers, [1]);
        assert_eq!(manifest.loop_members, [BTreeSet::from([1, 2, 3])]);
        assert_eq!(
            manifest.sites.iter().filter(|s| s.kind == "branch").count(),
            2
        );
        let frame = manifest
            .sites
            .iter()
            .find(|s| s.kind == "decoder_frame")
            .unwrap();
        let conditional = manifest
            .sites
            .iter()
            .find(|s| s.kind == "branch" && s.block == 1)
            .unwrap();
        assert_eq!(
            conditional.input_dependencies,
            BTreeSet::from([frame.site_id])
        );
        assert!(
            manifest
                .sites
                .iter()
                .find(|s| s.kind == "branch" && s.block == 3)
                .unwrap()
                .input_dependencies
                .contains(&frame.site_id)
        );
        assert!(ir.contains(&manifest.module_sha256));
        assert!(
            instrument(&ir)
                .unwrap_err()
                .to_string()
                .contains("preexisting")
        );
    }
    #[test]
    fn control_compiler_rejects_unsupported_or_unresolved_provenance() {
        let good = source();
        let cases = [
            good.replace("store i8 %bit, ptr %data", "; missing initialization"),
            good.replace("ptr %data, i64 1, ptr @ctx", "ptr %data, i64 2, ptr @ctx"),
            good.replace("ptr @ctx, i64 2", "ptr @ctx, i64 3"),
            good.replace("ptr @label, i64 5", "ptr @label, i64 1000"),
            good.replace(
                "%correct = icmp ne i64 %frame, 0",
                "%correct = icmp ne i64 %m, 0",
            ),
            good.replace("declare i64 @___qalloc()", "declare i32 @___qalloc()"),
            good.replace("@label = private constant", "@label = private global"),
            good.replace(
                "%next = add i64 %instance, 1",
                "%next = udiv i64 %instance, 1",
            ),
            good.replace("br label %loop", "br i1 true, label %loop, label %apply"),
            good.replace(
                "%data = alloca i8",
                "%data = alloca i8\n    %extra = load i8, ptr %data",
            ),
            format!("{good}\ndefine void @helper() {{ ret void }}"),
            format!("declare void @{HOOK}(i64, i64)\n{good}"),
            good.replace(
                "%value = call i1 @___read_future_bool(i64 %after)",
                "%value = call i1 @hidden(i64 %after)",
            )
            .replace(
                "declare i1 @___read_future_bool(i64)",
                "declare i1 @___read_future_bool(i64)\ndeclare i1 @hidden(i64)",
            ),
        ];
        for bad in cases {
            assert!(instrument(&bad).is_err(), "admitted unsupported module");
        }
    }
    #[test]
    fn control_compiler_tracks_phi_selector_inputs() {
        let source = source().replace("%bit = zext i1 %observed to i8", "br i1 %observed, label %yes, label %no\nyes:\n br label %merge\nno:\n br label %merge\nmerge:\n %p = phi i1 [true, %yes], [false, %no]\n br i1 %p, label %join, label %join\njoin:\n %bit = zext i1 %p to i8");
        let (_, manifest) = instrument(&source).unwrap();
        let input = manifest
            .sites
            .iter()
            .find(|s| s.kind == "read_bool")
            .unwrap();
        let phi_branch = manifest
            .sites
            .iter()
            .find(|s| s.kind == "branch" && s.block == 4)
            .unwrap();
        assert!(phi_branch.input_dependencies.contains(&input.site_id));
    }
    #[test]
    fn control_compiler_recognizes_nested_loops_and_rejects_irreducible_cycles() {
        let (headers, members) =
            loops(&[vec![1], vec![2, 5], vec![3], vec![2, 4], vec![1], vec![]]).unwrap();
        assert_eq!(headers, [1, 2]);
        assert_eq!(
            members,
            [BTreeSet::from([1, 2, 3, 4]), BTreeSet::from([2, 3])]
        );
        assert!(loops(&[vec![1, 2], vec![2], vec![1, 3], vec![]]).is_err());
    }
}
