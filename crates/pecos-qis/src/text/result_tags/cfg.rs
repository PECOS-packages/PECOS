//! Block validation, reachability, cycles and immediate dominators.

use super::{Call, Result, relevant};
use crate::text::reader::{self, Token};
use std::collections::{BTreeMap, BTreeSet};

fn terminator(opcode: &str) -> bool {
    matches!(
        opcode,
        "ret"
            | "br"
            | "switch"
            | "unreachable"
            | "invoke"
            | "callbr"
            | "indirectbr"
            | "resume"
            | "catchswitch"
            | "cleanupret"
            | "catchret"
    )
}

pub(super) fn instruction_start(tokens: &[Token<'_>]) -> Result<usize> {
    let start = if tokens.get(1).is_some_and(|t| t.text == "=") {
        2
    } else {
        0
    };
    if start >= tokens.len() {
        return Err("incomplete instruction".into());
    }
    Ok(start)
}

fn opcode(instruction: &str) -> Result<&str> {
    let tokens = reader::tokens(instruction);
    Ok(tokens[instruction_start(&tokens)?].text)
}

fn validate_opcode(instruction: &str) -> Result<()> {
    let tokens = reader::tokens(instruction);
    let mut opcode = instruction_start(&tokens)?;
    if matches!(tokens[opcode].text, "tail" | "musttail" | "notail") {
        opcode += 1;
    }
    let mut depth = 0;
    for (index, token) in tokens.iter().enumerate() {
        if (token.text == "call" || terminator(token.text)) && depth == 0 && index != opcode {
            return Err(format!("{} outside instruction opcode", token.text).into());
        }
        depth += reader::nesting(token.text);
    }
    Ok(())
}

struct Block {
    label: String,
    instructions: Vec<String>,
}

fn blocks(body: &str) -> Result<Vec<Block>> {
    let mut blocks = Vec::<Block>::new();
    let mut current = Block {
        label: "0".into(),
        instructions: Vec::new(),
    };
    let mut explicit = false;
    let mut pending = String::new();
    for line in body.lines() {
        let tokens = reader::tokens(line);
        if tokens.is_empty() {
            continue;
        }
        let mut start = 0;
        if tokens.get(1).is_some_and(|t| t.text == ":") {
            if !pending.is_empty() {
                return Err("unterminated instruction before block label".into());
            }
            if explicit || !current.instructions.is_empty() {
                blocks.push(current);
            }
            let label = tokens[0].text.trim_matches('"').to_owned();
            if blocks.iter().any(|block| block.label == label) {
                return Err(format!("duplicate block %{label}").into());
            }
            current = Block {
                label,
                instructions: Vec::new(),
            };
            explicit = true;
            start = 2;
            if start == tokens.len() {
                continue;
            }
        }
        pending.push(' ');
        pending.push_str(&reader::joined(&tokens[start..]));
        let depth: i32 = reader::tokens(&pending)
            .iter()
            .map(|t| match t.text {
                "(" | "[" | "{" => 1,
                ")" | "]" | "}" => -1,
                _ => 0,
            })
            .sum();
        if depth != 0 {
            continue;
        }
        current.instructions.push(pending.trim().to_owned());
        pending.clear();
    }
    if !pending.is_empty() {
        return Err("unterminated instruction".into());
    }
    if explicit || !current.instructions.is_empty() {
        blocks.push(current);
    }
    if blocks.is_empty() {
        return Err("empty function body".into());
    }
    for block in &blocks {
        for instruction in &block.instructions {
            validate_opcode(instruction)?;
        }
        for instruction in block
            .instructions
            .iter()
            .take(block.instructions.len().saturating_sub(1))
        {
            let opcode = opcode(instruction)?;
            if terminator(opcode) {
                return Err(
                    format!("terminator {opcode} is not last in block %{}", block.label).into(),
                );
            }
        }
        if block
            .instructions
            .last()
            .map(|s| opcode(s))
            .transpose()?
            .is_none_or(|op| !terminator(op))
        {
            return Err(format!("block %{} has no terminator", block.label).into());
        }
    }
    Ok(blocks)
}

fn successors(instruction: &str) -> Result<Vec<String>> {
    let opcode = opcode(instruction)?;
    if !matches!(opcode, "ret" | "br" | "switch" | "unreachable") {
        return Err(format!("unsupported terminator {opcode}").into());
    }
    if matches!(opcode, "ret" | "unreachable") {
        return Ok(Vec::new());
    }
    let tokens = reader::tokens(instruction);
    let successors: Vec<_> = tokens
        .windows(2)
        .filter(|pair| pair[0].text == "label")
        .map(|pair| {
            pair[1]
                .text
                .get(1..)
                .unwrap_or_default()
                .trim_matches('"')
                .to_owned()
        })
        .collect();
    if successors.is_empty() || (opcode == "br" && !matches!(successors.len(), 1 | 2)) {
        return Err(format!("unsupported {opcode} targets").into());
    }
    Ok(successors)
}

pub(super) struct FunctionBody {
    blocks: Vec<Block>,
    graph: Vec<Vec<usize>>,
    reachable: Vec<usize>,
    pub calls: Vec<Vec<Call>>,
}

impl FunctionBody {
    pub fn parse(body: &str) -> Result<Self> {
        let blocks = blocks(body)?;
        let indices: BTreeMap<_, _> = blocks
            .iter()
            .enumerate()
            .map(|(index, block)| (block.label.as_str(), index))
            .collect();
        let mut graph = vec![Vec::new(); blocks.len()];
        let mut calls = vec![Vec::new(); blocks.len()];
        let mut seen = BTreeSet::new();
        let mut reachable = Vec::new();
        let mut pending = vec![0];
        while let Some(block) = pending.pop() {
            if !seen.insert(block) {
                continue;
            }
            reachable.push(block);
            let instructions = &blocks[block].instructions;
            for successor in successors(instructions.last().expect("validated terminator"))? {
                graph[block].push(
                    *indices
                        .get(successor.as_str())
                        .ok_or_else(|| format!("missing block %{successor}"))?,
                );
            }
            pending.extend(&graph[block]);
            for instruction in instructions {
                if let Some(call) = Call::parse(instruction)? {
                    calls[block].push(call);
                }
            }
        }
        Ok(Self {
            blocks,
            graph,
            reachable,
            calls,
        })
    }

    pub fn ordered_blocks(&self, functions: &BTreeSet<String>) -> Result<Vec<usize>> {
        let mut relevant: Vec<_> = self
            .reachable
            .iter()
            .copied()
            .filter(|block| {
                self.calls[*block]
                    .iter()
                    .any(|call| relevant(&call.target) || functions.contains(&call.target))
            })
            .collect();
        if relevant.is_empty() {
            return Ok(relevant);
        }
        let returns: BTreeSet<_> = self
            .reachable
            .iter()
            .copied()
            .filter(|block| {
                self.blocks[*block]
                    .instructions
                    .last()
                    .is_some_and(|s| s.starts_with("ret "))
            })
            .collect();
        if returns.is_empty() {
            return Err("function has no reachable ret".into());
        }
        let mut predecessors = vec![Vec::new(); self.blocks.len()];
        for block in &self.reachable {
            for successor in &self.graph[*block] {
                predecessors[*successor].push(*block);
            }
        }
        let postorder = postorder(&self.graph);
        let cyclic = cycle_blocks(&self.graph, &predecessors, &postorder);
        let reverse_postorder: Vec<_> = postorder.into_iter().rev().collect();
        let idoms = immediate_dominators(&predecessors, &reverse_postorder);
        let mut depth = vec![0; self.blocks.len()];
        let mut returns_below: Vec<_> = (0..self.blocks.len())
            .map(|block| usize::from(returns.contains(&block)))
            .collect();
        for block in reverse_postorder.iter().skip(1) {
            depth[*block] = depth[idoms[*block]] + 1;
        }
        for block in reverse_postorder.iter().skip(1).rev() {
            returns_below[idoms[*block]] += returns_below[*block];
        }
        for block in &relevant {
            let label = &self.blocks[*block].label;
            if cyclic.contains(block) {
                return Err(format!("order-relevant block %{label} is on a CFG cycle").into());
            }
            if returns_below[*block] != returns.len() {
                return Err(format!(
                    "order-relevant block %{label} does not dominate every reachable ret"
                )
                .into());
            }
        }
        relevant.sort_by_key(|block| depth[*block]);
        Ok(relevant)
    }
}

fn postorder(graph: &[Vec<usize>]) -> Vec<usize> {
    let mut seen = vec![false; graph.len()];
    let mut order = Vec::new();
    let mut pending = vec![(0, false)];
    while let Some((block, finished)) = pending.pop() {
        if finished {
            order.push(block);
        } else if !seen[block] {
            seen[block] = true;
            pending.push((block, true));
            pending.extend(
                graph[block]
                    .iter()
                    .rev()
                    .filter(|successor| !seen[**successor])
                    .map(|successor| (*successor, false)),
            );
        }
    }
    order
}

fn cycle_blocks(
    graph: &[Vec<usize>],
    predecessors: &[Vec<usize>],
    postorder: &[usize],
) -> BTreeSet<usize> {
    let mut seen = BTreeSet::new();
    let mut cyclic = BTreeSet::new();
    for root in postorder.iter().rev().copied() {
        if !seen.insert(root) {
            continue;
        }
        let mut component = Vec::new();
        let mut pending = vec![root];
        while let Some(block) = pending.pop() {
            component.push(block);
            for predecessor in &predecessors[block] {
                if seen.insert(*predecessor) {
                    pending.push(*predecessor);
                }
            }
        }
        if component.len() > 1 || graph[root].contains(&root) {
            cyclic.extend(component);
        }
    }
    cyclic
}

fn common_dominator(mut first: usize, mut second: usize, idoms: &[usize], rank: &[usize]) -> usize {
    while first != second {
        while rank[first] > rank[second] {
            first = idoms[first];
        }
        while rank[second] > rank[first] {
            second = idoms[second];
        }
    }
    first
}

fn immediate_dominators(predecessors: &[Vec<usize>], reverse_postorder: &[usize]) -> Vec<usize> {
    // Cooper, Harvey, Kennedy: intersect immediate dominator chains in RPO.
    let mut rank = vec![0; predecessors.len()];
    for (index, block) in reverse_postorder.iter().enumerate() {
        rank[*block] = index;
    }
    let mut idoms = vec![usize::MAX; predecessors.len()];
    idoms[0] = 0;
    loop {
        let mut changed = false;
        for block in reverse_postorder.iter().skip(1).copied() {
            let mut processed = predecessors[block]
                .iter()
                .copied()
                .filter(|parent| idoms[*parent] != usize::MAX);
            let mut parent = processed.next().expect("earlier RPO predecessor");
            for other in processed {
                parent = common_dominator(parent, other, &idoms, &rank);
            }
            if idoms[block] != parent {
                idoms[block] = parent;
                changed = true;
            }
        }
        if !changed {
            return idoms;
        }
    }
}
