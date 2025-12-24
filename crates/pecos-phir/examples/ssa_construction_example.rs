//! Example showing SSA construction during parsing
//!
//! This demonstrates how we build SSA form incrementally while parsing,
//! without needing a separate AST.

use pecos_phir::{
    ops::{ClassicalOp, Operation, SSAValue},
    phir::{Block, Instruction, Terminator},
    types::{IntWidth, Type},
};
use std::collections::BTreeMap;

fn main() {
    println!("=== SSA Construction During Parsing ===\n");

    example_basic_ssa();
    example_phi_nodes();
    example_dominance_frontier();
}

/// Basic SSA construction
fn example_basic_ssa() {
    struct SSABuilder {
        next_id: u32,
        current_block: Block,
        value_map: BTreeMap<String, SSAValue>,
    }

    impl SSABuilder {
        fn new() -> Self {
            Self {
                next_id: 1,
                current_block: Block::new(Some("entry".to_string())),
                value_map: BTreeMap::new(),
            }
        }

        fn new_ssa_value(&mut self) -> SSAValue {
            let val = SSAValue::new(self.next_id);
            self.next_id += 1;
            val
        }

        fn define(&mut self, name: &str) -> SSAValue {
            let ssa = self.new_ssa_value();
            self.value_map.insert(name.to_string(), ssa);
            println!("  Defined {name} = {ssa}");
            ssa
        }

        fn lookup(&self, name: &str) -> Option<&SSAValue> {
            self.value_map.get(name)
        }
    }

    println!("1. Basic SSA Construction");
    println!("------------------------");

    // Parsing: x = 5; y = x + 10; return y
    let mut builder = SSABuilder::new();

    // Parse: x = 5
    let x_ssa = builder.define("x");
    let const_5 = Instruction::new(
        Operation::Classical(ClassicalOp::ConstInt(5)),
        vec![],
        vec![x_ssa],
        vec![Type::Int(IntWidth::I32)],
    );
    builder.current_block.add_instruction(const_5);

    // Parse: y = x + 10
    let y_ssa = builder.define("y");
    let x_use = *builder.lookup("x").unwrap();
    let const_10_ssa = builder.new_ssa_value();

    let const_10 = Instruction::new(
        Operation::Classical(ClassicalOp::ConstInt(10)),
        vec![],
        vec![const_10_ssa],
        vec![Type::Int(IntWidth::I32)],
    );
    builder.current_block.add_instruction(const_10);

    let add = Instruction::new(
        Operation::Classical(ClassicalOp::Add),
        vec![x_use, const_10_ssa],
        vec![y_ssa],
        vec![Type::Int(IntWidth::I32)],
    );
    builder.current_block.add_instruction(add);
    println!("  Used {} in addition", builder.lookup("x").unwrap());

    // Parse: return y
    let y_use = *builder.lookup("y").unwrap();
    builder.current_block.set_terminator(Terminator::Return {
        values: vec![y_use],
    });
    println!("  Returned {y_use}");

    println!("\n  SSA form constructed during parsing!\n");
}

#[derive(Debug)]
struct BranchDefs {
    then_defs: BTreeMap<String, SSAValue>,
    else_defs: BTreeMap<String, SSAValue>,
}

/// Handling control flow with phi nodes
fn example_phi_nodes() {
    println!("2. Phi Nodes for Control Flow");
    println!("-----------------------------");

    // Parsing:
    // During parsing, we track which variables are defined in each branch

    // ```
    // if (cond) {
    //   x = 1
    // } else {
    //   x = 2
    // }
    // return x  // Which x?
    // ```

    println!("  Parsing if-else with variable definitions:");

    let mut branch_defs = BranchDefs {
        then_defs: BTreeMap::new(),
        else_defs: BTreeMap::new(),
    };

    // In then branch: x = 1
    let x_then = SSAValue::new(10);
    branch_defs.then_defs.insert("x".to_string(), x_then);
    println!("    Then branch: x = {x_then}");

    // In else branch: x = 2
    let x_else = SSAValue::new(11);
    branch_defs.else_defs.insert("x".to_string(), x_else);
    println!("    Else branch: x = {x_else}");

    // At merge point, create phi node
    println!("\n  Creating merge block with phi node:");

    let mut merge_block = Block::new(Some("merge".to_string()));

    // Phi node for x
    let x_phi = SSAValue::new(12);
    merge_block.arguments.push(pecos_phir::phir::BlockArgument {
        value: x_phi,
        ty: Type::Int(IntWidth::I32),
        name: Some("x.phi".to_string()),
    });

    println!("    {x_phi} = phi [{x_then} from then], [{x_else} from else]");

    // Now 'return x' uses the phi node
    println!("    return {x_phi} (the phi node)");

    println!("\n  Phi nodes created at control flow merge points!\n");
}

#[derive(Debug)]
#[allow(dead_code)]
struct DefSite {
    block: String,
    ssa_value: SSAValue,
}

/// Example with dominance frontiers
fn example_dominance_frontier() {
    println!("3. Dominance Frontiers and Phi Placement");
    println!("----------------------------------------");

    // More complex example:
    // Track variable definitions and their dominance frontiers

    // ```
    // x = 0
    // while (cond) {
    //   x = x + 1
    // }
    // return x
    // ```

    println!("  Parsing while loop with mutations:");

    let mut var_defs: BTreeMap<String, Vec<DefSite>> = BTreeMap::new();

    // Entry block: x = 0
    let x_init = SSAValue::new(20);
    var_defs.entry("x".to_string()).or_default().push(DefSite {
        block: "entry".to_string(),
        ssa_value: x_init,
    });
    println!("    entry: x = {x_init} (initial value)");

    // Loop header needs phi node (dominance frontier of loop body)
    let x_phi = SSAValue::new(21);
    println!("    loop.header: {x_phi} = phi [{x_init} from entry], [%x.next from loop.body]");

    // Loop body: x = x + 1
    let x_next = SSAValue::new(22);
    var_defs.entry("x".to_string()).or_default().push(DefSite {
        block: "loop.body".to_string(),
        ssa_value: x_next,
    });
    println!("    loop.body: {x_next} = {x_phi} + 1");

    // Exit block uses the phi node
    println!("    exit: return {x_phi} (from loop header phi)");

    println!("\n  Algorithm:");
    println!("  1. Compute dominance tree");
    println!("  2. Find dominance frontiers");
    println!("  3. Place phi nodes at frontiers");
    println!("  4. Rename variables in SSA form");

    println!("\n  SSA construction complete with minimal phi nodes!\n");
}
