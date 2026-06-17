//! Bell port — the round-2 decision gate for build-PHIR-on-pliron.
//! See pecos-docs design/slr-phir-vision.md §7.1 + slr-phir-vision-qis-port-sketch.md.
//!
//! Milestone 0: prove the pecos-engines sim seam with a hand-built Bell ByteMessage.
//! Milestone 1: build the SAME Bell program as pliron `qec` IR (allocator/slot model),
//!   emit the ByteMessage from a pliron op-walk using an EXPLICIT `Value -> qubit index`
//!   side-table (NOT the `SSAValue`-as-u32 overload that pecos-phir uses), then run it
//!   through the SAME unchanged pecos-engines StateVecEngine and assert the Bell invariant.
//!
//! This is the concrete proof the round-2 decision is gated on: typed pliron ops feed the
//! real backend through explicit qubit/result maps, and the ByteMessage seam is untouched.

use std::any::Any;
use std::collections::{BTreeSet, HashMap};

use awint::bw;
use pecos_engines::byte_message::ByteMessage;
use pecos_engines::hybrid::builder::HybridEngineBuilder;
use pecos_engines::quantum::StateVecEngine;
use pecos_engines::{ClassicalEngine, ControlEngine, Data, Engine, EngineStage, PecosError, Shot};
use pliron::{
    builtin::{
        attributes::IntegerAttr,
        op_interfaces::{
            IsTerminatorInterface, NOpdsInterface, NResultsInterface, OneResultInterface,
            SingleBlockRegionInterface,
        },
        ops::{FuncOp, ModuleOp},
        types::{FunctionType, IntegerType, Signedness},
    },
    attribute::AttrObj,
    basic_block::BasicBlock,
    common_traits::Verify,
    context::{Context, Ptr},
    derive::{pliron_op, pliron_type},
    linked_list::ContainsLinkedList,
    op::{verify_op, Op},
    operation::Operation,
    printable::Printable,
    result::Result,
    r#type::{TypeObj, Typed},
    utils::apint::APInt,
    value::Value,
    verify_err,
};

// ===================== Milestone 0: hand-built Bell ByteMessage =====================

fn bell_message() -> ByteMessage {
    let mut b = ByteMessage::quantum_operations_builder();
    b.h(&[0]);
    b.cx(&[(0, 1)]);
    b.mz(&[0]);
    b.mz(&[1]);
    b.build()
}

/// Run a Bell ByteMessage through the real state-vector simulator; assert each shot is 00 or 11.
fn run_and_check(label: &str, msg: ByteMessage, shots: usize) {
    let mut engine = StateVecEngine::new(2);
    let (mut saw0, mut saw3) = (false, false);
    for _ in 0..shots {
        engine.reset().unwrap();
        let out = engine.process(msg.clone()).unwrap();
        let o = out.outcomes().unwrap();
        assert_eq!(o.len(), 2, "expected 2 measurement outcomes");
        let combined = (o[1] << 1) | o[0]; // q1 q0
        assert!(combined == 0 || combined == 3, "{label}: Bell must be 00 or 11, got {combined}");
        saw0 |= combined == 0;
        saw3 |= combined == 3;
    }
    assert!(saw0 && saw3, "{label}: expected both 00 and 11 over {shots} shots");
    println!("[{label}] OK -- all {shots} shots in {{0,3}}, saw both 00 and 11");
}

// ===================== the qec pliron dialect (allocator/slot model) =====================

#[pliron_type(name = "qec.alloc", generate_get = true, format, verifier = "succ")]
#[derive(Hash, PartialEq, Eq, Debug)]
pub struct AllocType;

#[pliron_type(name = "qec.qubitref", generate_get = true, format, verifier = "succ")]
#[derive(Hash, PartialEq, Eq, Debug)]
pub struct QubitRefType;

fn alloc_ty(ctx: &Context) -> Ptr<TypeObj> {
    AllocType::get(ctx).into()
}
fn qubitref_ty(ctx: &Context) -> Ptr<TypeObj> {
    QubitRefType::get(ctx).into()
}

mod slot_attr {
    use pliron::dict_key;
    dict_key!(INDEX, "qec_slot_index");
}

#[pliron_op(name = "qec.qalloc", format, interfaces = [NOpdsInterface<0>, OneResultInterface, NResultsInterface<1>], verifier = "succ")]
pub struct QallocOp;
impl QallocOp {
    pub fn new(ctx: &mut Context) -> Self {
        let a = alloc_ty(ctx);
        let op = Operation::new(ctx, Self::get_concrete_op_info(), vec![a], vec![], vec![], 0);
        QallocOp { op }
    }
}

#[pliron_op(name = "qec.slot", format, interfaces = [NOpdsInterface<1>, OneResultInterface, NResultsInterface<1>])]
pub struct SlotOp;
impl SlotOp {
    pub fn new(ctx: &mut Context, alloc: Value, index: u64) -> Self {
        let r = qubitref_ty(ctx);
        let op = Operation::new(ctx, Self::get_concrete_op_info(), vec![r], vec![alloc], vec![], 0);
        let i64_ty = IntegerType::get(ctx, 64, Signedness::Signless);
        let attr = IntegerAttr::new(i64_ty, APInt::from_u64(index, bw(64)));
        op.deref_mut(ctx).attributes.0.insert(slot_attr::INDEX.clone(), Box::new(attr));
        SlotOp { op }
    }
    pub fn index(&self, ctx: &Context) -> u64 {
        let op = self.get_operation().deref(ctx);
        let attr: AttrObj = op.attributes.0.get(&*slot_attr::INDEX).expect("slot index attr").clone();
        let int_attr = attr.downcast::<IntegerAttr>().unwrap_or_else(|_| panic!("not IntegerAttr"));
        Into::<APInt>::into(*int_attr).to_u64()
    }
}
impl Verify for SlotOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        if self.get_operation().deref(ctx).get_operand(0).get_type(ctx) != alloc_ty(ctx) {
            return verify_err!(self.loc(ctx), "qec.slot operand must be a qec.alloc");
        }
        Ok(())
    }
}

#[pliron_op(name = "qec.prepare", format, interfaces = [NOpdsInterface<1>, NResultsInterface<0>])]
pub struct PrepareOp;
impl PrepareOp {
    pub fn new(ctx: &mut Context, qref: Value) -> Self {
        let op = Operation::new(ctx, Self::get_concrete_op_info(), vec![], vec![qref], vec![], 0);
        PrepareOp { op }
    }
}
impl Verify for PrepareOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        if self.get_operation().deref(ctx).get_operand(0).get_type(ctx) != qubitref_ty(ctx) {
            return verify_err!(self.loc(ctx), "qec.prepare operand must be a qec.qubitref");
        }
        Ok(())
    }
}

#[pliron_op(name = "qec.h", format, interfaces = [NOpdsInterface<1>, NResultsInterface<0>])]
pub struct HOp;
impl HOp {
    pub fn new(ctx: &mut Context, qref: Value) -> Self {
        let op = Operation::new(ctx, Self::get_concrete_op_info(), vec![], vec![qref], vec![], 0);
        HOp { op }
    }
}
impl Verify for HOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        if self.get_operation().deref(ctx).get_operand(0).get_type(ctx) != qubitref_ty(ctx) {
            return verify_err!(self.loc(ctx), "qec.h operand must be a qec.qubitref");
        }
        Ok(())
    }
}

#[pliron_op(name = "qec.cx", format, interfaces = [NOpdsInterface<2>, NResultsInterface<0>])]
pub struct CxOp;
impl CxOp {
    pub fn new(ctx: &mut Context, ctrl: Value, tgt: Value) -> Self {
        let op = Operation::new(ctx, Self::get_concrete_op_info(), vec![], vec![ctrl, tgt], vec![], 0);
        CxOp { op }
    }
}
impl Verify for CxOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        for i in 0..2 {
            if self.get_operation().deref(ctx).get_operand(i).get_type(ctx) != qubitref_ty(ctx) {
                return verify_err!(self.loc(ctx), "qec.cx operands must be qec.qubitref");
            }
        }
        Ok(())
    }
}

#[pliron_op(name = "qec.measure", format, interfaces = [NOpdsInterface<1>, OneResultInterface, NResultsInterface<1>])]
pub struct MeasureOp;
impl MeasureOp {
    pub fn new(ctx: &mut Context, qref: Value) -> Self {
        let i1 = IntegerType::get(ctx, 1, Signedness::Signless);
        let op = Operation::new(ctx, Self::get_concrete_op_info(), vec![i1.into()], vec![qref], vec![], 0);
        MeasureOp { op }
    }
}
impl Verify for MeasureOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        if self.get_operation().deref(ctx).get_operand(0).get_type(ctx) != qubitref_ty(ctx) {
            return verify_err!(self.loc(ctx), "qec.measure operand must be a qec.qubitref");
        }
        Ok(())
    }
}

/// `qec.cond_x(cond: i1, target: qubitref)` -- apply X to `target` iff the measurement result
/// `cond` is 1. The measurement-conditioned gate that forces a classical decision between batches.
#[pliron_op(name = "qec.cond_x", format, interfaces = [NOpdsInterface<2>, NResultsInterface<0>])]
pub struct CondXOp;
impl CondXOp {
    pub fn new(ctx: &mut Context, cond: Value, target: Value) -> Self {
        let op = Operation::new(ctx, Self::get_concrete_op_info(), vec![], vec![cond, target], vec![], 0);
        CondXOp { op }
    }
}
impl Verify for CondXOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        // target (operand 1) must be a qubitref; the condition (operand 0) is a measurement-result
        // i1 by construction (constructing the i1 type needs &mut Context, unavailable in verify).
        if self.get_operation().deref(ctx).get_operand(1).get_type(ctx) != qubitref_ty(ctx) {
            return verify_err!(self.loc(ctx), "qec.cond_x target must be a qec.qubitref");
        }
        Ok(())
    }
}

#[pliron_op(name = "qec.end", format, interfaces = [IsTerminatorInterface, NOpdsInterface<0>, NResultsInterface<0>], verifier = "succ")]
pub struct EndOp;
impl EndOp {
    pub fn new(ctx: &mut Context) -> Self {
        let op = Operation::new(ctx, Self::get_concrete_op_info(), vec![], vec![], vec![], 0);
        EndOp { op }
    }
}

// ===================== Milestone 1: build Bell as pliron IR, emit ByteMessage =====================

/// Build `fn main() { q = qalloc; q0 = slot(q,0); q1 = slot(q,1); prepare q0; prepare q1;
/// h q0; cx q0,q1; m0 = measure q0; m1 = measure q1; end }` as pliron IR; return the func block.
fn build_bell_ir(ctx: &mut Context) -> (ModuleOp, Ptr<BasicBlock>) {
    let module = ModuleOp::new(ctx, "bell".try_into().unwrap());
    let func_ty = FunctionType::get(ctx, vec![], vec![]);
    let func = FuncOp::new(ctx, "main".try_into().unwrap(), func_ty);
    module.append_operation(ctx, func.get_operation(), 0);
    let bb = func.get_entry_block(ctx);

    macro_rules! push {
        ($op:expr) => {{ let o = $op; o.get_operation().insert_at_back(bb, ctx); o }};
    }

    let q = push!(QallocOp::new(ctx));
    let qv = q.get_result(ctx);
    let s0 = push!(SlotOp::new(ctx, qv, 0));
    let s0v = s0.get_result(ctx);
    let s1 = push!(SlotOp::new(ctx, qv, 1));
    let s1v = s1.get_result(ctx);
    push!(PrepareOp::new(ctx, s0v));
    push!(PrepareOp::new(ctx, s1v));
    push!(HOp::new(ctx, s0v));
    push!(CxOp::new(ctx, s0v, s1v));
    push!(MeasureOp::new(ctx, s0v));
    push!(MeasureOp::new(ctx, s1v));
    push!(EndOp::new(ctx));
    (module, bb)
}

/// Walk the pliron func block and emit a ByteMessage. The ONLY qubit-identity source is an
/// explicit `Value(qubitref) -> qubit index` map populated from each `qec.slot`'s index
/// attribute -- pliron Values are op-defined handles, never reused as dense qubit indices.
fn emit_bytemessage(ctx: &Context, block: Ptr<BasicBlock>) -> ByteMessage {
    let mut qubit_of: HashMap<Value, usize> = HashMap::new();
    let mut b = ByteMessage::quantum_operations_builder();
    let ops: Vec<Ptr<Operation>> = block.deref(ctx).iter(ctx).collect();
    for op in ops {
        if let Some(s) = Operation::get_op::<SlotOp>(op, ctx) {
            qubit_of.insert(s.get_result(ctx), s.index(ctx) as usize);
        } else if let Some(p) = Operation::get_op::<PrepareOp>(op, ctx) {
            let q = qubit_of[&p.get_operation().deref(ctx).get_operand(0)];
            b.pz(&[q]);
        } else if let Some(h) = Operation::get_op::<HOp>(op, ctx) {
            let q = qubit_of[&h.get_operation().deref(ctx).get_operand(0)];
            b.h(&[q]);
        } else if let Some(cx) = Operation::get_op::<CxOp>(op, ctx) {
            let opn = cx.get_operation();
            let c = qubit_of[&opn.deref(ctx).get_operand(0)];
            let t = qubit_of[&opn.deref(ctx).get_operand(1)];
            b.cx(&[(c, t)]);
        } else if let Some(m) = Operation::get_op::<MeasureOp>(op, ctx) {
            let q = qubit_of[&m.get_operation().deref(ctx).get_operand(0)];
            b.mz(&[q]); // outcome order = mz-emission order
        }
        // qalloc / end: nothing to emit
    }
    b.build()
}

fn run_milestone_1() {
    let ctx = &mut Context::new();
    let (module, bb) = build_bell_ir(ctx);
    println!("=== Bell as pliron qec IR ===");
    println!("{}", module.get_operation().disp(ctx));
    match verify_op(&module, ctx) {
        Ok(()) => println!("[milestone-1 verify] OK"),
        Err(e) => panic!("[milestone-1 verify] FAILED: {}", e.disp(ctx)),
    }
    let msg = emit_bytemessage(ctx, bb);
    run_and_check("milestone-1 pliron-emitted Bell", msg, 200);
}

// ===================== Milestone 2: pliron op-walk AS a real ClassicalEngine =====================

/// A `ClassicalControlEngine` whose command stream is the ByteMessage emitted from the pliron
/// `qec` IR walk. Bell is straight-line (one batch, no classical feedback), so it sends the
/// whole program once, stores the outcomes, and records register "c" = (q1<<1)|q0 as a Shot.
#[derive(Clone)]
struct PlironBellEngine {
    msg: ByteMessage,
    num_qubits: usize,
    sent: bool,
    outcomes: Vec<u32>,
}

impl Engine for PlironBellEngine {
    type Input = ();
    type Output = Shot;
    fn process(&mut self, _input: ()) -> std::result::Result<Shot, PecosError> {
        self.get_results()
    }
    fn reset(&mut self) -> std::result::Result<(), PecosError> {
        self.sent = false;
        self.outcomes.clear();
        Ok(())
    }
}

impl ClassicalEngine for PlironBellEngine {
    fn num_qubits(&self) -> usize {
        self.num_qubits
    }
    fn generate_commands(&mut self) -> std::result::Result<ByteMessage, PecosError> {
        if self.sent {
            Ok(ByteMessage::create_empty())
        } else {
            self.sent = true;
            Ok(self.msg.clone())
        }
    }
    fn handle_measurements(&mut self, message: ByteMessage) -> std::result::Result<(), PecosError> {
        self.outcomes = message.outcomes()?;
        Ok(())
    }
    fn get_results(&self) -> std::result::Result<Shot, PecosError> {
        let combined = if self.outcomes.len() >= 2 {
            (self.outcomes[1] << 1) | self.outcomes[0]
        } else {
            0
        };
        let mut shot = Shot::default();
        shot.add_register("c", combined, 2);
        Ok(shot)
    }
    fn compile(&self) -> std::result::Result<(), PecosError> {
        Ok(())
    }
    fn reset(&mut self) -> std::result::Result<(), PecosError> {
        Engine::reset(self)
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

impl ControlEngine for PlironBellEngine {
    type Input = ();
    type Output = Shot;
    type EngineInput = ByteMessage;
    type EngineOutput = ByteMessage;
    fn start(&mut self, _input: ()) -> std::result::Result<EngineStage<ByteMessage, Shot>, PecosError> {
        self.sent = false;
        self.outcomes.clear();
        let cmds = self.generate_commands()?;
        if cmds.as_bytes().is_empty() {
            Ok(EngineStage::Complete(self.get_results()?))
        } else {
            Ok(EngineStage::NeedsProcessing(cmds))
        }
    }
    fn continue_processing(&mut self, measurements: ByteMessage) -> std::result::Result<EngineStage<ByteMessage, Shot>, PecosError> {
        self.handle_measurements(measurements)?;
        // Bell is a single batch with no classical feedback: once the (only) batch's measurements
        // are in, we are done. (A general engine would loop on generate_commands + a `finished`
        // flag like PhirEngine; `create_empty().as_bytes()` is NOT byte-empty, so it is not a
        // reliable termination signal on its own.)
        Ok(EngineStage::Complete(self.get_results()?))
    }
    fn reset(&mut self) -> std::result::Result<(), PecosError> {
        Engine::reset(self)
    }
}

fn run_milestone_2() {
    let ctx = &mut Context::new();
    let (module, bb) = build_bell_ir(ctx);
    verify_op(&module, ctx).unwrap_or_else(|e| panic!("[milestone-2 verify] FAILED: {}", e.disp(ctx)));
    let msg = emit_bytemessage(ctx, bb); // pliron-emitted Bell ByteMessage

    let mut engine = PlironBellEngine { msg, num_qubits: 2, sent: false, outcomes: Vec::new() };
    let mut qsys = StateVecEngine::new(2);

    // Drive the ControlEngine protocol by hand (start -> {process, continue_processing} loop ->
    // Complete) -- this IS the engine seam HybridEngine automates; doing it explicitly proves the
    // pliron-backed ClassicalControlEngine feeds the real StateVecEngine and yields a Shot.
    let (mut saw0, mut saw3) = (false, false);
    for _ in 0..200 {
        qsys.reset().unwrap();
        let mut stage = ControlEngine::start(&mut engine, ()).unwrap();
        let shot = loop {
            match stage {
                EngineStage::Complete(s) => break s,
                EngineStage::NeedsProcessing(cmd) => {
                    let meas = qsys.process(cmd).unwrap();
                    stage = engine.continue_processing(meas).unwrap();
                }
            }
        };
        let v = shot.data.get("c").and_then(Data::as_u32).expect("register c");
        assert!(v == 0 || v == 3, "milestone-2: Bell via ClassicalEngine must be 00 or 11, got {v}");
        saw0 |= v == 0;
        saw3 |= v == 3;
    }
    assert!(saw0 && saw3, "milestone-2: expected both 00 and 11 over 200 shots");
    println!("[milestone-2 pliron ClassicalControlEngine -> StateVecEngine] OK -- 200 shots in {{0,3}}, saw both (register \"c\")");
}

// ===================== Milestone 3: parse a real bell.ll into pliron qec IR =====================

/// Minimal QIS-LLVM-IR -> pliron `qec` IR parser for the Bell fixture (`examples/llvm/bell.ll`).
/// Recognizes `__quantum__qis__{h,cx,m}__body` calls with integer qubit args. This proves the
/// LLVM-IR frontend ports onto pliron ops (not just hand-built IR) -- the last gate-fidelity item.
fn parse_bell_ll(ctx: &mut Context, src: &str) -> (ModuleOp, Ptr<BasicBlock>) {
    fn i64_args(line: &str) -> Vec<usize> {
        match (line.find('('), line.rfind(')')) {
            (Some(l), Some(r)) if r > l => line[l + 1..r]
                .split(',')
                .filter_map(|t| t.trim().strip_prefix("i64 ").and_then(|n| n.trim().parse::<usize>().ok()))
                .collect(),
            _ => Vec::new(),
        }
    }
    // pass 1: collect the gate/measure stream + the set of qubits referenced
    let mut parsed: Vec<(&str, Vec<usize>)> = Vec::new();
    let mut qubits: BTreeSet<usize> = BTreeSet::new();
    for line in src.lines() {
        let l = line.trim();
        if !(l.starts_with("call ") || l.contains("= call ")) {
            continue;
        }
        if l.contains("__quantum__qis__h__body") {
            let a = i64_args(l);
            qubits.insert(a[0]);
            parsed.push(("h", a));
        } else if l.contains("__quantum__qis__cx__body") {
            let a = i64_args(l);
            qubits.insert(a[0]);
            qubits.insert(a[1]);
            parsed.push(("cx", a));
        } else if l.contains("__quantum__qis__m__body") {
            let a = i64_args(l);
            qubits.insert(a[0]);
            parsed.push(("m", a));
        }
    }
    // pass 2: build the pliron qec IR
    let module = ModuleOp::new(ctx, "bell_from_ll".try_into().unwrap());
    let func_ty = FunctionType::get(ctx, vec![], vec![]);
    let func = FuncOp::new(ctx, "qmain".try_into().unwrap(), func_ty);
    module.append_operation(ctx, func.get_operation(), 0);
    let bb = func.get_entry_block(ctx);
    macro_rules! push {
        ($op:expr) => {{ let o = $op; o.get_operation().insert_at_back(bb, ctx); o }};
    }

    let q = push!(QallocOp::new(ctx));
    let qv = q.get_result(ctx);
    let mut slot_of: HashMap<usize, Value> = HashMap::new();
    for &idx in &qubits {
        let s = push!(SlotOp::new(ctx, qv, idx as u64));
        let sv = s.get_result(ctx);
        slot_of.insert(idx, sv);
        push!(PrepareOp::new(ctx, sv)); // QIS qubits start |0>; that implicit init IS a prepare in the qec model
    }
    for (op, a) in &parsed {
        match *op {
            "h" => {
                push!(HOp::new(ctx, slot_of[&a[0]]));
            }
            "cx" => {
                push!(CxOp::new(ctx, slot_of[&a[0]], slot_of[&a[1]]));
            }
            "m" => {
                push!(MeasureOp::new(ctx, slot_of[&a[0]]));
            }
            _ => {}
        }
    }
    push!(EndOp::new(ctx));
    (module, bb)
}

fn run_milestone_3() {
    let ctx = &mut Context::new();
    let src = include_str!("../../../examples/llvm/bell.ll");
    let (module, bb) = parse_bell_ll(ctx, src);
    println!("=== bell.ll parsed into pliron qec IR ===");
    println!("{}", module.get_operation().disp(ctx));
    verify_op(&module, ctx).unwrap_or_else(|e| panic!("[milestone-3 verify] FAILED: {}", e.disp(ctx)));
    let msg = emit_bytemessage(ctx, bb);
    run_and_check("milestone-3 bell.ll -> pliron qec -> sim", msg, 200);
}

// ===================== Milestone 4: adaptive, multi-batch, measurement-conditioned =====================
// Program: prepare q0,q1; h q0; m0 = measure q0; cond_x(m0, q1); mf1 = measure q1.
// The engine must send batch1 (gates + mz q0), get m0, make a CLASSICAL decision, then send a
// SECOND batch (conditionally x q1, then mz q1). Invariant: mf1 == m0 (q1 flips iff m0==1) -- which
// only holds if the measurement-conditioned feedback works across the two quantum batches.

#[derive(Clone, Copy)]
enum Cmd {
    Pz(usize),
    H(usize),
    Cx(usize, usize),
    Mz(usize),
}

#[derive(Clone)]
struct AdaptivePlan {
    batch1: Vec<Cmd>,        // gates + the conditioning Mz
    cond_outcome_idx: usize, // index in batch1's mz-outcomes that gates cond_x
    cond_target: usize,      // qubit to X iff that outcome == 1
    batch2: Vec<Cmd>,        // post-condition ops (final Mz)
}

/// Walk a `qec` block once and split it into the two-batch adaptive plan at the `cond_x` boundary.
/// Qubit identity comes only from the explicit slot-index map (same discipline as the Bell port).
fn plan_from_ir(ctx: &Context, block: Ptr<BasicBlock>) -> AdaptivePlan {
    let mut qubit_of: HashMap<Value, usize> = HashMap::new();
    let (mut batch1, mut batch2): (Vec<Cmd>, Vec<Cmd>) = (Vec::new(), Vec::new());
    let mut after_cond = false;
    let mut mz_b1 = 0usize;
    let (mut cond_outcome_idx, mut cond_target) = (0usize, 0usize);
    let ops: Vec<Ptr<Operation>> = block.deref(ctx).iter(ctx).collect();
    for op in ops {
        if let Some(s) = Operation::get_op::<SlotOp>(op, ctx) {
            qubit_of.insert(s.get_result(ctx), s.index(ctx) as usize);
        } else if let Some(p) = Operation::get_op::<PrepareOp>(op, ctx) {
            let q = qubit_of[&p.get_operation().deref(ctx).get_operand(0)];
            if after_cond { batch2.push(Cmd::Pz(q)) } else { batch1.push(Cmd::Pz(q)) }
        } else if let Some(h) = Operation::get_op::<HOp>(op, ctx) {
            let q = qubit_of[&h.get_operation().deref(ctx).get_operand(0)];
            if after_cond { batch2.push(Cmd::H(q)) } else { batch1.push(Cmd::H(q)) }
        } else if let Some(cx) = Operation::get_op::<CxOp>(op, ctx) {
            let opn = cx.get_operation();
            let c = qubit_of[&opn.deref(ctx).get_operand(0)];
            let t = qubit_of[&opn.deref(ctx).get_operand(1)];
            if after_cond { batch2.push(Cmd::Cx(c, t)) } else { batch1.push(Cmd::Cx(c, t)) }
        } else if let Some(m) = Operation::get_op::<MeasureOp>(op, ctx) {
            let q = qubit_of[&m.get_operation().deref(ctx).get_operand(0)];
            if after_cond {
                batch2.push(Cmd::Mz(q));
            } else {
                cond_outcome_idx = mz_b1;
                mz_b1 += 1;
                batch1.push(Cmd::Mz(q));
            }
        } else if let Some(c) = Operation::get_op::<CondXOp>(op, ctx) {
            cond_target = qubit_of[&c.get_operation().deref(ctx).get_operand(1)];
            after_cond = true;
        }
    }
    AdaptivePlan { batch1, cond_outcome_idx, cond_target, batch2 }
}

fn emit_cmds(b: &mut pecos_engines::byte_message::ByteMessageBuilder, cmds: &[Cmd]) {
    for c in cmds {
        match *c {
            Cmd::Pz(q) => { b.pz(&[q]); }
            Cmd::H(q) => { b.h(&[q]); }
            Cmd::Cx(c0, t) => { b.cx(&[(c0, t)]); }
            Cmd::Mz(q) => { b.mz(&[q]); }
        }
    }
}

#[derive(Clone)]
struct PlironAdaptiveEngine {
    plan: AdaptivePlan,
    stage: u8, // 0 fresh, 1 batch1 sent, 2 batch2 sent
    b1: Vec<u32>,
    b2: Vec<u32>,
}

impl PlironAdaptiveEngine {
    fn batch1_msg(&self) -> ByteMessage {
        let mut b = ByteMessage::quantum_operations_builder();
        emit_cmds(&mut b, &self.plan.batch1);
        b.build()
    }
    fn batch2_msg(&self) -> ByteMessage {
        let mut b = ByteMessage::quantum_operations_builder();
        // the classical decision: apply X iff the conditioning measurement was 1
        if self.b1.get(self.plan.cond_outcome_idx).copied() == Some(1) {
            b.x(&[self.plan.cond_target]);
        }
        emit_cmds(&mut b, &self.plan.batch2);
        b.build()
    }
}

impl Engine for PlironAdaptiveEngine {
    type Input = ();
    type Output = Shot;
    fn process(&mut self, _i: ()) -> std::result::Result<Shot, PecosError> {
        self.get_results()
    }
    fn reset(&mut self) -> std::result::Result<(), PecosError> {
        self.stage = 0;
        self.b1.clear();
        self.b2.clear();
        Ok(())
    }
}

impl ClassicalEngine for PlironAdaptiveEngine {
    fn num_qubits(&self) -> usize {
        2
    }
    fn generate_commands(&mut self) -> std::result::Result<ByteMessage, PecosError> {
        if self.stage == 0 {
            self.stage = 1;
            Ok(self.batch1_msg())
        } else {
            Ok(ByteMessage::create_empty())
        }
    }
    fn handle_measurements(&mut self, m: ByteMessage) -> std::result::Result<(), PecosError> {
        let o = m.outcomes()?;
        if self.stage == 1 { self.b1 = o } else { self.b2 = o }
        Ok(())
    }
    fn get_results(&self) -> std::result::Result<Shot, PecosError> {
        let mut shot = Shot::default();
        shot.add_register("mid", self.b1.first().copied().unwrap_or(0), 1);
        shot.add_register("final", self.b2.first().copied().unwrap_or(0), 1);
        Ok(shot)
    }
    fn compile(&self) -> std::result::Result<(), PecosError> {
        Ok(())
    }
    fn reset(&mut self) -> std::result::Result<(), PecosError> {
        Engine::reset(self)
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

impl ControlEngine for PlironAdaptiveEngine {
    type Input = ();
    type Output = Shot;
    type EngineInput = ByteMessage;
    type EngineOutput = ByteMessage;
    fn start(&mut self, _i: ()) -> std::result::Result<EngineStage<ByteMessage, Shot>, PecosError> {
        self.stage = 0;
        self.b1.clear();
        self.b2.clear();
        self.stage = 1;
        Ok(EngineStage::NeedsProcessing(self.batch1_msg()))
    }
    fn continue_processing(&mut self, meas: ByteMessage) -> std::result::Result<EngineStage<ByteMessage, Shot>, PecosError> {
        if self.stage == 1 {
            self.b1 = meas.outcomes()?;
            self.stage = 2;
            Ok(EngineStage::NeedsProcessing(self.batch2_msg()))
        } else {
            self.b2 = meas.outcomes()?;
            Ok(EngineStage::Complete(self.get_results()?))
        }
    }
    fn reset(&mut self) -> std::result::Result<(), PecosError> {
        Engine::reset(self)
    }
}

fn build_adaptive_ir(ctx: &mut Context) -> (ModuleOp, Ptr<BasicBlock>) {
    let module = ModuleOp::new(ctx, "adaptive".try_into().unwrap());
    let func_ty = FunctionType::get(ctx, vec![], vec![]);
    let func = FuncOp::new(ctx, "main".try_into().unwrap(), func_ty);
    module.append_operation(ctx, func.get_operation(), 0);
    let bb = func.get_entry_block(ctx);
    macro_rules! push {
        ($op:expr) => {{ let o = $op; o.get_operation().insert_at_back(bb, ctx); o }};
    }
    let q = push!(QallocOp::new(ctx));
    let qv = q.get_result(ctx);
    let s0 = push!(SlotOp::new(ctx, qv, 0));
    let s0v = s0.get_result(ctx);
    let s1 = push!(SlotOp::new(ctx, qv, 1));
    let s1v = s1.get_result(ctx);
    push!(PrepareOp::new(ctx, s0v));
    push!(PrepareOp::new(ctx, s1v));
    push!(HOp::new(ctx, s0v));
    let m0 = push!(MeasureOp::new(ctx, s0v));
    let m0v = m0.get_result(ctx);
    push!(CondXOp::new(ctx, m0v, s1v)); // X q1 iff m0 == 1
    push!(MeasureOp::new(ctx, s1v));
    push!(EndOp::new(ctx));
    (module, bb)
}

fn run_milestone_4() {
    let ctx = &mut Context::new();
    let (module, bb) = build_adaptive_ir(ctx);
    println!("=== adaptive (mid-measure -> cond_x -> final) pliron qec IR ===");
    println!("{}", module.get_operation().disp(ctx));
    verify_op(&module, ctx).unwrap_or_else(|e| panic!("[milestone-4 verify] FAILED: {}", e.disp(ctx)));
    let plan = plan_from_ir(ctx, bb);
    let engine = PlironAdaptiveEngine { plan, stage: 0, b1: Vec::new(), b2: Vec::new() };

    // Drive through the REAL HybridEngine this time (closes the "wrapper unrun" gap from round 3).
    let mut hybrid = HybridEngineBuilder::new()
        .with_classical_engine(Box::new(engine))
        .with_quantum_engine(Box::new(StateVecEngine::new(2)))
        .build();

    let (mut all_eq, mut saw_mid0, mut saw_mid1) = (true, false, false);
    for _ in 0..200 {
        let shot = hybrid.run_shot().unwrap();
        let mid = shot.data.get("mid").and_then(Data::as_u32).expect("mid");
        let fin = shot.data.get("final").and_then(Data::as_u32).expect("final");
        if fin != mid {
            all_eq = false;
        }
        saw_mid0 |= mid == 0;
        saw_mid1 |= mid == 1;
        Engine::reset(&mut hybrid).unwrap();
    }
    assert!(all_eq, "milestone-4: final must equal mid in every shot (measurement-conditioned X feedback)");
    assert!(saw_mid0 && saw_mid1, "milestone-4: expected both mid=0 and mid=1 over 200 shots");
    println!("[milestone-4 adaptive multi-batch via HybridEngine] OK -- final==mid in all 200 shots, saw mid 0 and 1");
}

fn main() {
    run_and_check("milestone-0 hand-built Bell", bell_message(), 200);
    run_milestone_1();
    run_milestone_2();
    run_milestone_3();
    run_milestone_4();
}
