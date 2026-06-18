//! Parallel, non-default QIS-LLVM-IR -> pliron-PHIR path (the strangler crate, scoped to the
//! covered QIS-LLVM subset). See pecos-docs design/slr-phir-pliron-strangler-scope.md.
//!
//! A QIS-LLVM-IR program is lowered to a pliron `qec` dialect (allocator/slot model; measurements
//! are SSA values with metadata in a side-table registry; `result_record_output -> qec.record`
//! export) and run through the UNCHANGED pecos-engines seam (`ClassicalControlEngine` /
//! `ByteMessage` / `Shot`) -- no pliron type leaks into the engine API. Qubit identity comes from an
//! explicit `Value -> qubit index` map, NOT the `SSAValue`-as-u32 overload the incumbent uses.
//!
//! [`from_qis_llvm_ir_pliron`] is the narrow opt-in entry point: `.ll` text -> a boxed engine ready
//! for `HybridEngineBuilder`. The milestones M0-M7 + differential remain as the regression suite.

use std::any::Any;
use std::collections::{BTreeMap, BTreeSet, HashMap};

use awint::bw;
use pecos_core::Angle64;
use pecos_engines::byte_message::ByteMessage;
use pecos_engines::hybrid::builder::HybridEngineBuilder;
use pecos_engines::quantum::StateVecEngine;
use pecos_engines::{ClassicalControlEngine, ClassicalEngine, ControlEngine, Data, Engine, EngineStage, PecosError, Shot};
use pliron::{
    builtin::{
        attributes::IntegerAttr,
        op_interfaces::{
            IsTerminatorInterface, NOpdsInterface, NRegionsInterface, NResultsInterface,
            NoTerminatorInterface, OneResultInterface, SingleBlockRegionInterface,
        },
        ops::{FuncOp, ModuleOp},
        types::{FunctionType, IntegerType, Signedness},
    },
    attribute::AttrObj,
    basic_block::BasicBlock,
    common_traits::Verify,
    context::{Context, Ptr},
    derive::{pliron_attr, pliron_op, pliron_type},
    linked_list::ContainsLinkedList,
    op::{verify_op, Op},
    operation::Operation,
    printable::Printable,
    result::Result,
    r#type::{TypeObj, TypePtr, Typed},
    utils::apint::APInt,
    value::Value,
    verify_err,
};

// ===================== Milestone 0: hand-built Bell ByteMessage =====================

pub fn bell_message() -> ByteMessage {
    let mut b = ByteMessage::quantum_operations_builder();
    b.h(&[0]);
    b.cx(&[(0, 1)]);
    b.mz(&[0]);
    b.mz(&[1]);
    b.build()
}

/// Run a Bell ByteMessage through the real state-vector simulator; assert each shot is 00 or 11.
pub fn run_and_check(label: &str, msg: ByteMessage, shots: usize) {
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

pub fn alloc_ty(ctx: &Context) -> Ptr<TypeObj> {
    AllocType::get(ctx).into()
}
pub fn qubitref_ty(ctx: &Context) -> Ptr<TypeObj> {
    QubitRefType::get(ctx).into()
}

mod slot_attr {
    use pliron::dict_key;
    dict_key!(INDEX, "qec_slot_index");
}

mod angle_attr {
    use pliron::dict_key;
    dict_key!(ANGLE, "qec_angle");
}

/// A PECOS-native angle attribute: stores `Angle64`'s fixed-point *fraction* (a fraction of a full
/// turn, full circle = `2^64`) exactly -- no lossy f64 round-trip in the IR. `qec.rz/rx/ry` carry one.
#[pliron_attr(name = "qec.angle", format = "$0", verifier = "succ")]
#[derive(PartialEq, Eq, Clone, Debug, Hash)]
pub struct Angle64Attr(u64);
impl From<Angle64> for Angle64Attr {
    fn from(a: Angle64) -> Self {
        Angle64Attr(a.fraction())
    }
}
impl From<Angle64Attr> for Angle64 {
    fn from(a: Angle64Attr) -> Self {
        Angle64::new(a.0)
    }
}

// ===================== measurement-SSA registry (measurement-id-system.md, Phase 1) =====================
// A `qec.measure` op produces an SSA `Value` -- that value IS the measurement's identity. Per the
// design note, all per-measurement metadata (qubit, basis, export label) lives in a side table
// keyed by that Value, NOT bolted onto the op as attributes: ops stay lightweight, and the hot
// path (plan/engine) keys lookups by Value. No position-dependent indices anywhere.

/// Measurement basis. Every measurement in the current QIS path is a Z measurement (`mz`); the field
/// gives the side-table a home for basis so a future X/Y measurement needs no schema change.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Basis {
    X,
    Y,
    Z,
}

/// Side-table metadata for one measurement-SSA value (kept off the op, per the design note).
#[derive(Clone, Copy, Debug)]
pub struct MeasurementInfo {
    pub qubit: usize,
    pub basis: Basis,
    pub export_label: u64, // QIS result-id: the name this measurement records under
}

/// The circuit's measurement side-table: `measurement-SSA Value -> MeasurementInfo`. Populated as
/// `qec.measure` ops are built; the plan/engine look measurements up by Value. `HashMap` (not
/// `BTreeMap`) because pliron `Value` is not `Ord`, and the table is only point-queried -- never
/// iterated for output, so there is no determinism concern (export order comes from `qec.record`).
#[derive(Clone, Default)]
pub struct MeasurementRegistry {
    info: HashMap<Value, MeasurementInfo>,
}
impl MeasurementRegistry {
    /// Record a freshly built measurement-SSA value and its metadata.
    pub fn record(&mut self, value: Value, info: MeasurementInfo) {
        self.info.insert(value, info);
    }
    /// Look up a measurement-SSA value; panics (fail loud) if the value was never registered.
    pub fn get(&self, value: Value) -> MeasurementInfo {
        *self
            .info
            .get(&value)
            .unwrap_or_else(|| panic!("measurement-SSA value not in the measurement registry"))
    }
}

/// Store a fixed-point `Angle64` on an op as a `qec.angle` attribute.
pub fn set_angle(ctx: &Context, op: Ptr<Operation>, angle: Angle64) {
    op.deref_mut(ctx).attributes.0.insert(angle_attr::ANGLE.clone(), Box::new(Angle64Attr::from(angle)));
}
pub fn get_angle(ctx: &Context, op: Ptr<Operation>) -> Angle64 {
    let o = op.deref(ctx);
    let a: AttrObj = o.attributes.0.get(&*angle_attr::ANGLE).expect("angle attr").clone();
    let aa = a.downcast::<Angle64Attr>().unwrap_or_else(|_| panic!("angle not Angle64Attr"));
    Angle64::from(*aa)
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

/// `%result = qec.measure(qubitref)` -- a measurement. Its result `%result` IS the measurement-SSA
/// value (the identity); all per-measurement metadata (qubit, basis, export label) lives in the
/// `MeasurementRegistry` side-table keyed by that value, not on the op (ops stay lightweight).
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

/// `qec.record(measurement-SSA)` -- mark a `qec.measure` result as a recorded program output
/// (lowered from QIS `__quantum__rt__result_record_output`). The textual order of these ops IS the
/// program's classical-output order; the operand makes the recorded measurement-SSA explicit.
#[pliron_op(name = "qec.record", format, interfaces = [NOpdsInterface<1>, NResultsInterface<0>])]
pub struct RecordOp;
impl RecordOp {
    pub fn new(ctx: &mut Context, result: Value) -> Self {
        let op = Operation::new(ctx, Self::get_concrete_op_info(), vec![], vec![result], vec![], 0);
        RecordOp { op }
    }
}
impl Verify for RecordOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let opnd = self.get_operation().deref(ctx).get_operand(0);
        let records_a_measure = opnd.defining_op().is_some_and(|o| Operation::get_op::<MeasureOp>(o, ctx).is_some());
        if !records_a_measure {
            return verify_err!(self.loc(ctx), "qec.record operand must be a qec.measure result");
        }
        // Region scope: the recorded measurement must be visible from this record op -- i.e. its
        // defining block must be the record's block or an enclosing one. Recording a measurement
        // defined inside a `qec.if` region from the outer block is a cross-region SSA escape that
        // needs block-args/yield (measurement-SSA Phase 2); reject it rather than silently allow it.
        let meas_block = opnd.get_defining_block(ctx);
        let mut cur = self.get_operation().deref(ctx).get_parent_block();
        let visible = loop {
            match cur {
                Some(b) if Some(b) == meas_block => break true,
                Some(b) => cur = b.deref(ctx).get_parent_block(ctx),
                None => break false,
            }
        };
        if !visible {
            return verify_err!(self.loc(ctx), "qec.record operand is defined in a non-enclosing region (cross-region escape needs yield)");
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
        // condition (operand 0) must be an i1 (a measurement result); target (operand 1) a qubitref.
        // We inspect the operand's actual type read-only (TypePtr::from_ptr + width) rather than
        // construct the expected i1 -- constructing a parameterized type needs &mut Context.
        let cond_ty = self.get_operation().deref(ctx).get_operand(0).get_type(ctx);
        let cond_is_i1 = TypePtr::<IntegerType>::from_ptr(cond_ty, ctx).is_ok_and(|tp| tp.deref(ctx).width() == 1);
        if !cond_is_i1 {
            return verify_err!(self.loc(ctx), "qec.cond_x condition (operand 0) must be an i1 measurement result");
        }
        if self.get_operation().deref(ctx).get_operand(1).get_type(ctx) != qubitref_ty(ctx) {
            return verify_err!(self.loc(ctx), "qec.cond_x target must be a qec.qubitref");
        }
        Ok(())
    }
}

/// `qec.x(qubitref)` -- plain Pauli-X (used inside `qec.if` region blocks).
#[pliron_op(name = "qec.x", format, interfaces = [NOpdsInterface<1>, NResultsInterface<0>])]
pub struct XOp;
impl XOp {
    pub fn new(ctx: &mut Context, qref: Value) -> Self {
        let op = Operation::new(ctx, Self::get_concrete_op_info(), vec![], vec![qref], vec![], 0);
        XOp { op }
    }
}
impl Verify for XOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        if self.get_operation().deref(ctx).get_operand(0).get_type(ctx) != qubitref_ty(ctx) {
            return verify_err!(self.loc(ctx), "qec.x operand must be a qec.qubitref");
        }
        Ok(())
    }
}

/// `qec.if(cond: i1) { then } { else }` -- region-based conditional control flow (vision-aligned;
/// no CFG flattening). Two single-block regions of qec ops, no terminators.
#[pliron_op(name = "qec.if", format, interfaces = [NOpdsInterface<1>, NResultsInterface<0>, NRegionsInterface<2>, SingleBlockRegionInterface, NoTerminatorInterface])]
pub struct IfOp;
impl IfOp {
    pub fn new(ctx: &mut Context, cond: Value) -> Self {
        let op = Operation::new(ctx, Self::get_concrete_op_info(), vec![], vec![cond], vec![], 2);
        IfOp { op }
    }
    /// Create + attach a fresh single block for region `idx` and return it (for building).
    pub fn make_region_block(&self, ctx: &mut Context, idx: usize) -> Ptr<BasicBlock> {
        let region = self.get_operation().deref(ctx).get_region(idx);
        let block = BasicBlock::new(ctx, None, vec![]);
        block.insert_at_front(region, ctx);
        block
    }
}
impl Verify for IfOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        // the condition (operand 0) must be an i1 measurement result (region count is enforced by
        // NRegionsInterface<2>); inspect the operand's type read-only, do not construct it.
        let cond_ty = self.get_operation().deref(ctx).get_operand(0).get_type(ctx);
        let is_i1 = TypePtr::<IntegerType>::from_ptr(cond_ty, ctx).is_ok_and(|tp| tp.deref(ctx).width() == 1);
        if !is_i1 {
            return verify_err!(self.loc(ctx), "qec.if condition (operand 0) must be an i1 measurement result");
        }
        Ok(())
    }
}

/// Single-qubit rotations carrying a fixed-point `Angle64` (qprog.ll's rz/rx/ry).
#[pliron_op(name = "qec.rz", format, interfaces = [NOpdsInterface<1>, NResultsInterface<0>])]
pub struct RzOp;
impl RzOp {
    pub fn new(ctx: &mut Context, q: Value, angle: Angle64) -> Self {
        let op = Operation::new(ctx, Self::get_concrete_op_info(), vec![], vec![q], vec![], 0);
        set_angle(ctx, op, angle);
        RzOp { op }
    }
}
impl Verify for RzOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        if self.get_operation().deref(ctx).get_operand(0).get_type(ctx) != qubitref_ty(ctx) {
            return verify_err!(self.loc(ctx), "qec.rz operand must be a qec.qubitref");
        }
        Ok(())
    }
}
#[pliron_op(name = "qec.rx", format, interfaces = [NOpdsInterface<1>, NResultsInterface<0>])]
pub struct RxOp;
impl RxOp {
    pub fn new(ctx: &mut Context, q: Value, angle: Angle64) -> Self {
        let op = Operation::new(ctx, Self::get_concrete_op_info(), vec![], vec![q], vec![], 0);
        set_angle(ctx, op, angle);
        RxOp { op }
    }
}
impl Verify for RxOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        if self.get_operation().deref(ctx).get_operand(0).get_type(ctx) != qubitref_ty(ctx) {
            return verify_err!(self.loc(ctx), "qec.rx operand must be a qec.qubitref");
        }
        Ok(())
    }
}
#[pliron_op(name = "qec.ry", format, interfaces = [NOpdsInterface<1>, NResultsInterface<0>])]
pub struct RyOp;
impl RyOp {
    pub fn new(ctx: &mut Context, q: Value, angle: Angle64) -> Self {
        let op = Operation::new(ctx, Self::get_concrete_op_info(), vec![], vec![q], vec![], 0);
        set_angle(ctx, op, angle);
        RyOp { op }
    }
}
impl Verify for RyOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        if self.get_operation().deref(ctx).get_operand(0).get_type(ctx) != qubitref_ty(ctx) {
            return verify_err!(self.loc(ctx), "qec.ry operand must be a qec.qubitref");
        }
        Ok(())
    }
}
/// Two-qubit sqrt-ZZ entangler (PECOS `SZZ`; qprog.ll's no-angle `zz` maps here).
#[pliron_op(name = "qec.szz", format, interfaces = [NOpdsInterface<2>, NResultsInterface<0>], verifier = "succ")]
pub struct SzzOp;
impl SzzOp {
    pub fn new(ctx: &mut Context, a: Value, b: Value) -> Self {
        let op = Operation::new(ctx, Self::get_concrete_op_info(), vec![], vec![a, b], vec![], 0);
        SzzOp { op }
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
pub fn build_bell_ir(ctx: &mut Context) -> (ModuleOp, Ptr<BasicBlock>) {
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
pub fn emit_bytemessage(ctx: &Context, block: Ptr<BasicBlock>) -> ByteMessage {
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

pub fn run_milestone_1() {
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
pub struct PlironBellEngine {
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

pub fn run_milestone_2() {
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

/// Structured rejection for anything outside the covered QIS-LLVM subset (see the strangler scope
/// doc) -- used instead of silently dropping unrecognized calls or panicking on malformed structure.
fn unsupported_qis(msg: impl Into<String>) -> PecosError {
    PecosError::Feature(format!("pecos-phir-pliron: unsupported QIS-LLVM-IR -- {}", msg.into()))
}

/// Minimal QIS-LLVM-IR -> pliron `qec` IR parser for the Bell-style straight-line subset. Recognizes
/// `__quantum__qis__{h,cx,m}__body` + `__quantum__rt__result_record_output`; any other `__quantum__`
/// call (or a malformed operand list) is rejected with a structured error rather than dropped.
pub fn parse_bell_ll(ctx: &mut Context, src: &str) -> std::result::Result<(ModuleOp, Ptr<BasicBlock>, MeasurementRegistry), PecosError> {
    fn i64_args(line: &str) -> Vec<usize> {
        match (line.find('('), line.rfind(')')) {
            (Some(l), Some(r)) if r > l => line[l + 1..r]
                .split(',')
                .filter_map(|t| t.trim().strip_prefix("i64 ").and_then(|n| n.trim().parse::<usize>().ok()))
                .collect(),
            _ => Vec::new(),
        }
    }
    // pass 1: collect the gate/measure/record stream + the set of qubits referenced
    let mut parsed: Vec<(&str, Vec<usize>)> = Vec::new();
    let mut qubits: BTreeSet<usize> = BTreeSet::new();
    for line in src.lines() {
        let l = line.trim();
        if !(l.starts_with("call ") || l.contains("= call ")) {
            continue;
        }
        let a = i64_args(l);
        if l.contains("__quantum__qis__h__body") {
            let q = *a.first().ok_or_else(|| unsupported_qis("h__body missing qubit operand"))?;
            qubits.insert(q);
            parsed.push(("h", a));
        } else if l.contains("__quantum__qis__cx__body") {
            if a.len() < 2 {
                return Err(unsupported_qis("cx__body needs two qubit operands"));
            }
            qubits.insert(a[0]);
            qubits.insert(a[1]);
            parsed.push(("cx", a));
        } else if l.contains("__quantum__qis__m__body") {
            if a.len() < 2 {
                return Err(unsupported_qis("m__body needs (qubit, result_id) operands"));
            }
            qubits.insert(a[0]);
            parsed.push(("m", a));
        } else if l.contains("__quantum__rt__result_record_output") {
            if a.is_empty() {
                return Err(unsupported_qis("result_record_output missing result_id operand"));
            }
            parsed.push(("record", a));
        } else if l.contains("__quantum__") {
            return Err(unsupported_qis(format!("operation not in the covered subset: {l}")));
        }
    }
    // pass 2: build the pliron qec IR + the measurement-SSA registry
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
    let mut reg = MeasurementRegistry::default();
    let mut measured: HashMap<u64, Value> = HashMap::new(); // result-id -> measurement-SSA Value
    for (op, a) in &parsed {
        match *op {
            "h" => {
                push!(HOp::new(ctx, slot_of[&a[0]]));
            }
            "cx" => {
                push!(CxOp::new(ctx, slot_of[&a[0]], slot_of[&a[1]]));
            }
            "m" => {
                let m = push!(MeasureOp::new(ctx, slot_of[&a[0]]));
                let v = m.get_result(ctx);
                let rid = a[1] as u64;
                measured.insert(rid, v);
                reg.record(v, MeasurementInfo { qubit: a[0], basis: Basis::Z, export_label: rid });
            }
            "record" => {
                let rid = a[0] as u64;
                let v = *measured
                    .get(&rid)
                    .ok_or_else(|| unsupported_qis(format!("result_record_output references unknown result-id {rid}")))?;
                push!(RecordOp::new(ctx, v));
            }
            _ => {}
        }
    }
    push!(EndOp::new(ctx));
    Ok((module, bb, reg))
}

pub fn run_milestone_3() {
    let ctx = &mut Context::new();
    let src = include_str!("../../../examples/llvm/bell.ll");
    let (module, bb, _reg) = parse_bell_ll(ctx, src).expect("milestone-3: parse bell.ll");
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
pub enum Cmd {
    Pz(usize),
    H(usize),
    X(usize),
    Rz(usize, Angle64),
    Rx(usize, Angle64),
    Ry(usize, Angle64),
    Szz(usize, usize),
    Cx(usize, usize),
    Mz(usize, u64), // (qubit, QIS result-id); the id rides through to the export/register mapping
}

#[derive(Clone)]
pub struct AdaptivePlan {
    batch1: Vec<Cmd>,        // gates + the conditioning Mz
    cond_outcome_idx: usize, // index in batch1's mz-outcomes that gates cond_x
    cond_target: usize,      // qubit to X iff that outcome == 1
    batch2: Vec<Cmd>,        // post-condition ops (final Mz)
}

/// Build the `Cmd::Mz` for a `qec.measure`, asserting the registry's qubit agrees with the IR slot
/// index. The registry is a side-table, so `verify` can't catch a mismatch; this fails loud at
/// plan-build time if the registry and the IR ever drift apart.
fn measure_to_mz(ctx: &Context, m: MeasureOp, qubit_of: &HashMap<Value, usize>, reg: &MeasurementRegistry) -> Cmd {
    let operand = m.get_operation().deref(ctx).get_operand(0);
    let slot_qubit = qubit_of[&operand];
    let info = reg.get(m.get_result(ctx));
    assert_eq!(
        info.qubit, slot_qubit,
        "measurement registry qubit ({}) disagrees with the IR slot index ({}) -- registry/IR drift",
        info.qubit, slot_qubit
    );
    Cmd::Mz(info.qubit, info.export_label)
}

/// Walk a `qec` block once and split it into the two-batch adaptive plan at the `cond_x` boundary.
/// Gate qubits come from the explicit slot-index map; measurement metadata (qubit + export label)
/// comes from the `MeasurementRegistry`, keyed by the measure op's SSA value.
pub fn plan_from_ir(ctx: &Context, block: Ptr<BasicBlock>, reg: &MeasurementRegistry) -> AdaptivePlan {
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
            let mz = measure_to_mz(ctx, m, &qubit_of, reg);
            if after_cond {
                batch2.push(mz);
            } else {
                cond_outcome_idx = mz_b1;
                mz_b1 += 1;
                batch1.push(mz);
            }
        } else if let Some(c) = Operation::get_op::<CondXOp>(op, ctx) {
            cond_target = qubit_of[&c.get_operation().deref(ctx).get_operand(1)];
            after_cond = true;
        }
    }
    AdaptivePlan { batch1, cond_outcome_idx, cond_target, batch2 }
}

/// Number of qubits a Cmd stream touches = max referenced qubit index + 1 (0 if it touches none).
/// Lets an engine report its real `num_qubits()` instead of a hard-coded value.
pub fn cmds_num_qubits(batches: &[&[Cmd]]) -> usize {
    let mut n = 0;
    for cmds in batches {
        for c in *cmds {
            let hi = match *c {
                Cmd::Pz(q) | Cmd::H(q) | Cmd::X(q) | Cmd::Rz(q, _) | Cmd::Rx(q, _) | Cmd::Ry(q, _) | Cmd::Mz(q, _) => q,
                Cmd::Szz(a, b) | Cmd::Cx(a, b) => a.max(b),
            };
            n = n.max(hi + 1);
        }
    }
    n
}

pub fn emit_cmds(b: &mut pecos_engines::byte_message::ByteMessageBuilder, cmds: &[Cmd]) {
    for c in cmds {
        match *c {
            Cmd::Pz(q) => { b.pz(&[q]); }
            Cmd::H(q) => { b.h(&[q]); }
            Cmd::X(q) => { b.x(&[q]); }
            Cmd::Rz(q, a) => { b.rz(a, &[q]); }
            Cmd::Rx(q, a) => { b.rx(a, &[q]); }
            Cmd::Ry(q, a) => { b.ry(a, &[q]); }
            Cmd::Szz(a, c0) => { b.szz(&[(a, c0)]); }
            Cmd::Cx(c0, t) => { b.cx(&[(c0, t)]); }
            Cmd::Mz(q, _rid) => { b.mz(&[q]); } // result-id is bookkeeping, not a simulator op
        }
    }
}

#[derive(Clone)]
pub struct PlironAdaptiveEngine {
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
        // gates/measures across both batches, plus the conditional-X target qubit.
        cmds_num_qubits(&[&self.plan.batch1, &self.plan.batch2]).max(self.plan.cond_target + 1)
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

pub fn build_adaptive_ir(ctx: &mut Context) -> (ModuleOp, Ptr<BasicBlock>, MeasurementRegistry) {
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
    let mut reg = MeasurementRegistry::default();
    let m0 = push!(MeasureOp::new(ctx, s0v)); // mid measure (conditioning)
    let m0v = m0.get_result(ctx);
    reg.record(m0v, MeasurementInfo { qubit: 0, basis: Basis::Z, export_label: 0 });
    push!(CondXOp::new(ctx, m0v, s1v)); // X q1 iff m0 == 1
    let m1 = push!(MeasureOp::new(ctx, s1v)); // final measure
    reg.record(m1.get_result(ctx), MeasurementInfo { qubit: 1, basis: Basis::Z, export_label: 1 });
    push!(EndOp::new(ctx));
    (module, bb, reg)
}

pub fn run_milestone_4() {
    let ctx = &mut Context::new();
    let (module, bb, reg) = build_adaptive_ir(ctx);
    println!("=== adaptive (mid-measure -> cond_x -> final) pliron qec IR ===");
    println!("{}", module.get_operation().disp(ctx));
    verify_op(&module, ctx).unwrap_or_else(|e| panic!("[milestone-4 verify] FAILED: {}", e.disp(ctx)));
    let plan = plan_from_ir(ctx, bb, &reg);
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

// ===================== Milestone 5: region-based conditional control flow (qec.if) =====================
// Same adaptive program as M4, but the conditional is now a real pliron REGION op
// (`qec.if(cond) { then } { else }`), not a single cond_x op. The engine interprets the chosen
// region across batches. This proves region-based control flow (vision-aligned; no CFG flattening).

/// Lower the ops of a single block (a `qec.if` region body) to a Cmd list, reusing the slot map.
/// Gate qubits come from `qubit_of`; measurement qubit + export label come from the registry.
pub fn block_to_cmds(ctx: &Context, block: Ptr<BasicBlock>, qubit_of: &HashMap<Value, usize>, reg: &MeasurementRegistry) -> Vec<Cmd> {
    let mut cmds = Vec::new();
    for op in block.deref(ctx).iter(ctx).collect::<Vec<_>>() {
        if let Some(x) = Operation::get_op::<XOp>(op, ctx) {
            cmds.push(Cmd::X(qubit_of[&x.get_operation().deref(ctx).get_operand(0)]));
        } else if let Some(h) = Operation::get_op::<HOp>(op, ctx) {
            cmds.push(Cmd::H(qubit_of[&h.get_operation().deref(ctx).get_operand(0)]));
        } else if let Some(p) = Operation::get_op::<PrepareOp>(op, ctx) {
            cmds.push(Cmd::Pz(qubit_of[&p.get_operation().deref(ctx).get_operand(0)]));
        } else if let Some(cx) = Operation::get_op::<CxOp>(op, ctx) {
            let opn = cx.get_operation();
            cmds.push(Cmd::Cx(qubit_of[&opn.deref(ctx).get_operand(0)], qubit_of[&opn.deref(ctx).get_operand(1)]));
        } else if let Some(m) = Operation::get_op::<MeasureOp>(op, ctx) {
            cmds.push(measure_to_mz(ctx, m, qubit_of, reg));
        }
    }
    cmds
}

pub struct IfPlan {
    batch1: Vec<Cmd>,
    cond_outcome_idx: usize,
    then_cmds: Vec<Cmd>,
    else_cmds: Vec<Cmd>,
    post: Vec<Cmd>,
    export: Vec<u64>, // QIS result-ids in `qec.record` (result_record_output) order
}

/// Walk the func block; split at the `qec.if` boundary, reading the two region bodies as the
/// then/else command lists. Measurement metadata + export order come from the registry.
pub fn plan_from_if_ir(ctx: &Context, block: Ptr<BasicBlock>, reg: &MeasurementRegistry) -> IfPlan {
    let mut qubit_of: HashMap<Value, usize> = HashMap::new();
    let (mut batch1, mut post, mut then_cmds, mut else_cmds) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let mut after_if = false;
    let mut mz_b1 = 0usize;
    let mut cond_outcome_idx = 0usize;
    let mut export: Vec<u64> = Vec::new();
    for op in block.deref(ctx).iter(ctx).collect::<Vec<_>>() {
        if let Some(s) = Operation::get_op::<SlotOp>(op, ctx) {
            qubit_of.insert(s.get_result(ctx), s.index(ctx) as usize);
        } else if let Some(p) = Operation::get_op::<PrepareOp>(op, ctx) {
            let q = qubit_of[&p.get_operation().deref(ctx).get_operand(0)];
            if after_if { post.push(Cmd::Pz(q)) } else { batch1.push(Cmd::Pz(q)) }
        } else if let Some(h) = Operation::get_op::<HOp>(op, ctx) {
            let q = qubit_of[&h.get_operation().deref(ctx).get_operand(0)];
            if after_if { post.push(Cmd::H(q)) } else { batch1.push(Cmd::H(q)) }
        } else if let Some(cx) = Operation::get_op::<CxOp>(op, ctx) {
            let opn = cx.get_operation();
            let c = qubit_of[&opn.deref(ctx).get_operand(0)];
            let t = qubit_of[&opn.deref(ctx).get_operand(1)];
            if after_if { post.push(Cmd::Cx(c, t)) } else { batch1.push(Cmd::Cx(c, t)) }
        } else if let Some(r) = Operation::get_op::<RzOp>(op, ctx) {
            let q = qubit_of[&r.get_operation().deref(ctx).get_operand(0)];
            let t = get_angle(ctx, r.get_operation());
            if after_if { post.push(Cmd::Rz(q, t)) } else { batch1.push(Cmd::Rz(q, t)) }
        } else if let Some(r) = Operation::get_op::<RxOp>(op, ctx) {
            let q = qubit_of[&r.get_operation().deref(ctx).get_operand(0)];
            let t = get_angle(ctx, r.get_operation());
            if after_if { post.push(Cmd::Rx(q, t)) } else { batch1.push(Cmd::Rx(q, t)) }
        } else if let Some(r) = Operation::get_op::<RyOp>(op, ctx) {
            let q = qubit_of[&r.get_operation().deref(ctx).get_operand(0)];
            let t = get_angle(ctx, r.get_operation());
            if after_if { post.push(Cmd::Ry(q, t)) } else { batch1.push(Cmd::Ry(q, t)) }
        } else if let Some(z) = Operation::get_op::<SzzOp>(op, ctx) {
            let opn = z.get_operation();
            let a = qubit_of[&opn.deref(ctx).get_operand(0)];
            let bq = qubit_of[&opn.deref(ctx).get_operand(1)];
            if after_if { post.push(Cmd::Szz(a, bq)) } else { batch1.push(Cmd::Szz(a, bq)) }
        } else if let Some(m) = Operation::get_op::<MeasureOp>(op, ctx) {
            let mz = measure_to_mz(ctx, m, &qubit_of, reg);
            if after_if {
                post.push(mz);
            } else {
                cond_outcome_idx = mz_b1;
                mz_b1 += 1;
                batch1.push(mz);
            }
        } else if let Some(ifop) = Operation::get_op::<IfOp>(op, ctx) {
            then_cmds = block_to_cmds(ctx, ifop.get_body(ctx, 0), &qubit_of, reg);
            else_cmds = block_to_cmds(ctx, ifop.get_body(ctx, 1), &qubit_of, reg);
            after_if = true;
        } else if let Some(rec) = Operation::get_op::<RecordOp>(op, ctx) {
            // export order = textual order of qec.record; resolve the recorded measurement-SSA value
            // to its export label via the registry (the value is the identity).
            let recorded = rec.get_operation().deref(ctx).get_operand(0);
            export.push(reg.get(recorded).export_label);
        }
    }
    IfPlan { batch1, cond_outcome_idx, then_cmds, else_cmds, post, export }
}

#[derive(Clone)]
pub struct PlironIfEngine {
    batch1: Vec<Cmd>,
    cond_outcome_idx: usize,
    then_cmds: Vec<Cmd>,
    else_cmds: Vec<Cmd>,
    post: Vec<Cmd>,
    export: Vec<u64>, // QIS result-ids to record, in result_record_output order
    stage: u8,
    b1: Vec<u32>,
    b2: Vec<u32>,
}
impl PlironIfEngine {
    fn b1_msg(&self) -> ByteMessage {
        let mut b = ByteMessage::quantum_operations_builder();
        emit_cmds(&mut b, &self.batch1);
        b.build()
    }
    fn taken_cmds(&self) -> &[Cmd] {
        if self.b1.get(self.cond_outcome_idx).copied() == Some(1) { &self.then_cmds } else { &self.else_cmds }
    }
    fn b2_msg(&self) -> ByteMessage {
        let mut b = ByteMessage::quantum_operations_builder();
        emit_cmds(&mut b, self.taken_cmds());
        emit_cmds(&mut b, &self.post);
        b.build()
    }
    /// Reconstruct `result-id -> outcome` by replaying the emission order: batch1's `Mz`s consume
    /// `b1` in order; batch2 (the runtime-taken branch, then `post`) consume `b2` in order. This is
    /// the explicit measurement-SSA mapping -- each measurement's QIS result-id keyed to its value.
    fn outcome_by_result_id(&self) -> BTreeMap<u64, u32> {
        let mut map = BTreeMap::new();
        let mut i = 0;
        for c in &self.batch1 {
            if let Cmd::Mz(_, rid) = *c {
                map.insert(rid, self.b1.get(i).copied().unwrap_or(0));
                i += 1;
            }
        }
        let mut j = 0;
        for c in self.taken_cmds().iter().chain(self.post.iter()) {
            if let Cmd::Mz(_, rid) = *c {
                map.insert(rid, self.b2.get(j).copied().unwrap_or(0));
                j += 1;
            }
        }
        map
    }
}
impl Engine for PlironIfEngine {
    type Input = ();
    type Output = Shot;
    fn process(&mut self, _i: ()) -> std::result::Result<Shot, PecosError> { self.get_results() }
    fn reset(&mut self) -> std::result::Result<(), PecosError> { self.stage = 0; self.b1.clear(); self.b2.clear(); Ok(()) }
}
impl ClassicalEngine for PlironIfEngine {
    fn num_qubits(&self) -> usize {
        cmds_num_qubits(&[&self.batch1, &self.then_cmds, &self.else_cmds, &self.post])
    }
    fn generate_commands(&mut self) -> std::result::Result<ByteMessage, PecosError> {
        if self.stage == 0 { self.stage = 1; Ok(self.b1_msg()) } else { Ok(ByteMessage::create_empty()) }
    }
    fn handle_measurements(&mut self, m: ByteMessage) -> std::result::Result<(), PecosError> {
        let o = m.outcomes()?;
        if self.stage == 1 { self.b1 = o } else { self.b2 = o }
        Ok(())
    }
    fn get_results(&self) -> std::result::Result<Shot, PecosError> {
        let map = self.outcome_by_result_id();
        let mut s = Shot::default();
        // one 1-bit register per recorded result-id ("r<id>"), in result_record_output order.
        for &rid in &self.export {
            s.add_register(&format!("r{rid}"), map.get(&rid).copied().unwrap_or(0), 1);
        }
        Ok(s)
    }
    fn compile(&self) -> std::result::Result<(), PecosError> { Ok(()) }
    fn reset(&mut self) -> std::result::Result<(), PecosError> { Engine::reset(self) }
    fn as_any(&self) -> &dyn Any { self }
    fn as_any_mut(&mut self) -> &mut dyn Any { self }
}
impl ControlEngine for PlironIfEngine {
    type Input = ();
    type Output = Shot;
    type EngineInput = ByteMessage;
    type EngineOutput = ByteMessage;
    fn start(&mut self, _i: ()) -> std::result::Result<EngineStage<ByteMessage, Shot>, PecosError> {
        self.stage = 1;
        self.b1.clear();
        self.b2.clear();
        Ok(EngineStage::NeedsProcessing(self.b1_msg()))
    }
    fn continue_processing(&mut self, meas: ByteMessage) -> std::result::Result<EngineStage<ByteMessage, Shot>, PecosError> {
        if self.stage == 1 {
            self.b1 = meas.outcomes()?;
            self.stage = 2;
            Ok(EngineStage::NeedsProcessing(self.b2_msg()))
        } else {
            self.b2 = meas.outcomes()?;
            Ok(EngineStage::Complete(self.get_results()?))
        }
    }
    fn reset(&mut self) -> std::result::Result<(), PecosError> { Engine::reset(self) }
}

pub fn build_if_ir(ctx: &mut Context) -> (ModuleOp, Ptr<BasicBlock>, MeasurementRegistry) {
    let module = ModuleOp::new(ctx, "adaptive_if".try_into().unwrap());
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
    let mut reg = MeasurementRegistry::default();
    let m0 = push!(MeasureOp::new(ctx, s0v)); // mid measure (conditioning)
    let m0v = m0.get_result(ctx);
    reg.record(m0v, MeasurementInfo { qubit: 0, basis: Basis::Z, export_label: 0 });
    let ifop = push!(IfOp::new(ctx, m0v)); // if m0 { x q1 } else { }
    let then_bb = ifop.make_region_block(ctx, 0);
    XOp::new(ctx, s1v).get_operation().insert_at_back(then_bb, ctx);
    let _else_bb = ifop.make_region_block(ctx, 1);
    let m1 = push!(MeasureOp::new(ctx, s1v)); // final measure
    let m1v = m1.get_result(ctx);
    reg.record(m1v, MeasurementInfo { qubit: 1, basis: Basis::Z, export_label: 1 });
    push!(RecordOp::new(ctx, m0v)); // record mid  -> register r0
    push!(RecordOp::new(ctx, m1v)); // record final -> register r1
    push!(EndOp::new(ctx));
    (module, bb, reg)
}

pub fn run_milestone_5() {
    let ctx = &mut Context::new();
    let (module, bb, reg) = build_if_ir(ctx);
    println!("=== region-based conditional (qec.if) pliron qec IR ===");
    println!("{}", module.get_operation().disp(ctx));
    verify_op(&module, ctx).unwrap_or_else(|e| panic!("[milestone-5 verify] FAILED: {}", e.disp(ctx)));
    let plan = plan_from_if_ir(ctx, bb, &reg);
    let engine = PlironIfEngine {
        batch1: plan.batch1,
        cond_outcome_idx: plan.cond_outcome_idx,
        then_cmds: plan.then_cmds,
        else_cmds: plan.else_cmds,
        post: plan.post,
        export: plan.export,
        stage: 0,
        b1: Vec::new(),
        b2: Vec::new(),
    };
    let mut hybrid = HybridEngineBuilder::new()
        .with_classical_engine(Box::new(engine))
        .with_quantum_engine(Box::new(StateVecEngine::new(2)))
        .build();
    let (mut all_eq, mut saw0, mut saw1) = (true, false, false);
    for _ in 0..200 {
        let shot = hybrid.run_shot().unwrap();
        // explicit export mapping: r0 = mid (result-id 0), r1 = final (result-id 1)
        let mid = shot.data.get("r0").and_then(Data::as_u32).expect("r0 (mid)");
        let fin = shot.data.get("r1").and_then(Data::as_u32).expect("r1 (final)");
        if fin != mid {
            all_eq = false;
        }
        saw0 |= mid == 0;
        saw1 |= mid == 1;
        Engine::reset(&mut hybrid).unwrap();
    }
    assert!(all_eq, "milestone-5: final must equal mid (region-based qec.if feedback)");
    assert!(saw0 && saw1, "milestone-5: expected both mid=0 and mid=1");
    println!("[milestone-5 region-based qec.if via HybridEngine] OK -- r1(final)==r0(mid) in all 200 shots, saw mid 0 and 1");
}

// ===================== Milestone 6: parse the literal qprog.ll (adaptive) =====================
// qprog.ll: rz/rx/ry/zz; mid-measure; icmp+br (diamond CFG); conditional x; final measures.
// We lift the diamond CFG into a `qec.if` during the parse, producing real rotations + region
// control flow, and run it end-to-end through the real HybridEngine.

#[derive(Clone, Copy)]
pub enum ParsedOp {
    H(usize),
    Rz(usize, f64),
    Rx(usize, f64),
    Ry(usize, f64),
    Szz(usize, usize),
    X(usize),
    M(usize, u64),  // (qubit, QIS result-id = 2nd i64 of m__body)
    Record(u64),    // result_record_output(result-id): export this measurement-SSA, in this order
}

pub fn inner_parens(l: &str) -> &str {
    match (l.find('('), l.rfind(')')) {
        (Some(a), Some(b)) if b > a => &l[a + 1..b],
        _ => "",
    }
}
pub fn doubles_and_ints(inner: &str) -> (Vec<f64>, Vec<usize>) {
    let (mut ds, mut is) = (Vec::new(), Vec::new());
    for t in inner.split(',') {
        let t = t.trim();
        if let Some(d) = t.strip_prefix("double ")
            && let Ok(v) = d.trim().parse::<f64>()
        {
            ds.push(v);
        } else if let Some(n) = t.strip_prefix("i64 ")
            && let Ok(v) = n.trim().parse::<usize>()
        {
            is.push(v);
        }
    }
    (ds, is)
}
pub fn br_labels(l: &str) -> Vec<String> {
    l.split("label %")
        .skip(1)
        .filter_map(|s| s.split([',', ' ']).next().filter(|x| !x.is_empty()).map(str::to_string))
        .collect()
}
pub fn collect_qubits(ops: &[ParsedOp], set: &mut BTreeSet<usize>) {
    for p in ops {
        match *p {
            ParsedOp::H(q) | ParsedOp::Rz(q, _) | ParsedOp::Rx(q, _) | ParsedOp::Ry(q, _) | ParsedOp::X(q) | ParsedOp::M(q, _) => { set.insert(q); }
            ParsedOp::Szz(a, b) => { set.insert(a); set.insert(b); }
            ParsedOp::Record(_) => {}
        }
    }
}
/// Emit one parsed op into `block`. `measured` accumulates `result-id -> measurement-SSA Value` so a
/// later `ParsedOp::Record` resolves exactly which `qec.measure` becomes a program output; `reg`
/// gets each measurement's metadata (qubit, basis, export label) keyed by its SSA value.
pub fn emit_parsed(ctx: &mut Context, p: &ParsedOp, block: Ptr<BasicBlock>, slot_of: &HashMap<usize, Value>, measured: &mut HashMap<u64, Value>, reg: &mut MeasurementRegistry) -> std::result::Result<(), PecosError> {
    match *p {
        ParsedOp::H(q) => { HOp::new(ctx, slot_of[&q]).get_operation().insert_at_back(block, ctx); }
        // .ll angles are f64 radians (the wire format); convert to fixed-point Angle64 at the boundary.
        ParsedOp::Rz(q, t) => { RzOp::new(ctx, slot_of[&q], Angle64::from_radians(t)).get_operation().insert_at_back(block, ctx); }
        ParsedOp::Rx(q, t) => { RxOp::new(ctx, slot_of[&q], Angle64::from_radians(t)).get_operation().insert_at_back(block, ctx); }
        ParsedOp::Ry(q, t) => { RyOp::new(ctx, slot_of[&q], Angle64::from_radians(t)).get_operation().insert_at_back(block, ctx); }
        ParsedOp::Szz(a, b) => { SzzOp::new(ctx, slot_of[&a], slot_of[&b]).get_operation().insert_at_back(block, ctx); }
        ParsedOp::X(q) => { XOp::new(ctx, slot_of[&q]).get_operation().insert_at_back(block, ctx); }
        ParsedOp::M(q, rid) => {
            let m = MeasureOp::new(ctx, slot_of[&q]);
            m.get_operation().insert_at_back(block, ctx);
            let v = m.get_result(ctx);
            measured.insert(rid, v);
            reg.record(v, MeasurementInfo { qubit: q, basis: Basis::Z, export_label: rid });
        }
        ParsedOp::Record(rid) => {
            let v = *measured
                .get(&rid)
                .ok_or_else(|| unsupported_qis(format!("result_record_output references unknown result-id {rid}")))?;
            RecordOp::new(ctx, v).get_operation().insert_at_back(block, ctx);
        }
    }
    Ok(())
}

/// Parse the adaptive single-diamond QIS-LLVM subset, lifting the conditional branch into a `qec.if`.
/// Rejects (structured error, not silent-drop/panic) anything outside the subset: unrecognized
/// `__quantum__` calls, malformed operand lists, and more than one conditional branch.
pub fn parse_qprog_ll(ctx: &mut Context, src: &str) -> std::result::Result<(ModuleOp, Ptr<BasicBlock>, MeasurementRegistry), PecosError> {
    // pass 1: collect ops per block label (entry = "") and the conditional-branch targets.
    let mut blocks: Vec<(String, Vec<ParsedOp>)> = Vec::new();
    let mut cur_label = String::new();
    let mut cur_ops: Vec<ParsedOp> = Vec::new();
    let (mut then_label, mut else_label) = (None::<String>, None::<String>);
    let mut in_func = false;
    for raw in src.lines() {
        let l = raw.trim();
        if l.starts_with("define ") && l.contains("@qmain") { in_func = true; continue; }
        if !in_func { continue; }
        if l == "}" { blocks.push((std::mem::take(&mut cur_label), std::mem::take(&mut cur_ops))); break; }
        if l.ends_with(':') && !l.contains(' ') {
            blocks.push((std::mem::take(&mut cur_label), std::mem::take(&mut cur_ops)));
            cur_label = l.trim_end_matches(':').to_string();
            continue;
        }
        let (ds, is) = doubles_and_ints(inner_parens(l));
        let iq = |i: usize| is.get(i).copied().ok_or_else(|| unsupported_qis(format!("missing i64 operand {i}: {l}")));
        let da = |i: usize| ds.get(i).copied().ok_or_else(|| unsupported_qis(format!("missing double operand {i}: {l}")));
        if l.contains("__quantum__qis__h__body") {
            cur_ops.push(ParsedOp::H(iq(0)?));
        } else if l.contains("__quantum__qis__rz__body") {
            cur_ops.push(ParsedOp::Rz(iq(0)?, da(0)?));
        } else if l.contains("__quantum__qis__rx__body") {
            cur_ops.push(ParsedOp::Rx(iq(0)?, da(0)?));
        } else if l.contains("__quantum__qis__ry__body") {
            cur_ops.push(ParsedOp::Ry(iq(0)?, da(0)?));
        } else if l.contains("__quantum__qis__zz__body") {
            cur_ops.push(ParsedOp::Szz(iq(0)?, iq(1)?));
        } else if l.contains("__quantum__qis__x__body") {
            cur_ops.push(ParsedOp::X(iq(0)?));
        } else if l.contains("__quantum__qis__m__body") {
            cur_ops.push(ParsedOp::M(iq(0)?, iq(1)? as u64)); // m__body(i64 qubit, i64 result_id)
        } else if l.contains("__quantum__rt__result_record_output") {
            cur_ops.push(ParsedOp::Record(iq(0)? as u64)); // result_record_output(i64 result_id, i8* null)
        } else if l.starts_with("br ") && l.contains("label %") {
            let labels = br_labels(l);
            if labels.len() == 2 {
                if then_label.is_some() {
                    return Err(unsupported_qis("more than one conditional branch -- only a single diamond is supported"));
                }
                then_label = Some(labels[0].clone());
                else_label = Some(labels[1].clone());
            }
        } else if l.contains("__quantum__") {
            return Err(unsupported_qis(format!("operation not in the covered subset: {l}")));
        }
    }
    let find = |lab: &str| blocks.iter().find(|(l, _)| l == lab).map(|(_, o)| o.clone()).unwrap_or_default();
    let entry_ops = find("");
    let then_ops = then_label.as_deref().map(find).unwrap_or_default();
    let else_ops = else_label.as_deref().map(find).unwrap_or_default();
    let merge_ops = blocks
        .iter()
        .find(|(l, _)| !l.is_empty() && Some(l) != then_label.as_ref() && Some(l) != else_label.as_ref())
        .map(|(_, o)| o.clone())
        .unwrap_or_default();

    // pass 2: build the pliron qec IR.
    let mut qubits = BTreeSet::new();
    for ops in [&entry_ops, &then_ops, &else_ops, &merge_ops] { collect_qubits(ops, &mut qubits); }

    let module = ModuleOp::new(ctx, "qprog".try_into().unwrap());
    let func_ty = FunctionType::get(ctx, vec![], vec![]);
    let func = FuncOp::new(ctx, "qmain".try_into().unwrap(), func_ty);
    module.append_operation(ctx, func.get_operation(), 0);
    let bb = func.get_entry_block(ctx);

    let q = QallocOp::new(ctx);
    q.get_operation().insert_at_back(bb, ctx);
    let qv = q.get_result(ctx);
    let mut slot_of: HashMap<usize, Value> = HashMap::new();
    for &idx in &qubits {
        let s = SlotOp::new(ctx, qv, idx as u64);
        s.get_operation().insert_at_back(bb, ctx);
        let sv = s.get_result(ctx);
        slot_of.insert(idx, sv);
        PrepareOp::new(ctx, sv).get_operation().insert_at_back(bb, ctx);
    }
    // result-id -> measurement-SSA Value, so result_record_output can name the recorded measurement.
    let mut measured: HashMap<u64, Value> = HashMap::new();
    let mut reg = MeasurementRegistry::default();
    // entry gates (everything before the trailing mid-measure), then the mid measure (the cond).
    let (mid_q, mid_rid) = match entry_ops.last() {
        Some(ParsedOp::M(q, rid)) => (*q, *rid),
        _ => return Err(unsupported_qis("entry block must end with a mid-measurement (single-diamond adaptive shape)")),
    };
    for p in &entry_ops[..entry_ops.len() - 1] { emit_parsed(ctx, p, bb, &slot_of, &mut measured, &mut reg)?; }
    let m0 = MeasureOp::new(ctx, slot_of[&mid_q]);
    m0.get_operation().insert_at_back(bb, ctx);
    let m0v = m0.get_result(ctx);
    measured.insert(mid_rid, m0v);
    reg.record(m0v, MeasurementInfo { qubit: mid_q, basis: Basis::Z, export_label: mid_rid });
    // lift the diamond into qec.if(mid) { then } { else }
    let ifop = IfOp::new(ctx, m0v);
    ifop.get_operation().insert_at_back(bb, ctx);
    let then_bb = ifop.make_region_block(ctx, 0);
    for p in &then_ops { emit_parsed(ctx, p, then_bb, &slot_of, &mut measured, &mut reg)?; }
    let else_bb = ifop.make_region_block(ctx, 1);
    for p in &else_ops { emit_parsed(ctx, p, else_bb, &slot_of, &mut measured, &mut reg)?; }
    // final measurements + result_record_output ops (the export list)
    for p in &merge_ops { emit_parsed(ctx, p, bb, &slot_of, &mut measured, &mut reg)?; }
    EndOp::new(ctx).get_operation().insert_at_back(bb, ctx);
    Ok((module, bb, reg))
}

pub fn run_milestone_6() {
    let ctx = &mut Context::new();
    let src = include_str!("../../../examples/llvm/qprog.ll");
    let (module, bb, reg) = parse_qprog_ll(ctx, src).expect("milestone-6: parse qprog.ll");
    println!("=== qprog.ll parsed into pliron qec IR (rotations + qec.if) ===");
    println!("{}", module.get_operation().disp(ctx));
    verify_op(&module, ctx).unwrap_or_else(|e| panic!("[milestone-6 verify] FAILED: {}", e.disp(ctx)));
    let plan = plan_from_if_ir(ctx, bb, &reg);
    let engine = PlironIfEngine {
        batch1: plan.batch1,
        cond_outcome_idx: plan.cond_outcome_idx,
        then_cmds: plan.then_cmds,
        else_cmds: plan.else_cmds,
        post: plan.post,
        export: plan.export,
        stage: 0,
        b1: Vec::new(),
        b2: Vec::new(),
    };
    let mut hybrid = HybridEngineBuilder::new()
        .with_classical_engine(Box::new(engine))
        .with_quantum_engine(Box::new(StateVecEngine::new(2)))
        .build();
    // qprog records result-ids 0,1,2 -> registers r0=final_q0, r1=final_q1, r2=mid (export order).
    let mut seen: BTreeSet<(u32, u32, u32)> = BTreeSet::new();
    let mut n = 0;
    for _ in 0..200 {
        let shot = hybrid.run_shot().unwrap();
        let f0 = shot.data.get("r0").and_then(Data::as_u32).expect("r0 (final_q0)");
        let f1 = shot.data.get("r1").and_then(Data::as_u32).expect("r1 (final_q1)");
        let mid = shot.data.get("r2").and_then(Data::as_u32).expect("r2 (mid)");
        // q0 only sees Z-diagonal ops (rz, szz), so its mid and final measures are deterministically 0.
        assert_eq!(mid, 0, "milestone-6: q0 is Z-diagonal, mid (r2) must be 0");
        assert_eq!(f0, 0, "milestone-6: q0 is Z-diagonal, final_q0 (r0) must be 0");
        seen.insert((mid, f0, f1));
        n += 1;
        Engine::reset(&mut hybrid).unwrap();
    }
    assert_eq!(n, 200, "milestone-6: expected 200 shots");
    assert!(!seen.is_empty(), "milestone-6: qprog.ll must produce results");
    // The conditional branch is never taken here (mid is deterministically 0) -- branch-firing is
    // exercised by M5/M7; the quantum variety is on q1 (rx(pi)+ry+szz), so r1 varies across shots.
    println!("[milestone-6 qprog.ll -> pliron qec (rotations + qec.if) -> HybridEngine] OK -- {n} shots, observed (r2_mid,r0_final_q0,r1_final_q1): {seen:?}");
}

/// M7 step 2: a branch-*taken* fixture parsed from `.ll` — proves `qec.if` firing from parsed input
/// (unlike qprog.ll, whose q0 is deterministic). h q0 -> measure -> if 1 { x q1 } -> measure q1.
pub fn run_milestone_7() {
    let ctx = &mut Context::new();
    let src = include_str!("../fixtures/adaptive_branch.ll");
    let (module, bb, reg) = parse_qprog_ll(ctx, src).expect("milestone-7: parse adaptive_branch.ll");
    println!("=== adaptive_branch.ll parsed into pliron qec IR ===");
    println!("{}", module.get_operation().disp(ctx));
    verify_op(&module, ctx).unwrap_or_else(|e| panic!("[milestone-7 verify] FAILED: {}", e.disp(ctx)));
    let plan = plan_from_if_ir(ctx, bb, &reg);
    assert_eq!(plan.export, vec![2, 1], "milestone-7: result_record_output order must be [2,1] (mid, final_q1)");
    let engine = PlironIfEngine {
        batch1: plan.batch1,
        cond_outcome_idx: plan.cond_outcome_idx,
        then_cmds: plan.then_cmds,
        else_cmds: plan.else_cmds,
        post: plan.post,
        export: plan.export,
        stage: 0,
        b1: Vec::new(),
        b2: Vec::new(),
    };
    let mut hybrid = HybridEngineBuilder::new()
        .with_classical_engine(Box::new(engine))
        .with_quantum_engine(Box::new(StateVecEngine::new(2)))
        .build();
    // adaptive_branch records result-ids 2,1 -> r2 = mid (q0 after h), r1 = final_q1. Invariant r1==r2.
    let (mut all_eq, mut saw0, mut saw1) = (true, false, false);
    for _ in 0..200 {
        let shot = hybrid.run_shot().unwrap();
        let mid = shot.data.get("r2").and_then(Data::as_u32).expect("r2 (mid)");
        let fin = shot.data.get("r1").and_then(Data::as_u32).expect("r1 (final_q1)");
        if fin != mid {
            all_eq = false;
        }
        saw0 |= mid == 0;
        saw1 |= mid == 1;
        Engine::reset(&mut hybrid).unwrap();
    }
    assert!(all_eq, "milestone-7: final_q1 (r1) must equal mid (r2) -- qec.if firing from parsed input");
    assert!(saw0 && saw1, "milestone-7: branch must both fire (mid=1) and not (mid=0)");
    println!("[milestone-7 adaptive_branch.ll -> qec.if branch TAKEN from parsed input] OK -- r1(final_q1)==r2(mid) all 200 shots, mid 0 and 1 both seen");
}

/// Coverage for a measurement INSIDE the conditional branch: `b2`'s length varies with the taken
/// branch (taken: `[m1, f0, f1]`; skipped: `[f0, f1]`), so the engine's outcome reconstruction must
/// walk the actually-emitted measures, not a fixed layout. The in-branch measure is not recorded
/// (a cross-region SSA escape needs yield -- Phase 2); recorded outputs are the unconditional finals.
pub fn run_branch_measure() {
    let ctx = &mut Context::new();
    let src = include_str!("../fixtures/branch_measure.ll");
    let (module, bb, reg) = parse_qprog_ll(ctx, src).expect("branch_measure: parse");
    verify_op(&module, ctx).unwrap_or_else(|e| panic!("[branch_measure verify] FAILED: {}", e.disp(ctx)));
    let plan = plan_from_if_ir(ctx, bb, &reg);
    // structural: the then-branch really does contain a measurement (this is what we are covering),
    // and the in-branch measure (result-id 3) is NOT in the export list.
    assert!(plan.then_cmds.iter().any(|c| matches!(c, Cmd::Mz(..))), "then-branch must contain a measurement");
    assert_eq!(plan.export, vec![0, 1], "only the unconditional finals are recorded (in-branch m1 is not)");
    let engine = PlironIfEngine {
        batch1: plan.batch1,
        cond_outcome_idx: plan.cond_outcome_idx,
        then_cmds: plan.then_cmds,
        else_cmds: plan.else_cmds,
        post: plan.post,
        export: plan.export,
        stage: 0,
        b1: Vec::new(),
        b2: Vec::new(),
    };
    let mut hybrid = HybridEngineBuilder::new()
        .with_classical_engine(Box::new(engine))
        .with_quantum_engine(Box::new(StateVecEngine::new(2)))
        .build();
    let (mut all_eq, mut saw0, mut saw1) = (true, false, false);
    for _ in 0..200 {
        let shot = hybrid.run_shot().unwrap();
        // r0 = final_q0 (== mid), r1 = final_q1 (== mid). The in-branch measure of q1 leaves q1 == mid.
        let r0 = shot.data.get("r0").and_then(Data::as_u32).expect("r0 (final_q0)");
        let r1 = shot.data.get("r1").and_then(Data::as_u32).expect("r1 (final_q1)");
        if r0 != r1 {
            all_eq = false;
        }
        saw0 |= r0 == 0;
        saw1 |= r0 == 1;
        Engine::reset(&mut hybrid).unwrap();
    }
    assert!(all_eq, "branch_measure: final_q0 (r0) must equal final_q1 (r1) -- variable-length b2 reconstructed correctly");
    assert!(saw0 && saw1, "branch_measure: branch must both fire and not (r0 both 0 and 1)");
    println!("[branch_measure measure-inside-branch -> variable-length b2 reconstruction] OK -- r0==r1 all 200 shots, both 0 and 1 seen");
}

// ===================== public adapter: the opt-in QIS-LLVM-IR -> pliron call path =====================

/// True if the program has a conditional branch (`br i1 ...`) -- i.e. the adaptive single-diamond
/// shape (qprog-style). Its absence means straight-line (bell-style). This is the dispatch over the
/// covered QIS-LLVM subset; richer CFGs are out of scope (see the strangler scope doc).
fn has_conditional_branch(src: &str) -> bool {
    src.lines().any(|l| l.trim().starts_with("br i1"))
}

/// Lower a QIS-LLVM-IR program to the pliron `qec` dialect and return a boxed
/// `ClassicalControlEngine` ready for `HybridEngineBuilder` -- the narrow, opt-in entry point for the
/// pliron path. The incumbent `pecos-phir` stays the default; callers select this explicitly.
///
/// Scope: the covered QIS-LLVM subset -- Bell-style straight-line and the single-diamond adaptive
/// shape (`h`/`x`/`cx`/`rz`/`rx`/`ry`/`zz`/`m`, one `icmp`+`br` lifted to `qec.if`,
/// `result_record_output` export). The returned engine reports its own `num_qubits()` for sizing the
/// quantum backend. Returns a structured error if the lowered IR fails verification.
///
/// Known limits (tracked in the scope doc, not yet closed): the parser still panics on malformed
/// input outside the covered subset (parse-path hardening to structured errors is a separate item),
/// and `num_qubits()` is currently fixed at 2 (correct for the covered fixtures).
pub fn from_qis_llvm_ir_pliron(src: &str) -> std::result::Result<Box<dyn ClassicalControlEngine>, PecosError> {
    let ctx = &mut Context::new();
    let (module, bb, reg) = if has_conditional_branch(src) {
        parse_qprog_ll(ctx, src)?
    } else {
        parse_bell_ll(ctx, src)?
    };
    verify_op(&module, ctx)
        .map_err(|e| PecosError::Compilation(format!("pliron qec verification failed: {}", e.disp(ctx))))?;
    let plan = plan_from_if_ir(ctx, bb, &reg);
    Ok(Box::new(PlironIfEngine {
        batch1: plan.batch1,
        cond_outcome_idx: plan.cond_outcome_idx,
        then_cmds: plan.then_cmds,
        else_cmds: plan.else_cmds,
        post: plan.post,
        export: plan.export,
        stage: 0,
        b1: Vec::new(),
        b2: Vec::new(),
    }))
}

// ===================== regression tests (the milestones, run via `cargo test`) =====================
#[cfg(test)]
mod tests {
    use super::*;

    // ---- strangler differential vs the existing pecos-phir QIS->PHIR->sim path ----
    // Not an equivalence proof: on these fixtures pecos-phir produces nothing / errors, so the
    // honest result is a *characterization* of where the pliron port supersedes the murky path.
    // These assert pecos-phir's CURRENT behavior so a future change there trips the test and we
    // re-examine the cutover criterion.

    /// bell.ll export divergence: the port lowers `result_record_output` to `qec.record` ops and
    /// exports `r{id}` registers through the measurement registry (here exercising that real export
    /// path, not a manual pack); pecos-phir *elides* `result_record_output` (and bell.ll has no other
    /// export source), so its `Shot` is empty.
    #[test]
    fn differential_bell_ll_export_convention() {
        let bell = include_str!("../../../examples/llvm/bell.ll");

        // port side: bell.ll -> pliron qec (incl. qec.record + registry) -> the SAME export path as
        // qprog (plan_from_if_ir + PlironIfEngine) -> r0/r1 registers. No IfOp -> a single batch.
        let ctx = &mut Context::new();
        let (module, bb, reg) = parse_bell_ll(ctx, bell).expect("port lowers bell.ll");
        verify_op(&module, ctx).expect("port lowers bell.ll (with qec.record) and verifies");
        let plan = plan_from_if_ir(ctx, bb, &reg);
        assert_eq!(plan.export, vec![0, 1], "bell.ll records result-ids 0 (q0) then 1 (q1)");
        let engine = PlironIfEngine {
            batch1: plan.batch1,
            cond_outcome_idx: plan.cond_outcome_idx,
            then_cmds: plan.then_cmds,
            else_cmds: plan.else_cmds,
            post: plan.post,
            export: plan.export,
            stage: 0,
            b1: Vec::new(),
            b2: Vec::new(),
        };
        let mut hybrid = pecos_engines::hybrid::HybridEngineBuilder::new()
            .with_classical_engine(Box::new(engine))
            .with_quantum_engine(Box::new(StateVecEngine::with_seed(2, 7)))
            .build();
        let (mut saw0, mut saw1) = (false, false);
        for _ in 0..200 {
            let shot = hybrid.run_shot().unwrap();
            let r0 = shot.data.get("r0").and_then(Data::as_u32).expect("r0 (q0)");
            let r1 = shot.data.get("r1").and_then(Data::as_u32).expect("r1 (q1)");
            assert_eq!(r0, r1, "port bell.ll via qec.record export must be Bell-correlated, got r0={r0} r1={r1}");
            saw0 |= r0 == 0;
            saw1 |= r0 == 1;
            Engine::reset(&mut hybrid).unwrap();
        }
        assert!(saw0 && saw1, "port bell.ll must see both 00 and 11");

        // pecos-phir side: same bell.ll, run through its real engine -> empty Shot (no exports).
        let module = pecos_phir::parse_qis_to_quantum(bell).expect("pecos-phir parses bell.ll");
        let pe = pecos_phir::PhirEngine::new(module).expect("pecos-phir builds an engine");
        let mut hybrid = pecos_engines::hybrid::HybridEngineBuilder::new()
            .with_classical_engine(Box::new(pe))
            .with_quantum_engine(Box::new(StateVecEngine::with_seed(2, 7)))
            .build();
        let shot = hybrid.run_shot().unwrap();
        assert!(
            shot.data.is_empty(),
            "DIVERGENCE: pecos-phir elides result_record_output, so bell.ll yields no exported \
             registers; the port exports r0/r1. pecos-phir gave: {:?}",
            shot.data.keys().collect::<Vec<_>>()
        );
        println!("[differential bell.ll] port -> r0==r1 Bell pair via qec.record export; pecos-phir -> empty Shot (record_output elided)");
    }

    /// qprog.ll rotation divergence: the port lowers `rz/rx/ry/zz` (M6 runs it end-to-end), while
    /// pecos-phir's `qis_to_quantum` currently cannot resolve the `rz` angle to a constant and errors.
    #[test]
    fn differential_qprog_ll_rotation_support() {
        let qprog = include_str!("../../../examples/llvm/qprog.ll");

        // port side: lowers without error (full end-to-end physics is m6).
        let ctx = &mut Context::new();
        let (module, _bb, _reg) = parse_qprog_ll(ctx, qprog).expect("port lowers qprog.ll");
        verify_op(&module, ctx).expect("port lowers qprog.ll (rotations + qec.if) and verifies");

        // pecos-phir side: errors lowering the rotation angle.
        let err = pecos_phir::parse_qis_to_quantum(qprog)
            .expect_err("DIVERGENCE: pecos-phir is expected to fail lowering qprog.ll's rotations");
        let msg = format!("{err:?}");
        assert!(
            msg.contains("angle") || msg.contains("rz"),
            "expected the divergence to be the rz-angle resolution; got: {msg}"
        );
        println!("[differential qprog.ll] port lowers rotations + qec.if; pecos-phir errors: {msg}");
    }

    /// The real call path: drive the PUBLIC `from_qis_llvm_ir_pliron` adapter (not the internal
    /// parse/plan/engine pieces) through `HybridEngine`, for both the straight-line (bell) and
    /// diamond (qprog) shapes -- proving the opt-in entry point produces a usable engine.
    #[test]
    fn adapter_real_call_path_bell_and_qprog() {
        use pecos_engines::hybrid::HybridEngineBuilder;

        // bell.ll (straight-line) -> r0==r1 Bell pair via the registry/qec.record export.
        let eng = from_qis_llvm_ir_pliron(include_str!("../../../examples/llvm/bell.ll")).expect("adapter lowers bell.ll");
        let n = eng.num_qubits();
        let mut hybrid = HybridEngineBuilder::new()
            .with_classical_engine(eng)
            .with_quantum_engine(Box::new(StateVecEngine::with_seed(n, 11)))
            .build();
        let (mut saw0, mut saw1) = (false, false);
        for _ in 0..200 {
            let shot = hybrid.run_shot().unwrap();
            let r0 = shot.data.get("r0").and_then(Data::as_u32).expect("r0");
            let r1 = shot.data.get("r1").and_then(Data::as_u32).expect("r1");
            assert_eq!(r0, r1, "bell via adapter must be Bell-correlated, got r0={r0} r1={r1}");
            saw0 |= r0 == 0;
            saw1 |= r0 == 1;
            Engine::reset(&mut hybrid).unwrap();
        }
        assert!(saw0 && saw1, "bell via adapter must see both 00 and 11");

        // qprog.ll (diamond) -> records r0/r1/r2; q0 is Z-diagonal so mid (r2) and final_q0 (r0) are 0.
        let eng = from_qis_llvm_ir_pliron(include_str!("../../../examples/llvm/qprog.ll")).expect("adapter lowers qprog.ll");
        let n = eng.num_qubits();
        let mut hybrid = HybridEngineBuilder::new()
            .with_classical_engine(eng)
            .with_quantum_engine(Box::new(StateVecEngine::with_seed(n, 11)))
            .build();
        for _ in 0..50 {
            let shot = hybrid.run_shot().unwrap();
            assert_eq!(shot.data.get("r2").and_then(Data::as_u32), Some(0), "qprog mid (r2) deterministically 0");
            assert_eq!(shot.data.get("r0").and_then(Data::as_u32), Some(0), "qprog final_q0 (r0) deterministically 0");
            assert!(shot.data.get("r1").and_then(Data::as_u32).is_some(), "qprog final_q1 (r1) present");
            Engine::reset(&mut hybrid).unwrap();
        }
    }
    /// Dynamic qubit count: a 3-qubit GHZ through the adapter must report `num_qubits()==3` (not the
    /// old hard-coded 2) and produce a perfectly correlated triple (r0==r1==r2).
    #[test]
    fn adapter_ghz3_dynamic_qubit_count() {
        use pecos_engines::hybrid::HybridEngineBuilder;
        let eng = from_qis_llvm_ir_pliron(include_str!("../fixtures/ghz3.ll")).expect("adapter lowers ghz3.ll");
        assert_eq!(eng.num_qubits(), 3, "GHZ-3 engine must report 3 qubits (dynamic, not hard-coded 2)");
        let n = eng.num_qubits();
        let mut hybrid = HybridEngineBuilder::new()
            .with_classical_engine(eng)
            .with_quantum_engine(Box::new(StateVecEngine::with_seed(n, 13)))
            .build();
        let (mut saw0, mut saw1) = (false, false);
        for _ in 0..200 {
            let shot = hybrid.run_shot().unwrap();
            let r0 = shot.data.get("r0").and_then(Data::as_u32).expect("r0");
            let r1 = shot.data.get("r1").and_then(Data::as_u32).expect("r1");
            let r2 = shot.data.get("r2").and_then(Data::as_u32).expect("r2");
            assert!(r0 == r1 && r1 == r2, "GHZ-3 must be fully correlated, got r0={r0} r1={r1} r2={r2}");
            saw0 |= r0 == 0;
            saw1 |= r0 == 1;
            Engine::reset(&mut hybrid).unwrap();
        }
        assert!(saw0 && saw1, "GHZ-3 must see both 000 and 111");
    }
    #[test]
    fn m0_hand_built_bell_seam() {
        run_and_check("milestone-0 hand-built Bell", bell_message(), 200);
    }
    #[test]
    fn m1_pliron_emitted_bell() {
        run_milestone_1();
    }
    #[test]
    fn m2_pliron_classical_control_engine() {
        run_milestone_2();
    }
    #[test]
    fn m3_bell_ll_parse() {
        run_milestone_3();
    }
    #[test]
    fn m4_adaptive_multi_batch() {
        run_milestone_4();
    }
    #[test]
    fn m5_region_based_qec_if() {
        run_milestone_5();
    }
    #[test]
    fn m6_literal_qprog_ll() {
        run_milestone_6();
    }
    #[test]
    fn m7_branch_taken_qec_if() {
        run_milestone_7();
    }
    #[test]
    fn measure_inside_branch_variable_length_b2() {
        run_branch_measure();
    }

    /// The measurement-SSA registry is the metadata home: every `qec.measure` value resolves to its
    /// (qubit, basis, export-label) via the side-table, with no attribute on the op. Keyed by the
    /// SSA value -- the value IS the identity.
    #[test]
    fn registry_holds_measurement_metadata() {
        let ctx = &mut Context::new();
        let src = include_str!("../../../examples/llvm/qprog.ll");
        let (_module, bb, reg) = parse_qprog_ll(ctx, src).expect("parse qprog.ll");
        let mut infos: Vec<(usize, Basis, u64)> = bb
            .deref(ctx)
            .iter(ctx)
            .collect::<Vec<_>>()
            .into_iter()
            .filter_map(|op| Operation::get_op::<MeasureOp>(op, ctx))
            .map(|m| {
                let i = reg.get(m.get_result(ctx));
                (i.qubit, i.basis, i.export_label)
            })
            .collect();
        infos.sort_by_key(|&(q, _, label)| (label, q));
        // qprog.ll: result-id 0 = final q0, 1 = final q1, 2 = mid q0 -- all Z measurements.
        assert_eq!(infos, vec![(0, Basis::Z, 0), (1, Basis::Z, 1), (0, Basis::Z, 2)]);
    }

    // ---- negative tests: prove the verifiers and seam invariants actually bite ----

    /// `qec.h` on a `qec.alloc` handle (an alloc, not a qubitref) must be rejected by verification.
    /// Guards against the gate verifiers silently regressing to `verifier="succ"` no-ops.
    #[test]
    fn negative_h_on_alloc_rejected() {
        let ctx = &mut Context::new();
        let alloc = QallocOp::new(ctx);
        let bad = HOp::new(ctx, alloc.get_result(ctx));
        let res = verify_op(&bad, ctx);
        assert!(res.is_err(), "qec.h on a qec.alloc handle must fail verification, got Ok");
    }

    /// `qec.cond_x`'s condition (operand 0) must be an `i1` measurement result. A non-`i1` condition
    /// (here a qubitref) must be rejected -- guards the read-only type-inspection check in verify.
    #[test]
    fn negative_cond_x_non_i1_condition_rejected() {
        let ctx = &mut Context::new();
        let alloc = QallocOp::new(ctx);
        let s0 = SlotOp::new(ctx, alloc.get_result(ctx), 0);
        let s1 = SlotOp::new(ctx, alloc.get_result(ctx), 1);
        // use a qubitref (operand 0) as the condition -- it is not an i1 measurement result.
        let bad = CondXOp::new(ctx, s0.get_result(ctx), s1.get_result(ctx));
        let err = verify_op(&bad, ctx).expect_err("qec.cond_x with a non-i1 condition must fail verification");
        assert!(
            format!("{}", err.disp(ctx)).contains("i1"),
            "expected the i1-condition rejection, got: {}",
            err.disp(ctx)
        );
    }

    /// `qec.rz` (and rx/ry) must reject a non-qubitref operand -- guards the rotation verifiers that
    /// were tightened from `verifier = "succ"`.
    #[test]
    fn negative_rz_on_alloc_rejected() {
        let ctx = &mut Context::new();
        let alloc = QallocOp::new(ctx);
        // rz on a qec.alloc handle (not a qubitref).
        let bad = RzOp::new(ctx, alloc.get_result(ctx), Angle64::from_radians(0.5));
        assert!(verify_op(&bad, ctx).is_err(), "qec.rz on a qec.alloc handle must fail verification");
    }

    /// `qec.if`'s condition must be an `i1`; a non-`i1` (here a qubitref) must be rejected -- guards
    /// the IfOp verifier tightened from `verifier = "succ"`.
    #[test]
    fn negative_if_non_i1_condition_rejected() {
        let ctx = &mut Context::new();
        let alloc = QallocOp::new(ctx);
        let s0v = SlotOp::new(ctx, alloc.get_result(ctx), 0).get_result(ctx);
        let ifop = IfOp::new(ctx, s0v); // qubitref condition -- not an i1
        ifop.make_region_block(ctx, 0);
        ifop.make_region_block(ctx, 1);
        let err = verify_op(&ifop, ctx).expect_err("qec.if with a non-i1 condition must fail verification");
        assert!(
            format!("{}", err.disp(ctx)).contains("i1"),
            "expected the i1-condition rejection, got: {}",
            err.disp(ctx)
        );
    }

    /// The plan-build assertion catches registry/IR drift: a measurement whose registry qubit
    /// disagrees with its IR slot index must panic (the registry is a side-table `verify` can't check).
    #[test]
    #[should_panic(expected = "registry/IR drift")]
    fn registry_ir_drift_panics_at_plan_build() {
        let ctx = &mut Context::new();
        let module = ModuleOp::new(ctx, "drift".try_into().unwrap());
        let func_ty = FunctionType::get(ctx, vec![], vec![]);
        let func = FuncOp::new(ctx, "main".try_into().unwrap(), func_ty);
        module.append_operation(ctx, func.get_operation(), 0);
        let bb = func.get_entry_block(ctx);
        macro_rules! push {
            ($op:expr) => {{ let o = $op; o.get_operation().insert_at_back(bb, ctx); o }};
        }
        let qv = push!(QallocOp::new(ctx)).get_result(ctx);
        let sv = push!(SlotOp::new(ctx, qv, 0)).get_result(ctx); // slot index 0
        push!(PrepareOp::new(ctx, sv));
        let m = push!(MeasureOp::new(ctx, sv));
        let mut reg = MeasurementRegistry::default();
        reg.record(m.get_result(ctx), MeasurementInfo { qubit: 1, basis: Basis::Z, export_label: 0 }); // qubit 1 != slot 0
        let _ = plan_from_if_ir(ctx, bb, &reg); // measure_to_mz must assert and panic
    }

    /// The `qec.angle` attribute round-trips an `Angle64` through the IR exactly (fixed-point, no
    /// f64 bit-pattern hack): the fraction stored on a `qec.rz` reads back bit-identical.
    #[test]
    fn angle_attr_roundtrips_fixed_point() {
        let ctx = &mut Context::new();
        let module = ModuleOp::new(ctx, "ang".try_into().unwrap());
        let func_ty = FunctionType::get(ctx, vec![], vec![]);
        let func = FuncOp::new(ctx, "main".try_into().unwrap(), func_ty);
        module.append_operation(ctx, func.get_operation(), 0);
        let bb = func.get_entry_block(ctx);
        macro_rules! push {
            ($op:expr) => {{ let o = $op; o.get_operation().insert_at_back(bb, ctx); o }};
        }
        let q = push!(QallocOp::new(ctx));
        let sv = push!(SlotOp::new(ctx, q.get_result(ctx), 0)).get_result(ctx);
        let angle = Angle64::from_radians(1.07);
        let rz = push!(RzOp::new(ctx, sv, angle));
        let got = get_angle(ctx, rz.get_operation());
        assert_eq!(got.fraction(), angle.fraction(), "qec.angle must round-trip the Angle64 fixed-point fraction exactly");
    }

    /// Recording a measurement defined *inside* a `qec.if` region from the OUTER block is a
    /// cross-region SSA escape (it needs block-args/yield -- Phase 2). `qec.record`'s verifier must
    /// reject it, not silently accept it.
    #[test]
    fn negative_record_of_in_region_measurement_rejected() {
        let ctx = &mut Context::new();
        let module = ModuleOp::new(ctx, "bad_region".try_into().unwrap());
        let func_ty = FunctionType::get(ctx, vec![], vec![]);
        let func = FuncOp::new(ctx, "main".try_into().unwrap(), func_ty);
        module.append_operation(ctx, func.get_operation(), 0);
        let bb = func.get_entry_block(ctx);
        macro_rules! push {
            ($op:expr) => {{ let o = $op; o.get_operation().insert_at_back(bb, ctx); o }};
        }
        let q = push!(QallocOp::new(ctx));
        let qv = q.get_result(ctx);
        let s0v = push!(SlotOp::new(ctx, qv, 0)).get_result(ctx);
        let s1v = push!(SlotOp::new(ctx, qv, 1)).get_result(ctx);
        push!(PrepareOp::new(ctx, s0v));
        push!(PrepareOp::new(ctx, s1v));
        push!(HOp::new(ctx, s0v));
        let m0v = push!(MeasureOp::new(ctx, s0v)).get_result(ctx);
        let ifop = push!(IfOp::new(ctx, m0v));
        let then_bb = ifop.make_region_block(ctx, 0);
        let m1 = MeasureOp::new(ctx, s1v); // measure INSIDE the then-region
        m1.get_operation().insert_at_back(then_bb, ctx);
        let m1v = m1.get_result(ctx);
        let _else_bb = ifop.make_region_block(ctx, 1);
        push!(RecordOp::new(ctx, m1v)); // record it from the OUTER block -- the cross-region escape
        push!(EndOp::new(ctx));
        let err = verify_op(&module, ctx).expect_err(
            "recording an in-qec.if-region measurement from the outer block must fail verification",
        );
        // bite for the RIGHT reason: the region-scope check, not some incidental failure.
        assert!(
            format!("{}", err.disp(ctx)).contains("cross-region escape"),
            "expected the cross-region-escape rejection, got: {}",
            err.disp(ctx)
        );
    }

    /// `ByteMessage::create_empty()` is *semantically* empty (`is_empty() == Ok(true)`) even though
    /// its `as_bytes()` is not byte-empty -- the M2-era footgun that made single-batch engines loop
    /// forever until they signalled `EngineStage::Complete` explicitly. Lock the invariant down.
    #[test]
    fn empty_message_is_semantically_empty() {
        let empty = ByteMessage::create_empty();
        assert!(empty.is_empty().unwrap(), "create_empty() must be semantically empty (is_empty()==Ok(true))");
        assert!(!empty.as_bytes().is_empty(), "create_empty().as_bytes() is NOT byte-empty -- that is the footgun");
    }

    /// A `result_record_output` that names a result-id no measurement produced must return a
    /// structured error (not silently record a default 0, not panic). Guards the export resolution.
    #[test]
    fn negative_record_of_unknown_result_id_errors() {
        const BAD: &str = "\
define i64 @qmain(i64 %arg) #0 {
    call void @__quantum__qis__h__body(i64 0)
    %mid = call i32 @__quantum__qis__m__body(i64 0, i64 2)
    %cond = icmp eq i32 %mid, 1
    br i1 %cond, label %apply_x, label %skip_x
apply_x:
    call void @__quantum__qis__x__body(i64 1)
    br label %final
skip_x:
    br label %final
final:
    %f1 = call i32 @__quantum__qis__m__body(i64 1, i64 1)
    call void @__quantum__rt__result_record_output(i64 9, i8* null)
    ret i64 0
}
";
        let ctx = &mut Context::new();
        let Err(err) = parse_qprog_ll(ctx, BAD) else {
            panic!("recording an unknown result-id must error, but the parse succeeded");
        };
        assert!(
            format!("{err}").contains("unknown result-id 9"),
            "expected the unknown-result-id rejection, got: {err}"
        );
    }

    /// A QIS call outside the covered subset (here an `s` gate) must be rejected with a structured
    /// error, NOT silently dropped (the old behavior).
    #[test]
    fn negative_unsupported_gate_errors() {
        const BAD: &str = "\
define i64 @qmain(i64 %arg) #0 {
    call void @__quantum__qis__h__body(i64 0)
    call void @__quantum__qis__s__body(i64 0)
    %r0 = call i32 @__quantum__qis__m__body(i64 0, i64 0)
    call void @__quantum__rt__result_record_output(i64 0, i8* null)
    ret i64 0
}
";
        let ctx = &mut Context::new();
        let Err(err) = parse_bell_ll(ctx, BAD) else {
            panic!("an unsupported gate must error, but the parse succeeded");
        };
        assert!(
            format!("{err}").contains("not in the covered subset"),
            "expected the unsupported-operation rejection, got: {err}"
        );
    }

    /// More than one conditional branch is outside the single-diamond subset and must be rejected
    /// (not misparsed by taking only the first branch).
    #[test]
    fn negative_multiple_conditional_branches_errors() {
        const BAD: &str = "\
define i64 @qmain(i64 %arg) #0 {
    %m0 = call i32 @__quantum__qis__m__body(i64 0, i64 0)
    %c0 = icmp eq i32 %m0, 1
    br i1 %c0, label %a, label %b
a:
    %m1 = call i32 @__quantum__qis__m__body(i64 1, i64 1)
    %c1 = icmp eq i32 %m1, 1
    br i1 %c1, label %c, label %d
b:
    br label %d
c:
    br label %d
d:
    ret i64 0
}
";
        let ctx = &mut Context::new();
        let Err(err) = parse_qprog_ll(ctx, BAD) else {
            panic!("more than one conditional branch must error, but the parse succeeded");
        };
        assert!(
            format!("{err}").contains("more than one conditional branch"),
            "expected the unsupported-control-flow rejection, got: {err}"
        );
    }
}
