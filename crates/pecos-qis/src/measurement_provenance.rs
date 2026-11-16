//! Post-scheduling byte-shadow instrumentation, verified by LLVM's CLI tools.
//! No LLVM library is linked into this crate for this pass.
mod symbols;
mod syntax;

use crate::qis_interface::{InterfaceError, ProgramFormat};
use pecos_qis_ffi_types::provenance::{CONSTANT, DERIVED, UNKNOWN, memory, op};
use std::collections::{BTreeMap, BTreeSet};
use symbols::Policy;
use syntax::{size, split, text, tokens, type_end, typed};
type Result<T> = std::result::Result<T, InterfaceError>;
fn error(reason: impl std::fmt::Display) -> InterfaceError {
    InterfaceError::InvalidFormat(format!("measurement provenance: {reason}"))
}
struct Instruction {
    raw: String,
    name: Option<String>,
    op: String,
    args: Vec<String>,
}
impl Instruction {
    fn parse(raw: &str) -> Self {
        let mut t = tokens(raw);
        let name = if t.get(1).is_some_and(|v| v == "=") {
            let name = t.remove(0);
            t.remove(0);
            Some(name)
        } else {
            None
        };
        if t.first()
            .is_some_and(|s| matches!(s.as_str(), "tail" | "musttail" | "notail"))
        {
            t.remove(0);
        }
        let op = if t.is_empty() {
            String::new()
        } else {
            t.remove(0)
        };
        // Semantic flags do not change the shadow transfer rule.
        while t.first().is_some_and(|s| {
            matches!(
                s.as_str(),
                "nuw"
                    | "nsw"
                    | "exact"
                    | "nneg"
                    | "fast"
                    | "reassoc"
                    | "nnan"
                    | "ninf"
                    | "nsz"
                    | "arcp"
                    | "contract"
                    | "afn"
                    | "samesign"
                    | "disjoint"
            )
        }) {
            t.remove(0);
        }
        // Metadata is not an instruction operand.
        if let Some(i) = t
            .windows(2)
            .position(|w| w[0] == "," && w[1].starts_with('!'))
        {
            t.truncate(i);
        }
        Self {
            raw: raw.to_string(),
            name,
            op,
            args: t,
        }
    }
}
struct Function {
    header: String,
    signature: syntax::Call,
    lines: Vec<String>,
}
struct Writer<'a> {
    out: String,
    next: usize,
    shadows: BTreeMap<String, String>,
    internal: &'a BTreeSet<String>,
    types: &'a BTreeMap<String, String>,
    bools: BTreeSet<String>,
}
impl Writer<'_> {
    fn temp(&mut self) -> String {
        let n = self.next;
        self.next += 1;
        format!("%pecos.prov.t{n}")
    }
    fn line(&mut self, s: impl AsRef<str>) {
        self.out.push_str("  ");
        self.out.push_str(s.as_ref());
        self.out.push('\n');
    }
    fn shadow(&self, value: &str) -> Result<String> {
        if value.starts_with('%') {
            self.shadows
                .get(value)
                .cloned()
                .ok_or_else(|| error(format!("unresolved SSA operand {value}")))
        } else {
            Ok(CONSTANT.to_string())
        }
    }
    fn op(&mut self, code: u64, args: &[String]) -> String {
        let n = self.temp();
        self.op_as(&n, code, args);
        n
    }
    fn op_as(&mut self, n: &str, code: u64, args: &[String]) {
        let a = (0..4)
            .map(|i| format!("i64 {}", args.get(i).map_or("0", String::as_str)))
            .collect::<Vec<_>>()
            .join(", ");
        self.line(format!("{n} = call i64 @pecos_prov_op_v1(i64 {code}, {a})"));
    }
    fn memory(
        &mut self,
        code: u64,
        pointer: &str,
        source: &str,
        length: &str,
        h: &str,
        ph: &str,
    ) -> String {
        let n = self.temp();
        self.line(format!("{n} = call i64 @pecos_prov_memory_v1(i64 {code}, ptr {pointer}, ptr {source}, i64 {length}, i64 {h}, i64 {ph})"));
        n
    }
    fn assign(&mut self, name: &str, h: &str) {
        self.line(format!("{name} = add i64 {h}, 0"));
    }
    fn merge(&mut self, values: impl IntoIterator<Item = String>) -> Result<String> {
        let mut h = CONSTANT.to_string();
        for v in values {
            let s = self.shadow(&v)?;
            h = self.op(op::MERGE, &[h, s]);
        }
        Ok(h)
    }
    fn wide(&mut self, h: &str, ty: &str) -> String {
        self.op(op::SIZED, &[h.to_string(), size(ty)])
    }
    fn insert(
        &mut self,
        base: &str,
        member: &str,
        member_ty: &str,
        offset: &str,
        total_ty: &str,
    ) -> String {
        let member = self.wide(member, member_ty);
        self.op(
            op::INSERT,
            &[base.into(), member, offset.into(), size(total_ty)],
        )
    }
    // Arrays read a range, including the dense descriptor's x/data fields.
    // y/mask are deliberately excluded because the output runtime ignores them.
    fn array_range(&mut self, callee: &syntax::Call, dense: bool) -> Result<String> {
        let range = if dense {
            let descriptor = &callee.args[2].1;
            let pointer_shadow = self.shadow(descriptor)?;
            let mut range = pointer_shadow.clone();
            for (index, ty) in [(0, "i32"), (2, "ptr")] {
                let field = self.temp();
                self.line(format!("{field} = getelementptr {{i32, i32, ptr, ptr}}, ptr {descriptor}, i32 0, i32 {index}"));
                let field_shadow = self.memory(
                    memory::LOAD,
                    &field,
                    "null",
                    &size(ty),
                    &CONSTANT.to_string(),
                    &pointer_shadow,
                );
                range = self.op(op::MERGE, &[range, field_shadow]);
            }
            range
        } else {
            self.merge(
                callee
                    .args
                    .iter()
                    .rev()
                    .take(2)
                    .map(|(_, value)| value.clone()),
            )?
        };
        Ok(self.op(op::ADDRESS, &[range]))
    }
    fn integer64(&mut self, ty: &str, v: &str) -> Result<String> {
        if ty == "i64" {
            return Ok(v.into());
        }
        let bits = ty
            .strip_prefix('i')
            .and_then(|v| v.parse::<usize>().ok())
            .ok_or_else(|| error("expected integer length"))?;
        let n = self.temp();
        self.line(format!(
            "{n} = {} {ty} {v} to i64",
            if bits < 64 { "zext" } else { "trunc" }
        ));
        Ok(n)
    }
    fn member(&self, ty: &str, indices: &[Vec<String>]) -> Result<(String, String)> {
        let mut current = ty.to_string();
        let mut path = String::new();
        for index in indices {
            let i = text(index)
                .parse::<usize>()
                .map_err(|_| error("aggregate index is not constant"))?;
            while current.starts_with('%') {
                current = self
                    .types
                    .get(&current)
                    .cloned()
                    .ok_or_else(|| error(format!("unresolved aggregate type {current}")))?;
            }
            let t = tokens(&current);
            current = if t.first().is_some_and(|s| s == "[") {
                text(&t[3..t.len() - 1])
            } else {
                let start = if t.first().is_some_and(|s| s == "<") {
                    2
                } else {
                    1
                };
                let end = t.len() - start;
                text(
                    split(&t[start..end])
                        .get(i)
                        .ok_or_else(|| error("aggregate index out of bounds"))?,
                )
            };
            path.push_str(", i32 ");
            path.push_str(&i.to_string());
        }
        Ok((
            current,
            format!("ptrtoint (ptr getelementptr ({ty}, ptr null, i32 0{path}) to i64)"),
        ))
    }
    fn instruction(&mut self, ins: &Instruction) -> Result<()> {
        let n = ins.name.as_ref().and_then(|n| self.shadows.get(n)).cloned();
        let a = &ins.args;
        if a.iter().any(|v| matches!(v.as_str(), "atomic" | "asm")) {
            return Err(error("atomic or inline asm"));
        }
        let raw = if ins.op == "call" {
            let mut call_tokens = tokens(&ins.raw);
            let opcode = if ins.name.is_some() { 2 } else { 0 };
            if call_tokens
                .get(opcode)
                .is_some_and(|s| matches!(s.as_str(), "tail" | "musttail" | "notail"))
            {
                call_tokens.remove(opcode);
            }
            syntax::relax_effects(&text(&call_tokens))?
        } else {
            ins.raw.clone()
        };
        if ins.op == "phi" {
            let end = type_end(a, 0)?;
            let ty = text(&a[..end]);
            reject_vector(&ty, "SSA")?;
            let pairs = split(&a[end..])
                .iter()
                .map(|p| {
                    let pair = split(&p[1..p.len() - 1]);
                    Ok(format!(
                        "[ {}, {} ]",
                        self.shadow(&text(&pair[0]))?,
                        text(&pair[1])
                    ))
                })
                .collect::<Result<Vec<_>>>()?
                .join(", ");
            self.line(raw);
            self.line(format!(
                "{} = phi i64 {pairs}",
                n.as_deref().ok_or_else(|| error("phi without result"))?
            ));
            return Ok(());
        }
        // Instructions with effects on control/memory/call frames must instrument
        // before the original operation; all other shadows follow it.
        let h = match ins.op.as_str() {
            "call" => {
                let callee = syntax::call(a)?;
                reject_vector(&callee.ty, "SSA")?;
                for (ty, value) in &callee.args {
                    reject_vector(ty, value)?;
                }
                if self.internal.contains(&callee.symbol) {
                    self.op(op::PUSH_FRAME, &[]);
                    for (i, (_, v)) in callee.args.iter().enumerate() {
                        let h = self.shadow(v)?;
                        self.op(op::SET_ARGUMENT, &[i.to_string(), h]);
                    }
                    self.line(raw);
                    self.op(op::POP_FRAME, &[])
                } else if callee.symbol.starts_with("@llvm.") {
                    let name = &callee.symbol;
                    if name.starts_with("@llvm.lifetime.") || name == "@llvm.assume" {
                        self.line(raw);
                        CONSTANT.to_string()
                    } else if name.starts_with("@llvm.memcpy.")
                        || name.starts_with("@llvm.memmove.")
                        || name.starts_with("@llvm.memset.")
                    {
                        let args = &callee.args;
                        if args.len() != 4 {
                            return Err(error("memory intrinsic signature"));
                        }
                        let len = self.integer64(&args[2].0, &args[2].1)?;
                        let is_fill = name.starts_with("@llvm.memset.");
                        let ph = self.merge(if is_fill {
                            vec![args[0].1.clone(), args[2].1.clone()]
                        } else {
                            vec![args[0].1.clone(), args[1].1.clone(), args[2].1.clone()]
                        })?;
                        let h = self.shadow(&args[1].1)?;
                        self.memory(
                            if is_fill { memory::FILL } else { memory::COPY },
                            &args[0].1,
                            if is_fill { "null" } else { &args[1].1 },
                            &len,
                            &h,
                            &ph,
                        );
                        self.line(raw);
                        CONSTANT.to_string()
                    } else {
                        return Err(error(format!("unlisted intrinsic {}", callee.symbol)));
                    }
                } else {
                    let policy = symbols::policy(&callee.symbol);
                    if policy == Policy::Unknown
                        && callee.args.iter().any(|(t, _)| self.pointer_type(t))
                    {
                        return Err(error(format!(
                            "unlisted external {} receives pointer",
                            callee.symbol
                        )));
                    }
                    if !callee.args.is_empty()
                        && (matches!(
                            policy,
                            Policy::Quantum | Policy::QuantumSeed | Policy::QuantumArray(..)
                        ) || callee.symbol == "@selene_qubit_measure")
                    {
                        let h = self.merge(callee.args.iter().map(|(_, v)| v.clone()))?;
                        self.op(op::CONTROL, &[h]);
                    }
                    if let Policy::QuantumArray(pointer, length) = policy {
                        let (ty, len) = &callee.args[length];
                        let len = self.integer64(ty, len)?;
                        let bytes = self.temp();
                        self.line(format!("{bytes} = mul i64 {len}, {}", size("i64")));
                        let p = &callee.args[pointer].1;
                        let ph = self.merge([p.clone(), callee.args[length].1.clone()])?;
                        let h = self.memory(
                            memory::LOAD,
                            p,
                            "null",
                            &bytes,
                            &CONSTANT.to_string(),
                            &ph,
                        );
                        self.op(op::CONTROL, &[h]);
                    }
                    if let Policy::DerivedBuffers(indices) = policy {
                        let range =
                            self.merge(indices.iter().map(|&i| callee.args[i].1.clone()))?;
                        self.op(op::ADDRESS, &[range]);
                    }
                    let label_arguments = match policy {
                        Policy::Bool | Policy::ScalarSink => callee.args.len().checked_sub(1),
                        Policy::Array { dense, .. } => {
                            callee.args.len().checked_sub(if dense { 1 } else { 2 })
                        }
                        _ => None,
                    };
                    if let Some(count) = label_arguments {
                        let label = self
                            .merge(callee.args[..count].iter().map(|(_, value)| value.clone()))?;
                        self.op(op::ADDRESS, &[label]);
                    }
                    if policy == Policy::Bool {
                        let value = &callee
                            .args
                            .last()
                            .ok_or_else(|| error("bool output missing value"))?
                            .1;
                        let h = self.shadow(value)?;
                        self.line(format!("call void @pecos_prov_bind_v1(i64 {h})"));
                    }
                    if let Policy::Array { dense, boolean } = policy {
                        let address_shadow = self.array_range(&callee, dense)?;
                        if boolean {
                            self.line(format!(
                                "call void @pecos_prov_bind_v1(i64 {address_shadow})"
                            ));
                        }
                    }
                    self.line(raw);
                    if policy == Policy::Allocation {
                        let count = self.shadow(&callee.args[0].1)?;
                        self.op(op::ADDRESS, &[count])
                    } else if matches!(
                        policy,
                        Policy::Seed | Policy::QuantumSeed | Policy::SeleneSeed
                    ) {
                        let h = self.temp();
                        self.line(format!("{h} = call i64 @pecos_prov_take_seed_v1()"));
                        self.insert(
                            &CONSTANT.to_string(),
                            &h,
                            "i8",
                            if policy == Policy::SeleneSeed {
                                "4"
                            } else {
                                "0"
                            },
                            &callee.ty,
                        )
                    } else if policy == Policy::Config {
                        let address_shadow = self.shadow(&callee.args[0].1)?;
                        self.memory(
                            memory::STORE,
                            &callee.args[0].1,
                            "null",
                            &size("ptr"),
                            &CONSTANT.to_string(),
                            &address_shadow,
                        );
                        CONSTANT.to_string()
                    } else if matches!(policy, Policy::Derived | Policy::DerivedBuffers(_)) {
                        DERIVED.to_string()
                    } else if policy == Policy::Unknown {
                        UNKNOWN.to_string()
                    } else {
                        CONSTANT.to_string()
                    }
                }
            }
            "ret" => {
                {
                    let h = if a[0] == "void" {
                        CONSTANT.to_string()
                    } else {
                        let (ty, value) = typed(a)?;
                        reject_vector(&ty, &value)?;
                        self.shadow(&value)?
                    };
                    self.op(op::SET_RETURN, &[h]);
                }
                self.line(raw);
                CONSTANT.to_string()
            }
            "br" | "switch" => {
                if a.first().is_some_and(|s| s != "label") {
                    let first = split(a);
                    let (_, v) = typed(&first[0])?;
                    let h = self.shadow(&v)?;
                    self.op(op::CONTROL, &[h]);
                }
                self.line(raw);
                CONSTANT.to_string()
            }
            "unreachable" => {
                self.line(raw);
                CONSTANT.to_string()
            }
            "store" => {
                let args = split(a);
                let start = usize::from(args[0].first().is_some_and(|s| s == "volatile"));
                let (ty, v) = typed(&args[0][start..])?;
                reject_memory_vector(&ty, &v)?;
                let (_, p) = typed(&args[1])?;
                let h = self.shadow(&v)?;
                let ph = self.shadow(&p)?;
                self.memory(memory::STORE, &p, "null", &size(&ty), &h, &ph);
                self.line(raw);
                CONSTANT.to_string()
            }
            "load" => {
                let args = split(a);
                let start = usize::from(args[0].first().is_some_and(|s| s == "volatile"));
                let ty = text(&args[0][start..]);
                reject_memory_vector(&ty, "SSA")?;
                let (_, p) = typed(&args[1])?;
                let ph = self.shadow(&p)?;
                self.line(raw);
                self.memory(
                    memory::LOAD,
                    &p,
                    "null",
                    &size(&ty),
                    &CONSTANT.to_string(),
                    &ph,
                )
            }
            "alloca" => {
                let args = split(a);
                let ty = text(&args[0]);
                let mut len = syntax::allocation_size(&ty);
                let mut count_shadow = CONSTANT.to_string();
                if let Some(count) = args
                    .get(1)
                    .filter(|v| v.first().is_some_and(|s| s.starts_with('i')))
                {
                    let (ct, cv) = typed(count)?;
                    count_shadow = self.shadow(&cv)?;
                    let count = self.integer64(&ct, &cv)?;
                    let bytes = self.temp();
                    self.line(format!("{bytes} = mul i64 {len}, {count}"));
                    len = bytes;
                }
                self.line(raw);
                let ptr = ins
                    .name
                    .as_deref()
                    .ok_or_else(|| error("alloca without result"))?;
                let address_shadow = self.op(op::ADDRESS, &[count_shadow]);
                self.memory(
                    memory::CLEAR,
                    ptr,
                    "null",
                    &len,
                    &CONSTANT.to_string(),
                    &address_shadow,
                );
                address_shadow
            }
            "getelementptr" => {
                let mut args = split(a);
                while args[0]
                    .first()
                    .is_some_and(|s| matches!(s.as_str(), "inbounds" | "nuw" | "nusw"))
                {
                    args[0].remove(0);
                }
                let values = args
                    .iter()
                    .skip(1)
                    .map(|v| typed(v).map(|(_, v)| v))
                    .collect::<Result<Vec<_>>>()?;
                let h = self.merge(values)?;
                let h = self.op(op::ADDRESS, &[h]);
                self.line(raw);
                h
            }
            "select" => {
                let args = split(a);
                let (_, condition) = typed(&args[0])?;
                let (ty, yes) = typed(&args[1])?;
                let (_, no) = typed(&args[2])?;
                reject_vector(&ty, "SSA")?;
                let ch = self.shadow(&condition)?;
                let yh = self.shadow(&yes)?;
                let nh = self.shadow(&no)?;
                self.line(raw);
                let chosen = self.temp();
                self.line(format!(
                    "{chosen} = select i1 {condition}, i64 {yh}, i64 {nh}"
                ));
                self.op(op::SELECT, &[ch, chosen, yh, nh])
            }
            "freeze" => {
                let (ty, v) = typed(a)?;
                reject_vector(&ty, "SSA")?;
                self.line(raw);
                self.shadow(&v)?
            }
            "insertvalue" | "extractvalue" => {
                let args = split(a);
                let (ty, v) = typed(&args[0])?;
                let h = self.shadow(&v)?;
                let start = if ins.op == "insertvalue" { 2 } else { 1 };
                let (member, offset) = self.member(&ty, &args[start..])?;
                if ins.op == "extractvalue" {
                    reject_vector(&member, "SSA")?;
                }
                self.line(raw);
                if ins.op == "insertvalue" {
                    let (value_ty, v) = typed(&args[1])?;
                    reject_vector(&value_ty, &v)?;
                    let mh = self.shadow(&v)?;
                    self.insert(&h, &mh, &member, &offset, &ty)
                } else {
                    self.op(op::EXTRACT, &[h, offset, size(&member)])
                }
            }
            "trunc" | "zext" | "sext" | "fptrunc" | "fpext" | "fptoui" | "fptosi" | "uitofp"
            | "sitofp" | "bitcast" | "addrspacecast" => {
                let to = a
                    .iter()
                    .position(|s| s == "to")
                    .ok_or_else(|| error("cast has no destination type"))?;
                let (ty, v) = typed(&a[..to])?;
                let dest = text(&a[to + 1..]);
                reject_vector(&ty, "SSA")?;
                reject_vector(&dest, "SSA")?;
                let h = self.shadow(&v)?;
                self.line(raw);
                let widths = ty
                    .strip_prefix('i')
                    .and_then(|s| s.parse::<u64>().ok())
                    .zip(dest.strip_prefix('i').and_then(|s| s.parse::<u64>().ok()));
                if ins.op == "trunc"
                    && widths.is_some_and(|(a, b)| b == 1 || (a % 8 == 0 && b % 8 == 0))
                {
                    self.op(op::EXTRACT, &[h, CONSTANT.to_string(), size(&dest)])
                } else if ins.op == "zext"
                    && widths.is_some_and(|(a, b)| a == 1 || (a % 8 == 0 && b % 8 == 0))
                {
                    self.insert(&CONSTANT.to_string(), &h, &ty, "0", &dest)
                } else {
                    self.op(op::MERGE, &[h, CONSTANT.to_string()])
                }
            }
            "add" | "sub" | "mul" | "udiv" | "sdiv" | "urem" | "srem" | "fadd" | "fsub"
            | "fmul" | "fdiv" | "frem" | "fneg" | "and" | "or" | "xor" | "shl" | "lshr"
            | "ashr" | "icmp" | "fcmp" => {
                let mut start = 0;
                while start < a.len() && type_end(a, start).is_err() {
                    start += 1;
                }
                let args = split(&a[start..]);
                let (ty, v) = typed(&args[0])?;
                reject_vector(&ty, "SSA")?;
                let second = args.get(1).map(|v| text(v));
                self.line(raw);
                if matches!(ins.op.as_str(), "shl" | "lshr")
                    && second
                        .as_ref()
                        .and_then(|s| s.parse::<i64>().ok())
                        .is_some_and(|n| n % 8 == 0)
                {
                    let count = second
                        .as_ref()
                        .and_then(|s| s.parse::<i64>().ok())
                        .ok_or_else(|| error("invalid shift"))?
                        / 8;
                    let h = self.shadow(&v)?;
                    self.op(
                        op::SHIFT,
                        &[
                            h,
                            (if ins.op == "shl" { count } else { -count }).to_string(),
                            size(&ty),
                        ],
                    )
                } else if ins.op == "icmp"
                    && self.bools.contains(&v)
                    && ((a[0] == "ne" && second.as_deref() == Some("0"))
                        || (a[0] == "eq" && second.as_deref() == Some("1")))
                {
                    let h = self.shadow(&v)?;
                    self.op(op::EXTRACT, &[h, CONSTANT.to_string(), "1".into()])
                } else {
                    self.merge(std::iter::once(v).chain(second))?
                }
            }
            _ => return Err(error(format!("unlisted instruction {}", ins.op))),
        };
        if let Some(n) = n {
            self.assign(&n, &h);
        }
        Ok(())
    }
    fn pointer_type(&self, ty: &str) -> bool {
        if ty.contains("ptr") {
            return true;
        }
        if let Some(t) = self.types.get(ty) {
            return self.pointer_type(t);
        }
        false
    }
}

// Fixed-width integer loads/stores copy representation bytes unchanged. Keep
// all computation on vector SSA values unsupported, including calls/returns.
fn reject_memory_vector(ty: &str, value: &str) -> Result<()> {
    let t = tokens(ty);
    if let [open, count, x, element, close] = t.as_slice()
        && open == "<"
        && close == ">"
        && x == "x"
        && count.parse::<usize>().is_ok_and(|n| n > 0)
        && element
            .strip_prefix('i')
            .is_some_and(|width| width.parse::<usize>().is_ok_and(|n| n > 0))
    {
        return Ok(());
    }
    // Preserve the existing constant-store path for other element types.
    // Only integer vectors gain support for loaded SSA values.
    reject_vector(ty, value)
}
fn reject_vector(ty: &str, value: &str) -> Result<()> {
    if ty.starts_with('<') && !ty.starts_with("< {") && (value == "SSA" || value.starts_with('%')) {
        return Err(error(format!("non-constant vector value {value}")));
    }
    Ok(())
}
fn boolean_values(instructions: &[Instruction]) -> BTreeSet<String> {
    let mut dependencies = BTreeMap::<String, Vec<String>>::new();
    for i in instructions {
        let Some(name) = &i.name else {
            continue;
        };
        let base = (i.op == "zext" && i.args.first().is_some_and(|s| s == "i1"))
            || (i.op == "call"
                && syntax::call(&i.args).is_ok_and(|c| {
                    matches!(
                        symbols::policy(&c.symbol),
                        Policy::Seed | Policy::QuantumSeed
                    ) && (c.ty == "i1" || c.symbol == "@__quantum__rt__result_get_one")
                }));
        let copied = if i.op == "phi" {
            type_end(&i.args, 0).ok().map(|end| {
                split(&i.args[end..])
                    .iter()
                    .filter_map(|p| split(&p[1..p.len() - 1]).first().map(|v| text(v)))
                    .collect::<Vec<_>>()
            })
        } else if i.op == "select" {
            Some(
                split(&i.args)
                    .iter()
                    .skip(1)
                    .filter_map(|p| typed(p).ok().map(|(_, v)| v))
                    .collect(),
            )
        } else {
            None
        };
        if base {
            dependencies.insert(name.clone(), Vec::new());
        } else if let Some(inputs) = copied {
            dependencies.insert(name.clone(), inputs);
        }
    }
    // Greatest fixed point admits loop-carried copies, but removes every chain
    // that can be entered from a load, extract, arithmetic, or other non-proof.
    let mut known = dependencies.keys().cloned().collect::<BTreeSet<_>>();
    loop {
        let bad = known
            .iter()
            .filter(|name| dependencies[*name].iter().any(|v| !known.contains(v)))
            .cloned()
            .collect::<Vec<_>>();
        if bad.is_empty() {
            return known;
        }
        for name in bad {
            known.remove(&name);
        }
    }
}
/// Instrument a module only at the trace-enabled engine build boundary.
/// # Errors
/// Rejects invalid LLVM, unsupported instructions, and unsafe external pointer escapes.
pub fn instrument(source: &str) -> Result<String> {
    let canonical =
        crate::executor::canonical_qis_ir(source.as_bytes(), ProgramFormat::LlvmIrText)?;
    if canonical.contains("pecos_prov_") || canonical.contains("pecos.prov.") {
        return Err(error("preexisting protected instrumentation"));
    }
    let mut functions = Vec::<Function>::new();
    let mut outside = String::new();
    let mut types = BTreeMap::new();
    let mut globals = Vec::new();
    for line in canonical.lines() {
        let t = tokens(line);
        if t.iter()
            .any(|token| matches!(token.as_str(), "ptrtoint" | "inttoptr"))
        {
            let context = functions
                .last()
                .filter(|f| f.lines.last().is_none_or(|s| s != "}"))
                .map_or_else(
                    || t.first().map_or("module", String::as_str),
                    |f| f.signature.symbol.as_str(),
                );
            return Err(error(format!(
                "function/global {context}, instruction `{}`: input integer/pointer conversion is unsupported",
                line.trim()
            )));
        }
        if let Some(function) = functions
            .last_mut()
            .filter(|f| f.lines.last().is_none_or(|s| s != "}"))
        {
            function.lines.push(line.to_string());
            continue;
        }
        if t.first().is_some_and(|s| s == "define") {
            functions.push(Function {
                header: syntax::relax_effects(line)?,
                signature: syntax::call(&t)?,
                lines: Vec::new(),
            });
        } else {
            if t.get(2).is_some_and(|s| s == "type") {
                types.insert(t[0].clone(), text(&t[3..]));
            }
            if t.first().is_some_and(|s| s.starts_with('@'))
                && let Some(i) = t.iter().position(|s| s == "constant")
            {
                let end = type_end(&t, i + 1)?;
                globals.push((t[0].clone(), text(&t[i + 1..end])));
            }
            if t.first().is_some_and(|s| s == "attributes") {
                outside.push_str(&syntax::relax_effects(line)?);
            } else {
                outside.push_str(line);
            }
            outside.push('\n');
        }
    }
    let internal = functions
        .iter()
        .map(|f| f.signature.symbol.clone())
        .collect::<BTreeSet<_>>();
    let entry = if internal.contains("@qmain") {
        "@qmain"
    } else if internal.contains("@main") {
        "@main"
    } else {
        return Err(error("no defined qmain or main; module is unrunnable"));
    };
    outside.push_str("\ndeclare void @pecos_prov_begin_v1()\ndeclare void @pecos_prov_bind_v1(i64)\ndeclare i64 @pecos_prov_take_seed_v1()\ndeclare i64 @pecos_prov_op_v1(i64, i64, i64, i64, i64)\ndeclare i64 @pecos_prov_memory_v1(i64, ptr, ptr, i64, i64, i64)\n");
    for f in functions {
        let is_entry = f.signature.symbol == entry;
        let mut lines = Vec::<String>::new();
        let mut switch = None::<String>;
        for line in &f.lines[..f.lines.len() - 1] {
            let t = tokens(line);
            if t.is_empty() {
                continue;
            }
            if let Some(pending) = switch.as_mut() {
                pending.push(' ');
                pending.push_str(&text(&t));
                if t.last().is_some_and(|s| s == "]") {
                    lines.push(
                        switch
                            .take()
                            .ok_or_else(|| error("switch assembly invariant"))?,
                    );
                }
            } else if t.first().is_some_and(|s| s == "switch") && t.last().is_some_and(|s| s == "[")
            {
                switch = Some(text(&t));
            } else {
                lines.push(line.clone());
            }
        }
        if switch.is_some() {
            return Err(error("unterminated switch"));
        }
        let instructions = lines
            .iter()
            .filter(|l| tokens(l).get(1).is_none_or(|s| s != ":"))
            .map(|l| Instruction::parse(l))
            .collect::<Vec<_>>();
        let mut w = Writer {
            out: String::new(),
            next: 0,
            shadows: BTreeMap::new(),
            internal: &internal,
            types: &types,
            bools: boolean_values(&instructions),
        };
        for (ty, _) in &f.signature.args {
            reject_vector(ty, "SSA")?;
        }
        for (_, v) in &f.signature.args {
            if v.starts_with('%') {
                let n = format!("%pecos.prov.s{}", w.shadows.len());
                w.shadows.insert(v.clone(), n);
            }
        }
        for i in &instructions {
            if let Some(v) = &i.name {
                let n = format!("%pecos.prov.s{}", w.shadows.len());
                w.shadows.insert(v.clone(), n);
            }
        }
        w.out.push_str(&f.header);
        w.out.push('\n');
        let mut first = true;
        for raw in lines {
            let t = tokens(&raw);
            if t.is_empty() {
                continue;
            }
            if t.get(1).is_some_and(|s| s == ":") {
                w.out.push_str(&raw);
                w.out.push('\n');
                continue;
            }
            if first {
                first = false;
                if is_entry {
                    w.line("call void @pecos_prov_begin_v1()");
                    for (g, ty) in &globals {
                        w.memory(
                            memory::STORE,
                            g,
                            "null",
                            &size(ty),
                            &CONSTANT.to_string(),
                            &CONSTANT.to_string(),
                        );
                    }
                }
                for (index, (_, v)) in f.signature.args.iter().enumerate() {
                    if let Some(n) = w.shadows.get(v).cloned() {
                        w.op_as(&n, op::GET_ARGUMENT, &[index.to_string()]);
                    }
                }
            }
            let ins = Instruction::parse(&raw);
            w.instruction(&ins).map_err(|e| {
                error(format!(
                    "function {}, instruction `{}`: {e}",
                    f.signature.symbol,
                    raw.trim()
                ))
            })?;
        }
        w.out.push_str("}\n");
        outside.push_str(&w.out);
    }
    let verified =
        crate::executor::canonical_qis_ir(outside.as_bytes(), ProgramFormat::LlvmIrText)?;
    Ok(verified
        .lines()
        .filter(|line| !line.starts_with("; ModuleID") && !line.starts_with("source_filename ="))
        .collect::<Vec<_>>()
        .join("\n"))
}

#[cfg(test)]
mod tests;
