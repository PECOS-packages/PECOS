//! Shot-local byte shadows. Handles are immutable vectors, so SSA copies and
//! recursive call frames never alias mutable shadow values.
use pecos_qis_ffi_types::{
    ValueProvenance as V,
    provenance::{CONSTANT, DERIVED, UNKNOWN, VECTOR_BASE, memory, op},
};
use std::collections::BTreeMap;

#[derive(Default)]
struct Frame {
    arguments: Vec<u64>,
    result: u64,
}

#[derive(Default)]
pub(crate) struct Shadows {
    pub memory: BTreeMap<usize, V>,
    vectors: Vec<Vec<V>>,
    frames: Vec<Frame>,
    pub seed: Option<V>,
    pub binding: Option<V>,
    pub instrumented: bool,
    pub control_dependent: bool,
}

impl Shadows {
    fn byte(&self, handle: u64, index: usize) -> Result<V, String> {
        match handle {
            CONSTANT => Ok(V::Constant),
            UNKNOWN => Ok(V::Unknown),
            DERIVED => Ok(V::Derived),
            _ => self
                .vectors
                .get(extent(handle - VECTOR_BASE)?)
                .map(|v| v.get(index).copied().unwrap_or(V::Constant))
                .ok_or_else(|| format!("invalid provenance handle {handle}")),
        }
    }
    fn bytes(&self, handle: u64, size: usize) -> Result<Vec<V>, String> {
        (0..size).map(|i| self.byte(handle, i)).collect()
    }
    fn intern(&mut self, bytes: Vec<V>) -> u64 {
        if bytes.iter().all(|b| *b == V::Constant) {
            return CONSTANT;
        }
        self.vectors.push(bytes);
        self.vectors.len() as u64 - 1 + VECTOR_BASE
    }
    fn class(&self, handle: u64) -> Result<u64, String> {
        if handle < VECTOR_BASE {
            return Ok(handle);
        }
        let bytes = self
            .vectors
            .get(extent(handle - VECTOR_BASE)?)
            .ok_or("invalid shadow vector")?;
        if bytes
            .iter()
            .any(|v| matches!(v, V::Measurement { .. } | V::Derived))
        {
            Ok(DERIVED)
        } else if bytes.contains(&V::Unknown) {
            Ok(UNKNOWN)
        } else {
            Ok(CONSTANT)
        }
    }
    pub fn clear_memory(&mut self, address: usize, size: usize) {
        // Remove only tracked bytes, without allocating a shadow for fresh memory.
        let end = address.saturating_add(size);
        let keys: Vec<_> = self.memory.range(address..end).map(|(&a, _)| a).collect();
        for key in keys {
            self.memory.remove(&key);
        }
    }
    pub fn output(
        &self,
        binding: Option<V>,
        array: Option<usize>,
        count: usize,
    ) -> Result<Option<Vec<V>>, String> {
        if !self.instrumented {
            return Ok(None);
        }
        if let Some(address) = array {
            let dependent =
                binding.ok_or("instrumented bool array output has no range binding")? == V::Derived;
            Ok(Some(
                (0..count)
                    .map(|i| {
                        if dependent {
                            return V::Derived;
                        }
                        self.memory
                            .get(&(address + i))
                            .copied()
                            .unwrap_or(V::Unknown)
                    })
                    .collect(),
            ))
        } else {
            let value = binding.ok_or("instrumented bool output has no provenance binding")?;
            Ok(Some(vec![value]))
        }
    }
    fn take_seed(&mut self) -> Result<u64, String> {
        let value = self
            .seed
            .take()
            .ok_or("seed read did not populate provenance stash")?;
        Ok(self.intern(vec![value]))
    }
    fn begin(&mut self) {
        self.instrumented = true;
    }
    fn bind(&mut self, handle: u64) -> Result<(), String> {
        if self.binding.is_some() {
            return Err("unconsumed output provenance binding".into());
        }
        self.binding = Some(self.byte(handle, 0)?);
        Ok(())
    }
    fn memory(
        &mut self,
        operation: u64,
        pointer: *const u8,
        source: *const u8,
        size: u64,
        handle: u64,
        address_shadow: u64,
    ) -> Result<u64, String> {
        let address = pointer as usize;
        let size = extent(size)?;
        address
            .checked_add(size)
            .ok_or("shadow memory address overflow")?;
        let dependent = self.class(address_shadow)? == DERIVED;
        // A zero-length operation still computes a measurement-dependent range.
        self.control_dependent |= dependent;
        match operation {
            memory::LOAD => {
                let bytes = (0..size)
                    .map(|i| {
                        if dependent {
                            V::Derived
                        } else {
                            self.memory
                                .get(&(address + i))
                                .copied()
                                .unwrap_or(V::Unknown)
                        }
                    })
                    .collect();
                Ok(self.intern(bytes))
            }
            memory::STORE | memory::COPY | memory::FILL => {
                let bytes = if dependent {
                    vec![V::Derived; size]
                } else if operation == memory::COPY {
                    let source = source as usize;
                    source
                        .checked_add(size)
                        .ok_or("shadow source address overflow")?;
                    (0..size)
                        .map(|i| {
                            self.memory
                                .get(&(source + i))
                                .copied()
                                .unwrap_or(V::Unknown)
                        })
                        .collect()
                } else if operation == memory::FILL {
                    vec![self.byte(handle, 0)?; size]
                } else {
                    self.bytes(handle, size)?
                };
                for (i, value) in bytes.into_iter().enumerate() {
                    self.memory.insert(address + i, value);
                }
                Ok(CONSTANT)
            }
            memory::CLEAR => {
                self.clear_memory(address, size);
                // Preserve allocation dependence even through a constant-shadow alias:
                // branches and phi nodes selecting literal bits can reconstruct a
                // stored pointer without retaining its explicit data dependence.
                if dependent {
                    for offset in 0..size {
                        self.memory.insert(address + offset, V::Derived);
                    }
                }
                Ok(CONSTANT)
            }
            _ => Err(format!("unknown shadow memory operation {operation}")),
        }
    }
    fn op(&mut self, op: u64, a: u64, b: u64, c: u64, d: u64) -> Result<u64, String> {
        match op {
            op::MERGE => Ok(self.class(a)?.max(self.class(b)?)),
            op::EXTRACT => {
                let bytes = (0..extent(c)?)
                    .map(|i| self.byte(a, extent(b)? + i))
                    .collect::<Result<_, _>>()?;
                Ok(self.intern(bytes))
            }
            op::SHIFT => {
                let size = extent(c)?;
                let shift = b.cast_signed();
                let bytes = (0..size)
                    .map(|i| {
                        let source = i64::try_from(i)
                            .ok()
                            .and_then(|i| i.checked_sub(shift))
                            .and_then(|source| usize::try_from(source).ok())
                            .filter(|&source| source < size);
                        source.map_or(Ok(V::Constant), |source| self.byte(a, source))
                    })
                    .collect::<Result<_, _>>()?;
                Ok(self.intern(bytes))
            }
            op::SELECT => {
                let class = self.class(a)?;
                self.control_dependent |= class == DERIVED;
                if class == CONSTANT {
                    Ok(b)
                } else {
                    Ok(class.max(self.class(c)?).max(self.class(d)?))
                }
            }
            op::CONTROL => {
                self.control_dependent |= self.class(a)? == DERIVED;
                Ok(CONSTANT)
            }
            op::ADDRESS => {
                let derived = self.class(a)? == DERIVED;
                self.control_dependent |= derived;
                Ok(if derived { DERIVED } else { CONSTANT })
            }
            op::PUSH_FRAME => {
                self.frames.push(Frame::default());
                Ok(CONSTANT)
            }
            op::SET_ARGUMENT => {
                let frame = self.frames.last_mut().ok_or("missing call shadow frame")?;
                let i = extent(a)?;
                frame.arguments.resize(i + 1, CONSTANT);
                frame.arguments[i] = b;
                Ok(CONSTANT)
            }
            op::GET_ARGUMENT => match self.frames.last() {
                Some(frame) => Ok(*frame
                    .arguments
                    .get(extent(a)?)
                    .ok_or("missing parameter shadow")?),
                None => Ok(CONSTANT),
            },
            op::SET_RETURN => {
                if let Some(frame) = self.frames.last_mut() {
                    frame.result = a;
                }
                Ok(CONSTANT)
            }
            op::POP_FRAME => Ok(self
                .frames
                .pop()
                .ok_or("missing completed shadow frame")?
                .result),
            // Insert an explicitly sized member, including constant/unknown members.
            op::INSERT => {
                let mut bytes = self.bytes(a, extent(d)?)?;
                let offset = extent(c)?;
                let values = self
                    .vectors
                    .get(extent(
                        b.checked_sub(VECTOR_BASE).ok_or("unsized member shadow")?,
                    )?)
                    .ok_or("missing sized member shadow")?
                    .clone();
                if offset + values.len() > bytes.len() {
                    return Err("member shadow exceeds aggregate".into());
                }
                bytes[offset..offset + values.len()].copy_from_slice(&values);
                Ok(self.intern(bytes))
            }
            op::SIZED => {
                let bytes = self.bytes(a, extent(b)?)?;
                self.vectors.push(bytes);
                Ok(self.vectors.len() as u64 - 1 + VECTOR_BASE)
            }
            _ => Err(format!("unknown provenance operation {op}")),
        }
    }
}
fn extent(n: u64) -> Result<usize, String> {
    usize::try_from(n)
        .ok()
        .filter(|&n| isize::try_from(n).is_ok())
        .ok_or_else(|| format!("invalid shadow extent {n}"))
}
fn with_shadows<T>(f: impl FnOnce(&mut Shadows) -> Result<T, String>) -> Result<T, String> {
    let ctx = crate::get_execution_context().ok_or("no execution context for provenance")?;
    let mut shadows = unsafe { &*ctx }
        .shadows
        .lock()
        .map_err(|_| "poisoned provenance state")?;
    f(&mut shadows)
}
// Each outer ABI frame owns only scalars at checkpoint/transfer. All guards and
// vectors are confined to the returning helper, as in ffi::dynamic_read.
macro_rules! boundary {
    ($name:ident($($arg:ident: $ty:ty),*) $body:expr) => {
        /// Execute a compiler-inserted provenance operation.
        /// # Safety
        /// Execution must obey the runtime's non-nested checkpoint discipline.
        #[unsafe(no_mangle)]
        pub unsafe extern "C-unwind" fn $name($($arg: $ty),*) -> u64 {
            if unsafe { crate::ffi::cancellation_checkpoint() } { return CONSTANT; }
            match with_shadows($body) {
                Ok(value) => value,
                Err(reason) => { unsafe { crate::ffi::fatal_ffi_input(stringify!($name), reason) }; 0 }
            }
        }
    }
}
boundary!(pecos_prov_op_v1(op: u64, a: u64, b: u64, c: u64, d: u64)
    |shadows| shadows.op(op, a, b, c, d));
boundary!(pecos_prov_take_seed_v1() Shadows::take_seed);

/// Mark this shot as instrumented. Does not reset setup-independent state.
/// # Safety
/// Must run within the runtime's non-nested execution guard.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn pecos_prov_begin_v1() {
    if unsafe { crate::ffi::cancellation_checkpoint() } {
        return;
    }
    if let Err(reason) = with_shadows(|shadows| {
        shadows.begin();
        Ok(())
    }) {
        unsafe { crate::ffi::fatal_ffi_input("pecos_prov_begin_v1", reason) };
    }
}
/// Bind the next bool output: a scalar byte-zero shadow or an array range class.
/// Both scalar and array calls require a binding in instrumented shots.
/// # Safety
/// Must run within the runtime's non-nested execution guard.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn pecos_prov_bind_v1(handle: u64) {
    if unsafe { crate::ffi::cancellation_checkpoint() } {
        return;
    }
    if let Err(reason) = with_shadows(|shadows| shadows.bind(handle)) {
        unsafe { crate::ffi::fatal_ffi_input("pecos_prov_bind_v1", reason) };
    }
}
boundary!(pecos_prov_memory_v1(operation: u64, pointer: *const u8, source: *const u8, size: u64, handle: u64, address_shadow: u64)
    |shadows| shadows.memory(operation, pointer, source, size, handle, address_shadow));

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reset_clears_all_shadow_state() {
        let context = crate::ExecutionContext::new();
        {
            let mut s = context.shadows.lock().unwrap();
            s.instrumented = true;
            s.control_dependent = true;
            s.seed = Some(V::Unknown);
            s.binding = Some(V::Derived);
            s.memory.insert(123, V::Derived);
            s.vectors.push(vec![V::Derived]);
            s.frames.push(Frame::default());
        }
        context.reset_outputs();
        let s = context.shadows.lock().unwrap();
        assert!(!s.instrumented);
        assert!(!s.control_dependent);
        assert!(s.seed.is_none());
        assert!(s.binding.is_none());
        assert_eq!(s.memory, BTreeMap::new());
        assert_eq!(s.vectors.len(), 0);
        assert_eq!(s.frames.len(), 0);
    }

    #[test]
    fn heap_free_clears_the_entire_allocation_shadow() {
        let mut context = crate::ExecutionContext::new();
        unsafe { crate::pecos_register_execution_context(&raw mut context) };
        let pointer = unsafe { crate::ffi::heap_alloc(4) };
        assert!(!pointer.is_null());
        {
            let mut s = context.shadows.lock().unwrap();
            for i in 0..4 {
                s.memory.insert(pointer as usize + i, V::Derived);
            }
        }
        unsafe { crate::ffi::heap_free(pointer) };
        assert_eq!(context.shadows.lock().unwrap().memory, BTreeMap::new());
        unsafe { crate::pecos_register_execution_context(std::ptr::null_mut()) };
    }

    #[test]
    fn missing_binding_is_a_sticky_output_error() {
        let context = crate::ExecutionContext::new();
        context.shadows.lock().unwrap().instrumented = true;
        context.store_named_bool("missing", true);
        let error = context.program_error.lock().unwrap().clone().unwrap();
        assert!(format!("{error:?}").contains("missing"));
        assert!(format!("{error:?}").contains("binding"));
        context.store_named_bool("later", false);
        assert_eq!(*context.program_error.lock().unwrap(), Some(error));
    }
    #[test]
    fn reserved_sources_are_unknown() {
        let context = crate::ExecutionContext::new();
        for source in [
            None,
            Some(pecos_qis_ffi_types::RUNTIME_REQUEST_RESULT_ID),
            Some(usize::MAX),
        ] {
            context.reset_outputs();
            context.stash_result_source(source);
            {
                let mut shadows = context.shadows.lock().unwrap();
                shadows.begin();
                let handle = shadows.take_seed().unwrap();
                shadows.bind(handle).unwrap();
            }
            context.store_named_bool("reserved", true);
            let trace = &context.get_named_result_traces()[0];
            assert_eq!(trace.provenance, Some(vec![V::Unknown]));
            assert_eq!(trace.result_ids, [] as [usize; 0]);
        }
    }

    #[test]
    fn missing_array_address_is_a_program_error() {
        let context = crate::ExecutionContext::new();
        context.shadows.lock().unwrap().begin();
        context.shadows.lock().unwrap().bind(CONSTANT).unwrap();
        context.store_named_result("missing_array", crate::NamedResult::Bool(vec![true]), false);
        let error = context
            .program_error
            .lock()
            .unwrap()
            .clone()
            .unwrap()
            .to_string();
        assert!(error.contains("missing_array"));
        assert!(error.contains("address"));
        assert_eq!(
            context.get_named_result_traces(),
            [] as [crate::NamedResultTrace; 0]
        );
    }

    #[test]
    fn poisoned_trace_lock_is_a_program_error() {
        let context = crate::ExecutionContext::new();
        let _ = std::panic::catch_unwind(|| {
            let _guard = context.named_result_traces.lock().unwrap();
            panic!("poison test trace lock");
        });
        context.store_named_bool("poisoned_trace", true);
        let error = context
            .program_error
            .lock()
            .unwrap()
            .clone()
            .unwrap()
            .to_string();
        assert!(error.contains("poisoned_trace"));
        assert!(error.contains("Failed to trace"));
    }
    #[test]
    fn missing_array_binding_is_a_sticky_output_error() {
        for values in [vec![], vec![true, false]] {
            let context = crate::ExecutionContext::new();
            context.shadows.lock().unwrap().begin();
            context.store_named_array("missing_range", &values);
            let error = context.program_error.lock().unwrap().clone().unwrap();
            assert!(error.to_string().contains("missing_range"));
            assert!(error.to_string().contains("range binding"));
            assert_eq!(
                context.get_named_result_traces(),
                [] as [crate::NamedResultTrace; 0]
            );
            context.store_named_array("later", &values);
            assert_eq!(*context.program_error.lock().unwrap(), Some(error));
        }
    }

    #[test]
    fn output_errors_consume_bindings_and_preserve_first_error() {
        let context = crate::ExecutionContext::new();
        context.shadows.lock().unwrap().begin();
        context.store_named_result("conflict", crate::NamedResult::F64(vec![0.5]), true);
        context.shadows.lock().unwrap().bind(CONSTANT).unwrap();
        context.store_named_bool("conflict", true);
        let error = context.program_error.lock().unwrap().clone().unwrap();
        assert!(error.to_string().contains("element type"));
        assert!(context.shadows.lock().unwrap().binding.is_none());
        for (values, scalar) in [
            (crate::NamedResult::Bool(vec![true]), true),
            (crate::NamedResult::Bool(vec![false, true]), false),
            (crate::NamedResult::I64(vec![1]), true),
            (crate::NamedResult::F64(vec![0.5]), true),
        ] {
            // bind is the fallible body of pecos_prov_bind_v1: an error here
            // would make that export transfer control out of the shot.
            if matches!(&values, crate::NamedResult::Bool(_)) {
                context.shadows.lock().unwrap().bind(CONSTANT).unwrap();
            }
            context.store_named_result("later", values, scalar);
            assert!(context.shadows.lock().unwrap().binding.is_none());
            assert_eq!(*context.program_error.lock().unwrap(), Some(error.clone()));
        }
    }

    #[test]
    fn output_lock_errors_consume_bindings() {
        for lock in [
            "shadows",
            "program_error",
            "named_results",
            "named_result_traces",
        ] {
            let context = crate::ExecutionContext::new();
            context.shadows.lock().unwrap().begin();
            context.shadows.lock().unwrap().bind(CONSTANT).unwrap();
            let _ = std::panic::catch_unwind(|| match lock {
                "shadows" => {
                    let _guard = context.shadows.lock().unwrap();
                    panic!("poison shadows");
                }
                "program_error" => {
                    let _guard = context.program_error.lock().unwrap();
                    panic!("poison error");
                }
                "named_results" => {
                    let _guard = context.named_results.lock().unwrap();
                    panic!("poison results");
                }
                _ => {
                    let _guard = context.named_result_traces.lock().unwrap();
                    panic!("poison traces");
                }
            });
            context.store_named_bool("poisoned", true);
            let shadows = context
                .shadows
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            assert!(shadows.binding.is_none(), "{lock}");
        }
    }

    #[test]
    fn dependent_clear_covers_constant_shadow_alias() {
        let mut shadows = Shadows::default();
        // These pointers are shadow-map keys only; the runtime never dereferences
        // them. A branch/phi reconstruction can retain the address with constant
        // bit shadows, so memory dependence must survive independently.
        let pointer = std::ptr::without_provenance::<u8>(4096);
        shadows
            .memory(
                memory::CLEAR,
                pointer,
                std::ptr::null(),
                2,
                CONSTANT,
                DERIVED,
            )
            .unwrap();
        let loaded = shadows
            .memory(
                memory::LOAD,
                pointer,
                std::ptr::null(),
                2,
                CONSTANT,
                CONSTANT,
            )
            .unwrap();
        assert_eq!(shadows.bytes(loaded, 2).unwrap(), vec![V::Derived; 2]);
    }
    #[test]
    fn non_bool_outputs_preserve_pending_binding() {
        for instrumented in [false, true] {
            for scalar in [false, true] {
                for values in [
                    crate::NamedResult::I64(vec![1]),
                    crate::NamedResult::U64(vec![0]),
                    crate::NamedResult::F64(vec![0.5]),
                ] {
                    let context = crate::ExecutionContext::new();
                    context.shadows.lock().unwrap().instrumented = instrumented;
                    context.shadows.lock().unwrap().bind(CONSTANT).unwrap();
                    context.store_named_result("numeric", values, scalar);
                    assert!(context.program_error.lock().unwrap().is_none());
                    assert_eq!(
                        context.shadows.lock().unwrap().bind(CONSTANT),
                        Err("unconsumed output provenance binding".into())
                    );
                }
            }
        }
    }

    #[test]
    fn non_bool_outputs_ignore_poisoned_shadows() {
        let context = crate::ExecutionContext::new();
        let _ = std::panic::catch_unwind(|| {
            let _guard = context.shadows.lock().unwrap();
            panic!("poison shadows");
        });
        for (name, values) in [
            ("int", crate::NamedResult::I64(vec![1])),
            ("uint", crate::NamedResult::U64(vec![0])),
            ("float", crate::NamedResult::F64(vec![0.5])),
        ] {
            context.store_named_result(name, values.clone(), true);
            assert_eq!(
                context.named_results.lock().unwrap().get(name),
                Some(&values)
            );
            assert!(context.program_error.lock().unwrap().is_none());
        }
    }
}
