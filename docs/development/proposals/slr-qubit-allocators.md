# SLR Qubit Allocator Proposal

## Status

**Design decisions finalized.** Ready for implementation.

---

## Motivation

SLR needs to bridge two different models of qubit management:

1. **QASM model**: Static registers declared upfront (`qreg q[5];`), all qubits exist from program start
2. **Guppy model**: Dynamic allocation with linear types (`q = qubit()`), qubits appear on demand

Both have drawbacks:
- QASM's static registers lack ownership semantics and lifecycle tracking
- Guppy's dynamic allocation feels disconnected from physical hardware constraints ("qubits from nowhere")

### The Allocator Model

Inspired by Zig's allocator pattern and NASA's Power of 10 rules (particularly Rule 3: no dynamic allocation after initialization), we propose a hierarchical qubit allocator model that:
- Grounds allocation in physical resource constraints (total qubit budget declared upfront)
- Provides explicit ownership and natural scoping
- Tracks qubit slot states (unprepared vs prepared)
- Maintains array/register-oriented access patterns (effectively implementing QRegs)
- Enables compiler optimizations through abstracted physical identity

### NASA Power of 10 Alignment

| Power of 10 Rule | Allocator Model |
|------------------|-----------------|
| Rule 3: No dynamic allocation after init | Base allocator declares total capacity in `main` |
| Rule 6: Smallest possible scope | Child allocators scoped to functions/blocks |
| Rule 2: Fixed loop bounds | Allocator capacity is bounded and known |
| Predictability | All resource usage visible from `main` |

---

## Core Concepts

### 1. Base Allocator

Every program declares a base allocator in `main` representing the total physical qubit capacity:

```python
def main():
    base = QAlloc(capacity=100)  # "I have 100 physical qubits"

    # All other allocators derive from this
    data = base.child(7)
    ancilla = base.child(6)

    run_qec(data, ancilla)
```

This is the root of all qubit ownership. It represents the actual hardware constraint. Functions that need qubits receive allocators (or children thereof) as parameters.

### 2. Child Allocators (Hierarchical Ownership)

Any allocator can create child allocators that reserve slots from its capacity:

```python
base = QAlloc(100)

# First level partitioning
data = base.child(7)       # Reserve 7 slots for data
ancilla = base.child(6)    # Reserve 6 slots for ancilla
# base now has 87 available

# Nested partitioning - any allocator can have children
workspace = ancilla.child(2)  # Borrow 2 from ancilla
# ancilla now has 4 available
```

**Key properties:**
- Child allocators exclusively reserve slots from their parent
- Parent cannot use reserved slots until child releases them
- Children can create their own children (unlimited depth)
- Natural scoping: unreturned allocators automatically release to parent

### 3. Slots and Qubit Association

An allocator has N **slots**. Each slot is in one of two states:

```
┌─────────────┐
│ unprepared  │  ← Not ready for gates (initial state, or after measurement)
└──────┬──────┘
       │ prepare()  ← Request qubit, associate with slot, initialize to |0⟩
       ▼
┌─────────────┐
│  prepared   │  ← Ready for gates
└──────┬──────┘
       │ measure()
       ▼
┌─────────────┐
│ unprepared  │  ← Back to unprepared
└─────────────┘
```

Two states, not three. Whether a slot has "never been used" or "was measured" doesn't matter - both are **unprepared** and require `prepare()` before gates.

**Key insight**: `prepare()` means "request a qubit to be associated with this slot and prepared for use." The slot becomes usable. After measurement, the slot returns to unprepared.

**Rules (enforced at compile time):**
- Gates can only be applied to **prepared** slots → compile error if unprepared
- Measurement transitions slots to **unprepared**
- Slots can be prepared individually or in batches at different times

### 4. Slot-Based Access (Not Physical Identity)

Qubits are accessed through their allocator via slot indices:

```python
ancilla = base.child(4)
ancilla.prepare(0, 1)  # Prepare slots 0 and 1

H(ancilla[0])                 # Apply H to slot 0
CNOT(ancilla[0], ancilla[1])  # CNOT between slots 0 and 1
```

**Important**: `ancilla[0]` refers to "slot 0 in the ancilla allocator" - not a fixed physical qubit. After measure + prepare cycles, the physical qubit backing slot 0 may change. The compiler manages the mapping.

```python
ancilla.prepare_all()
# ancilla[0] → physical qubit 42

Measure(ancilla)
ancilla.prepare_all()
# ancilla[0] → might now be physical qubit 37 (compiler's choice)
```

This abstraction enables:
- Qubit recycling and reuse optimizations
- Routing around defective qubits
- Connectivity-aware mapping

The programmer thinks in logical slots; the compiler handles physical mapping.

### 5. Ownership and Natural Scoping

Allocators follow ownership rules similar to Zig:

```python
def syndrome_round(ancilla: QAlloc[6]) -> Bits:
    ancilla.prepare_all()
    # ... syndrome extraction ...
    return Measure(ancilla)
    # ancilla NOT returned → consumed → released to parent

def apply_logical_gate(data: QAlloc[7]) -> QAlloc[7]:
    # ... apply gate ...
    return data  # returned → caller retains ownership
```

**Scoping rules:**
- If an allocator is not returned from a function/block, it is automatically released
- Released resources flow back to the parent allocator
- No explicit `free()` or `release()` needed - scope handles it

---

## API Design

### QAlloc Class

```python
class QAlloc[N]:
    """
    A qubit allocator managing N qubit slots.

    Type parameter N is the capacity (known at compile time for type checking,
    but can also be runtime-determined).
    """

    # --- Creation ---

    def __init__(self, capacity: int):
        """Create a base allocator with given capacity."""
        ...

    def child(self, size: int) -> QAlloc:
        """
        Create a child allocator with `size` slots.

        Reserves `size` qubits from this allocator's available pool.
        Raises if insufficient capacity available.
        """
        ...

    # --- Lifecycle Operations ---

    def prepare(self, *indices: int) -> None:
        """Prepare specific slots (unprepared → prepared)."""
        ...

    def prepare_all(self) -> None:
        """Prepare all slots in this allocator."""
        ...

    # --- Access ---

    def __getitem__(self, index: int) -> QubitRef:
        """
        Access slot `index` for use in gates.

        Returns a QubitRef that can be passed to gate operations.
        The qubit must be in 'prepared' state.
        """
        ...

    # --- Information ---

    @property
    def capacity(self) -> int:
        """Total number of slots in this allocator."""
        ...

    @property
    def available(self) -> int:
        """Number of slots not reserved by children."""
        ...

    def state(self, index: int) -> SlotState:
        """Get the state of a specific slot (unprepared or prepared)."""
        ...
```

### QubitRef (Reference to a Slot)

```python
class QubitRef:
    """
    A reference to a qubit slot in an allocator.

    Used as arguments to gate operations. Not a standalone qubit -
    always tied to its parent allocator.
    """
    allocator: QAlloc
    index: int
```

### SlotState Enum

```python
class SlotState(Enum):
    UNPREPARED = "unprepared"  # Not ready for gates (initial or post-measurement)
    PREPARED = "prepared"       # Ready for gate operations
```

Two states only. Simple.

### Gate Operations

Gates accept `QubitRef` arguments:

```python
# Single qubit gates
H(alloc[0])
X(alloc[1])
Rz(alloc[2], angle=0.5)

# Two qubit gates
CNOT(alloc[0], alloc[1])
CZ(data[0], ancilla[0])  # Can span different allocators

# Measurement (transitions to unprepared)
result = Measure(alloc[0])        # Single qubit
results = Measure(alloc)          # All qubits in allocator
results = Measure(alloc[0:3])     # Slice of allocator
```

---

## Edge Cases and Considerations

### 1. Cross-Allocator Entanglement

**Scenario**: Qubits from different allocators become entangled.

```python
data = base.child(7)
ancilla = base.child(6)

data.prepare_all()
ancilla.prepare_all()

# Entangle across allocators
CNOT(data[0], ancilla[0])
```

**Decision**: Allowed. Allocators manage ownership and slot lifecycle, not entanglement tracking. If `ancilla` is released while entangled with `data`, the slots return to the parent (unprepared). No compiler warning needed - we're not tracking entanglement.

### 2. Partial Measurement

**Scenario**: Only some slots in an allocator are measured.

```python
ancilla = base.child(4)
ancilla.prepare_all()

# Measure only slots 0 and 1
result = Measure(ancilla[0], ancilla[1])

# ancilla[0], ancilla[1] are now unprepared
# ancilla[2], ancilla[3] are still prepared
```

**Resolution**: The allocator tracks per-slot state. This is fully supported.

### 3. Capacity Exhaustion

**Scenario**: Requesting more qubits than available.

```python
base = QAlloc(10)
a = base.child(6)
b = base.child(6)  # ERROR: only 4 available
```

**Resolution**:
- **Compile-time**: If sizes are known statically, this is a compile error
- **Runtime**: Raises an exception (e.g., `AllocationError`)

The type system can help: `QAlloc[N]` carries capacity information.

### 4. Returning Partial Allocators

**Scenario**: Function receives an allocator, creates children, returns some.

```python
def split_and_process(alloc: QAlloc[10]) -> QAlloc[3]:
    a = alloc.child(3)
    b = alloc.child(7)

    process(b)  # b consumed (not returned)

    return a  # Only a returned - COMPILE ERROR
```

**Decision**: Must return parent OR all children. Enforced at compile time.

If you create children from a received allocator, you must either:
1. Return the parent allocator (children are released back to it)
2. Return ALL children (parent is consumed, all resources accounted for)

This ensures clear resource contracts and prevents orphaned slots.

```python
# Valid: return parent
def process_and_return(alloc: QAlloc[10]) -> QAlloc[10]:
    child = alloc.child(5)
    use(child)  # child released back to alloc
    return alloc

# Valid: return all children
def split_evenly(alloc: QAlloc[10]) -> tuple[QAlloc[5], QAlloc[5]]:
    a = alloc.child(5)
    b = alloc.child(5)
    return a, b  # alloc consumed, all slots accounted for
```

### 5. Conditional Allocation

**Scenario**: Allocation inside conditional blocks.

```python
base = QAlloc(10)

if condition:
    extra = base.child(5)
    extra.prepare_all()
    # ... use extra ...
    # extra released at end of block
else:
    # no allocation
```

**Resolution**: This is fine. The allocation is scoped to the if-block. After the block, resources are back in `base`. Both branches end with `base` having the same available capacity.

### 6. Allocation in Loops

**Scenario**: Creating allocators inside loops.

```python
base = QAlloc(10)

for round in range(1000):
    ancilla = base.child(4)
    ancilla.prepare_all()
    syndrome = Measure(ancilla)
    # ancilla released, back to base

    # Next iteration can allocate again
```

**Resolution**: This is a primary use case. Each iteration allocates, uses, and releases. The pool is recycled.

### 7. Escaping References (Zig-inspired)

**Scenario**: Storing a `QubitRef` beyond the allocator's lifetime.

```python
stored_ref = None

def bad_function(alloc: QAlloc[5]):
    global stored_ref
    alloc.prepare_all()
    stored_ref = alloc[0]  # Store reference
    # alloc released at end

bad_function(base.child(5))
H(stored_ref)  # ERROR: dangling reference
```

**Decision**: `QubitRef` is ephemeral, like Zig slices/pointers into allocator memory.

In Zig, when you get memory from an allocator, you get a slice that's valid only while the allocator owns that memory. Similarly, `QubitRef` is valid only while its allocator is alive.

- `alloc[i]` creates a `QubitRef` for immediate use in gate operations
- `QubitRef` should not be stored in data structures or globals
- The compile-time analysis detects when a `QubitRef` escapes its allocator's scope
- If used after allocator release: compile error (if detectable) or runtime error

### 8. Allocator Merging

**Scenario**: Combining two sibling allocators.

```python
base = QAlloc(10)
a = base.child(3)
b = base.child(3)

# Can we merge a and b into a single allocator of 6?
merged = merge(a, b)  # ???
```

**Resolution**: Not supported in initial design. If you need a combined view:
- Release both back to parent
- Allocate a new child of desired size

Merging adds complexity (different qubit states, index remapping). YAGNI for now.

### 9. Slicing Allocators

**Scenario**: Creating a view into part of an allocator.

```python
data = base.child(7)
first_three = data[0:3]  # Is this a new allocator or just refs?
```

**Resolution**: Two options:

**Option A**: Slicing returns a tuple of `QubitRef`
```python
first_three = (data[0], data[1], data[2])  # Just refs
```

**Option B**: Slicing creates a child allocator (view)
```python
first_three = data.slice(0, 3)  # New child allocator
```

**Recommendation**: Option A for simplicity. Slicing is just syntactic sugar for multiple refs. Use explicit `child()` for ownership transfer.

### 10. Classical Data - Do We Need Allocators?

**Decision**: No. Keep `CReg` as-is.

Classical data doesn't have:
- The same physical scarcity constraints
- Lifecycle states (bits don't need "preparation")
- The same ownership complexity

Classical registers remain as simple `CReg` arrays. The allocator pattern is specifically for quantum resources.

### 11. Interaction with Existing SLR Constructs

The allocator model effectively implements `QReg` with additional semantics:

| Current | New |
|---------|-----|
| `QReg("q", 5)` | `q = parent.child(5)` |
| `q[0]` (Qubit) | `q[0]` (QubitRef) |
| `Prep(q[0])` | `q.prepare(0)` |
| `Measure(q[0])` | `Measure(q[0])` (unchanged) |

**Key difference**: Base allocator declared in main, passed to functions:

```python
def main():
    base = QAlloc(100)  # Declare capacity in main

    data = base.child(7)
    ancilla = base.child(6)

    data.prepare_all()
    ancilla.prepare_all()

    run_qec_rounds(data, ancilla)

def run_qec_rounds(data: QAlloc[7], ancilla: QAlloc[6]):
    # Receives allocators as parameters
    # ...
```

This replaces the current pattern where `QReg` is declared inside `Block` classes.

---

## Type System Integration

### Static Capacity Tracking

```python
QAlloc[N]  # Allocator with capacity N

def syndrome_extraction(
    data: QAlloc[7],
    ancilla: QAlloc[6]
) -> tuple[Bits, QAlloc[7]]:
    # Type system knows:
    # - data has 7 slots
    # - ancilla has 6 slots
    # - ancilla is consumed (not in return type)
    # - data is returned (ownership transferred back)
    ...
```

### Lowering to Target Formats

| Target | Allocator Becomes |
|--------|-------------------|
| QASM 2.0 | `qreg` declarations with index mapping |
| QASM 3.0 | `qubit[N]` arrays |
| Guppy | `array[qubit, N]` with linear ownership |
| HUGR | Qubit allocation ops with region tracking |
| QIR | Qubit allocation intrinsics |

---

## Implementation Plan

### Phase 1: Core Data Structures

1. Define `QAlloc` class with:
   - Capacity tracking
   - Child creation and management
   - Per-slot state tracking (unprepared/prepared)

2. Define `QubitRef` class as thin wrapper

3. Define `SlotState` enum (two states: unprepared, prepared)

### Phase 2: Integration with SLR Operations

1. Modify gate classes to accept `QubitRef`
2. Add `prepare()` method/gate
3. Modify `Measure` to transition slots to unprepared
4. Update `Block` to require base allocator declaration

### Phase 3: Code Generation Updates

1. Update `SlrConverter` to handle allocator-based programs
2. Update QASM generator to map allocators to registers
3. Update Guppy generator to map allocators to arrays with linear semantics
4. Update resource planner to understand allocator hierarchy

### Phase 4: Validation and Analysis

1. Add compile-time checks for:
   - Base allocator requirement
   - Capacity overflow detection
   - Lifecycle violations (gate on unprepared slot)
   - Ownership violations (use after release)

2. Add warnings for:
   - Releasing entangled qubits
   - Unused allocator capacity

**Integration Points for State Checking:**

The existing data flow analysis infrastructure can be leveraged:
- `DataFlowAnalyzer` (data_flow.py:37-354) - already tracks consumption and replacement
- `IRAnalyzer` (ir_analyzer.py:114) - integration point after `_integrate_data_flow()`
- `IRBuilder` (ir_builder.py:3078, 4137) - gate conversion with validation
- `ScopeManager` (scope_manager.py:27-145) - runtime state tracking

A new `QubitStateValidator` module would:
1. Initialize from DataFlowAnalysis with all elements as "unprepared"
2. Mark elements as "prepared" when `Prep`/`Init`/`Reset` operations occur
3. Mark elements as "unprepared" when `Measure` operations occur
4. Validate that gates only operate on "prepared" elements

### Phase 5: Documentation and Migration

1. Document the allocator model
2. Provide migration guide from `QReg` to allocators
3. Update examples

---

## Design Decisions Summary

| Question | Decision |
|----------|----------|
| Slot states | Two: `unprepared` and `prepared` (no separate "dirty") |
| Prepare syntax | Method: `alloc.prepare(0, 1, 2)` or `alloc.prepare_all()` |
| Gate on unprepared slot | Compile-time error |
| Base allocator location | Declared in `main`, passed to functions |
| Returning allocators | Must return parent OR all children |
| Cross-allocator entanglement | Allowed, not tracked |
| Classical registers | Keep `CReg` as-is |
| QubitRef lifetime | Ephemeral, Zig-inspired (no escaping scope) |
| Philosophy | NASA Power of 10 inspired (no dynamic alloc after init) |

---

## Implementation Decisions

| Question | Decision |
|----------|----------|
| Migration | Dual support: `QReg` as alias/wrapper for `QAlloc` |
| Naming | `QAlloc` (follows `QReg`/`CReg` convention) |
| Prepare return | `void` - keeps refs ephemeral, allocator is source of truth |

---

## Complete Example: QEC Round

```python
def main():
    # Declare physical resource budget
    base = QAlloc(capacity=17)

    # Partition into logical groupings
    data = base.child(9)      # 9 data qubits for surface code
    ancilla = base.child(8)   # 8 ancilla for syndrome extraction

    # Initialize data qubits
    data.prepare_all()
    encode_logical_zero(data)

    # Run QEC rounds
    for round in range(100):
        syndrome = extract_syndrome(data, ancilla)
        if needs_correction(syndrome):
            apply_correction(data, syndrome)

    # Final readout
    result = decode_and_measure(data)
    return result


def extract_syndrome(
    data: QAlloc[9],
    ancilla: QAlloc[8]
) -> Bits:
    """
    Extract syndrome without consuming data.
    Ancilla is consumed (not returned).
    """
    ancilla.prepare_all()

    # X stabilizers
    for i in range(4):
        H(ancilla[i])
        CNOT(ancilla[i], data[stabilizer_x_targets(i)])
        H(ancilla[i])

    # Z stabilizers
    for i in range(4):
        CNOT(data[stabilizer_z_targets(i)], ancilla[4 + i])

    # Measure all ancilla - slots become unprepared
    syndrome = Measure(ancilla)

    # ancilla not returned → released back to caller's scope
    # data returned implicitly via not being consumed
    return syndrome


def encode_logical_zero(data: QAlloc[9]) -> QAlloc[9]:
    """
    Encode logical |0⟩. Returns the data allocator.
    """
    # data already prepared by caller
    H(data[0])
    CNOT(data[0], data[1])
    # ... encoding circuit ...
    return data  # Ownership returned to caller


def decode_and_measure(data: QAlloc[9]) -> Bits:
    """
    Decode and measure. Consumes the data allocator.
    """
    # ... decoding circuit ...
    result = Measure(data)
    # data not returned → consumed
    return result
```

### What This Demonstrates

1. **Base in main**: `QAlloc(17)` declares total capacity upfront
2. **Child allocators as "registers"**: `data` and `ancilla` are like QRegs
3. **Slot preparation**: `prepare_all()` makes slots usable
4. **Natural consumption**: `ancilla` not returned from `extract_syndrome` → released
5. **Explicit return for ownership**: `encode_logical_zero` returns `data` to maintain ownership
6. **Loop reuse**: Each round re-prepares ancilla, reusing the same slots

---

## Summary

The qubit allocator model provides:

- **Physical grounding**: Resources come from a declared capacity, not thin air
- **Hierarchical ownership**: Clear parent-child relationships with natural scoping
- **Lifecycle tracking**: Two states (unprepared/prepared) enforced at compile time
- **Slot abstraction**: Logical indices, not physical identity - enabling optimizations
- **Clean semantics**: Ownership rules similar to Rust/Zig for resource safety

This bridges QASM's register model and Guppy's linear types while feeling more connected to physical hardware constraints.
