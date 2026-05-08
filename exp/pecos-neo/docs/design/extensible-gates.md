# Extensible Gate System Design

## Problem Statement

The current gate system uses a closed `GateType` enum in pecos-core. This means:
- Adding new gates requires modifying pecos-core
- Users cannot define custom gates for domain-specific operations
- Simulators/noise models can't declare what gates they support
- No adaptor mechanism for decomposing unsupported gates

## Goals

1. **User-defined gates**: Users can specify gate name, qubit count, angle count, whether it returns results
2. **Mandatory validation**: All compatibility checked before execution - can't skip it
3. **Zero-cost for core gates**: No performance regression for standard gate set
4. **Adaptor pattern**: Decompose unsupported gates into supported primitives
5. **DOD/ECS-friendly**: Data-oriented, cache-friendly, minimal indirection

## Validation Philosophy

**Compile-time where natural, build-time for everything else - but mandatory.**

### What's Checked at Compile Time (Rust type system)

These checks happen automatically via the type system without viral generics:

| Check | How | Example |
|-------|-----|---------|
| Gate construction | API enforces arity | `Gate::cx()` requires qubit pairs |
| Simulator capabilities | Trait bounds | `SparseStab: CliffordGateable` |
| Primitive composition | Type safety | `prob(0.01, depolarize())` type-checks |
| Registry type safety | Generics | `world.register::<GateSpec>(...)` |

### What's Checked at Build Time (Mandatory)

These checks happen when constructing the simulation - **cannot be skipped**:

| Check | When | Error |
|-------|------|-------|
| Circuit gates supported | `build(circuit)` | `UnsupportedGates([...])` |
| Noise covers all gates | `build(circuit)` | `UnhandledByNoise([...])` |
| User gates registered | `build(circuit)` | `UnknownGateId(...)` |
| Adaptor coverage | `build(circuit)` | `CannotAdapt([...])` |

### No Way to Skip Validation

The API is designed so you **cannot** get a runnable simulation without passing validation.

The outer builder (Tool) already has the circuit/program, so it passes it to sub-builders internally:

```rust
impl ToolBuilder {
    // Circuit set on outer builder
    pub fn with_program(mut self, program: Program) -> Self {
        self.program = Some(program);
        self
    }

    // build() validates internally - no way to skip
    pub fn build(self) -> Result<Tool, ValidationError> {
        let program = self.program.ok_or(ValidationError::NoProgram)?;

        // Outer builder passes circuit to sub-builders for validation
        let gate_registry = self.gate_registry;
        let sim_context = self.simulator_builder.validate_and_build(&program, &gate_registry)?;
        let noise_context = self.noise_builder.validate_and_build(&program, &gate_registry)?;
        let adaptor_context = self.adaptor_builder.validate_and_build(&program, &gate_registry)?;

        Ok(Tool::new_validated(sim_context, noise_context, adaptor_context, program))
    }
}

impl SimulatorBuilder {
    /// Called by outer builder with circuit - validates coverage
    fn validate_and_build(
        self,
        program: &Program,
        registry: &GateRegistry,
    ) -> Result<SimContext, ValidationError> {
        let required = program.gate_set();
        let unsupported = &required.bits & !&self.supported_bits;
        if unsupported.any() {
            return Err(ValidationError::UnsupportedGates(...));
        }
        Ok(SimContext { ... })
    }
}

impl Tool {
    // Private - only created via validated build
    fn new_validated(...) -> Self { ... }

    // Run is infallible - already validated
    pub fn run(&mut self) -> Outcomes {
        self.run_inner()  // No checks needed
    }
}
```

### Why Build-Time, Not Runtime?

Validation happens **once at build**, not on every gate:

```rust
// BAD: Check every gate during execution (slow)
for gate in circuit.gates() {
    if !self.supports(gate) {  // Called millions of times
        return Err(...);
    }
    self.execute(gate);
}

// GOOD: Check once at build (fast), execute without checks
let sim = builder.build(&circuit)?;  // Validated once
for _ in 0..1_000_000 {
    sim.run();  // No validation overhead
}
```

## Data Structure Principles

**No HashMap/BTreeMap in the hot path.** All gate lookups must be O(1) with minimal indirection:

| Operation | Data Structure | Complexity |
|-----------|----------------|------------|
| GateId → GateSpec | `Vec<GateSpec>` indexed by ID | O(1), single pointer chase |
| GateId → Handler | `Vec<Option<fn>>` indexed by ID | O(1), single pointer chase |
| "Supports gate?" | `BitVec` indexed by ID | O(1), bit test |
| "Can adapt gate?" | `BitVec` indexed by ID | O(1), bit test |
| Name → GateId | Only at parse time, not hot path | - |

For the rare case of name lookup (parsing circuits from text), we can use a sorted array with binary search or a perfect hash at initialization time - but this is never on the execution hot path.

## Strings vs IDs: Phase Separation

Users work with **strings** (readable, ergonomic). Execution uses **IDs** (fast, compact). The separation is clear:

| Phase | What Happens | Data Used |
|-------|--------------|-----------|
| **Authoring** | User writes circuit | Strings: `"H"`, `"MyRotation"` |
| **Parsing/Loading** | Resolve names, validate signatures | Strings → IDs (once) |
| **Build/Validation** | Check coverage, compatibility | IDs + BitVec |
| **Execution (Hot Path)** | Run gates | IDs only, no strings |

### Authoring: Strings for Ergonomics

Users write circuits with readable gate names:

```rust
// User-facing API uses strings
let circuit = CircuitBuilder::new()
    .gate("PZ", &[0, 1])
    .gate("H", &[0])
    .gate("MyRotation", &[0, 1], &[0.5, 0.25, 0.1])  // Custom gate by name
    .gate("MZ", &[0, 1])
    .build();
```

Or in a file format:

```yaml
gates:
  - { gate: "H", qubits: [0] }
  - { gate: "MyRotation", qubits: [0, 1], angles: [0.5, 0.25, 0.1] }
```

### Loading: String Resolution (Once)

When building the circuit, strings are resolved to IDs **once**:

```rust
impl CircuitBuilder {
    pub fn gate(mut self, name: &str, qubits: &[usize], angles: &[f64]) -> Self {
        // String lookup happens HERE - once per gate definition
        let gate_id = self.registry.lookup(name)  // O(log n) binary search
            .ok_or_else(|| BuildError::UnknownGate(name.to_string()))?;

        // Validate signature matches spec
        let spec = self.registry.get(gate_id).unwrap();
        if qubits.len() != spec.quantum_arity as usize {
            return Err(BuildError::WrongQubitCount {
                gate: name.to_string(),
                expected: spec.quantum_arity,
                got: qubits.len(),
            });
        }
        if angles.len() != spec.angle_arity as usize {
            return Err(BuildError::WrongAngleCount {
                gate: name.to_string(),
                expected: spec.angle_arity,
                got: angles.len(),
            });
        }

        // Store ONLY the ID - string is gone
        self.gates.push(Gate {
            gate_id,  // GateId(u16), not String
            qubits: qubits.iter().map(|&q| QubitId(q as u32)).collect(),
            angles: angles.iter().map(|&a| Angle64::from_turns(a)).collect(),
        });

        self
    }
}
```

After this point, **no strings exist** in the circuit data structure.

### Validation: IDs and BitVecs

Validation uses IDs and bitwise operations - no strings:

```rust
impl ToolBuilder {
    fn validate(&self, program: &Program) -> Result<(), ValidationError> {
        // All operations use IDs, not strings
        let required: BitVec = program.gate_set().bits;
        let supported: BitVec = self.simulator.supported_bits();

        // Bitwise ANDNOT - no string comparison
        let unsupported = &required & !&supported;

        if unsupported.any() {
            // Only convert to strings for error messages
            let names: Vec<_> = unsupported.iter_ones()
                .map(|id| self.registry.get(GateId(id as u16)).unwrap().name)
                .collect();
            return Err(ValidationError::UnsupportedGates(names));
        }

        Ok(())
    }
}
```

### Execution: IDs Only (Hot Path)

The hot path uses only integer IDs - no strings, no lookups:

```rust
impl Tool {
    pub fn run(&mut self) -> Outcomes {
        for gate in &self.program.gates {
            // gate.gate_id is GateId(u16) - fits in register
            // Direct array index - O(1), no hashing, no string ops
            let handler = self.dispatch_table[gate.gate_id.0 as usize];

            if let Some(f) = handler {
                // Native execution
                f(&mut self.simulator, &gate.qubits, &gate.angles);
            } else {
                // Adaptor path - still uses IDs
                self.run_adapted(gate);
            }
        }

        self.collect_outcomes()
    }

    fn run_adapted(&mut self, gate: &Gate) {
        // Adaptor lookup by ID, not string
        let adaptor = &self.adaptors[gate.gate_id.0 as usize];
        let decomposed = adaptor.adapt(gate);

        for sub_gate in decomposed {
            // Recursive, still all IDs
            let handler = self.dispatch_table[sub_gate.gate_id.0 as usize];
            handler.unwrap()(&mut self.simulator, &sub_gate.qubits, &sub_gate.angles);
        }
    }
}
```

### Summary: Where Strings Live

```
┌─────────────────────────────────────────────────────────────────┐
│  User Code / Files                                               │
│    "H", "CX", "MyRotation"  ←── Strings here (ergonomic)        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼ parse/build (once)
┌─────────────────────────────────────────────────────────────────┐
│  GateRegistry                                                    │
│    name_index: [("CX", 50), ("H", 10), ("MyRotation", 256)]     │
│                 ↑ strings for lookup only                        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼ resolved to IDs
┌─────────────────────────────────────────────────────────────────┐
│  Program / Circuit                                               │
│    gates: [Gate { id: 10, ... }, Gate { id: 256, ... }]         │
│            ↑ IDs only, no strings                                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼ execution
┌─────────────────────────────────────────────────────────────────┐
│  Hot Path                                                        │
│    dispatch_table[gate.id] → handler                             │
│    ↑ array index, O(1), no strings                               │
└─────────────────────────────────────────────────────────────────┘
```

Strings are **never** in the hot path. They exist only for:
1. User-facing APIs (authoring)
2. Error messages (debugging)
3. Serialization (file formats)

## Unified Registration Pattern

Gates, noise channels, and adaptors all follow the same pattern - they're **resources** registered with a World/Tool context. This ECS-style approach keeps things consistent and fast.

### The Pattern

All registerable items share a common structure:

| Item | Spec Type | ID Type | Registry | Hot-Path Lookup |
|------|-----------|---------|----------|-----------------|
| Gate | `GateSpec` | `GateId(u16)` | `GateRegistry` | `Vec` indexed by ID |
| Noise Channel | `ChannelSpec` | `ChannelId(u16)` | `ChannelRegistry` | `Vec` indexed by ID |
| Adaptor | `AdaptorSpec` | `AdaptorId(u16)` | `AdaptorRegistry` | `Vec` indexed by ID |

### Trait-Based Registration

```rust
/// Marker trait for items that can be registered
pub trait Registerable: Sized + 'static {
    /// Compact ID type (u16 for cache-friendly indexing)
    type Id: Copy + Into<usize>;

    /// Registry that stores these items
    type Registry: Registry<Item = Self>;
}

/// Base trait for all registries - uniform interface
pub trait Registry: Resource + Default {
    type Item: Registerable;
    type Id: Copy + Into<usize>;

    /// Register item, returns ID - O(1) amortized
    fn register(&mut self, item: Self::Item) -> Self::Id;

    /// Get by ID - O(1) direct indexing
    fn get(&self, id: Self::Id) -> Option<&Self::Item>;

    /// Check if ID is valid - O(1)
    fn contains(&self, id: Self::Id) -> bool;
}
```

### Unified World API

```rust
impl World {
    /// Register any registerable item - type-safe, uniform API
    pub fn register<T: Registerable>(&mut self, item: T) -> T::Id {
        self.resource_mut::<T::Registry>().register(item)
    }

    /// Get registered item by ID
    pub fn get<T: Registerable>(&self, id: T::Id) -> Option<&T> {
        self.resource::<T::Registry>().get(id)
    }
}

// Usage - all follow the same pattern
let gate_id = world.register(GateSpec { name: "MyGate", ... });
let channel_id = world.register(ChannelSpec { name: "MyNoise", ... });
let adaptor_id = world.register(AdaptorSpec { ... });
```

### Registry Implementation (Same for All)

All registries use the same high-performance structure:

```rust
/// Generic registry implementation - works for gates, channels, adaptors
pub struct VecRegistry<T, Id> {
    /// Core items (ID 0-255) - pre-allocated, no resize
    core: [Option<T>; 256],

    /// User items (ID 256+) - grows dynamically
    user: Vec<T>,

    /// Support bitset for fast "contains" queries
    present: BitVec,

    _marker: PhantomData<Id>,
}

impl<T, Id: From<u16> + Into<usize>> VecRegistry<T, Id> {
    /// O(1) lookup by ID
    #[inline]
    pub fn get(&self, id: Id) -> Option<&T> {
        let idx: usize = id.into();
        if idx < 256 {
            self.core[idx].as_ref()
        } else {
            self.user.get(idx - 256)
        }
    }

    /// O(1) amortized registration
    pub fn register(&mut self, item: T) -> Id {
        let id = 256 + self.user.len();
        self.user.push(item);

        if id >= self.present.len() {
            self.present.resize(id + 64, false);
        }
        self.present.set(id, true);

        Id::from(id as u16)
    }

    /// O(1) presence check
    #[inline]
    pub fn contains(&self, id: Id) -> bool {
        self.present.get(id.into()).map(|b| *b).unwrap_or(false)
    }
}
```

### Why This Matters for Performance

1. **Uniform memory layout**: All registries use the same Vec+BitVec structure, predictable cache behavior

2. **ID-based dispatch**: All lookups are `registry[id]` - no hashing, no string comparison

3. **Compile-time specialization**: Generic over `T` and `Id`, monomorphized for each type

4. **Core items pre-allocated**: Gates 0-255, channels 0-255, etc. never cause allocation in hot path

5. **BitVec for set operations**: "Does this noise model handle all gates in this circuit?" becomes bitwise AND

### Matching Gates to Handlers

At execution time, matching a gate to its handlers is pure index arithmetic:

```rust
impl ExecutionContext {
    /// Find all applicable noise channels for a gate - O(channels)
    /// But each check is O(1) bit test
    #[inline]
    pub fn channels_for_gate(&self, gate_id: GateId) -> impl Iterator<Item = &CompositeChannel> {
        self.channel_registry.iter()
            .filter(|ch| ch.filter.matches_gate(gate_id, &self.gate_registry))
    }

    /// Find adaptor for unsupported gate - O(1) bit test
    #[inline]
    pub fn adaptor_for_gate(&self, gate_id: GateId) -> Option<&dyn GateAdaptor> {
        // Each adaptor has a BitVec of gates it can handle
        self.adaptor_registry.iter()
            .find(|a| a.can_adapt_bits.get(gate_id.0 as usize).unwrap_or(false))
    }
}
```

### Bevy-Style System Integration

Following Bevy patterns, systems can query registries:

```rust
/// System that applies noise - queries registries as resources
fn apply_noise_system(
    gate_registry: Res<GateRegistry>,
    channel_registry: Res<ChannelRegistry>,
    mut simulators: Query<&mut Simulator>,
    circuits: Query<&Circuit>,
) {
    for circuit in circuits.iter() {
        for gate in circuit.gates() {
            // All lookups are O(1) indexed
            let spec = gate_registry.get(gate.gate_id);

            for channel in channel_registry.iter() {
                if channel.matches(gate, &gate_registry) {
                    // Apply noise
                }
            }
        }
    }
}
```

## Design Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        GateRegistry                              │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  GateId(0) → GateSpec { name: "I", qubits: 1, ... }      │   │
│  │  GateId(1) → GateSpec { name: "X", qubits: 1, ... }      │   │
│  │  GateId(10) → GateSpec { name: "H", qubits: 1, ... }     │   │
│  │  ...                                                      │   │
│  │  GateId(256) → GateSpec { name: "MyGate", ... }  (user)  │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                         Simulator                                │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  GateSupport: I, X, Y, Z, H, CX, CZ, RZ, MZ, ...           │ │
│  │  handlers[GateId] → Option<ExecuteFn>                      │ │
│  └────────────────────────────────────────────────────────────┘ │
│                              │                                   │
│              unsupported?    │                                   │
│                              ▼                                   │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  GateAdaptor: decomposes MyGate → [H, CX, RZ, CX, H]       │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## Core Types

### GateId

A compact identifier for gate types. Core gates use reserved IDs (0-255), user gates start at 256.

```rust
/// Compact gate type identifier
///
/// Core gates have reserved IDs in range 0-255.
/// User-defined gates are assigned IDs >= 256.
#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
#[repr(transparent)]
pub struct GateId(u16);

impl GateId {
    /// Check if this is a core (built-in) gate
    #[inline]
    pub const fn is_core(self) -> bool {
        self.0 < 256
    }

    /// Check if this is a user-defined gate
    #[inline]
    pub const fn is_user_defined(self) -> bool {
        self.0 >= 256
    }
}

// Core gate constants (matches current GateType enum values)
pub mod gates {
    use super::GateId;

    pub const I: GateId = GateId(0);
    pub const X: GateId = GateId(1);
    pub const Z: GateId = GateId(2);
    pub const Y: GateId = GateId(3);
    pub const H: GateId = GateId(10);
    pub const RZ: GateId = GateId(32);
    pub const CX: GateId = GateId(50);
    pub const CZ: GateId = GateId(52);
    pub const RZZ: GateId = GateId(82);
    pub const MEASURE: GateId = GateId(104);
    pub const PREP: GateId = GateId(134);
    // ... etc, matching GateType enum
}
```

### GateSpec

Describes the properties of a gate type.

```rust
/// Gate specification - describes what a gate IS
#[derive(Clone, Debug)]
pub struct GateSpec {
    /// Human-readable name
    pub name: &'static str,

    /// Number of qubits this gate operates on
    pub quantum_arity: u8,

    /// Number of angle parameters (Angle64)
    pub angle_arity: u8,

    /// Number of other parameters (f64, e.g., duration for Idle)
    pub param_arity: u8,

    /// Whether this gate produces measurement outcomes
    pub returns_result: bool,

    /// Optional: semantic category for noise model matching
    pub category: GateCategory,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum GateCategory {
    /// Single-qubit unitary (X, Y, Z, H, RZ, etc.)
    SingleQubitUnitary,
    /// Two-qubit unitary (CX, CZ, RZZ, etc.)
    TwoQubitUnitary,
    /// Multi-qubit unitary (CCX, etc.)
    MultiQubitUnitary,
    /// State preparation
    Preparation,
    /// Measurement
    Measurement,
    /// Idle/wait operation
    Idle,
    /// User-defined category
    Custom(u8),
}
```

### GateRegistry

Central registry for gate definitions.

```rust
/// Registry of gate specifications
///
/// NOT a global - this is a Resource attached to a World or Tool.
/// All hot-path lookups are O(1) via direct indexing.
///
/// In ECS terms: `world.resource::<GateRegistry>()`
pub struct GateRegistry {
    /// Core gate specs (indices 0-255, pre-populated)
    /// Direct array indexing - no indirection
    core_specs: [Option<GateSpec>; 256],

    /// User gate specs (indices >= 256)
    /// Vec indexing: specs[id - 256]
    user_specs: Vec<GateSpec>,

    /// Name to ID lookup - ONLY for parsing, never hot path
    /// Sorted by name for binary search. Rebuilt on registration.
    name_index: Vec<(&'static str, GateId)>,
}

impl GateRegistry {
    /// Create a new registry with core gates pre-populated
    pub fn new() -> Self {
        let mut registry = Self {
            core_specs: [const { None }; 256],
            user_specs: Vec::new(),
            name_index: Vec::new(),
        };
        registry.init_core_gates();
        registry
    }

    /// Register a user-defined gate, returns its ID
    pub fn register(&mut self, spec: GateSpec) -> GateId {
        let id = GateId(256 + self.user_specs.len() as u16);
        let name = spec.name;
        self.user_specs.push(spec);

        // Maintain sorted order for binary search
        let pos = self.name_index.binary_search_by_key(&name, |(n, _)| *n)
            .unwrap_or_else(|p| p);
        self.name_index.insert(pos, (name, id));

        id
    }

    /// Look up spec by ID - O(1) direct indexing
    #[inline]
    pub fn get(&self, id: GateId) -> Option<&GateSpec> {
        if id.is_core() {
            self.core_specs[id.0 as usize].as_ref()
        } else {
            self.user_specs.get((id.0 - 256) as usize)
        }
    }

    /// Look up ID by name - O(log n) binary search
    /// Only used at parse time, never in simulation hot path
    pub fn lookup(&self, name: &str) -> Option<GateId> {
        self.name_index.binary_search_by_key(&name, |(n, _)| *n)
            .ok()
            .map(|i| self.name_index[i].1)
    }
}

// GateRegistry is a Resource in the ECS sense
impl Resource for GateRegistry {}
```

### Scoping to World/Tool

The registry is **not global**. It's attached to the simulation context:

```rust
// In World (ECS style)
impl World {
    pub fn gate_registry(&self) -> &GateRegistry {
        self.resource::<GateRegistry>()
    }

    pub fn gate_registry_mut(&mut self) -> &mut GateRegistry {
        self.resource_mut::<GateRegistry>()
    }
}

// In Tool/SimNeoBuilder
impl SimNeoBuilder {
    /// Register a custom gate for this simulation
    pub fn register_gate(&mut self, spec: GateSpec) -> GateId {
        self.gate_registry.register(spec)
    }
}

// Usage
let mut builder = SimNeoBuilder::new();
let my_gate = builder.register_gate(GateSpec {
    name: "MyRotation",
    quantum_arity: 2,
    angle_arity: 3,
    ..Default::default()
});
```

This avoids global state and allows different simulations to have different gate sets.

### Updated Gate Struct

Replace `GateType` with `GateId`:

```rust
/// Quantum gate command
#[derive(Debug, Clone, PartialEq)]
pub struct Gate {
    /// Gate type identifier (core or user-defined)
    pub gate_id: GateId,

    /// Rotation angles for parameterized gates
    pub angles: GateAngles,

    /// Other parameters (e.g., duration)
    pub params: GateParams,

    /// Target qubits
    pub qubits: GateQubits,
}

impl Gate {
    /// Get the gate specification - requires registry reference
    #[inline]
    pub fn spec<'a>(&self, registry: &'a GateRegistry) -> &'a GateSpec {
        registry.get(self.gate_id)
            .expect("Gate ID not registered")
    }

    /// Convenience methods - for core gates, these are O(1) without registry lookup
    /// because core gate metadata is compile-time known
    #[inline]
    pub fn quantum_arity(&self, registry: &GateRegistry) -> usize {
        // Fast path for core gates: could use const lookup table
        if self.gate_id.is_core() {
            CORE_QUANTUM_ARITY[self.gate_id.0 as usize] as usize
        } else {
            self.spec(registry).quantum_arity as usize
        }
    }

    #[inline]
    pub fn returns_result(&self, registry: &GateRegistry) -> bool {
        if self.gate_id.is_core() {
            CORE_RETURNS_RESULT[self.gate_id.0 as usize]
        } else {
            self.spec(registry).returns_result
        }
    }
}

// Compile-time lookup tables for core gates - zero overhead
static CORE_QUANTUM_ARITY: [u8; 256] = {
    let mut table = [0u8; 256];
    table[gates::X.0 as usize] = 1;
    table[gates::H.0 as usize] = 1;
    table[gates::CX.0 as usize] = 2;
    // ... filled at compile time
    table
};

static CORE_RETURNS_RESULT: [bool; 256] = {
    let mut table = [false; 256];
    table[gates::MZ.0 as usize] = true;
    // ...
    table
};
```

## Simulator Support

### GateSupport Trait

Simulators declare what gates they support:

```rust
/// Trait for declaring gate support
pub trait GateSupport {
    /// Check if this simulator natively supports a gate
    fn supports(&self, gate_id: GateId) -> bool;

    /// List all natively supported gates
    fn supported_gates(&self) -> &[GateId];

    /// Get the gate category support (for noise models)
    /// Requires registry reference - not on hot path
    fn supports_category(&self, category: GateCategory, registry: &GateRegistry) -> bool {
        self.supported_gates().iter()
            .any(|&id| registry.get(id)
                .map(|s| s.category == category)
                .unwrap_or(false))
    }
}
```

### GateExecutor Trait

Generic execution interface:

```rust
/// Trait for executing gates
pub trait GateExecutor: GateSupport {
    /// Execute a gate, returning measurement outcomes if applicable
    fn execute(
        &mut self,
        gate_id: GateId,
        qubits: &[QubitId],
        angles: &[Angle64],
        params: &[f64],
    ) -> Option<SmallVec<[bool; 4]>>;

    /// Execute a Gate struct (convenience method)
    fn execute_gate(&mut self, gate: &Gate) -> Option<SmallVec<[bool; 4]>> {
        self.execute(gate.gate_id, &gate.qubits, &gate.angles, &gate.params)
    }
}
```

### Fast Dispatch Implementation

For hot-path performance, use a lookup table:

```rust
/// Function pointer type for gate execution
type GateFn<S> = fn(&mut S, &[QubitId], &[Angle64], &[f64]) -> Option<SmallVec<[bool; 4]>>;

/// Fast gate dispatcher using function pointer table
pub struct GateDispatcher<S> {
    /// Handlers indexed by GateId (None = not supported)
    handlers: Vec<Option<GateFn<S>>>,
}

impl<S: CliffordGateable> GateDispatcher<S> {
    pub fn for_clifford() -> Self {
        let mut handlers = vec![None; 256];

        // Map core gates to trait methods
        handlers[gates::X.0 as usize] = Some(|s, q, _, _| { s.x(q); None });
        handlers[gates::H.0 as usize] = Some(|s, q, _, _| { s.h(q); None });
        handlers[gates::CX.0 as usize] = Some(|s, q, _, _| { s.cx(q); None });
        handlers[gates::MZ.0 as usize] = Some(|s, q, _, _| {
            Some(s.mz(q).iter().map(|r| r.outcome).collect())
        });
        // ... etc

        Self { handlers }
    }

    /// Fast dispatch - O(1) lookup
    #[inline]
    pub fn dispatch(
        &self,
        sim: &mut S,
        gate: &Gate,
    ) -> Option<SmallVec<[bool; 4]>> {
        if let Some(handler) = self.handlers.get(gate.gate_id.0 as usize).copied().flatten() {
            handler(sim, &gate.qubits, &gate.angles, &gate.params)
        } else {
            None // Not supported, caller should use adaptor
        }
    }
}
```

## Gate Adaptors

### GateAdaptor Trait

```rust
/// Trait for decomposing gates into other gates
pub trait GateAdaptor {
    /// Check if this adaptor can decompose the given gate
    fn can_adapt(&self, gate_id: GateId) -> bool;

    /// Decompose a gate into a sequence of other gates
    fn adapt(
        &self,
        gate_id: GateId,
        qubits: &[QubitId],
        angles: &[Angle64],
        params: &[f64],
    ) -> Vec<Gate>;
}
```

### Standard Decompositions

```rust
/// Built-in adaptor with standard decompositions
pub struct StandardAdaptor {
    /// Bitset: can_adapt[gate_id] = true if we can decompose it
    can_adapt_bits: BitVec,
}

impl StandardAdaptor {
    /// Create adaptor targeting Clifford+RZ gate set
    pub fn stab_vec() -> Self {
        let mut bits = BitVec::repeat(false, 256);
        // Mark gates we can decompose
        bits.set(gates::RX.0 as usize, true);
        bits.set(gates::RY.0 as usize, true);
        bits.set(gates::RZZ.0 as usize, true);
        bits.set(gates::SWAP.0 as usize, true);
        bits.set(gates::T.0 as usize, true);
        Self { can_adapt_bits: bits }
    }
}

impl GateAdaptor for StandardAdaptor {
    /// O(1) bit test
    #[inline]
    fn can_adapt(&self, gate_id: GateId) -> bool {
        self.can_adapt_bits.get(gate_id.0 as usize).map(|b| *b).unwrap_or(false)
    }

    fn adapt(
        &self,
        gate_id: GateId,
        qubits: &[QubitId],
        angles: &[Angle64],
        _params: &[f64],
    ) -> Vec<Gate> {
        match gate_id {
            gates::T => {
                // T = RZ(1/8 turn)
                vec![Gate::rz(Angle64::from_turns(0.125), qubits)]
            }
            gates::RX => {
                // RX(θ) = H RZ(θ) H
                let theta = angles[0];
                vec![
                    Gate::h(qubits),
                    Gate::rz(theta, qubits),
                    Gate::h(qubits),
                ]
            }
            gates::RZZ => {
                // RZZ(θ) = CX(0,1) RZ(θ,1) CX(0,1)
                let theta = angles[0];
                let q0 = qubits[0];
                let q1 = qubits[1];
                vec![
                    Gate::cx(&[(q0, q1)]),
                    Gate::rz(theta, &[q1]),
                    Gate::cx(&[(q0, q1)]),
                ]
            }
            _ => panic!("No decomposition for {:?}", gate_id),
        }
    }
}
```

## Gate Canonicalization and Aliasing

Parameterized gates with specific angle values can be **canonicalized** to their fixed-gate equivalents. This happens at **build time**, not in the hot path.

### Why Canonicalize?

| Benefit | Example |
|---------|---------|
| **Faster execution** | `SZ` has no angle to process, `RZ(θ)` does |
| **Specialized implementations** | Simulator may have optimized `T` gate |
| **Numerical stability** | Exact gate vs floating-point angle |
| **Noise model matching** | Noise rules for `T` vs `RZ` may differ |

### Exact Comparison with Angle64

`Angle64` is a **fixed-point** representation where the full u64 range equals one turn. This means standard angles are **exactly representable**:

```rust
// These are EXACT - no floating point error
Angle64::QUARTER_TURN  // π/2 radians, exactly
Angle64::HALF_TURN     // π radians, exactly
Angle64::HALF_TURN / 4 // π/4 radians (T gate), exactly

// Equality is exact - no tolerance needed!
angle == Angle64::QUARTER_TURN  // ✓ Works perfectly
```

This eliminates floating-point tolerance issues entirely for standard gate angles.

### Standard Canonicalization Rules

```rust
/// Known angle → fixed gate mappings
/// Uses EXACT Angle64 comparison - no floating-point tolerance needed
pub struct GateCanonicalizer {
    /// Maps (parameterized_gate_id, angle) → fixed_gate_id
    /// Uses sorted vec indexed by (gate_id, angle.fraction) for fast lookup
    rules: Vec<CanonicalForm>,
}

pub struct CanonicalForm {
    pub from_gate: GateId,
    pub angle: Angle64,        // Exact fixed-point angle
    pub to_gate: GateId,
    // No tolerance field needed - Angle64 comparison is exact!
}

impl GateCanonicalizer {
    /// Standard rules for core gates
    /// All angles are EXACT Angle64 values - no approximation
    pub fn standard() -> Self {
        use Angle64 as A;

        Self {
            rules: vec![
                // RZ rules - exact angles
                CanonicalForm { from_gate: gates::RZ, angle: A::ZERO,                        to_gate: gates::I },
                CanonicalForm { from_gate: gates::RZ, angle: A::HALF_TURN / 4,               to_gate: gates::T },      // π/4
                CanonicalForm { from_gate: gates::RZ, angle: A::ZERO - A::HALF_TURN / 4,     to_gate: gates::Tdg },    // -π/4
                CanonicalForm { from_gate: gates::RZ, angle: A::QUARTER_TURN,                to_gate: gates::SZ },     // π/2
                CanonicalForm { from_gate: gates::RZ, angle: A::ZERO - A::QUARTER_TURN,      to_gate: gates::SZdg },   // -π/2
                CanonicalForm { from_gate: gates::RZ, angle: A::HALF_TURN,                   to_gate: gates::Z },      // π

                // RX rules
                CanonicalForm { from_gate: gates::RX, angle: A::ZERO,                        to_gate: gates::I },
                CanonicalForm { from_gate: gates::RX, angle: A::QUARTER_TURN,                to_gate: gates::SX },
                CanonicalForm { from_gate: gates::RX, angle: A::ZERO - A::QUARTER_TURN,      to_gate: gates::SXdg },
                CanonicalForm { from_gate: gates::RX, angle: A::HALF_TURN,                   to_gate: gates::X },

                // RY rules
                CanonicalForm { from_gate: gates::RY, angle: A::ZERO,                        to_gate: gates::I },
                CanonicalForm { from_gate: gates::RY, angle: A::QUARTER_TURN,                to_gate: gates::SY },
                CanonicalForm { from_gate: gates::RY, angle: A::ZERO - A::QUARTER_TURN,      to_gate: gates::SYdg },
                CanonicalForm { from_gate: gates::RY, angle: A::HALF_TURN,                   to_gate: gates::Y },

                // RZZ rules
                CanonicalForm { from_gate: gates::RZZ, angle: A::QUARTER_TURN,               to_gate: gates::SZZ },
                CanonicalForm { from_gate: gates::RZZ, angle: A::ZERO - A::QUARTER_TURN,     to_gate: gates::SZZdg },
            ],
        }
    }

    /// Try to canonicalize a gate - O(log n) lookup
    /// Uses EXACT Angle64 equality - no floating-point tolerance
    pub fn canonicalize(&self, gate_id: GateId, angles: &[Angle64]) -> Option<GateId> {
        if angles.len() != 1 {
            return None;  // Only single-angle gates for now
        }

        let angle = angles[0];

        // Exact equality check - Angle64 is fixed-point
        for canon in &self.rules {
            if canon.from_gate == gate_id && canon.angle == angle {
                return Some(canon.to_gate);
            }
        }

        None
    }
}
```

### Integration with Circuit Building

Canonicalization happens during circuit construction:

```rust
impl CircuitBuilder {
    pub fn gate(mut self, name: &str, qubits: &[usize], angles: &[f64]) -> Self {
        let gate_id = self.registry.lookup(name)?;
        let angle_vals: SmallVec<[Angle64; 3]> = angles.iter()
            .map(|&a| Angle64::from_turns(a))
            .collect();

        // Try to canonicalize
        let (final_id, final_angles) = if let Some(canonical) =
            self.canonicalizer.canonicalize(gate_id, &angle_vals)
        {
            // RZ(0.25) → SZ (no angles)
            (canonical, SmallVec::new())
        } else {
            // Keep as-is
            (gate_id, angle_vals)
        };

        self.gates.push(Gate {
            gate_id: final_id,
            angles: final_angles,
            qubits: qubits.iter().map(|&q| QubitId(q as u32)).collect(),
        });

        self
    }
}
```

### User-Defined Canonicalization Rules

Users can register their own rules for custom gates:

```rust
// Register a custom gate
let my_rot = builder.register_gate(GateSpec {
    name: "MyRotation",
    quantum_arity: 2,
    angle_arity: 1,
    ..Default::default()
});

// Register a fixed version of it
let my_rot_half = builder.register_gate(GateSpec {
    name: "MyRotationHalf",  // MyRotation at θ=0.5
    quantum_arity: 2,
    angle_arity: 0,  // No angles - it's fixed
    ..Default::default()
});

// Register canonicalization: MyRotation(0.5) → MyRotationHalf
builder.register_canonicalization(CanonicalForm {
    from_gate: my_rot,
    angle_turns: 0.5,
    tolerance: 1e-10,
    to_gate: my_rot_half,
});

// Now when building circuits:
circuit.gate("MyRotation", &[0, 1], &[0.5])  // Becomes MyRotationHalf
```

### Reverse Canonicalization (Expansion)

Sometimes you want the opposite - expand fixed gates to parameterized form:

```rust
impl GateCanonicalizer {
    /// Expand fixed gate to parameterized form
    pub fn expand(&self, gate_id: GateId) -> Option<(GateId, Angle64)> {
        // Reverse lookup
        for canon in &self.rules {
            if canon.to_gate == gate_id {
                return Some((canon.from_gate, Angle64::from_turns(canon.angle_turns)));
            }
        }
        None
    }
}

// Usage: T → RZ(π/4)
let (rz, angle) = canonicalizer.expand(gates::T).unwrap();
assert_eq!(rz, gates::RZ);
assert_eq!(angle.to_turns(), 0.125);
```

This is useful when:
- Simulator only supports parameterized gates (e.g., only RZ, not T)
- Exporting to formats that don't have fixed gates
- Optimization passes that work on parameterized form

### Canonicalization in the Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│  User writes: gate("RZ", [0], [0.25])                            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼ lookup name
┌─────────────────────────────────────────────────────────────────┐
│  Resolved: gate_id = gates::RZ, angles = [0.25 turns]           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼ canonicalize
┌─────────────────────────────────────────────────────────────────┐
│  Canonical: gate_id = gates::SZ, angles = []                    │
│  (RZ at π/2 is SZ, no angle needed)                             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼ store in circuit
┌─────────────────────────────────────────────────────────────────┐
│  Circuit: Gate { id: SZ, qubits: [0], angles: [] }              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼ execution (hot path)
┌─────────────────────────────────────────────────────────────────┐
│  dispatch_table[SZ] → optimized SZ handler (no angle parsing)   │
└─────────────────────────────────────────────────────────────────┘
```

### Practical Considerations

1. **No tolerance needed for Angle64**: Internal comparisons are exact
   ```rust
   // Angle64 is fixed-point - equality is exact
   angle == Angle64::QUARTER_TURN  // ✓ Always works

   // Tolerance only needed when PARSING from floating-point input
   fn parse_angle(radians: f64) -> Angle64 {
       // Convert float to fixed-point - this is where precision loss can occur
       Angle64::from_radians(radians)
   }
   ```

2. **Angle normalization is automatic**: Angle64 uses wrapping arithmetic
   ```rust
   // Angle64 automatically wraps - no manual normalization needed
   let a = Angle64::HALF_TURN + Angle64::HALF_TURN;  // = ZERO (wraps)
   let b = Angle64::ZERO - Angle64::QUARTER_TURN;    // = THREE_QUARTERS_TURN
   ```

3. **Parsing from floats**: When reading angles from files, convert early to Angle64
   ```rust
   // Parse floating-point, convert to fixed-point immediately
   let angle_f64: f64 = parse_from_file()?;
   let angle = Angle64::from_turns(angle_f64);  // Now it's exact

   // All subsequent comparisons are exact
   if angle == Angle64::QUARTER_TURN { ... }
   ```

4. **Multi-angle gates**: For gates with multiple angles (e.g., U gate), check all
   ```rust
   // U(0, 0, θ) = RZ(θ)
   // U(θ, -π/2, π/2) = RX(θ)
   if angles[0] == A::ZERO && angles[1] == A::ZERO {
       // U(0, 0, θ) → RZ(θ)
       return Some((gates::RZ, vec![angles[2]]));
   }
   ```

5. **Canonicalization is optional**: Can be disabled if exact gate preservation is needed
   ```rust
   let builder = CircuitBuilder::new()
       .with_canonicalization(false);  // Keep RZ(π/2) as RZ, don't convert to SZ
   ```

6. **Common exact angles**: These are all exactly representable in Angle64
   ```rust
   Angle64::ZERO                    // 0
   Angle64::HALF_TURN / 8           // π/8 (T/2)
   Angle64::HALF_TURN / 4           // π/4 (T gate)
   Angle64::QUARTER_TURN            // π/2 (S gate)
   Angle64::HALF_TURN               // π (Z gate)
   Angle64::THREE_QUARTERS_TURN     // 3π/2
   // Any fraction with power-of-2 denominator is exact
   ```

## Build-Time Validation

### Circuit Analysis

```rust
/// Set of gates used by a circuit - stored as bitset for O(1) operations
pub struct CircuitGateSet {
    /// Bitset indexed by GateId - bit is set if gate is used
    bits: BitVec,
    /// Max GateId seen (for iteration)
    max_id: u16,
}

impl CircuitGateSet {
    pub fn new() -> Self {
        Self {
            bits: BitVec::repeat(false, 256), // Start with core gates capacity
            max_id: 0,
        }
    }

    /// O(1) insertion
    #[inline]
    pub fn insert(&mut self, id: GateId) {
        let idx = id.0 as usize;
        if idx >= self.bits.len() {
            self.bits.resize(idx + 64, false); // Grow in chunks
        }
        self.bits.set(idx, true);
        self.max_id = self.max_id.max(id.0);
    }

    /// O(1) membership test
    #[inline]
    pub fn contains(&self, id: GateId) -> bool {
        self.bits.get(id.0 as usize).map(|b| *b).unwrap_or(false)
    }

    /// Iterate over all gates in the set
    pub fn iter(&self) -> impl Iterator<Item = GateId> + '_ {
        self.bits.iter_ones().map(|i| GateId(i as u16))
    }
}

impl Circuit {
    /// Analyze what gates this circuit uses - O(n) where n = gates in circuit
    pub fn gate_set(&self) -> CircuitGateSet {
        let mut set = CircuitGateSet::new();
        for gate in &self.gates {
            set.insert(gate.gate_id);
        }
        set
    }
}
```

### Builder Validation (BitVec Operations)

With all registries using BitVec, validation becomes fast bitwise operations:

```rust
impl ToolBuilder {
    /// Build a validated Tool - ONLY way to get runnable Tool
    ///
    /// The outer builder already has the program/circuit. It passes
    /// the circuit to sub-builders (simulator, noise, adaptor) for validation.
    ///
    /// Validates that:
    /// 1. Simulator (+ adaptor) supports all gates in circuit
    /// 2. Noise model handles all gates in circuit
    /// 3. All gate IDs in circuit are registered
    ///
    /// Returns Err if any validation fails - cannot proceed without fixing.
    pub fn build(self) -> Result<Tool, ValidationError> {
        let program = self.program.as_ref().ok_or(ValidationError::NoProgram)?;
        self.validate(program)?;
        Ok(Tool::new_validated(self))
    }

    // NOTE: No way to get Tool without validation!
    // Program is already on builder, validation happens internally.

    /// Internal validation - fast bitwise operations
    fn validate(&self, circuit: &Circuit) -> Result<(), ValidationError> {
        let required = circuit.gate_set();  // BitVec of gates in circuit

        // 1. Check all gate IDs are registered
        for gate_id in required.iter() {
            if !self.gate_registry.contains(gate_id) {
                return Err(ValidationError::UnknownGateId(gate_id));
            }
        }

        // 2. Check simulator + adaptor coverage (bitwise OR, then ANDNOT)
        let mut covered = self.simulator.supported_gates_bits().clone();
        if let Some(adaptor) = &self.adaptor {
            covered |= adaptor.can_adapt_bits();  // O(n/64) SIMD-friendly
        }

        let unsupported = &required.bits & !&covered;  // Bitwise ANDNOT
        if unsupported.any() {
            return Err(ValidationError::UnsupportedGates(
                unsupported.iter_ones().map(|i| GateId(i as u16)).collect()
            ));
        }

        // 3. Check noise model coverage (by gate ID or category)
        let noise_covered = self.compute_noise_coverage(&required);
        let unhandled = &required.bits & !&noise_covered;
        if unhandled.any() {
            return Err(ValidationError::UnhandledByNoise(
                unhandled.iter_ones().map(|i| GateId(i as u16)).collect()
            ));
        }

        Ok(())
    }

    /// Compute which gates the noise model covers (including category matching)
    fn compute_noise_coverage(&self, required: &CircuitGateSet) -> BitVec {
        let mut covered = self.noise_model.explicit_gate_bits().clone();

        // For gates not explicitly covered, check category coverage
        for gate_id in required.iter() {
            if covered.get(gate_id.0 as usize).map(|b| *b).unwrap_or(false) {
                continue;  // Already covered explicitly
            }

            // Check if category is covered
            if let Some(spec) = self.gate_registry.get(gate_id) {
                if self.noise_model.handles_category(spec.category) {
                    if gate_id.0 as usize >= covered.len() {
                        covered.resize(gate_id.0 as usize + 1, false);
                    }
                    covered.set(gate_id.0 as usize, true);
                }
            }
        }

        covered
    }
}

/// Validated Tool - can only be created via ToolBuilder::build()
pub struct Tool {
    // Private fields - no public constructor
    simulator: Box<dyn GateExecutor>,
    noise_model: ComposableNoiseModel,
    adaptor: Option<Box<dyn GateAdaptor>>,
    gate_registry: GateRegistry,
    program: Program,
    // ...
}

impl Tool {
    /// Private constructor - only callable from validated build
    fn new_validated(builder: ToolBuilder) -> Self {
        Self {
            simulator: builder.simulator,
            noise_model: builder.noise_model,
            adaptor: builder.adaptor,
            gate_registry: builder.gate_registry,
            program: builder.program.unwrap(),  // Validated to exist
        }
    }

    /// Run the simulation - no validation overhead, already checked at build
    pub fn run(&mut self) -> Outcomes {
        // Fast path - no checks needed, everything validated
        self.run_inner()
    }
}
```

## Noise Model Support

Noise models also need to declare gate support:

```rust
impl ComposableNoiseModel {
    /// Check if this noise model has rules for a gate
    /// Takes registry for category lookup (user gates need spec lookup)
    pub fn handles_gate(&self, gate_id: GateId, registry: &GateRegistry) -> bool {
        // Check by exact ID - O(1) bit test
        if self.gate_support_bits.get(gate_id.0 as usize).map(|b| *b).unwrap_or(false) {
            return true;
        }

        // Fall back to category matching
        if let Some(spec) = registry.get(gate_id) {
            self.category_support_bits.get(spec.category as usize)
                .map(|b| *b).unwrap_or(false)
        } else {
            false
        }
    }

    /// Validate noise model covers circuit gates
    pub fn validate_coverage(
        &self,
        circuit: &Circuit,
        registry: &GateRegistry,
    ) -> Result<(), Vec<GateId>> {
        let unhandled: Vec<GateId> = circuit.gate_set()
            .iter()
            .filter(|id| !self.handles_gate(*id, registry))
            .collect();

        if unhandled.is_empty() {
            Ok(())
        } else {
            Err(unhandled)
        }
    }
}
```

### Arity-Based Noise Filtering

Noise channels can match gates by their properties (arity, category) rather than specific IDs. This means user-defined gates automatically get appropriate noise:

```rust
/// Filter for which gates a noise channel applies to
pub enum CompositeEventFilter {
    // Match specific gate
    GateId(GateId),

    // Match by quantum arity (from GateSpec)
    SingleQubitGate,      // quantum_arity == 1
    TwoQubitGate,         // quantum_arity == 2
    MultiQubitGate,       // quantum_arity >= 3

    // Match by classical arity
    ParameterizedGate,    // angle_arity > 0

    // Match by category (from GateSpec)
    Category(GateCategory),

    // Compound filters
    And(Box<CompositeEventFilter>, Box<CompositeEventFilter>),
    Or(Box<CompositeEventFilter>, Box<CompositeEventFilter>),
    Not(Box<CompositeEventFilter>),
}

impl CompositeEventFilter {
    /// Check if gate matches - O(1) for core gates via const tables
    pub fn matches(&self, gate: &Gate, registry: &GateRegistry) -> bool {
        match self {
            Self::GateId(id) => gate.gate_id == *id,

            Self::SingleQubitGate => {
                if gate.gate_id.is_core() {
                    // Compile-time const table - no registry lookup
                    CORE_QUANTUM_ARITY[gate.gate_id.0 as usize] == 1
                } else {
                    registry.get(gate.gate_id)
                        .map(|s| s.quantum_arity == 1)
                        .unwrap_or(false)
                }
            }

            Self::TwoQubitGate => {
                if gate.gate_id.is_core() {
                    CORE_QUANTUM_ARITY[gate.gate_id.0 as usize] == 2
                } else {
                    registry.get(gate.gate_id)
                        .map(|s| s.quantum_arity == 2)
                        .unwrap_or(false)
                }
            }

            Self::ParameterizedGate => {
                if gate.gate_id.is_core() {
                    CORE_ANGLE_ARITY[gate.gate_id.0 as usize] > 0
                } else {
                    registry.get(gate.gate_id)
                        .map(|s| s.angle_arity > 0)
                        .unwrap_or(false)
                }
            }

            Self::Category(cat) => {
                if gate.gate_id.is_core() {
                    CORE_CATEGORY[gate.gate_id.0 as usize] == *cat
                } else {
                    registry.get(gate.gate_id)
                        .map(|s| s.category == *cat)
                        .unwrap_or(false)
                }
            }

            Self::And(a, b) => a.matches(gate, registry) && b.matches(gate, registry),
            Self::Or(a, b) => a.matches(gate, registry) || b.matches(gate, registry),
            Self::Not(f) => !f.matches(gate, registry),
        }
    }
}

// Compile-time const tables for core gates - zero-cost lookup
static CORE_QUANTUM_ARITY: [u8; 256] = const {
    let mut table = [0u8; 256];
    table[gates::X.0 as usize] = 1;
    table[gates::H.0 as usize] = 1;
    table[gates::RZ.0 as usize] = 1;
    table[gates::CX.0 as usize] = 2;
    table[gates::CZ.0 as usize] = 2;
    table[gates::RZZ.0 as usize] = 2;
    table[gates::CCX.0 as usize] = 3;
    // ... all core gates
    table
};

static CORE_ANGLE_ARITY: [u8; 256] = const {
    let mut table = [0u8; 256];
    table[gates::RX.0 as usize] = 1;
    table[gates::RY.0 as usize] = 1;
    table[gates::RZ.0 as usize] = 1;
    table[gates::RZZ.0 as usize] = 1;
    table[gates::R1XY.0 as usize] = 2;
    table[gates::U.0 as usize] = 3;
    // ...
    table
};

static CORE_CATEGORY: [GateCategory; 256] = const {
    let mut table = [GateCategory::SingleQubitUnitary; 256];
    table[gates::CX.0 as usize] = GateCategory::TwoQubitUnitary;
    table[gates::MZ.0 as usize] = GateCategory::Measurement;
    table[gates::PZ.0 as usize] = GateCategory::Preparation;
    // ...
    table
};
```

### Usage: Noise Automatically Applies to User Gates

```rust
// Define noise that applies to ALL 2-qubit gates
let two_qubit_noise = CompositeChannel::new("tq_depol", prob(0.01, two_qubit_pauli()))
    .with_filter(CompositeEventFilter::TwoQubitGate);

// Define noise for parameterized gates with angle-dependent probability
let angle_noise = CompositeChannel::new("angle_dep",
    prob_fn(|gate| gate.angles.first().map(|a| a.abs()).unwrap_or(0.0) * 0.01,
        depolarize()
    ))
    .with_filter(CompositeEventFilter::ParameterizedGate);

// User registers a custom 2-qubit parameterized gate
let my_gate = builder.register_gate(GateSpec {
    name: "MyRot",
    quantum_arity: 2,
    angle_arity: 1,
    category: GateCategory::TwoQubitUnitary,
    ..Default::default()
});

// The noise model automatically applies:
// - two_qubit_noise (because quantum_arity == 2)
// - angle_noise (because angle_arity > 0)
// No changes to noise model required!
```

## User-Defined Gates Example

```rust
// Create builder - owns its own GateRegistry
let mut builder = SimNeoBuilder::new();

// User defines and registers a custom gate (on the builder, not global)
let my_gate_id = builder.register_gate(GateSpec {
    name: "MyRotation",
    quantum_arity: 2,
    angle_arity: 3,
    param_arity: 0,
    returns_result: false,
    category: GateCategory::TwoQubitUnitary,
});

// Define decomposition adaptor
struct MyGateAdaptor {
    my_gate_id: GateId,
    can_adapt_bits: BitVec,
}

impl MyGateAdaptor {
    fn new(my_gate_id: GateId) -> Self {
        let mut bits = BitVec::repeat(false, my_gate_id.0 as usize + 1);
        bits.set(my_gate_id.0 as usize, true);
        Self { my_gate_id, can_adapt_bits: bits }
    }
}

impl GateAdaptor for MyGateAdaptor {
    #[inline]
    fn can_adapt(&self, id: GateId) -> bool {
        self.can_adapt_bits.get(id.0 as usize).map(|b| *b).unwrap_or(false)
    }

    fn adapt(&self, _id: GateId, qubits: &[QubitId], angles: &[Angle64], _: &[f64]) -> Vec<Gate> {
        // Decompose into native gates
        let (a, b, c) = (angles[0], angles[1], angles[2]);
        vec![
            Gate::rz(a, &[qubits[0]]),
            Gate::cx(&[(qubits[0], qubits[1])]),
            Gate::rz(b, &[qubits[1]]),
            Gate::cx(&[(qubits[0], qubits[1])]),
            Gate::rz(c, &[qubits[0]]),
        ]
    }
}

// Build a circuit using the custom gate
let circuit = CircuitBuilder::new()
    .pz(&[0, 1])
    .gate(my_gate_id, &[0, 1], &[angle1, angle2, angle3])  // Use custom gate
    .mz(&[0, 1])
    .build();

// Outer builder has circuit - passes to sub-builders internally for validation
let mut tool = ToolBuilder::new()
    .with_program(circuit)                           // Circuit set here
    .with_simulator(SparseStab::new(10))
    .with_adaptor(MyGateAdaptor::new(my_gate_id))
    .with_gate_registry(builder.gate_registry)       // Registry with custom gate
    .build()?;  // Validates internally - can't skip
    //    ^^^^ Returns Result, validation happens inside

// If we forgot the adaptor, build() would return:
// Err(ValidationError::UnsupportedGates([my_gate_id]))

// Run is infallible - already validated at build time
let outcomes = tool.run();
```

## Performance Considerations

### Hot Path Optimization

1. **GateId is u16**: Fits in register, fast comparison
2. **Lookup table dispatch**: O(1) function pointer lookup for core gates
3. **Inline specs for core gates**: No indirection for common case
4. **SmallVec for qubits/angles**: Stack allocation for typical sizes

### BitVec for Support Sets

All "does X support gate Y?" queries use `BitVec` indexed by `GateId`:

```rust
/// Compact support declaration - one bit per possible gate
pub struct GateSupportSet {
    /// bits[gate_id] = true if supported
    bits: BitVec,
}

impl GateSupportSet {
    /// O(1) membership test - single bit lookup
    #[inline]
    pub fn supports(&self, id: GateId) -> bool {
        self.bits.get(id.0 as usize).map(|b| *b).unwrap_or(false)
    }

    /// O(1) insertion
    #[inline]
    pub fn add(&mut self, id: GateId) {
        let idx = id.0 as usize;
        if idx >= self.bits.len() {
            self.bits.resize(idx + 64, false);
        }
        self.bits.set(idx, true);
    }

    /// Set union - for combining support sets
    pub fn union(&mut self, other: &GateSupportSet) {
        // Vectorized OR - very fast
        self.bits |= &other.bits;
    }
}
```

Memory: 256 core gates = 32 bytes. With user gates, grows dynamically.

### Comparison: BitVec vs HashSet

| Operation | BitVec | HashSet |
|-----------|--------|---------|
| Insert | O(1) bit set | O(1) amortized, hash + probe |
| Contains | O(1) bit test | O(1) amortized, hash + probe |
| Memory (256 gates) | 32 bytes | ~2KB (with overhead) |
| Cache behavior | Excellent (contiguous) | Poor (scattered) |
| Iteration | O(n) with `iter_ones()` | O(n) |

For gate sets (typically <500 gates), BitVec is faster and more cache-friendly.

### Benchmarking Strategy

```rust
#[bench]
fn bench_gate_dispatch_enum(b: &mut Bencher) {
    // Current: match on GateType enum
}

#[bench]
fn bench_gate_dispatch_id(b: &mut Bencher) {
    // New: lookup table by GateId
}

#[bench]
fn bench_gate_dispatch_with_adaptor(b: &mut Bencher) {
    // With adaptor fallback
}
```

Expected: ID-based dispatch should be within 5% of enum dispatch for core gates.

## Migration Path

### Phase 1: Add GateId alongside GateType

- Add `GateId` type and `GateRegistry`
- `GateType` gains `fn to_gate_id(&self) -> GateId`
- `Gate` struct adds optional `gate_id: Option<GateId>` field
- Simulators continue using trait methods

### Phase 2: Implement GateExecutor

- Add `GateExecutor` trait
- Implement for existing simulators (wrapping trait methods)
- Add `GateDispatcher` for fast lookup

### Phase 3: Add Adaptor Support

- Implement `GateAdaptor` trait
- Add `StandardAdaptor` with common decompositions
- Integrate into `SimNeoBuilder`

### Phase 4: Enable User Gates

- Document user gate registration API
- Add validation to builders
- Update noise model matching

### Phase 5: Deprecate GateType (Optional)

- If adoption is successful, consider deprecating `GateType` enum
- Or keep both: enum for compile-time-known gates, ID for runtime flexibility

## Circuit Headers for Custom Gates

Circuits/programs that use custom gates should be **self-describing** - they include a header declaring the custom gates they use. This enables:

1. **Serialization**: Save/load circuits without losing gate definitions
2. **Portability**: Send circuits to other systems that don't have the gates pre-registered
3. **Tooling**: Static analyzers can understand the circuit without execution context
4. **Translation**: When translating circuits, output includes new gate definitions

### Circuit Format

```rust
/// A circuit/program with its custom gate declarations
pub struct Program {
    /// Header: custom gate definitions used by this program
    /// Core gates (ID < 256) don't need to be declared
    pub custom_gates: Vec<GateSpec>,

    /// Body: the actual gate sequence
    pub gates: Vec<Gate>,
}

/// When loading a program, register its custom gates first
impl Program {
    pub fn load(bytes: &[u8]) -> Result<Self, ParseError> {
        let header = parse_header(bytes)?;
        let body = parse_body(bytes)?;

        Ok(Self {
            custom_gates: header.custom_gates,
            gates: body.gates,
        })
    }

    /// Register this program's custom gates with a registry
    pub fn register_custom_gates(&self, registry: &mut GateRegistry) -> Vec<GateId> {
        self.custom_gates
            .iter()
            .map(|spec| registry.register(spec.clone()))
            .collect()
    }
}
```

### Serialization Format (Example)

```yaml
# program.yaml
header:
  custom_gates:
    - id: 256
      name: "MyRotation"
      quantum_arity: 2
      angle_arity: 3
      param_arity: 0
      returns_result: false
      category: TwoQubitUnitary

    - id: 257
      name: "MyMeasure"
      quantum_arity: 1
      angle_arity: 0
      param_arity: 0
      returns_result: true
      category: Measurement

body:
  gates:
    - { gate: PZ, qubits: [0, 1] }
    - { gate: H, qubits: [0] }
    - { gate: 256, qubits: [0, 1], angles: [0.5, 0.25, 0.1] }  # MyRotation
    - { gate: 257, qubits: [0] }  # MyMeasure
    - { gate: 257, qubits: [1] }  # MyMeasure
```

### Loading Flow

```rust
// 1. Load program (includes header with custom gate specs)
let program = Program::load("circuit.yaml")?;

// 2. Build tool - custom gates registered from program header
let tool = ToolBuilder::new()
    .with_program(program)  // Registers custom gates internally
    .with_simulator(...)
    .with_adaptor(...)      // Must handle custom gates
    .build()?;              // Validates everything

// 3. Run
let outcomes = tool.run();
```

### Translation Output

When a translation produces custom gates, it includes them in the output:

```rust
impl Translator {
    pub fn translate(&self, input: &Program) -> Program {
        let mut output = Program::new();

        // Translation might introduce new custom gates
        let my_optimized_gate = GateSpec {
            name: "OptimizedBlock",
            quantum_arity: 4,
            ...
        };
        output.custom_gates.push(my_optimized_gate);

        // Translate gates, possibly using the new custom gate
        for gate in &input.gates {
            output.gates.extend(self.translate_gate(gate));
        }

        output  // Self-describing: includes custom gate definitions
    }
}
```

### Why Headers, Not Global Registration?

| Approach | Problem |
|----------|---------|
| Global registry | Different programs might use same ID for different gates |
| Pass registry everywhere | Verbose, easy to forget |
| **Header in program** | Self-describing, portable, no conflicts |

Each program declares exactly what custom gates it uses. When loading, those gates are registered into the execution context. No global state, no conflicts.

## Circuit-Type-Specific Validators

Different circuit types may have different constraints on what gates and angle values are acceptable. A circuit-type-specific validator can enforce these constraints **at build time**, before any execution.

### Why Circuit-Specific Validation?

| Use Case | Constraint | Benefit |
|----------|------------|---------|
| **Clifford-only circuits** | No parameterized gates, or only exact angles that canonicalize | Faster simulation, guaranteed exactness |
| **Hardware-targeted circuits** | Only native gates, specific angles | No runtime surprises |
| **Verification circuits** | Exact angle values only | Deterministic behavior |
| **Compiled circuits** | All angles pre-canonicalized | No runtime canonicalization overhead |

### The Problem: Angle Precision at Parse Time

When parsing from floating-point input, there's a precision boundary:

```rust
// User writes in file: angle = 0.25 (meaning 1/4 turn, π/2)
let angle_f64 = 0.25_f64;  // Exact in f64

// Convert to Angle64
let angle = Angle64::from_turns(angle_f64);

// Is this EXACTLY Angle64::QUARTER_TURN?
// It should be, but floating-point conversion could introduce error
```

With Angle64's fixed-point representation, `0.25` turns converts exactly. But for unusual values like `0.12345678901234567`, the conversion may not round-trip.

### Solution: Circuit Type Validators

Each circuit type can define a validator that runs at build time:

```rust
/// Trait for circuit-type-specific validation
pub trait CircuitValidator {
    /// Validate a circuit before execution
    /// Returns Ok if circuit meets this type's constraints
    fn validate(&self, circuit: &Circuit, registry: &GateRegistry) -> Result<(), ValidationError>;
}

/// Validator for Clifford-only circuits
pub struct CliffordValidator {
    /// Allowed gates (must be Clifford gates only)
    allowed_gates: BitVec,

    /// For parameterized gates, allowed exact angles
    allowed_angles: HashMap<GateId, Vec<Angle64>>,
}

impl CliffordValidator {
    pub fn new() -> Self {
        use Angle64 as A;

        let mut allowed_angles = HashMap::new();

        // RZ only at Clifford angles
        allowed_angles.insert(gates::RZ, vec![
            A::ZERO,              // Identity
            A::QUARTER_TURN,      // S gate
            -A::QUARTER_TURN,     // S† gate (same as 3/4 turn)
            A::HALF_TURN,         // Z gate
        ]);

        // Similar for RX, RY
        allowed_angles.insert(gates::RX, vec![
            A::ZERO,
            A::QUARTER_TURN,
            -A::QUARTER_TURN,
            A::HALF_TURN,
        ]);

        Self {
            allowed_gates: clifford_gate_bits(),
            allowed_angles,
        }
    }
}

impl CircuitValidator for CliffordValidator {
    fn validate(&self, circuit: &Circuit, registry: &GateRegistry) -> Result<(), ValidationError> {
        for (idx, gate) in circuit.gates.iter().enumerate() {
            // Check gate is allowed
            if !self.allowed_gates.get(gate.gate_id.0 as usize).unwrap_or(false) {
                return Err(ValidationError::ForbiddenGate {
                    gate_id: gate.gate_id,
                    position: idx,
                });
            }

            // Check angles are exact Clifford angles
            if !gate.angles.is_empty() {
                if let Some(allowed) = self.allowed_angles.get(&gate.gate_id) {
                    for (i, angle) in gate.angles.iter().enumerate() {
                        // EXACT comparison - Angle64 is fixed-point
                        if !allowed.contains(angle) {
                            return Err(ValidationError::ForbiddenAngle {
                                gate_id: gate.gate_id,
                                angle: *angle,
                                position: idx,
                                allowed: allowed.clone(),
                            });
                        }
                    }
                } else {
                    // Parameterized gate not in allowed_angles = not allowed
                    return Err(ValidationError::ForbiddenGate {
                        gate_id: gate.gate_id,
                        position: idx,
                    });
                }
            }
        }

        Ok(())
    }
}
```

### Exact-Angle-Only Validator

For circuits where all parameterized gates must be canonicalizable to fixed gates:

```rust
/// Validator that requires all angles to be exactly canonicalizable
pub struct ExactAngleValidator {
    canonicalizer: GateCanonicalizer,
}

impl ExactAngleValidator {
    pub fn new() -> Self {
        Self {
            canonicalizer: GateCanonicalizer::standard(),
        }
    }

    /// Check if angle is exactly one of the canonicalizable values
    fn is_exact_angle(&self, gate_id: GateId, angle: Angle64) -> bool {
        // Exact comparison - no tolerance needed
        self.canonicalizer.canonicalize(gate_id, &[angle]).is_some()
    }
}

impl CircuitValidator for ExactAngleValidator {
    fn validate(&self, circuit: &Circuit, _registry: &GateRegistry) -> Result<(), ValidationError> {
        for (idx, gate) in circuit.gates.iter().enumerate() {
            if gate.angles.len() == 1 {
                let angle = gate.angles[0];

                // Must be an angle that canonicalizes to a fixed gate
                if !self.is_exact_angle(gate.gate_id, angle) {
                    return Err(ValidationError::NonCanonicalAngle {
                        gate_id: gate.gate_id,
                        angle,
                        position: idx,
                        hint: "Angle must be a standard value (0, π/4, π/2, π, etc.)",
                    });
                }
            }
        }

        Ok(())
    }
}
```

### Integration with Builder

```rust
impl ToolBuilder {
    /// Add a circuit-type validator
    pub fn with_circuit_validator(mut self, validator: Box<dyn CircuitValidator>) -> Self {
        self.circuit_validators.push(validator);
        self
    }

    fn validate(&self, circuit: &Circuit) -> Result<(), ValidationError> {
        // ... existing validation ...

        // Run circuit-type-specific validators
        for validator in &self.circuit_validators {
            validator.validate(circuit, &self.gate_registry)?;
        }

        Ok(())
    }
}

// Usage
let tool = ToolBuilder::new()
    .with_program(circuit)
    .with_simulator(SparseStab::new(10))
    .with_circuit_validator(Box::new(CliffordValidator::new()))
    .build()?;
```

### Pre-Canonicalization Validation

To ensure no floating-point issues, validate that all angles are exact **before** canonicalization happens:

```rust
/// Validate that input angles match expected exact values
/// Run this BEFORE canonicalization
pub struct AnglePrecisionValidator {
    /// Known exact angles (in turns) that should be parsed exactly
    exact_values: Vec<(f64, Angle64)>,
}

impl AnglePrecisionValidator {
    pub fn new() -> Self {
        Self {
            exact_values: vec![
                (0.0,    Angle64::ZERO),
                (0.125,  Angle64::HALF_TURN / 4),      // T gate
                (0.25,   Angle64::QUARTER_TURN),       // S gate
                (0.5,    Angle64::HALF_TURN),          // Z gate
                (0.75,   Angle64::THREE_QUARTERS_TURN),
                (-0.125, Angle64::ZERO - Angle64::HALF_TURN / 4),
                (-0.25,  Angle64::ZERO - Angle64::QUARTER_TURN),
                // etc.
            ],
        }
    }

    /// Validate that a parsed angle matches an exact value
    pub fn validate_parsed(&self, parsed: Angle64) -> Result<(), ValidationError> {
        // Check if this matches ANY known exact value
        for &(_, exact) in &self.exact_values {
            if parsed == exact {  // Exact comparison
                return Ok(());
            }
        }

        Err(ValidationError::AngleNotExact {
            angle: parsed,
            hint: "Angle does not match any known exact value. Use fractions like 0.25 for π/2.",
        })
    }
}
```

### Workflow: Parse → Validate → Canonicalize → Execute

```
┌─────────────────────────────────────────────────────────────────┐
│  1. PARSE: Read angles from file                                │
│     "RZ(0.25)" → Angle64::from_turns(0.25)                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  2. VALIDATE: Circuit-type validator checks angles              │
│     Is Angle64 == QUARTER_TURN?  (exact comparison)             │
│     If not exact, REJECT with clear error                       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼ (only if validation passes)
┌─────────────────────────────────────────────────────────────────┐
│  3. CANONICALIZE: Convert to fixed gates                        │
│     RZ(QUARTER_TURN) → SZ                                       │
│     (Safe because validation confirmed exactness)               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  4. EXECUTE: Run with optimized fixed-gate handlers             │
│     dispatch_table[SZ] → fast_sz_handler                        │
└─────────────────────────────────────────────────────────────────┘
```

### Benefits

1. **No silent precision issues**: If an angle isn't exact, get a clear error at build time
2. **Guaranteed canonicalization**: After validation, canonicalization always succeeds
3. **Circuit type safety**: Clifford circuits can't accidentally contain non-Clifford gates
4. **Hardware targeting**: Validate that circuits only use native gate set

### Common Validators

| Validator | Allows |
|-----------|--------|
| `CliffordValidator` | Clifford gates only, S/Z angles only |
| `CliffordTValidator` | Clifford + T gate |
| `ExactAngleValidator` | Any gate, but angles must be canonicalizable |
| `NativeGateValidator` | Only gates natively supported by target |
| `AnyValidator` | No restrictions (default) |

## Angle Snapping (Tolerance-Based Canonicalization)

When angles come from floating-point sources (numerical optimization, external tools, noisy input), they may be *close* to exact values but not precisely equal. An explicit "snap to nearest" function can handle this at the parse boundary.

### When to Use Snapping

| Source | Precision | Approach |
|--------|-----------|----------|
| Hand-written code | Exact fractions | No snapping needed |
| Circuit files (QASM, etc.) | Usually exact | Validate, reject if not exact |
| Numerical optimizer output | Close but not exact | Snap with tolerance |
| Hardware calibration data | Floating-point | Snap with tolerance |
| User input (GUI slider) | Arbitrary | Snap or reject |

### AngleSnapper

```rust
/// Snaps angles to nearest exact canonical values within tolerance
pub struct AngleSnapper {
    /// Known exact angles to snap to
    targets: Vec<Angle64>,

    /// Maximum distance (in turns) to snap
    tolerance: f64,
}

impl AngleSnapper {
    /// Standard snapper for common angles (multiples of π/4)
    pub fn standard(tolerance: f64) -> Self {
        use Angle64 as A;

        Self {
            targets: vec![
                A::ZERO,
                A::HALF_TURN / 4,           // π/4 (T)
                A::QUARTER_TURN,            // π/2 (S)
                A::HALF_TURN / 4 * 3,       // 3π/4
                A::HALF_TURN,               // π (Z)
                A::HALF_TURN + A::HALF_TURN / 4,      // 5π/4
                A::THREE_QUARTERS_TURN,     // 3π/2
                A::HALF_TURN + A::HALF_TURN / 4 * 3,  // 7π/4
                // Negative equivalents handled by wraparound
            ],
            tolerance,
        }
    }

    /// Clifford-only snapper (multiples of π/2)
    pub fn clifford(tolerance: f64) -> Self {
        use Angle64 as A;

        Self {
            targets: vec![
                A::ZERO,
                A::QUARTER_TURN,
                A::HALF_TURN,
                A::THREE_QUARTERS_TURN,
            ],
            tolerance,
        }
    }

    /// Try to snap an angle to the nearest target
    /// Returns Ok(snapped) if within tolerance, Err if no target is close enough
    pub fn snap(&self, angle: Angle64) -> Result<SnapResult, SnapError> {
        let mut best_target: Option<Angle64> = None;
        let mut best_distance = f64::MAX;

        for &target in &self.targets {
            // Distance on the circle (handles wraparound)
            let distance = angle_distance(angle, target);

            if distance < best_distance {
                best_distance = distance;
                best_target = Some(target);
            }
        }

        if best_distance <= self.tolerance {
            Ok(SnapResult {
                original: angle,
                snapped: best_target.unwrap(),
                distance: best_distance,
            })
        } else {
            Err(SnapError::NoTargetWithinTolerance {
                angle,
                nearest: best_target.unwrap(),
                distance: best_distance,
                tolerance: self.tolerance,
            })
        }
    }

    /// Snap or return original (for permissive mode)
    pub fn snap_or_keep(&self, angle: Angle64) -> Angle64 {
        self.snap(angle).map(|r| r.snapped).unwrap_or(angle)
    }
}

/// Result of successful snap
pub struct SnapResult {
    pub original: Angle64,
    pub snapped: Angle64,
    pub distance: f64,  // How far we snapped (in turns)
}

/// Calculate angular distance (shortest path on circle)
fn angle_distance(a: Angle64, b: Angle64) -> f64 {
    let diff = (a.to_turns() - b.to_turns()).abs();
    diff.min(1.0 - diff)  // Handle wraparound
}
```

### Integration: Snap at Parse Boundary

Snapping should happen **once**, at the boundary where floating-point enters the system:

```rust
impl CircuitParser {
    pub fn with_angle_snapping(mut self, snapper: AngleSnapper) -> Self {
        self.snapper = Some(snapper);
        self
    }

    fn parse_angle(&self, value: f64) -> Result<Angle64, ParseError> {
        // Convert float to fixed-point
        let angle = Angle64::from_turns(value);

        // Optionally snap to nearest exact value
        if let Some(snapper) = &self.snapper {
            match snapper.snap(angle) {
                Ok(result) => {
                    if result.distance > 0.0 {
                        // Log that we snapped (useful for debugging)
                        log::debug!(
                            "Snapped angle {:.6} → {:.6} (distance: {:.2e})",
                            result.original.to_turns(),
                            result.snapped.to_turns(),
                            result.distance
                        );
                    }
                    Ok(result.snapped)
                }
                Err(e) => Err(ParseError::AngleSnapFailed(e)),
            }
        } else {
            // No snapping - use angle as-is
            Ok(angle)
        }
    }
}
```

### Workflow with Snapping

```
┌─────────────────────────────────────────────────────────────────┐
│  1. INPUT: Floating-point angle from optimizer                  │
│     angle_f64 = 0.2500000001  (close to π/2)                    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  2. SNAP: Within tolerance of 1e-9?                             │
│     Yes → snap to Angle64::QUARTER_TURN (exact π/2)             │
│     Log: "Snapped 0.2500000001 → 0.25 (distance: 1e-10)"        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  3. VALIDATE: Now angle is EXACTLY π/2                          │
│     ExactAngleValidator passes ✓                                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  4. CANONICALIZE: RZ(π/2) → SZ                                  │
│     Guaranteed to succeed (angle is exact)                      │
└─────────────────────────────────────────────────────────────────┘
```

### Snapping Policies

Different use cases need different policies:

```rust
pub enum SnapPolicy {
    /// No snapping - angles must be exact
    Exact,

    /// Snap within tolerance, fail if no target close enough
    SnapOrFail { tolerance: f64 },

    /// Snap within tolerance, keep original if no target close enough
    SnapOrKeep { tolerance: f64 },

    /// Always snap to nearest (no tolerance check)
    /// Warning: can silently change intentional angles!
    AlwaysSnap,
}

impl ToolBuilder {
    pub fn with_snap_policy(mut self, policy: SnapPolicy) -> Self {
        self.snap_policy = policy;
        self
    }
}

// Usage
let tool = ToolBuilder::new()
    .with_program(circuit)
    .with_snap_policy(SnapPolicy::SnapOrFail { tolerance: 1e-9 })
    .with_circuit_validator(Box::new(ExactAngleValidator::new()))
    .build()?;
```

### Recommended Tolerances

| Source | Recommended Tolerance |
|--------|----------------------|
| IEEE 754 f64 rounding | 1e-15 |
| Numerical optimizer | 1e-9 to 1e-6 |
| User input (GUI) | 1e-3 to 1e-2 |
| "Close enough" | 1e-2 |

### Warnings and Diagnostics

When snapping occurs, provide clear diagnostics:

```rust
pub struct SnapDiagnostics {
    /// Angles that were snapped
    pub snapped: Vec<SnapResult>,

    /// Angles that failed to snap (if policy allows keeping)
    pub unsnapped: Vec<Angle64>,

    /// Maximum distance snapped
    pub max_snap_distance: f64,
}

impl ToolBuilder {
    pub fn build_with_diagnostics(self) -> Result<(Tool, SnapDiagnostics), ValidationError> {
        // ...
    }
}

// Check diagnostics
let (tool, diag) = builder.build_with_diagnostics()?;
if diag.max_snap_distance > 1e-6 {
    log::warn!(
        "Large angle snaps detected (max {:.2e}). Verify input precision.",
        diag.max_snap_distance
    );
}
```

### When NOT to Snap

Snapping should be **opt-in** and used carefully:

1. **Don't snap by default**: Silent snapping can hide bugs
2. **Don't snap with large tolerance**: Could change intentional angles
3. **Don't snap after validation**: Snap at parse boundary only
4. **Log all snaps**: Make it visible when snapping occurs

The workflow should be:
- **Strict mode**: No snapping, reject imprecise angles
- **Permissive mode**: Snap with tight tolerance, log snaps
- **Import mode**: Snap with looser tolerance for external data

## Open Questions

1. **ID assignment on load**: When loading a program, should custom gate IDs from the file be preserved, or reassigned sequentially?
   - Preserving: Simpler, but risk of ID conflicts between programs
   - Reassigning: Safer, requires remapping gate references in body

2. **Gate equality**: Should two user-registered gates with identical specs be considered equal?
   - Probably not - IDs are identities, same spec can be registered twice with different IDs

3. **Gateable traits relationship**: Should `CliffordGateable` etc. implement `GateExecutor` automatically via macro?
   - Would reduce boilerplate for simulator authors

4. **Adaptor composition**: Can adaptors be chained? (A decomposes to B, B decomposes to C)
   - Need to prevent infinite loops
   - Should decomposition be cached?

5. **Hot-reloading**: Can custom gates be added after Tool is built?
   - Current design: No, validation happens at build time
   - Could support if we re-validate on new gates

### Resolved Questions

- **Compile-time vs build-time**: Build-time for circuit-specific checks (see Validation Philosophy)
- **Serialization**: Programs include header with custom gate specs (see Circuit Headers)
- **Registry scope**: Registry is per-World/Tool, not global (see Unified Registration)
- **Core gate metadata**: Compile-time const tables for O(1) lookup (see Performance)

## Summary

| Aspect | Current | Proposed |
|--------|---------|----------|
| Gate types | Closed enum | Open registry |
| Adding gates | Modify pecos-core | User registration |
| Simulator support | Implicit (trait methods) | Explicit declaration |
| Unsupported gates | Runtime panic | Adaptor decomposition |
| Validation | None | Build-time checks |
| Performance | Enum match | Lookup table (similar) |
