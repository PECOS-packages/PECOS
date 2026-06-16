# Zlup Standard Library Reference

The standard library provides common data structures and utilities for quantum programming with NASA Power of 10 compliance. All containers have bounded capacity specified at compile time.

## Importing the Standard Library

```zlup_nocheck
std := @import("std");

// Type-namespaced constants (preferred)
angle := std.a64.t_angle;         // T-gate angle (1/8 turn)
pi := std.f64.pi;                 // pi as f64
count := std.popcount_u8(syndrome);
```

---

## Module: f64

Float64 constants commonly used in calculations.

### Fundamental Constants

| Constant | Type | Value | Description |
|----------|------|-------|-------------|
| `std.f64.pi` | f64 | 3.14159... | Pi |
| `std.f64.tau` | f64 | 6.28318... | 2*pi (full rotation in radians) |
| `std.f64.e` | f64 | 2.71828... | Euler's number |
| `std.f64.sqrt2` | f64 | 1.41421... | Square root of 2 |
| `std.f64.sqrt2_inv` | f64 | 0.70710... | 1/sqrt(2), Hadamard normalization |

### Angle Fractions in Radians

| Constant | Value | Degrees | Common Use |
|----------|-------|---------|------------|
| `std.f64.pi_2` | pi/2 | 90 | Quarter turn |
| `std.f64.pi_3` | pi/3 | 60 | |
| `std.f64.pi_4` | pi/4 | 45 | T-gate |
| `std.f64.pi_6` | pi/6 | 30 | |
| `std.f64.pi_8` | pi/8 | 22.5 | |

### Conversion Factors

| Constant | Description | Example |
|----------|-------------|---------|
| `std.f64.deg_to_rad` | Multiply degrees to get radians | `45.0 * std.f64.deg_to_rad` |
| `std.f64.rad_to_deg` | Multiply radians to get degrees | `pi_4 * std.f64.rad_to_deg` |

---

## Module: a64

Angle constants in turns (the native unit for a64). All values are exact in the Angle64 fixed-point representation.

### Fundamental Turn Fractions

| Constant | Turns | Radians | Degrees | Common Use |
|----------|-------|---------|---------|------------|
| `std.a64.zero` | 0 | 0 | 0 | Identity |
| `std.a64.half_turn` | 1/2 | pi | 180 | Z-gate |
| `std.a64.quarter_turn` | 1/4 | pi/2 | 90 | S-gate |
| `std.a64.eighth_turn` | 1/8 | pi/4 | 45 | T-gate |
| `std.a64.sixteenth_turn` | 1/16 | pi/8 | 22.5 | |

### Gate-Named Aliases

| Constant | Equivalent | Description |
|----------|------------|-------------|
| `std.a64.t_angle` | 1/8 turn | T-gate rotation |
| `std.a64.tdg_angle` | 7/8 turn | T-dagger (negative T) |
| `std.a64.s_angle` | 1/4 turn | S-gate rotation |
| `std.a64.sdg_angle` | 3/4 turn | S-dagger (negative S) |
| `std.a64.z_angle` | 1/2 turn | Z-gate rotation |

### Example Usage

```zlup_nocheck
std := @import("std");

// Preferred: use a64 constants with turns unit
rz(std.a64.t_angle turns) q[0];       // T-gate
rz(std.a64.quarter_turn turns) q[0];  // S-gate

// Or use f64.pi with rad unit
rz(std.f64.pi/4 rad) q[0];            // Also T-gate

// Fraction literals work directly
rz(1/8 turns) q[0];                   // Also T-gate
```

---

## Module: bits

Bitwise utilities for working with measurement results. Essential for syndrome processing in QEC.

### Bit Counting (Population Count)

Count the number of set bits (1s) in a value.

```zlup_nocheck
fn popcount_u8(x: u8) -> u8
fn popcount_u16(x: u16) -> u16
fn popcount_u32(x: u32) -> u32
fn popcount_u64(x: u64) -> u64
```

**Example:**
```zlup_nocheck
result: u8 = 0b10110100;
weight := std.popcount_u8(result);  // 4
```

### Parity

Compute parity (XOR of all bits). Returns 0 if even number of 1s, 1 if odd.

```zlup_nocheck
fn parity_u8(x: u8) -> u1
fn parity_u16(x: u16) -> u1
fn parity_u32(x: u32) -> u1
fn parity_u64(x: u64) -> u1
```

**Example:**
```zlup_nocheck
// Syndrome parity check
syndrome: u8 = mz(pack u8) ancillas;
if std.parity_u8(syndrome) == 1 {
    // Odd parity - error detected
}
```

### Bit Extraction

Extract a single bit at the given index (0 = LSB).

```zlup_nocheck
fn get_bit_u8(x: u8, index: u8) -> u1
fn get_bit_u16(x: u16, index: u16) -> u1
fn get_bit_u32(x: u32, index: u32) -> u1
fn get_bit_u64(x: u64, index: u64) -> u1
```

**Example:**
```zlup_nocheck
result: u8 = 0b10110100;
bit2 := std.bits.get_bit_u8(result, 2);  // 1
bit0 := std.bits.get_bit_u8(result, 0);  // 0
```

### Bit Manipulation

```zlup_nocheck
fn set_bit_u8(x: u8, index: u8) -> u8      // Set bit to 1
fn clear_bit_u8(x: u8, index: u8) -> u8    // Set bit to 0
fn toggle_bit_u8(x: u8, index: u8) -> u8   // Flip bit
```

**Example:**
```zlup_nocheck
x: u8 = 0b00000000;
x = std.bits.set_bit_u8(x, 3);    // 0b00001000
x = std.bits.toggle_bit_u8(x, 0); // 0b00001001
x = std.bits.clear_bit_u8(x, 3);  // 0b00000001
```

### Byte Order

```zlup_nocheck
fn reverse_bits_u8(x: u8) -> u8           // Reverse bit order
fn swap_bytes_u16(x: u16) -> u16          // Swap bytes (endianness)
fn swap_bytes_u32(x: u32) -> u32          // Swap bytes
```

**Example:**
```zlup_nocheck
x: u8 = 0b10110100;
reversed := std.bits.reverse_bits_u8(x);  // 0b00101101
```

---

## Module: containers

Bounded data structures for deterministic memory usage.

### Stack(T, capacity)

Last-In-First-Out (LIFO) container.

```zlup_nocheck
stack: std.Stack(u32, 64) = .{};
```

**Methods:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `push` | `(&mut self, T) -> OverflowError!void` | Push item (fails if full) |
| `pop` | `(&mut self) -> ?T` | Pop item (none if empty) |
| `peek` | `(&mut self) -> ?T` | View top without removing |
| `is_empty` | `(&mut self) -> bool` | Check if empty |
| `is_full` | `(&mut self) -> bool` | Check if at capacity |
| `clear` | `(&mut self) -> unit` | Remove all items |
| `count` | `(&mut self) -> usize` | Current item count |
| `get_capacity` | `(&mut self) -> usize` | Maximum capacity |

**Example:**
```zlup_nocheck
stack: std.Stack(u32, 64) = .{};
try stack.push(10);
try stack.push(20);

if val := stack.pop() {
    // val is 20
}
```

### Queue(T, capacity)

First-In-First-Out (FIFO) container using a ring buffer.

```zlup_nocheck
queue: std.Queue(u32, 64) = .{};
```

**Methods:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `enqueue` | `(&mut self, T) -> OverflowError!void` | Add to back |
| `dequeue` | `(&mut self) -> ?T` | Remove from front |
| `peek_front` | `(&mut self) -> ?T` | View front without removing |
| `is_empty` | `(&mut self) -> bool` | Check if empty |
| `is_full` | `(&mut self) -> bool` | Check if at capacity |
| `clear` | `(&mut self) -> unit` | Remove all items |
| `count` | `(&mut self) -> usize` | Current item count |

**Example:**
```zlup_nocheck
queue: std.Queue(u32, 64) = .{};
try queue.enqueue(1);
try queue.enqueue(2);

if val := queue.dequeue() {
    // val is 1 (first in, first out)
}
```

### Deque(T, capacity)

Double-ended queue supporting insertion/removal at both ends.

```zlup_nocheck
deque: std.Deque(u32, 64) = .{};
```

**Methods:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `push_back` | `(&mut self, T) -> OverflowError!void` | Add to back |
| `push_front` | `(&mut self, T) -> OverflowError!void` | Add to front |
| `pop_back` | `(&mut self) -> ?T` | Remove from back |
| `pop_front` | `(&mut self) -> ?T` | Remove from front |
| `peek_front` | `(&mut self) -> ?T` | View front |
| `peek_back` | `(&mut self) -> ?T` | View back |
| `is_empty` | `(&mut self) -> bool` | Check if empty |
| `is_full` | `(&mut self) -> bool` | Check if at capacity |
| `clear` | `(&mut self) -> unit` | Remove all items |
| `count` | `(&mut self) -> usize` | Current item count |

### PriorityQueue(T, capacity)

Min-heap priority queue. Smallest element is always at front.

```zlup_nocheck
pq: std.PriorityQueue(u32, 64) = .{};
```

**Methods:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `insert` | `(&mut self, T) -> OverflowError!void` | Insert with priority |
| `extract_min` | `(&mut self) -> ?T` | Remove smallest |
| `peek_min` | `(&mut self) -> ?T` | View smallest |
| `is_empty` | `(&mut self) -> bool` | Check if empty |

**Example:**
```zlup_nocheck
pq: std.PriorityQueue(u32, 64) = .{};
try pq.insert(50);
try pq.insert(10);
try pq.insert(30);

if val := pq.extract_min() {
    // val is 10 (smallest)
}
```

---

## Module: qec

Quantum Error Correction utilities for decoder implementations.

### UnionFind(capacity)

Disjoint Set Union (DSU) data structure. Essential for MWPM decoders.

```zlup_nocheck
uf: std.UnionFind(256) = .{};
uf.init();
```

**Methods:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `init` | `(&mut self) -> unit` | Initialize all elements as separate sets |
| `find` | `(&mut self, usize) -> usize` | Find root of set (with path compression) |
| `union` | `(&mut self, usize, usize) -> bool` | Unite two sets (returns true if merged) |
| `connected` | `(&mut self, usize, usize) -> bool` | Check if in same set |
| `reset` | `(&mut self, usize) -> unit` | Reset element to own set |
| `reset_all` | `(&mut self) -> unit` | Reset all elements |

**Example:**
```zlup_nocheck
uf: std.UnionFind(256) = .{};
uf.init();

uf.union(0, 1);
uf.union(1, 2);

if uf.connected(0, 2) {
    // 0 and 2 are in the same set
}
```

### SyndromeBuffer(num_ancillas, max_rounds)

Storage for syndrome measurements across multiple rounds.

```zlup_nocheck
syndrome: std.SyndromeBuffer(16, 10) = .{};  // 16 ancillas, 10 rounds
```

**Methods:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `clear` | `(&mut self) -> unit` | Clear all data |
| `set` | `(&mut self, ancilla, round, bool) -> unit` | Set syndrome bit |
| `get` | `(&mut self, ancilla, round) -> bool` | Get syndrome bit |
| `record_round` | `(&mut self, [N]bool) -> OverflowError!void` | Record full round |
| `num_rounds` | `(&mut self) -> usize` | Rounds recorded |
| `has_error` | `(&mut self, round) -> bool` | Any syndrome in round? |
| `count_errors` | `(&mut self, round) -> usize` | Count triggered syndromes |

**Example:**
```zlup_nocheck
syndrome: std.SyndromeBuffer(4, 10) = .{};

// Record syndromes from measurement
ancilla_results: [4]bool = .{ true, false, false, true };
try syndrome.record_round(ancilla_results);

if syndrome.has_error(0) {
    // Process errors in round 0
}
```

### LookupDecoder(num_syndromes, num_corrections)

Table-based decoder for small codes.

```zlup_nocheck
decoder: std.LookupDecoder(16, 4) = .{};
```

**Methods:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `add_entry` | `(&mut self, syndrome, correction) -> OverflowError!void` | Add lookup entry |
| `decode` | `(&mut self, syndrome) -> ?Correction` | Lookup correction |

### PauliFrame(num_qubits)

Track Pauli frame for frame tracking decoders.

```zlup_nocheck
frame: std.PauliFrame(64) = .{};
```

**Methods:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `init` | `(&mut self) -> unit` | Initialize to identity |
| `apply_x` | `(&mut self, qubit) -> unit` | Apply X correction |
| `apply_z` | `(&mut self, qubit) -> unit` | Apply Z correction |
| `apply_y` | `(&mut self, qubit) -> unit` | Apply Y correction |
| `get_x` | `(&mut self, qubit) -> bool` | Check X component |
| `get_z` | `(&mut self, qubit) -> bool` | Check Z component |
| `reset` | `(&mut self, qubit) -> unit` | Reset qubit's frame |
| `reset_all` | `(&mut self) -> unit` | Reset entire frame |

### SparseGraph(max_nodes, max_edges)

Adjacency list graph for decoder algorithms.

```zlup_nocheck
graph: std.SparseGraph(256, 1024) = .{};
```

**Methods:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `init` | `(&mut self) -> unit` | Initialize empty graph |
| `add_edge` | `(&mut self, u, v, weight) -> OverflowError!void` | Add weighted edge |
| `get_neighbors` | `(&mut self, node) -> []Edge` | Get adjacent edges |
| `clear` | `(&mut self) -> unit` | Remove all edges |

---

## Module Structure

The main `std` module provides type-namespaced constants and utilities:

```zlup_nocheck
std := @import("std");

// Type-namespaced constants
std.f64.pi, std.f64.tau, std.f64.e      // f64 constants
std.f64.sqrt2, std.f64.sqrt2_inv
std.f64.pi_2, std.f64.pi_4, std.f64.pi_8

std.a64.quarter_turn, std.a64.eighth_turn  // a64 angle constants (in turns)
std.a64.t_angle, std.a64.s_angle           // Gate-named aliases

// Bit operations
std.popcount_u8, std.popcount_u16, std.popcount_u32, std.popcount_u64
std.parity_u8, std.parity_u16, std.parity_u32, std.parity_u64

// Container types (from std.containers)
std.Stack, std.Queue, std.Deque, std.PriorityQueue

// QEC utilities (from std.qec)
std.UnionFind, std.SyndromeBuffer, std.LookupDecoder
std.PauliFrame, std.SparseGraph
```

---

## Error Types

### OverflowError

Returned when a bounded container exceeds capacity.

```zlup_nocheck
OverflowError := error { Overflow };
```

**Handling:**
```zlup_nocheck
// Propagate with try
try stack.push(item);

// Handle with catch
stack.push(item) catch {
    // Handle overflow
};

// Check before operation
if !stack.is_full() {
    try stack.push(item);
}
```
