# PECOS PHIR v0.1

## Machine Operations

The PHIR format supports various machine operations for controlling the physical execution of quantum programs. These operations provide better control over timing, qubit movement, and other hardware-specific aspects.

### Supported Machine Operations

#### Idle Operation

The `Idle` operation specifies that certain qubits should remain idle for a specific duration.

```json
{
  "mop": "Idle",
  "args": [["q", 0], ["q", 1]],
  "duration": [5.0, "ms"]
}
```

- `args`: Specifies the qubits that will be in the idle state.
- `duration`: Specifies the duration as a tuple with value and unit (supported units: "s", "ms", "us", "ns").

#### Delay Operation

The `Delay` operation inserts a specific delay for the specified qubits.

```json
{
  "mop": "Delay",
  "args": [["q", 0]],
  "duration": [2.0, "us"]
}
```

- `args`: Specifies the qubits to apply the delay to.
- `duration`: Specifies the duration as a tuple with value and unit.

#### Transport Operation

The `Transport` operation represents moving qubits from one location to another.

```json
{
  "mop": "Transport",
  "args": [["q", 1]],
  "duration": [1.0, "ms"],
  "metadata": {"from_position": [0, 0], "to_position": [1, 0]}
}
```

- `args`: Specifies the qubits being transported.
- `duration`: Specifies the duration of the transport operation.
- `metadata`: Additional information about the transport, such as start and end positions.

#### Timing Operation

The `Timing` operation synchronizes operations in time, useful for choreographing complex sequences.

```json
{
  "mop": "Timing",
  "args": [["q", 0], ["q", 1]],
  "metadata": {"timing_type": "sync", "label": "sync_point_1"}
}
```

- `args`: Specifies the qubits affected by the timing operation.
- `metadata`: Additional information:
  - `timing_type`: The type of timing operation (e.g., "sync", "start", "end").
  - `label`: A label for the timing point for referencing in the program.

#### Reset Operation

The `Reset` operation resets qubits to the |0⟩ state.

```json
{
  "mop": "Reset",
  "args": [["q", 0]],
  "duration": [0.5, "us"]
}
```

- `args`: Specifies the qubits to reset.
- `duration`: Specifies the duration of the reset operation.

### Using Machine Operations in PHIR Programs

Machine operations can be combined with quantum and classical operations in PHIR programs. Here's an example showing a complete program using various machine operations:

```json
{
  "format": "PHIR/JSON",
  "version": "0.1.0",
  "metadata": {
    "num_qubits": 2,
    "source_program_type": ["Test", ["PECOS", "0.5.dev1"]]
  },
  "ops": [
    {"data": "qvar_define", "data_type": "qubits", "variable": "q", "size": 2},
    {"data": "cvar_define", "data_type": "i32", "variable": "result", "size": 32},

    {"qop": "H", "args": [["q", 0]]},

    {"mop": "Idle", "args": [["q", 0], ["q", 1]], "duration": [5.0, "ms"]},

    {"mop": "Delay", "args": [["q", 0]], "duration": [2.0, "us"]},

    {"mop": "Transport", "args": [["q", 1]], "duration": [1.0, "ms"], "metadata": {"from_position": [0, 0], "to_position": [1, 0]}},

    {"mop": "Timing", "args": [["q", 0], ["q", 1]], "metadata": {"timing_type": "sync", "label": "sync_point_1"}},

    {"mop": "Reset", "args": [["q", 0]], "duration": [0.5, "us"]},

    {"qop": "CX", "args": [["q", 0], ["q", 1]]},

    {"qop": "Measure", "args": [["q", 0]], "returns": [["m", 0]]},
    {"qop": "Measure", "args": [["q", 1]], "returns": [["m", 1]]},

    {"cop": "=", "args": [{"cop": "+", "args": [["m", 0], ["m", 1]]}], "returns": ["result"]},
    {"cop": "Result", "args": ["result"], "returns": ["output"]}
  ]
}
```

### Implementation Notes

- All time durations are converted to nanoseconds internally for consistent handling.
- Machine operations are processed in the order they appear in the program.
- Timing operations may be treated as no-ops on hardware that doesn't support them.
- The effect of machine operations depends on the capabilities of the target hardware.
