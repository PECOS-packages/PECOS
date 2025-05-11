# PECOS High-level Intermediate Representation (PHIR) Specification

Author: Ciarán Ryan-Anderson

PHIR (PECOS High-level Intermediate Representation), pronounced "fire," is a JSON-based format created specifically for
PECOS. Its primary purpose is to represent hybrid quantum-classical programs. This structure allows for capturing both
quantum/classical instructions as well as the nuances of machine state and noise, thereby enabling PECOS to offer a
realistic simulation experience.

This document sets out to outline the near-term implementation of PHIR, detailing its relationship with extended
OpenQASM 2.0 and its associated execution model.

## Program-level Structure

PHIR's top-level structure is represented as a dictionary, encompassing program-level metadata, version, and the actual
sequence of operations the program encapsulates.

```json5
{
  "format": "PHIR/JSON",
  "version": "0.1.0",
  "metadata": {  // optional
    "program_name": "Sample Program",
    "author": "Alice",
    // ... Other custom metadata fields can be added as required
  },
  "ops": [{...}, ...]
}
```

- `"format"`: Signifies the utilization of the PHIR/JSON format.
- `"version"`: Represents the semantic version number adhering to the PHIR spec.
- `"metadata"`: An optional segment where users can incorporate additional details. This segment holds potential for
future expansion, possibly to guide compilation processes and error modeling.
- `"ops": [{...}, ...]`: A linear sequence denoting the operations and blocks that constitute the program.

### Metadata Options

<!-- markdownlint-disable MD013 -->
| parameter              | options           | description                                                                                                                                                                                                                                                                                             |
| ---------------------- | ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `"strict_parallelism"` | `"true", "false"` | If `"true"`, tell emulator to interpret `"qop"`s with multiple arguments (outside a [qparallel block](#qparallel-block)) as parallel application of the `"qop"` to those arguments. If `"false"` (default), the emulator is free to decide how much parallelism to apply to multiple argument `"qop"`s. |
<!-- markdownlint-enable MD013 -->

## Comments

All entries in PHIR, whether instructions or blocks, adopt the dictionary format `{...}`. One
can intersperse comments in the form of `{"//": str }` that are inserted into a sequence of
operations/blocks `[{...}, ...]`

## General operation structure

Within the sequence of represented by the segment `"ops": {...}`, each operation and block has the form:

```json5
{
  "type": "op_name",
  "metadata": {...},  // optional
  "additional_keys": "values" | [{...}, ...],  // Such as inputs to a gate
  ...
}
```

Operations/blocks themselves may hold sequences or blocks, thus allowing for nesting.

At present, the principal types encompass: `"data"`, `"cop"`, `"qop"`, and `"block"`. Future iterations might include
other types, especially to bolster error modeling. Here's a quick breakdown of the `"type"`s:

- `"data"`: Directives specifically related to data handling such as the creation of variables.
- `"cop"`: Refers to classical operations. This includes actions like defining/assigning classical variables or
executing Boolean/arithmetic expressions.
- `"qop"`: Denotes quantum operations. These encompass actions like executing unitaries, measurements, and initializing
states.
- `"mop"`: A machine operation, which represents changes in the machine state. For example, idling, transport, etc.
- `"block"`: Facilitates grouping and control flow of the encapsulated blocks/operations.

A comprehensive explanation of these operations and blocks is given in the following sections.

## Data Management

These operations deal with data/variable handling such as data definition and exporting, helping to structure the
information flow in the program. In the future, the `"data"` type may be utilized to create and manipulate data
types/structures such as arrays, data de-allocation, scoping, etc.

### Defining Classical Variables

In the current implementation, classical variables are defined as globally accessible, meaning they exist in the
top-level scope. The lifespan of a classical variable extends until the program concludes. Once a classical variable and
its associated symbol are created, they remain accessible through the entirety of the program. The symbol consistently
refers to the same memory location. By default, classical variables are represented as i64s. The value of these
variables can be modified through assignment operations.

To define or create a variable, the following structure is employed:

```json5
{
  "data": "cvar_define",
  "data_type": str,  // Preferably "i64"
  "variable": str,  // The variable symbol
  "size": int  // Optional
}
```

- `"data_type"`: One of "i64", "i32", "u64", "u32".
- `"variable"`: Represents the symbol tied to the classical variable. By default, all variables are initialized with a
value of 0.
- `"size"`: Even though every variable is internally represented as an i64, a size can be specified. This correlates
with the size in OpenQASM 2.0's `creg sym[size];`. If omitted, `"size"` defaults to the bitsize of the integer
representation (e.g., 32 for i32, 64 for i64, etc.). During classical computations, all variables behave as complete
i64s. However, when assigned, the bits are restricted to the number defined by `"size"`. For instance, in OpenQASM 2.0,
executing `creg a[2]; a = 5;` results in `a` holding the value 3 (`0b111` becomes `0b011` since `a` is restricted to 2
bits).

To prevent runtime errors, ensure a classical variable is defined prior to its usage. Given their global scope and the
necessity for prior definition, it's advisable to declare variables at the program's onset.

### Exporting Classical Variables

At the conclusion of a simulation or quantum computation, you might want to extract or "export" certain classical
variables. This mechanism allows users to retrieve selected results from their computations. This mechanism also allows
a program to have an internal representation of variables and/or scratch space/helper variables and to then present the
user with only the information they requested. In PHIR, the structure to accomplish this is:

```json5
{
  "data": "cvar_export",
  "variables": [str, ...],  // List of classical variable symbols to export.
  "to": [str, ...],  // Optional; rename a variable upon export.
}
```

It's worth noting that if no specific export requests are made, PECOS will default to exporting all classical variables.
These will be made available in the user's final results dictionary post-simulation.

### Defining Quantum Variables

To define a set of qubits and associate them with quantum variables:

```json5
{
  "data": "qvar_define",
  "data_type": "qubits",  // Optional
  "variable": str,  // Symbol representing the quantum variable.
  "size": 10  // Number of qubits
}
```

Much like classical variables, quantum variables exist in the top-level scope. These are accessible throughout the
program and defined for its entirety. An individual qubit is denoted by the qubit ID `[variable_str, qubit_index]`. For
instance, the 1st qubit of the quantum variable `"q"` is represented as `["q", 1]`.

## Classical operations

Classical operations are all those operations that manipulate classical information. In the current PECOS
implementation, all classical variables are implemented as 64-bit signed integers (i64).

### Assigning values to Classical Variables

Assigning a value to a classical variable involves updating the underlying i64 to a new integer value. The structure for
this assignment in PHIR is:

```json5
{
  "cop": "=",
  "args": [int | int_expression],
  "returns": [str]  // variable symbol
}
```

Currently, only one variable can be assigned at a time; however, the `"args"` and `"returns"` syntax with a corresponding
list of variables is used to be consistent with the measurement and foreign function syntax discussed below, as well as
to leave open the possibility of supporting destructuring of tuples/arrays in the future.

In PHIR, specific bits of an integer can be addressed in an array-like syntax, mirroring the `a[0]` notation in OpenQASM
2.0. To reference a bit of a variable in PHIR, use the structure `["variable_symbol", bit_index]`. The assignment
structure then appears as:

```json5
{
  "cop": "=",
  "args": [int | int_expression],
  "returns": [ [str, int] ]  // bit_id -> [str, int]
}
```

Regardless of assigned `"value"`, when updating a single bit, only the least significant bit (0th bit) of the value is
taken into consideration.

The term `int_expression` has been introduced and will be elaborated upon in the upcoming sections. Essentially,
`int_expression` encompasses classical operations that ultimately yield an integer value.

### Integer Expressions

In PHIR, an integer expression encompasses any combination of arithmetic, comparison, or bitwise operations, including
classical variables or integer literals, that results in an integer value. While future iterations of PHIR and PECOS may
introduce other expression types (e.g., floating-point expressions), the current version strictly supports integer
expressions. The table provided below (Table I) details the list of supported classical operations (`cops`) that can be
used within these expressions as well as assignment operations.

Constructing these expressions follow an Abstract Syntax Tree (AST) style, utilizing the below formats:

#### General  Operations

```json5
{
  "cop": "op_name",
  "args": [cvariable | [cvariable, bit_index] | int | {int_expression...}, ...]
}
```

#### Unary Operations

```json5
{
  "cop": "op_name",
  "args": [cvariable | [cvariable, bit_index] | int | {int_expression...}]
}
```

#### Binary Operations

```json5
{
  "cop": "op_name",
  "args": [
    cvariable | [cvariable, bit_index] | int | {int_expression...},
    cvariable | [cvariable, bit_index] | int | {int_expression...}
  ]
}
```

**Important NOTE:** While PECOS is designed to handle comparison operations within expressions, extended OpenQASM is
not. Consequently, when translating from extended OpenQASM 2.0 to PHIR, restrict expressions to only arithmetic and
bitwise operations. For instance, `a = b ^ c;` is valid, whereas `a = b < c` is not. In OpenQASM 2.0's `if()`
statements, a direct comparison between a classical variable or bit and an integer is the only permitted configuration.
In PECOS implements true comparisons to evaluate to 1 and false ones to evaluate to 0.

#### Table I - Cop Assignment, arithmetic, comparison, & bitwise operations

| name   | # args | sub-type   | description            |
| ------ | ------ | ---------- | ---------------------- |
| `"="`  | 2      | Assignment | Assign                 |
| `"+"`  | 2      | Arithmetic | Addition               |
| `"-"`  | 1 / 2  | Arithmetic | Negation / Subtraction |
| `"*"`  | 2      | Arithmetic | Multiplication         |
| `"/"`  | 2      | Arithmetic | Division               |
| `"%"`  | 2      | Arithmetic | Modulus                |
| `"=="` | 2      | Comparison | Equal                  |
| `"!="` | 2      | Comparison | Not equal              |
| `">"`  | 2      | Comparison | Greater than           |
| `"<"`  | 2      | Comparison | Less than              |
| `">="` | 2      | Comparison | Greater than or equal  |
| `"<="` | 2      | Comparison | Less than or equal     |
| `"&"`  | 2      | Bitwise    | AND                    |
| `"\|"` | 2      | Bitwise    | OR                     |
| `"^"`  | 2      | Bitwise    | XOR                    |
| `"~"`  | 1      | Bitwise    | NOT                    |
| `"<<"` | 2      | Bitwise    | Left shift             |
| `">>"` | 2      | Bitwise    | Right shift            |

#### Integer Expression and Assignment Example

For illustrative purposes, let's explore how `b = (c[2] ^ d) | (e - 2 + (f == g));` would be represented in PHIR:

```json5
{
  "cop": "=",
  "args": [
    {"cop": "|",
    "args": [
      {"cop": "^", "args": [["c", 2], "d"]},
      {"cop": "+", "args": [
        {"cop": "-", "args": ["e", 2]},
        {"cop": "==", "args": ["f", "g"]}
      ]}
    ]
    }
  ],
  "returns": ["b"]
}
```

This example elucidates how intricate expressions are structured in a hierarchical, tree-like manner within PHIR.

### Exporting Results with the Result Command

The Result command enables mapping internal measurement results to exported variables in the final output. This operation is essential for specifying which measurement results should be exposed to the user and under what names.

```json5
{
  "cop": "Result",
  "args": [...],    // Source registers or bits to export
  "returns": [...]  // Names under which to export the results
}
```

The Result command supports several formats for both args and returns:

- Single bit export: `["m", 0]` for a specific bit
- Entire register export: `"m"` for the whole register
- Multiple variable export: using multiple entries in the arrays

When executed, this operation takes the current values from the source registers and makes them available in the final results under the specified export names. The Result command acts as an explicit declaration of program outputs, allowing internal implementation details and temporary values to remain hidden.

#### Examples

**Export a specific bit:**
```json5
// Export measurement result from bit 0 of register "m" as bit 0 of "q0_result"
{"cop": "Result", "args": [["m", 0]], "returns": [["q0_result", 0]]}
```

**Export entire registers:**
```json5
// Export entire "m" register as "results"
{"cop": "Result", "args": ["m"], "returns": ["results"]}
```

**Export multiple registers or bits:**
```json5
// Export multiple registers at once
{"cop": "Result", "args": ["m1", "m2"], "returns": ["results1", "results2"]}

// Export multiple bits with different mappings
{"cop": "Result",
 "args": [["m", 0], ["m", 1]],
 "returns": [["results", 0], ["results", 1]]}
```

The ability to export multiple values in a single Result command aligns with the general semantics of other commands in the PHIR specification, providing a consistent interface.

*Note:* Added in specification v0.1.1

### Calling External Classical Functions

In PECOS, it's possible to invoke external classical functions, especially using entities like WebAssembly (Wasm)
modules. This functionality broadens the expressive power of PECOS by tapping into the capabilities beyond quantum
operations. The structure for representing such "foreign function calls" in PHIR is:

PECOS can make foreign function calls utilizing objects such as Wasm modules. The structure is:

```json5
{
  "cop": "ffcall",
  "function": str,  // Name of the function to invoke
  "args": [...],  // List of input classical variables or bits
  "returns": [...],  // Optional; List of classical variables or bits to store the return values.
  "metadata": { // Optional
    "ff_object":  str, // Optional; hints at specific objects or modules providing the external function.
    ...
  }
}
```

When interacting with external classical functions in PHIR/PECOS, it's crucial to recognize that these external object
can maintain state. This means their behavior might depend on prior interactions, or they might retain information
between different calls. Here are some important considerations about such stateful interactions.

- *Stateful Operations in extended OpenQASM 2.0:* Extended OpenQASM 2.0 and its implementation recognizes the potential
statefulness of these objects. Therefore, foreign function calls in this environment are designed to be flexible. They
don't always mandate a return value. For instance, a QASM program can interact with the state of an external classical
object, possibly changing that state, without necessarily fetching any resultant data.
- *Asynchronous Processing:* These classical objects can process function calls asynchronously, operating alongside the
primary quantum or classical computation. This allows for efficient, non-blocking interactions.
- *Synchronization Points:* If a return value is eventually requested from a stateful object, it acts as a
synchronization point. The primary program will pause, ensuring that all preceding asynchronous calls to the external
object have fully resolved and that any required data is available before processing.

## Quantum operations

The generic qop gate structure is:

```json5
{
  "qop": str,
  "angles": [[float...], "rad" | "pi"],  // Include if gate has one or more angles.
  "args": [qubit_id, ... | [qubit_id, ... ], ...],  // Can be a list of qubit IDs or a list of lists for multi-qubit gates.
  "metadata": {}, // Optional metadata for potential auxiliary info or to be utilized by error models.
  "returns": [[str, int], ...]  // Include if gate produces output, e.g., a measurement.
}
```

`"angles"` is a tuple of a list of `float`s and a unit.
The units supported are radians (preferred) and multiples of ᴨ (pi radians).

Table II details the available qops.

For qops like `H q[0]; H q[1]; H q[4];` in QASM, it is translated as:

```json5
{
  "qop": "H",
  "args": [
    ["q", 0],
    ["q", 1],
    ["q", 4]
    ]
}
```

However, multi-qubit gates, such as `CX`, use a list of lists of qubit IDs. E.g.,
`CX q[0], q[1]; CX q[3], q[6]; CX q[2], q[7];` in QASM, can be represented as:

```json5
{
  "qop": "CX",
  "args": [
    [["q", 0], ["q", 1]],
    [["q", 3], ["q", 6]],
    [["q", 2], ["q", 7]]
    ]
}
```

PECOS ensures all qubit IDs in `"args"` are unique, meaning gates don't overlap on the same qubits.

For gates with one or multiple angles, angles are denoted as a list of floats and a unit in the `"angles"` field:

```json5
{
  "qop": "RZZ",
  "angles": [[0.173], "rad"],
  "args": [
    [ ["q", 0], ["q", 1] ],
    [ ["q", 2], ["q", 3] ]
  ],
  "metadata": {"duration": 100}
}
```

```json5
{
  "qop": "U1q",
  "angles": [[0.524, 1.834], "rad"],
  "args": [
    [ ["q", 0], ["q", 1], ["q", 2], ["q", 3] ]
  ],
  "metadata": {"duration":  40}
}
```

For a Z basis measurement on multiple qubits:

```json5
{
  "qop": "Measure",
  "args": [ ["q", 0], ["q", 1], ["q", 2], ["q", 3] ],
  "returns": [ ["m", 0], ["m", 1], ["m", 2], ["m", 3] ]
}
```

### Table II - Quantum operations

| name         | alt. names        | # angles | # qubits | matrix | description              |
| ------------ | ----------------- | -------- | -------- | ------ | ------------------------ |
| `"Init"`     |                   | 0        | 1        | ...    | Initialize qubit to \|0> |
| `"Measure"`  |                   | 0        | 1        | ...    | Measure qubit in Z basis |
| `"I"`        |                   | 0        | 1        | ...    | Identity                 |
| `"X"`        |                   | 0        | 1        | ...    | Pauli X                  |
| `"Y"`        |                   | 0        | 1        | ...    | Pauli Y                  |
| `"Z"`        |                   | 0        | 1        | ...    | Pauli Z                  |
| `"RX"`       |                   | 1        | 1        | ...    | Rotation about X         |
| `"RY"`       |                   | 1        | 1        | ...    | Rotation about Y         |
| `"RZ"`       |                   | 1        | 1        | ...    | Rotation about Z         |
| `"R1XY"`     | `"U1q"`           | 2        | 1        | ...    |                          |
| `"SX"`       |                   | 0        | 1        | ...    | Sqrt. of X               |
| `"SXdg"`     |                   | 0        | 1        | ...    | Adjoint of sqrt. of X    |
| `"SY"`       |                   | 0        | 1        | ...    | Sqrt. of Y               |
| `"SYdg"`     |                   | 0        | 1        | ...    | Adjoint of sqrt. of Y    |
| `"SZ"`       | `"S"`             | 0        | 1        | ...    | Sqrt. of Z               |
| `"SZdg"`     | `"Sdg"`           | 0        | 1        | ...    | Adjoint of sqrt. of Z    |
| `"H"`        |                   | 0        | 1        | ...    | Hadamard, X <-> Z        |
| `"F"`        |                   | 0        | 1        | ...    | X -> Y -> Z -> X         |
| `"Fdg"`      |                   | 0        | 1        | ...    |                          |
| `"T"`        |                   | 0        | 1        | ...    |                          |
| `"Tdg"`      |                   | 0        | 1        | ...    |                          |
| `"CX"`       | `"CNOT"`          | 0        | 2        | ...    |                          |
| `"CY"`       |                   | 0        | 2        | ...    |                          |
| `"CZ"`       |                   | 0        | 2        | ...    |                          |
| `"RXX"`      |                   | 1        | 2        | ...    | Rotation about XX        |
| `"RYY"`      |                   | 1        | 2        | ...    | Rotation about YY        |
| `"RZZ"`      | `"ZZPhase"`       | 1        | 2        | ...    | Rotation about ZZ        |
| `"R2XXYYZZ"` | `"RXXYYZZ"`       | 3        | 2        | ...    | RXX x RYY x RZZ          |
| `"SXX"`      |                   | 0        | 2        | ...    | Sqrt. of XX              |
| `"SXXdg"`    |                   | 0        | 2        | ...    | Adjoint of sqrt. of XX   |
| `"SYY"`      |                   | 0        | 2        | ...    | Sqrt. of YY              |
| `"SYYdg"`    |                   | 0        | 2        | ...    | Adjoint of sqrt. of YY   |
| `"SZZ"`      | `"ZZ"`, `"ZZMax"` | 0        | 2        | ...    | Sqrt. of ZZ              |
| `"SZZdg"`    |                   | 0        | 2        | ...    | Adjoint of sqrt. of ZZ   |
| `"SWAP"`     |                   | 0        | 2        | ...    | Swaps two qubits         |

## Machine operations

Machine operations (`"mop"`s) are operations that represent changes to the machine state such as the physical passage of
time or the movement of qubits as well as other aspects that are more directly related to a physical device although potentially
indirectly influencing the noise being applied via the error model.

The general form of `"mop"`s is:

```json5
{
  "mop": str,  // identifying name
  "args": [qubit_id, ... | [qubit_id, ... ], ...],  // optional
  "duration": [float, "s"|"ms"|"us"|"ns"],  // optional
  "metadata": {} // Optional metadata for potential auxiliary info or to be utilized by error models.
}
```

The `"duration"` field supports seconds (s), milliseconds (ms), microseconds (us), and nanoseconds (ns) as its units.

Currently, `"mop"`s are more defined by the implementation of the Machine and ErrorModel classes in PECOS. Therefore,
the `"metadata"` tag is heavily depended upon to supply values that these classes expect. An example of indicating
idling and transport include:

```json5
{
  "mop": "Idle",
  "args": [["q", 0], ["q", 5], ["w", 1] ],
  "duration": [0.000123, "s"] // typically in seconds
}
```

```json5
{
  "mop": "Transport",
  // potentially using "args" to indicate what qubits are being transported
  "duration": [0.5, "ms"]
  "metadata": {...} // potentially including what positions to and from qubits moved between or what path taken
}
```

The "Skip" `"mop"` is the empty operation that indicates *do nothing*. It is used in place of operations that will
have no effect on the machine state, such as the global phase operation.

```json5
{
  "mop": "Skip",
}
```

## Blocks

In the present version of PHIR/PECOS, blocks serve a dual purpose: they group operations and other blocks, and they
signify conditional operations and/or blocks. In the future, blocks may be utilized to represent more advanced control
flow. A notable aspect of blocks, is that they can encompass any other operation or block, offering a capability for
nesting.

### Basic block

The foundation block simply sequences operations and other blocks

```json5
{
  "block": "sequence",
  "ops": [{...}, ...],
  "metadata": {...}  // Optional
}
```

### QParallel block

A grouping of quantum operations to be performed in parallel.

```json5
{
  "block": "qparallel",
  "ops": [{...}, ...],
  "metadata": {...}  // Optional
}
```

The following example contains 6 RZ gate applications. There is 1 `"qop"` per unique gate angle, each with 2 qubit arguments.
All gates within the block will be applied in parallel.

```json5
{
  "block": "qparallel",
  "ops": [{"qop": "RZ", "angles": [[1.5], "pi"], "args": [["q", 0], ["q", 1]]},
          {"qop": "RZ", "angles": [[1.0], "pi"], "args": [["q", 2], ["q", 3]]},
          {"qop": "RZ", "angles": [[0.5], "pi"], "args": [["q", 4], ["q", 5]]}
          ]
}
```

### If/else block

An if-else block:

```json5
{
  "block": "if",
  "condition": {},
  "true_branch": [{...}, ...],
  "false_branch": [{...}, ...] // This is optional and should only be include if an 'else' branch exists.
}
```

The `"condition"` field houses a classical expression representable in PHIR. However, it's noteworthy that the extended
OpenQASM 2.0 restricts conditions to direct comparisons between classical variables or bits and integer literals. For
instance, when translating from extended OPenQASM 2.0, acceptable conditions would be `if(a > 3) ops` or
`if(a[0]>=1) ops;`. The extended OpenQASM 2.0 language explicitly avoids permitting multiple comparisons or
bitwise/logical operations, or comparisons between two variables and/or bits. In execution, if a comparison evaluates to
0, PECOS will initiate the `"false_branch"`; otherwise, the `"true_branch"` will be triggered.

*Note:* While PHIR/PECOS can effectively manage nested if/else statements, extended OpenQASM 2.0 strictly permits only
non-nested if statements. Consequently, such nesting should be sidestepped when converting from OpenQASM 2.0 to PHIR.

## Meta Instructions

Instructions that communicate information such as a compiler hints and debugging commands that have influence beyond
a quantum program.

### Barrier

A barrier instruction provides a hint to the compiler/emulator that qubits involved in barrier may not be optimized or
parallelized across the barrier. Effectively, it enforces an ordering in time for how quantum state is manipulated by
the machine.

```json5
{
  "meta": "barrier",
  "args": [qubit_id, ...] // list of qubit IDs
}
```

## Overall PHIR Example with Quantinuum's Extended OpenQASM 2.0

A simple quantum program might look like:

```qasm
OPENQASM 2.0;
include "hqslib1.inc";

qreg q[2];
qreg w[3];
qreg d[5];
creg m[2];
creg a[32];
creg b[32];
creg c[12];
creg d[10];
creg e[30];
creg f[5];
creg g[32];

h q[0];
CX q[0], q[1];

measure q -> m;

b = 5;
c = 3;

a[0] = add(b, c);  // FF call, e.g., Wasm call
if(m==1) a = (c[2] ^ d) | (e - 2 + (f & g));

if(m==2) sub(d, e);  // Conditioned void FF call. Void calls are assumed to update a separate classical state
// running asynchronously/in parallel.

if(a > 2) c = 7;
if(a > 2) x w[0];
if(a > 2) h w[1];
if(a > 2) CX w[1], w[2];
if(a > 2) measure w[1] -> g[0];
if(a > 2) measure w[2] -> g[1];

if(a[3]==1) h d;
measure d -> f;
```

Here is an equivalent version of the program using PHIR.

<!-- markdownlint-disable MD013 -->
```json5
{
  "format": "PHIR/JSON",
  "version": "0.1.0",
  "metadata": {
    "program_name": "example_prog",
    "description": "Program showing off PHIR",
    "num_qubits": 10
  },

  "ops": [
    {"//": "qreg q[2];"},
    {"//": "qreg w[3];"},
    {"//": "qreg d[5];"},
    {
      "data": "qvar_define",
      "data_type": "qubits",
      "variable": "q",
      "size": 2
    },
    {
      "data": "qvar_define",
      "data_type": "qubits",
      "variable": "w",
      "size": 3
    },
    {
      "data": "qvar_define",
      "data_type": "qubits",
      "variable": "d",
      "size": 5
    },

    {"//": "creg m[2];"},
    {"//": "creg a[32];"},
    {"//": "creg b[32];"},
    {"//": "creg c[12];"},
    {"//": "creg d[10];"},
    {"//": "creg e[30];"},
    {"//": "creg f[5];"},
    {"//": "creg g[32];"},
    {
      "data": "cvar_define",
      "data_type": "i64",
      "variable": "m",
      "size": 2
    },
    {
      "data": "cvar_define",
      "data_type": "i64",
      "variable": "a",
      "size": 32
    },
    {
      "data": "cvar_define",
      "data_type": "i64",
      "variable": "b",
      "size": 32
    },
    {
      "data": "cvar_define",
      "data_type": "i64",
      "variable": "c",
      "size": 12
    },
    {
      "data": "cvar_define",
      "data_type": "i64",
      "variable": "d",
      "size": 10
    },
    {
      "data": "cvar_define",
      "data_type": "i64",
      "variable": "e",
      "size": 30
    },
    {
      "data": "cvar_define",
      "data_type": "i64",
      "variable": "f",
      "size": 5
    },
    {
      "data": "cvar_define",
      "data_type": "i64",
      "variable": "g",
      "size": 32
    },

    {"//": "h q[0];"},
    {
      "qop": "H",
      "args": [ ["q", 0] ]
    },

    {"//": "CX q[0], q[1];"},
    {
      "qop": "CX",
      "args": [ [["q", 0], ["q", 1]] ]
    },

    {"//": "measure q -> m;"},
    {
      "qop": "Measure",
      "args": [ ["q", 0], ["q", 1] ],
      "returns": [ ["m", 0], ["m", 1] ]
    },

    {"//": "b = 5;"},
    {"cop": "=", "args": [5], "returns": ["b"]},

    {"//": "c = 3;"},
    {"cop": "=", "args": [3], "returns": ["c"]},

    {"//": "a[0] = add(b, c);  // FF call, e.g., Wasm call"},
    {
      "cop": "ffcall",
      "function": "add",
      "args": ["b", "c"],
      "returns": [ ["a", 0] ]
    },

    {"//": "if(m==1) a = (c[2] ^ d) | (e - 2 + (f & g));"},
    {
      "block": "if",
      "condition": {"cop": "==", "args": ["m", 1]},
      "true_branch": [{
        "cop": "=",
        "args": [{"cop": "|",
          "args": [
            {"cop": "^", "args": [["c", 2], "d"]},
            {"cop": "+", "args": [
              {"cop": "-", "args": ["e", 2]},
              {"cop": "&", "args": ["f", "g"]}
            ]}
          ]
        }],
        "returns": ["a"]
      }]
    },

    {"//": "if(m==2) sub(d, e);  // Conditioned void FF call. Void calls are assumed to update a separate classical state running asynchronously/in parallel."},
    {
      "block": "if",
      "condition": {"cop": "==", "args": ["m", 2]},
      "true_branch": [{
        "cop": "ffcall",
        "function": "sub",
        "args": ["d", "e"]
      }]
    },


    {"//": "if(a > 2) c = 7;"},
    {"//": "if(a > 2) x w[0];"},
    {"//": "if(a > 2) h w[1];"},
    {"//": "if(a > 2) CX w[1], w[2];"},
    {"//": "if(a > 2) measure w[1] -> g[0];"},
    {"//": "if(a > 2) measure w[2] -> g[1];"},
    {
      "block": "if",
      "condition": {"cop": ">", "args": ["a", 2]},
      "true_branch": [
        {
          "cop": "=",
          "args": [7],
          "returns": ["c"]
        },
        {
          "qop": "X",
          "args": [ ["w", 0] ]
        },
        {
          "qop": "H",
          "args": [ ["w", 1] ]
        },
        {
          "qop": "CX",
          "args": [ [["w", 1], ["w", 2]] ]
        },
        {
          "qop": "Measure",
          "args": [ ["w", 1], ["w", 2] ],
          "returns": [ ["g", 0], ["g", 1] ]
        }
      ]
    },


    {"//": "if(a[3]==1) h d;"},
    {
      "block": "if",
      "condition": {"cop": "==", "args": [ ["a", 3], 1]},
      "true_branch": [
        {
          "qop": "H",
          "args": [ ["d", 0], ["d", 1], ["d", 2], ["d", 3], ["d", 4] ]
        }
      ]
    },

    {"//": "measure d -> f;"},
    {
      "qop": "Measure",
      "args": [ ["d", 0], ["d", 1], ["d", 2], ["d", 3], ["d", 4] ],
      "returns": [ ["f", 0], ["f", 1], ["f", 2], ["f", 3], ["f", 4] ]
    },


    {
      "data": "cvar_export",
      "variables": ["m", "a", "b", "c", "d", "e", "f", "g"]
    }
  ]
}
```
<!-- markdownlint-enable MD013 -->
