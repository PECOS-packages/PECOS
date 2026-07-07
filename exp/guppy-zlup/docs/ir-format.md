# Guppy IR Format

The Guppy IR (Intermediate Representation) is a JSON format that represents
validated Guppy programs. It enables tool interoperability and caching of
validated programs.

## Schema Overview

```json
{
  "version": "0.1.0",
  "source_file": "path/to/source.py",
  "functions": [
    {
      "name": "function_name",
      "params": [...],
      "return_type": {...},
      "body": [...],
      "location": {...}
    }
  ]
}
```

## Top-Level Fields

| Field        | Type       | Required | Description                     |
|--------------|------------|----------|---------------------------------|
| `version`    | string     | Yes      | IR schema version ("0.1.0")     |
| `source_file`| string     | No       | Original source file path       |
| `functions`  | Function[] | Yes      | List of function definitions    |

## Function

```json
{
  "name": "bell_state",
  "params": [
    {"name": "n", "type": {"kind": "primitive", "name": "int"}}
  ],
  "return_type": {"kind": "primitive", "name": "None"},
  "body": [...],
  "is_pub": true,
  "location": {"line": 1, "column": 1, "end_line": 10, "end_column": 1}
}
```

| Field         | Type       | Required | Description                    |
|---------------|------------|----------|--------------------------------|
| `name`        | string     | Yes      | Function name                  |
| `params`      | Param[]    | No       | Function parameters            |
| `return_type` | TypeExpr   | No       | Return type annotation         |
| `body`        | Stmt[]     | Yes      | Function body statements       |
| `is_pub`      | boolean    | No       | Whether function is public     |
| `location`    | Location   | No       | Source location                |

## Parameter

```json
{"name": "q", "type": {"kind": "qalloc", "size": {"kind": "literal", "value": 4}}}
```

| Field  | Type     | Required | Description          |
|--------|----------|----------|----------------------|
| `name` | string   | Yes      | Parameter name       |
| `type` | TypeExpr | No       | Type annotation      |

## Type Expressions

### Primitive Types

```json
{"kind": "primitive", "name": "int"}
{"kind": "primitive", "name": "float"}
{"kind": "primitive", "name": "bool"}
{"kind": "primitive", "name": "str"}
{"kind": "primitive", "name": "None"}
```

### Qubit Allocation Type

```json
{"kind": "qalloc", "size": {"kind": "literal", "value": 4}}
{"kind": "qalloc", "size": {"kind": "ident", "name": "n"}}
```

### Array Type

```json
{"kind": "array", "element": {"kind": "primitive", "name": "int"}}
```

### Optional Type

```json
{"kind": "optional", "element": {"kind": "primitive", "name": "int"}}
```

### Named Type

```json
{"kind": "named", "name": "MyCustomType"}
```

## Statements

### Qubit Allocation

```json
{
  "kind": "qalloc",
  "name": "q",
  "size": {"kind": "literal", "value": 4}
}
```

Represents `q = qubit[4]`.

### Gate Application

```json
{
  "kind": "gate",
  "gate": "h",
  "targets": [
    {"kind": "index", "array": "q", "index": {"kind": "literal", "value": 0}}
  ]
}
```

Represents `h(q[0])`.

Supported gates: `h`, `x`, `y`, `z`, `t`, `tdg`, `sx`, `sy`, `sz`, `szdg`,
`rx`, `ry`, `rz`, `cx`, `cy`, `cz`, `swap`, `iswap`, `ccx`, `pz`

> **Note:** Gate names follow PECOS conventions. Use `sz` (S-gate, sqrt of Z) and `szdg` (S-dagger) rather than `s`/`sdg`.

For multi-qubit gates:

```json
{
  "kind": "gate",
  "gate": "cx",
  "targets": [
    {"kind": "index", "array": "q", "index": {"kind": "literal", "value": 0}},
    {"kind": "index", "array": "q", "index": {"kind": "literal", "value": 1}}
  ]
}
```

### Measurement

As a statement (result discarded):
```json
{
  "kind": "measure",
  "target": {"kind": "ident", "name": "q"}
}
```

As an assignment (result captured):
```json
{
  "kind": "assign",
  "target": {"kind": "ident", "name": "m"},
  "value": {
    "kind": "call",
    "callee": "measure",
    "args": [{"kind": "ident", "name": "q"}]
  }
}
```

**Note on Zlup measurement syntax:**

The generated Zlup supports flexible measurement syntax:

```zlup
// Per-qubit measurement into array
m := mz([4]u1) q;

// Single qubit measurement
bit := mz(u1) q[0];

// Pack into integer type
syndrome_byte := mz(pack u8) [q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7]];

// Pack into custom struct type
Syndrome := struct { x_parity: u1, z_parity: u1, flags: u2 };
syndrome := mz(pack Syndrome) [ancilla[0], ancilla[1], ancilla[2], ancilla[3]];
```

The `pack` modifier fills bits sequentially into the target type's bit layout (LSB first).

### Assignment

```json
{
  "kind": "assign",
  "target": {"kind": "ident", "name": "x"},
  "value": {"kind": "literal", "value": 42}
}
```

Target kinds: `ident`, `index`, `attr`, `tuple`

### For Loop

```json
{
  "kind": "for",
  "var": "i",
  "range": {
    "start": {"kind": "literal", "value": 0},
    "end": {"kind": "ident", "name": "n"}
  },
  "body": [...]
}
```

### If Statement

```json
{
  "kind": "if",
  "condition": {"kind": "binary", "left": ..., "op": "==", "right": ...},
  "then_body": [...],
  "else_body": [...]
}
```

### While Loop

```json
{
  "kind": "while",
  "condition": {"kind": "binary", ...},
  "body": [...]
}
```

### Return

```json
{
  "kind": "return",
  "return_value": {"kind": "ident", "name": "result"}
}
```

For entry/main functions, use `result()` instead of returning values.

### Result Emission

```json
{
  "kind": "result",
  "tag": "measurements",
  "value": {"kind": "ident", "name": "m"}
}
```

Represents `result("measurements", m)` - emits a tagged value to the quantum runtime.
Entry/main functions should use explicit `result()` calls rather than returning values.

## Expressions

### Literals

```json
{"kind": "literal", "value": 42}
{"kind": "literal", "value": 3.14}
{"kind": "literal", "value": "hello"}
{"kind": "literal", "value": true}
{"kind": "literal", "value": null}
```

### Identifier

```json
{"kind": "ident", "name": "variable_name"}
```

### Index Access

```json
{
  "kind": "index",
  "array": "q",
  "index": {"kind": "literal", "value": 0}
}
```

Or with expression base:
```json
{
  "kind": "index",
  "value": {"kind": "ident", "name": "arr"},
  "index": {"kind": "ident", "name": "i"}
}
```

### Binary Operations

```json
{
  "kind": "binary",
  "left": {"kind": "ident", "name": "a"},
  "op": "+",
  "right": {"kind": "literal", "value": 1}
}
```

Arithmetic operators: `+`, `-`, `*`, `/`, `//`, `%`, `**`, `<<`, `>>`, `|`, `^`, `&`, `@`

Comparison operators: `==`, `!=`, `<`, `<=`, `>`, `>=`, `is`, `is not`, `in`, `not in`

Both arithmetic and comparison operations use `"kind": "binary"`:

```json
{
  "kind": "binary",
  "left": {"kind": "ident", "name": "x"},
  "op": "<",
  "right": {"kind": "literal", "value": 10}
}

### Function Call

```json
{
  "kind": "call",
  "callee": "function_name",
  "args": [
    {"kind": "ident", "name": "arg1"},
    {"kind": "literal", "value": 42}
  ]
}
```

Or with expression callee:
```json
{
  "kind": "call",
  "func": {"kind": "field", "object": {...}, "field": "method"},
  "args": [...]
}
```

## Location

```json
{
  "line": 1,
  "column": 1,
  "end_line": 5,
  "end_column": 10
}
```

All fields are 1-indexed.

## Complete Example

```json
{
  "version": "0.1.0",
  "source_file": "bell.py",
  "functions": [
    {
      "name": "bell_state",
      "params": [],
      "return_type": {"kind": "primitive", "name": "None"},
      "body": [
        {
          "kind": "qalloc",
          "name": "q",
          "size": {"kind": "literal", "value": 2}
        },
        {
          "kind": "gate",
          "gate": "h",
          "targets": [
            {"kind": "index", "array": "q", "index": {"kind": "literal", "value": 0}}
          ]
        },
        {
          "kind": "gate",
          "gate": "cx",
          "targets": [
            {"kind": "index", "array": "q", "index": {"kind": "literal", "value": 0}},
            {"kind": "index", "array": "q", "index": {"kind": "literal", "value": 1}}
          ]
        },
        {
          "kind": "assign",
          "target": {"kind": "ident", "name": "m"},
          "value": {
            "kind": "call",
            "callee": "measure",
            "args": [{"kind": "ident", "name": "q"}]
          }
        },
        {
          "kind": "result",
          "tag": "measurements",
          "value": {"kind": "ident", "name": "m"}
        }
      ],
      "location": {"line": 1, "column": 1, "end_line": 6, "end_column": 13}
    }
  ]
}
```
