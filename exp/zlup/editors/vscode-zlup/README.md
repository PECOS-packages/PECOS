# Zlup for Visual Studio Code

Language support for Zlup - a quantum programming language.

## Features

- **Syntax Highlighting**: Full syntax highlighting for Zlup source files (.zlp)
- **Code Snippets**: Quick snippets for common patterns
- **Bracket Matching**: Automatic bracket and brace matching
- **Code Folding**: Fold functions and blocks
- **Commenting**: Toggle line and block comments

## Supported Constructs

### Quantum Gates
- Single-qubit: `h`, `x`, `y`, `z`, `s`, `t`, `rx`, `ry`, `rz`
- Two-qubit: `cx`, `cz`, `swap`, `iswap`, `rzz`
- Three-qubit: `ccx`
- Measurement: `mz`, `mx`, `my`
- Preparation: `pz`, `px`, `py`

### Types
- Integers: `u8`, `u16`, `u32`, `u64`, `i8`, `i16`, `i32`, `i64`
- Floats: `f32`, `f64`
- Angles: `a64`
- Quantum: `qubit`, `bit`, `qalloc`

### Control Flow
- `if`/`else`
- `for`/`while`
- `tick` (parallel quantum ops)
- `barrier`

## Snippets

| Prefix | Description |
|--------|-------------|
| `main` | Main function |
| `fn` | Function definition |
| `qalloc` | Qubit allocation |
| `bell` | Bell state preparation |
| `ghz` | GHZ state preparation |
| `for` | For loop |
| `if`/`ife` | If/if-else statement |
| `tick` | Tick block |
| `meas` | Typed measurement |

## Installation

### From VSIX
1. Download the `.vsix` file
2. In VS Code: Extensions > ... > Install from VSIX

### From Source
```bash
cd vscode-zlup
npm install
npm run package
```

## Example

```zlup
pub fn main() -> unit {
    q := qalloc(2);
    pz q;
    h q[0];
    cx (q[0], q[1]);
    results: [2]u1 = mz([2]u1) [q[0], q[1]];
    return;
}
```
