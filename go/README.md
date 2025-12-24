# PECOS Go Bindings

Go bindings for the PECOS quantum error correction simulator.

## Structure

- `pecos-go-ffi/` - Rust crate that exports C FFI functions
- `pecos/` - Go package that wraps the FFI

## Building

### 1. Build the Rust library

```bash
cd go/pecos-go-ffi
cargo build --release
```

This creates `libpecos_go.so` (Linux), `libpecos_go.dylib` (macOS), or `pecos_go.dll` (Windows) in `target/release/`.

### 2. Set library path

**Linux:**
```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)/target/release
```

**macOS:**
```bash
export DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:$(pwd)/target/release
```

### 3. Run Go tests

```bash
cd go/pecos
go test -v
```

## Usage

```go
package main

import (
    "fmt"
    "github.com/PECOS-packages/PECOS/go/pecos"
)

func main() {
    // Get version
    fmt.Println(pecos.Version())

    // Create a QubitId
    q := pecos.NewQubitId(0)
    fmt.Printf("Created: %s\n", q)
    fmt.Printf("Index: %d\n", q.Index())

    // Simple FFI test
    result := pecos.AddTwoNumbers(40, 2)
    fmt.Printf("40 + 2 = %d\n", result)
}
```

## Current Status

This is a proof-of-concept demonstrating the FFI bridge between Rust and Go. Currently exports:

- `Version()` - Get PECOS version string
- `NewQubitId(index)` - Create a QubitId
- `QubitId.Index()` - Get qubit index
- `QubitId.String()` - String representation
- `AddTwoNumbers(a, b)` - Test function

More functionality can be added by extending the FFI in `pecos-go-ffi/src/lib.rs` and the Go wrapper in `pecos/pecos.go`.
