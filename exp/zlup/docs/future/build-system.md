# Build System Design

This document outlines Zlup's build system, following Zig's philosophy: **the build system IS the language**.

## Philosophy

Like Zig's `build.zig`, Zlup uses `build.zlp` - a Zlup program that runs at compile time to configure the build. No separate DSL, no YAML, no TOML for build logic - just Zlup with comptime.

**Why?**
- **One language to learn**: Build logic uses the same syntax as regular code
- **Full power of comptime**: Conditional compilation, code generation
- **Type-safe configuration**: Compiler catches config errors
- **Debuggable**: Use the same tools to debug build scripts

## Basic Structure

### Project Layout

```
my-qec-project/
├── build.zlp           # Build configuration (Zlup code)
├── zlup.toml           # Simple metadata (name, version, deps)
├── src/
│   ├── main.zlp        # Entry point
│   └── lib/
│       └── syndrome.zlp
├── tests/
│   └── test_syndrome.zlp
└── ffi/
    └── decoder/        # Rust decoder crate
        ├── Cargo.toml
        └── src/lib.rs
```

### zlup.toml (Metadata Only)

Simple metadata that doesn't need comptime logic:

```toml
[package]
name = "my-qec-project"
version = "0.1.0"
authors = ["Alice <alice@example.com>"]
license = "Apache-2.0"

[dependencies]
# External Zlup packages (future)
# other-package = "0.1.0"

[ffi]
# Rust crates to build and link
decoder = { path = "ffi/decoder" }
```

### build.zlp (Build Logic)

```zlup_nocheck
//! Build configuration for my-qec-project
//!
//! This file runs at compile time to configure the build.

std := @import("std");
Build := @import("build");

pub fn build(b: *Build) -> unit {
    // Get target and optimization from CLI or defaults
    target := b.standardTargetOptions(.{});
    optimize := b.standardOptimizeOption(.{});

    // Main executable
    exe := b.addExecutable(.{
        name: "qec-sim",
        root_source: "src/main.zlp",
        target: target,
        optimize: optimize,
    });

    // Link Rust FFI library
    exe.linkLibrary("decoder");
    exe.addLibraryPath("ffi/decoder/target/release");

    // Install artifact
    b.installArtifact(exe);

    // Test step
    tests := b.addTest(.{
        root_source: "tests/test_syndrome.zlp",
    });

    test_step := b.step("test", "Run unit tests");
    test_step.dependOn(&tests.step);

    return;
}
```

## Build API

### Build Context

```zlup_nocheck
Build := struct {
    // Target configuration
    pub fn standardTargetOptions(&self, options: TargetOptions) -> Target { ... }
    pub fn standardOptimizeOption(&self, options: OptimizeOptions) -> Optimize { ... }

    // Add build artifacts
    pub fn addExecutable(&self, options: ExecutableOptions) -> *Executable { ... }
    pub fn addLibrary(&self, options: LibraryOptions) -> *Library { ... }
    pub fn addTest(&self, options: TestOptions) -> *Test { ... }

    // Build steps
    pub fn step(&self, name: []const u8, description: []const u8) -> *Step { ... }
    pub fn installArtifact(&self, artifact: *Artifact) -> unit { ... }

    // Options from command line
    pub fn option(&self, comptime T: type, name: []const u8, description: []const u8) -> ?T { ... }
};
```

### Executable Options

```zlup_nocheck
ExecutableOptions := struct {
    name: []const u8,
    root_source: []const u8,
    target: ?Target = none,
    optimize: ?Optimize = none,
    strict: bool = false,  // NASA Power of 10 strict mode
};
```

### Conditional Compilation

```zlup_nocheck
pub fn build(b: *Build) -> unit {
    // User-defined option
    enable_noise := b.option(bool, "noise", "Enable noise modeling") orelse false;

    exe := b.addExecutable(.{
        name: "qec-sim",
        root_source: "src/main.zlp",
    });

    // Conditional compilation flags
    if enable_noise {
        exe.addDefine("ENABLE_NOISE", "1");
        exe.linkLibrary("noise-model");
    }

    // Platform-specific
    if b.target.os == .linux {
        exe.linkSystemLibrary("pthread");
    }

    return;
}
```

### Code Generation

```zlup_nocheck
pub fn build(b: *Build) -> unit {
    // Generate lookup table at build time
    table := b.addGeneratedFile("syndrome_table.zlp");
    table.generator = generate_syndrome_table;

    exe := b.addExecutable(.{
        name: "decoder",
        root_source: "src/main.zlp",
    });
    exe.addModule("syndrome_table", table);

    return;
}

fn generate_syndrome_table(writer: *Writer) -> unit {
    writer.print("// Auto-generated syndrome lookup table\n");
    writer.print("pub table: [256]u8 = [\n");

    for i in 0..256 {
        correction := comptime compute_correction(i);
        writer.print("    {},\n", correction);
    }

    writer.print("];\n");
    return;
}
```

## CLI Integration

```bash
# Build using build.zlp
zlup build

# Build with options
zlup build -Dnoise=true -Doptimize=release

# Run tests
zlup build test

# Run specific step
zlup build run

# Show available steps and options
zlup build --help
```

## Rust FFI Integration

The build system handles Rust crate compilation:

```zlup_nocheck
pub fn build(b: *Build) -> unit {
    // Rust decoder crate
    decoder := b.addRustLibrary(.{
        name: "decoder",
        path: "ffi/decoder",
        profile: if b.optimize == .release { "release" } else { "debug" },
    });

    exe := b.addExecutable(.{
        name: "qec-sim",
        root_source: "src/main.zlp",
    });

    // Link the Rust library
    exe.linkRustLibrary(decoder);

    return;
}
```

This runs `cargo build` on the Rust crate and links the resulting `.a`/`.so`.

## Multi-Target Builds

```zlup_nocheck
pub fn build(b: *Build) -> unit {
    targets := [_]Target{
        .{ .os = .linux, .arch = .x86_64 },
        .{ .os = .macos, .arch = .aarch64 },
        .{ .os = .windows, .arch = .x86_64 },
    };

    for target in targets {
        exe := b.addExecutable(.{
            name: f"qec-sim-{target.os}-{target.arch}",
            root_source: "src/main.zlp",
            target: target,
        });
        b.installArtifact(exe);
    }

    return;
}
```

## Comparison with Alternatives

| Approach | Pros | Cons |
|----------|------|------|
| **build.zlp (Zlup)** | Full language power, type-safe, debuggable | Need Zlup knowledge |
| build.zig (Zig) | Proven approach, powerful | Different language |
| Cargo.toml (Rust) | Simple, declarative | Limited logic |
| CMake | Cross-platform | Complex DSL |
| Make | Universal | Arcane syntax |

## Implementation Plan

1. **Phase 1**: Basic build.zlp parsing and execution
2. **Phase 2**: Executable and library targets
3. **Phase 3**: Test integration
4. **Phase 4**: Rust FFI library linking
5. **Phase 5**: Code generation support
6. **Phase 6**: Multi-target builds

## Example: Complete QEC Project

```zlup_nocheck
//! build.zlp for a surface code simulator

std := @import("std");
Build := @import("build");

pub fn build(b: *Build) -> unit {
    target := b.standardTargetOptions(.{});
    optimize := b.standardOptimizeOption(.{});

    // Options
    distance := b.option(u32, "distance", "Code distance") orelse 3;
    strict := b.option(bool, "strict", "NASA Power of 10 strict mode") orelse true;

    // Rust MWPM decoder
    mwpm := b.addRustLibrary(.{
        name: "mwpm",
        path: "ffi/mwpm-decoder",
    });

    // Main simulator
    sim := b.addExecutable(.{
        name: "surface-sim",
        root_source: "src/main.zlp",
        target: target,
        optimize: optimize,
        strict: strict,
    });
    sim.addDefine("CODE_DISTANCE", std.fmt.comptimePrint("{}", distance));
    sim.linkRustLibrary(mwpm);

    b.installArtifact(sim);

    // Tests
    tests := b.addTest(.{
        root_source: "tests/all.zlp",
        strict: strict,
    });
    tests.linkRustLibrary(mwpm);

    test_step := b.step("test", "Run all tests");
    test_step.dependOn(&tests.step);

    // Benchmark step
    bench := b.addExecutable(.{
        name: "bench",
        root_source: "bench/main.zlp",
        optimize: .release,
    });
    bench.linkRustLibrary(mwpm);

    bench_step := b.step("bench", "Run benchmarks");
    bench_step.dependOn(&bench.run());

    return;
}
```
