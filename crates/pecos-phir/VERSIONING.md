# PHIR Versioning Strategy

This document outlines the strategy for handling multiple versions of the PHIR (PECOS High-level Intermediate
Representation) specification in the codebase.

## Overview

PHIR is a versioned specification, with each version potentially introducing new features, changes, or improvements. To
maintain backward compatibility while allowing for evolution, this crate implements a versioning strategy that:

1. Isolates each version's implementation in its own module
2. Provides version detection at runtime
3. Uses a consistent interface across versions
4. Enables selective compilation via feature flags

## Directory Structure

```
crates/pecos-phir/
├── src/
│   ├── lib.rs                   # Main entry point with version detection
│   ├── common.rs                # Shared utilities across versions
│   ├── version_traits.rs        # Version-agnostic interfaces
│   ├── v0_1.rs                  # v0.1 module definition
│   ├── v0_1/                    # v0.1 implementation
│   │   ├── ast.rs               # AST definitions for v0.1
│   │   ├── engine.rs            # Engine implementation for v0.1
│   │   └── operations.rs        # Operation handling for v0.1
│   ├── v0_2.rs                  # v0.2 module definition (future)
│   └── v0_2/                    # v0.2 implementation (future)
│       ├── ast.rs
│       ├── engine.rs
│       └── operations.rs
└── specification/
    ├── v0.1/
    │   └── spec.md              # v0.1 specification document
    └── v0.2/                    # Future version specification
        └── spec.md
```

## Version Management

### Feature Flags

The crate uses Cargo feature flags to control which versions are included in the build:

```toml
[features]
default = ["v0_1"]
v0_1 = []
v0_2 = []
all-versions = ["v0_1", "v0_2"]
```

This allows users to:
- Use the default (latest stable) version
- Explicitly select specific versions
- Include all versions for compatibility testing

### Version Detection

At runtime, the crate detects which version of PHIR is being used by examining the "version" field in the input JSON:

```rust
pub fn detect_version(json: &str) -> Result<PHIRVersion, PecosError> {
    let value: serde_json::Value = serde_json::from_str(json)?;

    if let Some(version) = value.get("version").and_then(|v| v.as_str()) {
        match version {
            "0.1.0" => Ok(PHIRVersion::V0_1),
            "0.2.0" => Ok(PHIRVersion::V0_2),
            _ => Err(PecosError::Input(format!("Unsupported PHIR version: {}", version))),
        }
    } else {
        Err(PecosError::Input("Missing version field in PHIR program".into()))
    }
}
```

### Version-agnostic Interface

Each version implements a common trait that defines the interface:

```rust
pub trait PHIRImplementation {
    type Program;
    type Engine;

    fn parse_program(json: &str) -> Result<Self::Program, PecosError>;
    fn create_engine(program: Self::Program) -> Box<dyn ClassicalEngine>;
    // ... other common operations
}
```

This ensures that regardless of the version, the same operations can be performed through a consistent interface.

## User API

### Automatic Version Detection

The primary API uses automatic version detection:

```rust
// Automatically detect and handle the version based on the PHIR program
let engine = setup_phir_engine(path_to_phir_file)?;
```

### Explicit Version Selection

Users can also explicitly select a version:

```rust
// Explicitly use v0.1
let engine = setup_phir_v0_1_engine(path_to_phir_file)?;

// Explicitly use v0.2 (when available)
let engine = setup_phir_v0_2_engine(path_to_phir_file)?;
```

## Adding a New Version

When adding a new version of the PHIR specification:

1. **Create the specification document**:
   - Add a new directory under `specification/` (e.g., `v0.2/`)
   - Document the new features, changes, and compatibility concerns

2. **Implement the new version**:
   - Create a new module entry file (e.g., `v0_2.rs`)
   - Create a new directory for implementation details (e.g., `v0_2/`)
   - Implement the `PHIRImplementation` trait for the new version

3. **Update version detection**:
   - Add the new version to the `PHIRVersion` enum
   - Update the `detect_version()` function to recognize the new version

4. **Add feature flags**:
   - Add a new feature flag in `Cargo.toml`
   - Consider updating the `default` feature if appropriate

5. **Add tests**:
   - Create version-specific tests
   - Add compatibility tests if backward compatibility is important

## Versioning Policy

### When to Create a New Version

- **Major Changes**: New versions should be created for significant changes to the specification that break backward
  compatibility
- **Minor Additions**: Minor additions that don't break backward compatibility might be added to the current version
- **Bug Fixes**: Bug fixes should be applied to all supported versions

### Version Numbering

- **v0.x**: Pre-stable versions, may have breaking changes between minor versions
- **v1.x**: Stable versions, major version increments for breaking changes, minor for non-breaking additions

### Version Support

- The crate will aim to support at least the two most recent versions of the specification
- Deprecated versions will be clearly marked and eventually removed from the default build (but may still be available
  via feature flags)

## Conclusion

This versioning strategy allows PHIR to evolve while maintaining backward compatibility when needed. By isolating each
version's implementation and providing a consistent interface, we can support multiple versions of the specification
within a single codebase.
