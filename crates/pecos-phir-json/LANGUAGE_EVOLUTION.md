# PHIR-JSON Language Evolution Strategy

This document outlines how the PHIR-JSON format evolves and what to expect from different types of changes.

## Versioning & Changes

- **Major Version** (0.1 → 0.2): Breaking changes
  - Creates a new implementation module
  - Preserves old implementations for backward compatibility
  - Provides migration documentation
  - We'll make breaking changes when needed to improve the language

- **Minor Version** (0.1.0 → 0.1.1): Non-breaking additions
  - Extends existing implementation
  - Maintains complete compatibility with previous minor versions

- **Preview Versions**: Early access to upcoming major versions
  - No stability guarantees between releases
  - Access through `setup_phir_json_engine_with_preview()`

- **Experimental Features**: Features being explored
  - May change or disappear at any time

## Feature Flags

Cargo features control which versions and capabilities are available:

- **Stable versions**: `v0_1`, `v0_2` - Enable specific stable versions
- **Preview versions**: `preview-v0_3` - Enable upcoming major versions
- **Experimental features**: `experimental-X` - Enable specific experimental features
- **Convenience groups**: `all-versions`, `all-preview`, `all` - Enable groups of features

Example in Cargo.toml:
```toml
# Use preview version
pecos-phir-json = { version = "0.1", features = ["preview-v0_3"] }

# Use stable version with experimental feature
pecos-phir-json = { version = "0.1", features = ["v0_1", "experimental-blocks"] }
```

## For Users

- **Stable code**: Use default features and `setup_phir_json_engine()`
- **Testing new versions**: Enable preview flags and use `setup_phir_json_engine_with_preview()`
- **Specific version**: Use version-specific functions (e.g., `setup_phir_v0_1_engine()`)

## For Developers

- **Non-breaking changes**: Extend existing version implementation
- **Breaking changes**: Create new version modules
- **Always**: Update documentation and add tests

## Support Policy

While we strive to minimize breaking changes between major versions, we will introduce them when necessary to improve
the language design, fix fundamental issues, or enable important new capabilities. When breaking changes are introduced,
we'll clearly document what breaks and why, along with migration guidance for users.
