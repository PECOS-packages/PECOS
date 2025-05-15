# Unified Include System

This document describes the unified include system implemented for the QASM parser.

## Overview

The QASM parser now uses a unified include resolution system that treats all includes consistently, regardless of their source (virtual/memory, filesystem, or system).

## Priority Order

When resolving an include file, the system searches in this order:

1. **User Virtual Includes** (highest priority)
   - Includes added programmatically via `ParseConfig.virtual_includes`
   - These override any other includes with the same name

2. **Filesystem Includes**
   - Files found in paths specified via `ParseConfig.include_paths`
   - Searched in the order paths were added

3. **System Includes** (lowest priority)
   - Built-in includes like `qelib1.inc` and `pecos.inc`
   - Always available unless overridden by higher priority includes

## Key Benefits

1. **Consistency**: All includes are handled the same way, regardless of source
2. **Override Capability**: Users can override system includes (like `qelib1.inc`) with their own versions
3. **Flexibility**: Mix and match includes from different sources in a single QASM file
4. **Simplicity**: Unified API through `IncludeResolver` class

## Example Usage

```rust
use pecos_qasm::{ParseConfig, QASMParser};

// Create config with custom virtual include
let mut config = ParseConfig::default();

// Add a virtual include that overrides system qelib1.inc
config.virtual_includes.push((
    "qelib1.inc".to_string(),
    "gate h a { U(pi/2,0,pi) a; }".to_string()
));

// Add filesystem search paths
config.include_paths.push("/custom/includes".into());

// Parse with custom configuration
let program = QASMParser::parse_with_config(qasm_source, config)?;
```

## Architecture

The system consists of:

1. **IncludeResolver**: Core component that manages include resolution
   - Maintains priority ordering
   - Handles circular dependency detection
   - Caches resolved includes

2. **Preprocessor**: Uses IncludeResolver to process QASM source
   - Handles include statement parsing
   - Performs recursive include resolution

3. **ParseConfig**: Configuration struct for parser
   - `virtual_includes`: User-provided in-memory includes
   - `include_paths`: Filesystem paths to search

## Migration

The following convenience methods were removed in favor of the unified approach:
- `parse_str_with_includes`
- `parse_str_with_virtual_includes`
- `parse_str_with_include_paths`
- etc.

Now use `ParseConfig` with `parse_with_config()` for all custom include scenarios.