# Zlup JetBrains Plugin

IntelliJ IDEA / JetBrains IDE plugin for the Zlup quantum programming language.

## Features

- Syntax highlighting for `.zlp` files
- Comment toggling (`Ctrl+/`)
- Brace matching

## LSP Support

For full LSP features (diagnostics, completion, hover), install the **LSP4IJ** plugin from the JetBrains Marketplace, then configure it to use the `zlups` language server.

### LSP4IJ Configuration

1. Install LSP4IJ from Marketplace
2. Go to **Settings > Languages & Frameworks > Language Servers**
3. Click **+** to add a new server:
   - **Name**: `Zlups`
   - **Command**: `/path/to/PECOS-alt/target/release/zlups`
   - **File patterns**: `*.zlp`

## Building the Plugin

### Requirements
- JDK 17 or later

### Build Commands

```bash
cd editors/jetbrains-zlup
./gradlew buildPlugin
```

The plugin ZIP will be in `build/distributions/`.

## Installation

### From pre-built ZIP

1. In your JetBrains IDE: **Settings > Plugins > Gear icon > Install Plugin from Disk...**
2. Select `jetbrains-zlup-0.1.0.zip`
3. Restart the IDE

### Development mode

Run the plugin in a sandbox IDE:
```bash
./gradlew runIde
```

## Building zlups (LSP server)

```bash
cd /path/to/PECOS-alt/exp/zlup
cargo build --features lsp --release
```

The binary will be at `target/release/zlups`.
