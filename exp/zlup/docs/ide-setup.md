# Zlup IDE Setup

This guide covers setting up IDE support for Zlup development, including syntax highlighting and LSP features (diagnostics, hover, completions).

## Building the LSP Server

The Zlup LSP server (`zlups`) provides diagnostics, hover information, and basic completions.

```bash
# From the repository root
cargo build --features "cli lsp" --bin zlups --release

# The binary will be at:
# target/release/zlups
```

## Neovim

### Prerequisites

- Neovim 0.8+ with LSP support
- [nvim-lspconfig](https://github.com/neovim/nvim-lspconfig)

### Setup

1. **Register the filetype** in your `init.lua`:

```lua
vim.filetype.add {
  extension = {
    zlp = 'zlup',
  },
}
```

2. **Configure the LSP** in your plugin configuration (e.g., `lua/custom/plugins/zlup.lua`):

```lua
return {
  'neovim/nvim-lspconfig',
  config = function()
    local lspconfig = require 'lspconfig'
    local configs = require 'lspconfig.configs'

    -- Register zlups as a custom LSP server
    if not configs.zlups then
      configs.zlups = {
        default_config = {
          cmd = { '/path/to/PECOS-alt/target/release/zlups' },
          filetypes = { 'zlup' },
          root_dir = lspconfig.util.root_pattern('zlup.toml', '.git'),
          settings = {},
        },
      }
    end

    -- Start the server
    lspconfig.zlups.setup {}
  end,
}
```

Replace `/path/to/PECOS-alt` with the actual path to your PECOS repository.

3. **Verify** by opening a `.zlp` file and running `:LspInfo`. You should see `zlups` attached.

### Troubleshooting

- **LSP not starting**: Check that the `zlups` binary exists and is executable
- **No diagnostics**: Restart the LSP with `:LspRestart zlups`
- **Check logs**: `:LspLog` shows LSP communication logs

## JetBrains IDEs (RustRover, IntelliJ, CLion, etc.)

There are two options for JetBrains IDE support:

### Option 1: LSP4IJ Plugin (Recommended)

[LSP4IJ](https://plugins.jetbrains.com/plugin/23257-lsp4ij) is a generic LSP client plugin that works with any JetBrains IDE.

1. **Install LSP4IJ** from the JetBrains Marketplace:
   - Settings → Plugins → Marketplace → Search "LSP4IJ" → Install

2. **Configure the LSP server**:
   - Settings → Languages & Frameworks → LSP4IJ → Language Servers
   - Add a new server:
     - **Name**: `zlups`
     - **Command**: `/path/to/PECOS-alt/target/release/zlups`
     - **File patterns**: `*.zlp`

3. **Associate the file type**:
   - Settings → Editor → File Types
   - Add `*.zlp` to a text-based file type (or create a new one called "Zlup")

### Option 2: Zlup Plugin (Syntax Highlighting Only)

A basic JetBrains plugin is available at `exp/zlup/editors/jetbrains-zlup/`. This provides syntax highlighting but not LSP features.

#### Building the Plugin

```bash
cd exp/zlup/editors/jetbrains-zlup
./gradlew buildPlugin

# The plugin ZIP will be at:
# build/distributions/jetbrains-zlup-0.1.0.zip
```

#### Installing

1. Settings → Plugins → Gear icon → Install Plugin from Disk
2. Select `jetbrains-zlup-0.1.0.zip`
3. Restart the IDE

#### Features

- Syntax highlighting for keywords, types, gates, comments
- File type registration for `.zlp` files

For full LSP support (diagnostics, hover), combine with LSP4IJ.

## VS Code

### Quick Setup (No Extension Required)

You can use a generic LSP client extension to get Zlup support without creating a custom extension.

#### Using vscode-glspc (Generic LSP Client)

1. **Install** [Generic LSP Client](https://marketplace.visualstudio.com/items?itemName=AaaronSun.vscode-glspc) from the marketplace

2. **Add to your `settings.json`**:

```json
{
  "glspc.serverPath": "/path/to/PECOS-alt/target/release/zlups",
  "glspc.languageId": "zlup",
  "files.associations": {
    "*.zlp": "zlup"
  }
}
```

#### Using lsp-client Extension

Alternatively, use [LSP Client](https://marketplace.visualstudio.com/items?itemName=AperiodicSierra.lsp-client):

1. **Install** the extension
2. **Configure in `settings.json`**:

```json
{
  "lsp-client.serverCommand": "/path/to/PECOS-alt/target/release/zlups",
  "lsp-client.fileExtensions": [".zlp"],
  "files.associations": {
    "*.zlp": "plaintext"
  }
}
```

### Basic Syntax Highlighting

For syntax highlighting without a full extension, add to `settings.json`:

```json
{
  "editor.tokenColorCustomizations": {
    "textMateRules": [
      {
        "scope": "keyword.control.zlup",
        "settings": { "foreground": "#C586C0" }
      }
    ]
  }
}
```

For full syntax highlighting, a TextMate grammar or custom extension would be needed.

### Creating a Full Extension (Advanced)

For a complete VS Code extension with syntax highlighting and LSP:

1. Use `yo code` to scaffold a language extension
2. Add a TextMate grammar for `.zlp` files
3. Configure the LSP client in `extension.ts`:

```typescript
import * as vscode from 'vscode';
import { LanguageClient, LanguageClientOptions, ServerOptions } from 'vscode-languageclient/node';

export function activate(context: vscode.ExtensionContext) {
  const serverOptions: ServerOptions = {
    command: '/path/to/PECOS-alt/target/release/zlups',
  };

  const clientOptions: LanguageClientOptions = {
    documentSelector: [{ scheme: 'file', language: 'zlup' }],
  };

  const client = new LanguageClient('zlups', 'Zlup Language Server', serverOptions, clientOptions);
  client.start();
}
```

Contributions for a full VS Code extension are welcome!

## LSP Features

The `zlups` server currently provides:

| Feature | Status |
|---------|--------|
| Diagnostics (errors/warnings) | Supported |
| Hover (type information) | Supported |
| Go to Definition | Supported |
| Completions | Context-aware |
| Formatting | Supported |

### Feature Details

**Go to Definition**: Jump to where variables, functions, and types are defined. Use `gd` in Neovim or Ctrl+Click in JetBrains/VS Code.

**Context-aware Completions**:
- After `.` on allocators: suggests `child()`, `release()`, etc.
- After `:` or `->`: suggests types (`u32`, `void`, `bool`, etc.)
- General context: keywords, quantum gates, built-in functions

**Formatting**: Formats code with consistent indentation and spacing. Use `:lua vim.lsp.buf.format()` in Neovim or the IDE's format command.

## Development

### Rebuilding After Changes

When modifying the parser, semantic analyzer, or LSP server:

```bash
cargo build --features "cli lsp" --bin zlups --release
```

Then restart the LSP in your editor:
- Neovim: `:LspRestart zlups`
- JetBrains: Restart the IDE or disable/enable the LSP server

### Testing LSP

A test file is provided at `exp/zlup/examples/test_lsp.zlp` for verifying LSP functionality.
