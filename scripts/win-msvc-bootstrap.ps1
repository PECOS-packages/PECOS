# Copyright 2025 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License. You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

# Local-capable Windows MSVC bootstrap for the Cargo build path.
#
# `just` pins `set shell := ["bash", "-cu"]`, so on Windows every recipe -- and
# the `cargo` / `link.exe` it spawns -- runs under git-bash, whose MSYS2 runtime
# (A) shadows MSVC `link.exe` with GNU coreutils `/usr/bin/link` on PATH and
# (B) mangles the semicolon/space/paren `LIB`/`INCLUDE` lists at the
# bash->native spawn. Both dissolve if the linker and LIB/INCLUDE live in
# `.cargo/config.toml`, which cargo reads *after* it starts -- bypassing the
# mangled shell env. This is the single, machine-local mechanism that makes
# `just build` behave identically on a developer's Windows box and in CI.
#
# This script is the ONLY writer of the `[target.x86_64-pc-windows-msvc]` table
# and the MSVC subset of `[env]` (LIB/INCLUDE/LIBPATH). It does a *scoped* merge
# so it never disturbs the keys the Rust writers own (LLVM_SYS_140_PREFIX,
# CUQUANTUM_ROOT) or any other table, and never emits a duplicate `[env]`.
#
# Inert (exit 0) on non-Windows so it is safe as an unconditional just prereq.
#
# PowerShell 7 (pwsh) is a hard requirement -- it is already the de facto repo
# requirement (every workflow uses `shell: pwsh`). Asserted below rather than
# papered over with a Windows-PowerShell-5.1 fallback.

#requires -Version 7.0

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if (-not $IsWindows) {
    exit 0
}

$Triple = "x86_64-pc-windows-msvc"
$StaleLinkerEnv = "CARGO_TARGET_X86_64_PC_WINDOWS_MSVC_LINKER"
$OwnedEnvKeys = @("LIB", "INCLUDE", "LIBPATH")

# Cargo env vars outrank .cargo/config.toml, so a stale pin would silently
# defeat everything written here. Fail loudly rather than mislead.
$pinned = [Environment]::GetEnvironmentVariable($StaleLinkerEnv)
if ($pinned) {
    throw "$StaleLinkerEnv is set ($pinned); it overrides the generated .cargo/config.toml linker. Unset it and re-run."
}

$repoRoot = Split-Path -Parent $PSScriptRoot
$cargoDir = Join-Path $repoRoot ".cargo"
$configPath = Join-Path $cargoDir "config.toml"

function Test-LibResolvesKernel32([string]$lib) {
    foreach ($dir in ($lib -split ';')) {
        $d = $dir.Trim()
        if ($d -and (Test-Path -LiteralPath (Join-Path $d "kernel32.lib"))) {
            return $true
        }
    }
    return $false
}

# Fast path: this script is a prerequisite of many recipes, so it runs several
# times per CI job (and per local build). vswhere + VsDevCmd cost ~10s each.
# If a prior run already wrote a config whose linker still exists and whose
# LIB still resolves kernel32.lib, the toolchain is unchanged -- skip the
# expensive discovery entirely. It self-heals: if VS moved or was removed the
# linker path is gone, so we fall through and regenerate. A fresh checkout
# (no config) or a stale path also falls through.
if (Test-Path -LiteralPath $configPath) {
    $cfg = Get-Content -LiteralPath $configPath -Raw
    if ($cfg) {
        $linkerMatch = [regex]::Match($cfg, '(?m)^\s*linker\s*=\s*"([^"]+)"')
        $libMatch = [regex]::Match($cfg, '(?m)^\s*LIB\s*=\s*(?:\{\s*value\s*=\s*"([^"]+)"|"([^"]+)")')
        if ($linkerMatch.Success -and $libMatch.Success) {
            $existingLinker = $linkerMatch.Groups[1].Value
            $existingLib = if ($libMatch.Groups[1].Success) {
                $libMatch.Groups[1].Value
            }
            else {
                $libMatch.Groups[2].Value
            }
            if ((Test-Path -LiteralPath $existingLinker) -and (Test-LibResolvesKernel32 $existingLib)) {
                Write-Host "win-msvc-bootstrap: $configPath already valid; skipping VsDevCmd"
                exit 0
            }
        }
    }
}

# --- Locate Visual Studio / the newest MSVC toolset -------------------------

$vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
if (-not (Test-Path -LiteralPath $vswhere)) {
    throw "vswhere.exe not found at $vswhere"
}

# -products * so Build Tools-only installs are found (the default omits them).
$vsPath = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $vsPath) {
    $vsPath = & $vswhere -latest -products * -property installationPath
}
if (-not $vsPath) {
    throw "vswhere found no Visual Studio installation with the x64 VC tools"
}

$devcmd = Join-Path $vsPath "Common7\Tools\VsDevCmd.bat"
if (-not (Test-Path -LiteralPath $devcmd)) {
    throw "VsDevCmd.bat not found at $devcmd"
}

# --- Capture the configured environment from VsDevCmd -----------------------

$command = "`"$devcmd`" -no_logo -arch=x64 -host_arch=x64 && set"
$lines = & cmd.exe /s /c $command
if ($LASTEXITCODE -ne 0) {
    throw "VsDevCmd.bat failed with exit code $LASTEXITCODE"
}

$vsenv = @{}
foreach ($line in $lines) {
    if ($line -match '^([^=]+)=(.*)$') {
        $vsenv[$Matches[1].ToUpperInvariant()] = $Matches[2]
    }
}

function Get-Required([string]$name) {
    $v = $vsenv[$name.ToUpperInvariant()]
    if (-not $v) { throw "VsDevCmd did not set $name" }
    return $v
}

$vcTools = Get-Required "VCToolsInstallDir"
$lib = Get-Required "LIB"
$include = Get-Required "INCLUDE"
$libpath = $vsenv["LIBPATH"]

# --- Derive + validate the linker and the SDK libs --------------------------

$linkExe = Join-Path ($vcTools.TrimEnd('\', '/')) "bin\Hostx64\x64\link.exe"
if (-not (Test-Path -LiteralPath $linkExe)) {
    throw "Derived MSVC linker does not exist: $linkExe"
}

# kernel32.lib lives in the Windows SDK um\x64 -- the exact entry LNK1181
# complains about when LIB is mangled. If it does not resolve here, refuse to
# write a config that would only fail later at link time.
if (-not (Test-LibResolvesKernel32 $lib)) {
    throw "Captured LIB does not resolve a real kernel32.lib (no Windows SDK um\x64?); refusing to write a config that would fail at link time"
}

# Forward slashes keep the values backslash-escape-free in TOML basic strings;
# link.exe accepts forward slashes.
function ConvertTo-TomlPath([string]$s) { return ($s -replace '\\', '/') }

$linkerValue = ConvertTo-TomlPath $linkExe
$envValues = [ordered]@{
    LIB     = ConvertTo-TomlPath $lib
    INCLUDE = ConvertTo-TomlPath $include
}
if ($libpath) { $envValues["LIBPATH"] = ConvertTo-TomlPath $libpath }

# --- Scoped merge into .cargo/config.toml -----------------------------------

$original = ""
if (Test-Path -LiteralPath $configPath) {
    $original = Get-Content -LiteralPath $configPath -Raw
    if ($null -eq $original) { $original = "" }
    # Strip a leading UTF-8 BOM so the first line parses as a header/comment.
    if ($original.Length -gt 0 -and $original[0] -eq [char]0xFEFF) {
        $original = $original.Substring(1)
    }
}

# This file is machine-managed (gitignored) and is written only by this script
# and the Rust toml_edit writer, both of which emit *canonical* tables: a bare
# `[name]` header alone on its line, no trailing comment, unquoted. The scoped
# merge below relies on that. Rather than risk a partial merge that yields a
# duplicate `[env]` or invalid TOML, fail fast if a header is in a form we
# can't safely round-trip (trailing comment, quoted/dotted `env`, etc.) --
# the user can delete the generated file and let it be regenerated.
$seenHeader = $false
foreach ($ln in ($original -split "`r?`n")) {
    if ($ln -match '^\s*\[') {
        $seenHeader = $true
        if ($ln -notmatch '^\s*\[([^\]\r\n]+)\]\s*$') {
            throw "Non-canonical TOML header in ${configPath}: '$ln'. This file is machine-managed; delete it and re-run."
        }
        $hn = $Matches[1].Trim()
        if ($hn -match '["'']') {
            throw "Quoted TOML header in ${configPath}: '$ln'. This file is machine-managed; delete it and re-run."
        }
        if ($hn -eq 'env' -or $hn -eq "target.$Triple") { continue }
        if ($hn -like 'env.*' -or $hn -like "target.$Triple.*") {
            throw "Dotted '$hn' table in ${configPath} conflicts with the keys this script owns. This file is machine-managed; delete it and re-run."
        }
        continue
    }
    # A top-level dotted key like `env.LIB = ...` or
    # `target.x86_64-pc-windows-msvc.linker = "..."` implicitly defines the
    # env/target table; our later `[env]` / `[target.<triple>]` header would
    # then redefine it -- invalid TOML. Only top-level (pre-header) lines can
    # define those tables, so check until the first header.
    if (-not $seenHeader -and $ln -match '^\s*("?)(env|target)\1\s*\.[^=\r\n]*=') {
        throw "Top-level dotted '$($Matches[2]).' key in ${configPath} conflicts with the [env] / [target.$Triple] tables this script writes. This file is machine-managed; delete it and re-run."
    }
}

# Parse into ordered sections. A table header is a line `[name]`; the block
# before the first header has header ''. We only ever rewrite the MSVC target
# table and the MSVC keys inside [env]; everything else is preserved verbatim.
$sections = [System.Collections.Generic.List[object]]::new()
$current = [pscustomobject]@{ Header = ''; Lines = [System.Collections.Generic.List[string]]::new() }
$sections.Add($current)
if ($original.Length -gt 0) {
    foreach ($ln in ($original -split "`r?`n")) {
        if ($ln -match '^\s*\[([^\]\r\n]+)\]\s*$') {
            $current = [pscustomobject]@{ Header = $Matches[1].Trim(); Lines = [System.Collections.Generic.List[string]]::new() }
            $sections.Add($current)
        }
        else {
            $current.Lines.Add($ln)
        }
    }
}

$out = [System.Collections.Generic.List[string]]::new()
$envEmitted = $false

function Add-EnvKeys([System.Collections.Generic.List[string]]$dst) {
    foreach ($k in $script:envValues.Keys) {
        $dst.Add(('{0} = {{ value = "{1}", force = true }}' -f $k, $script:envValues[$k]))
    }
}

foreach ($sec in $sections) {
    $h = $sec.Header
    if ($h -eq '') {
        foreach ($l in $sec.Lines) { $out.Add($l) }
        continue
    }
    if ($h -eq "target.$Triple") {
        continue  # script-owned: dropped here, re-emitted canonically below
    }
    if ($h -eq 'env') {
        $envEmitted = $true
        $out.Add('[env]')
        foreach ($l in $sec.Lines) {
            if ($l -match '^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=') {
                if ($OwnedEnvKeys -contains $Matches[1]) { continue }  # replace freshly
            }
            $out.Add($l)
        }
        Add-EnvKeys $out
        continue
    }
    $out.Add("[$h]")
    foreach ($l in $sec.Lines) { $out.Add($l) }
}

if (-not $envEmitted) {
    if ($out.Count -gt 0 -and $out[$out.Count - 1] -ne '') { $out.Add('') }
    $out.Add('[env]')
    Add-EnvKeys $out
}

if ($out.Count -gt 0 -and $out[$out.Count - 1] -ne '') { $out.Add('') }
$out.Add("[target.$Triple]")
$out.Add(('linker = "{0}"' -f $linkerValue))

$rendered = (($out -join "`n").TrimEnd()) + "`n"

if ($rendered -ne $original) {
    if (-not (Test-Path -LiteralPath $cargoDir)) {
        New-Item -ItemType Directory -Force -Path $cargoDir | Out-Null
    }
    # LF, no BOM -- matches what the Rust toml_edit writer produces.
    [System.IO.File]::WriteAllText($configPath, $rendered, (New-Object System.Text.UTF8Encoding $false))
    Write-Host "win-msvc-bootstrap: wrote $configPath (linker + LIB/INCLUDE for $Triple)"
}
else {
    Write-Host "win-msvc-bootstrap: $configPath already current for $Triple"
}
