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

# Used by the release workflow so nvcc can find cl.exe and the CUDA wheel's
# cargo build links against the right MSVC toolset. The Cargo build path in
# python-test.yml / local dev uses scripts/win-msvc-bootstrap.ps1 instead.
param(
    [string]$Arch = "x64",
    [string]$HostArch = "x64"
)

$ErrorActionPreference = "Stop"

if (-not $env:GITHUB_ENV -or -not $env:GITHUB_PATH) {
    throw "setup-msvc.ps1 must run inside GitHub Actions so GITHUB_ENV and GITHUB_PATH are available"
}

function Add-GitHubEnv {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [AllowEmptyString()][string]$Value
    )

    $delimiter = [Guid]::NewGuid().ToString("N")
    "$Name<<$delimiter" | Out-File -FilePath $env:GITHUB_ENV -Encoding utf8 -Append
    $Value | Out-File -FilePath $env:GITHUB_ENV -Encoding utf8 -Append
    "$delimiter" | Out-File -FilePath $env:GITHUB_ENV -Encoding utf8 -Append
}

$vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
if (-not (Test-Path $vswhere)) {
    throw "Could not find vswhere.exe at $vswhere"
}

# -products * so Build Tools-only installs are found (the default omits them).
$vsPath = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $vsPath) {
    $vsPath = & $vswhere -latest -products * -property installationPath
}
if (-not $vsPath) {
    throw "Could not find a Visual Studio installation"
}

$devcmd = Join-Path $vsPath "Common7\Tools\VsDevCmd.bat"
if (-not (Test-Path $devcmd)) {
    throw "Could not find VsDevCmd.bat at $devcmd"
}

$before = [System.Collections.Generic.Dictionary[string, string]]::new([System.StringComparer]::OrdinalIgnoreCase)
Get-ChildItem Env: | ForEach-Object {
    $before[$_.Name] = $_.Value
}

$command = "`"$devcmd`" -no_logo -arch=$Arch -host_arch=$HostArch && set"
$lines = & cmd.exe /s /c $command
if ($LASTEXITCODE -ne 0) {
    throw "VsDevCmd.bat failed with exit code $LASTEXITCODE"
}

$after = [System.Collections.Generic.Dictionary[string, string]]::new([System.StringComparer]::OrdinalIgnoreCase)
foreach ($line in $lines) {
    if ($line -match '^([^=][^=]*)=(.*)$') {
        $after[$Matches[1]] = $Matches[2]
    }
}

foreach ($name in ($after.Keys | Sort-Object)) {
    if ($name -ieq "Path") {
        continue
    }

    $value = $after[$name]
    if (-not $before.ContainsKey($name) -or $before[$name] -cne $value) {
        Add-GitHubEnv -Name $name -Value $value
    }
}

$oldPathParts = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::OrdinalIgnoreCase)
if ($before.ContainsKey("Path")) {
    foreach ($pathPart in ($before["Path"] -split ';')) {
        if ($pathPart) {
            [void]$oldPathParts.Add($pathPart)
        }
    }
}

if ($after.ContainsKey("Path")) {
    foreach ($pathPart in ($after["Path"] -split ';')) {
        if ($pathPart -and -not $oldPathParts.Contains($pathPart)) {
            $pathPart | Out-File -FilePath $env:GITHUB_PATH -Encoding utf8 -Append
        }
    }
}

# Derive link.exe from the toolset VsDevCmd actually configured
# (VCToolsInstallDir), so the pinned linker always matches the LIB / INCLUDE
# this script exports AND the .cargo/config.toml that python-release.yml
# derives from the same variable. (Cargo env vars outrank config.toml, so the
# env pin and any generated config MUST agree on the toolset.) This replaces a
# lexical MSVC-dir scan that could select a mismatched toolset and fail with
# `LNK1181: cannot open input file 'kernel32.lib'`.
$vcTools = $after["VCToolsInstallDir"]
if (-not $vcTools) {
    throw "VsDevCmd did not set VCToolsInstallDir"
}
$linkPath = Join-Path ($vcTools.TrimEnd('\', '/')) "bin\Hostx64\x64\link.exe"
if (-not (Test-Path $linkPath)) {
    throw "MSVC link.exe not found at $linkPath"
}

Add-GitHubEnv -Name "CARGO_TARGET_X86_64_PC_WINDOWS_MSVC_LINKER" -Value $linkPath

# The Justfile pins `set shell := ["bash", "-cu"]`, so every `just` recipe (and
# the `cargo` / `link.exe` it spawns) runs under git-bash, whose MSYS2 runtime
# rewrites "path-like" environment variables when crossing the bash<->native
# boundary. LIB / INCLUDE / LIBPATH from VsDevCmd.bat are semicolon-separated
# lists of Windows paths that contain spaces and parentheses (e.g.
# `C:\Program Files (x86)\Windows Kits\10\Lib\...\um\x64`). MSYS2's heuristic
# conversion corrupts these on the round trip, so the native linker receives a
# broken LIB and fails with `LNK1181: cannot open input file 'kernel32.lib'`
# (kernel32.lib lives in the Windows SDK's um\x64, which is exactly the entry
# that gets mangled). MSYS2_ENV_CONV_EXCL is the documented mechanism to opt
# specific variables out of that conversion so they pass through verbatim.
# Setting it here (via GITHUB_ENV) means every subsequent bash step -- and the
# nested `bash -cu` that `just` spawns for each recipe -- honors it, because the
# MSYS2 runtime reads MSYS2_ENV_CONV_EXCL from the process environment at
# startup. This is required for cold cargo builds (build scripts that haven't
# been pre-warmed in the rust-cache) to link on Windows.
Add-GitHubEnv -Name "MSYS2_ENV_CONV_EXCL" -Value "LIB;INCLUDE;LIBPATH"

Write-Host "Configured Visual Studio environment from $vsPath for $Arch"
Write-Host "Configured Cargo MSVC linker: $linkPath"
Write-Host "Excluded LIB;INCLUDE;LIBPATH from MSYS2 path conversion"
