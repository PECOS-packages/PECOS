param(
    [Parameter()]
    [string]$DemPath
)

$ErrorActionPreference = "Stop"
$workspace = Split-Path -Parent $PSScriptRoot
$arguments = @("build", "--release", "--target", "wasm32-unknown-unknown", "-p", "pecos-frontier-wasm")

$previousRustFlags = $env:RUSTFLAGS
try {
    $env:RUSTFLAGS = (@($previousRustFlags, '--cfg getrandom_backend="unsupported"') |
        Where-Object { $_ }) -join ' '
    if ($DemPath) {
        $env:FRONTIER_DEM_PATH = (Resolve-Path -LiteralPath $DemPath).Path
    } else {
        Remove-Item Env:FRONTIER_DEM_PATH -ErrorAction SilentlyContinue
    }
    & cargo @arguments --manifest-path (Join-Path $workspace "Cargo.toml")
    if ($LASTEXITCODE -ne 0) {
        throw "cargo build failed with exit code $LASTEXITCODE"
    }
} finally {
    if ($null -eq $previousRustFlags) {
        Remove-Item Env:RUSTFLAGS -ErrorAction SilentlyContinue
    } else {
        $env:RUSTFLAGS = $previousRustFlags
    }
    Remove-Item Env:FRONTIER_DEM_PATH -ErrorAction SilentlyContinue
}

$source = Join-Path $workspace "target\wasm32-unknown-unknown\release\pecos_frontier_wasm.wasm"
$dist = Join-Path $workspace "dist"
New-Item -ItemType Directory -Force -Path $dist | Out-Null
Copy-Item -LiteralPath $source -Destination (Join-Path $dist "pecos_frontier_wasm.wasm") -Force
Get-Item (Join-Path $dist "pecos_frontier_wasm.wasm") | Select-Object FullName, Length
