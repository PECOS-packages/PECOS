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

$vsPath = & $vswhere -latest -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $vsPath) {
    $vsPath = & $vswhere -latest -property installationPath
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

$linkPath = Get-ChildItem -Path (Join-Path $vsPath "VC\Tools\MSVC") -Recurse -Filter "link.exe" |
    Where-Object { $_.FullName -like "*\bin\Hostx64\x64\*" } |
    Select-Object -First 1 -ExpandProperty FullName

if (-not $linkPath) {
    throw "Could not find MSVC link.exe for x64"
}

Add-GitHubEnv -Name "CARGO_TARGET_X86_64_PC_WINDOWS_MSVC_LINKER" -Value $linkPath

Write-Host "Configured Visual Studio environment from $vsPath for $Arch"
Write-Host "Configured Cargo MSVC linker: $linkPath"
