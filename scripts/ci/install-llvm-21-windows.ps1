param(
    [string]$InstallDir = (Join-Path $env:USERPROFILE ".pecos\deps\llvm-21.1"),
    [string]$Version = ""
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

if (-not $Version) {
    if ($env:LLVM_RELEASE_VERSION) {
        $Version = $env:LLVM_RELEASE_VERSION
    }
    else {
        $Version = "21.1.8"
    }
}

$RequiredVersion = "21.1"
$MambaVersion = if ($env:MAMBA_VERSION) { $env:MAMBA_VERSION } else { "latest" }
$MambaRoot = if ($env:MAMBA_ROOT_PREFIX) {
    $env:MAMBA_ROOT_PREFIX
}
else {
    Join-Path $env:USERPROFILE ".cache\pecos-micromamba"
}
$LlvmPrefix = Join-Path $InstallDir "Library"
$LlvmConfig = Join-Path $LlvmPrefix "bin\llvm-config.exe"
$LlvmBin = Join-Path $LlvmPrefix "bin"
$LlvmLib = Join-Path $LlvmPrefix "lib"
$Libclang = Join-Path $LlvmPrefix "bin\libclang.dll"

function Find-SevenZip {
    foreach ($Name in @("7z.exe", "7zz.exe", "7za.exe")) {
        $Command = Get-Command $Name -ErrorAction SilentlyContinue
        if ($Command) {
            return $Command.Source
        }
    }

    $Candidates = @()
    if ($env:ProgramFiles) {
        $Candidates += Join-Path $env:ProgramFiles "7-Zip\7z.exe"
    }
    if (${env:ProgramFiles(x86)}) {
        $Candidates += Join-Path ${env:ProgramFiles(x86)} "7-Zip\7z.exe"
    }
    if ($env:ChocolateyInstall) {
        $Candidates += Join-Path $env:ChocolateyInstall "bin\7z.exe"
    }

    foreach ($Candidate in $Candidates) {
        if ($Candidate -and (Test-Path $Candidate)) {
            return $Candidate
        }
    }

    return $null
}

function Invoke-DownloadFile {
    param(
        [string]$Url,
        [string]$Output
    )

    $Curl = Get-Command curl.exe -ErrorAction SilentlyContinue
    if ($Curl) {
        & $Curl.Source --fail --location --retry 8 --retry-all-errors --retry-delay 5 --retry-max-time 300 --connect-timeout 30 --output $Output $Url
        if ($LASTEXITCODE -ne 0) {
            throw "curl failed to download $Url with exit code $LASTEXITCODE"
        }
    }
    else {
        $LastError = $null
        for ($Attempt = 1; $Attempt -le 8; $Attempt++) {
            try {
                Invoke-WebRequest -Uri $Url -OutFile $Output -TimeoutSec 300
                return
            }
            catch {
                $LastError = $_
                if ($Attempt -eq 8) {
                    break
                }

                Start-Sleep -Seconds 5
            }
        }

        throw "Invoke-WebRequest failed to download $Url after 8 attempts: $LastError"
    }
}

function Expand-TarBz2 {
    param(
        [string]$Archive,
        [string]$Destination
    )

    New-Item -ItemType Directory -Force -Path $Destination | Out-Null

    $SevenZip = Find-SevenZip
    if ($SevenZip) {
        $Stage = Join-Path $Destination "_stage"
        New-Item -ItemType Directory -Force -Path $Stage | Out-Null

        & $SevenZip x -y -bb0 "-o$Stage" $Archive | Out-Null
        if ($LASTEXITCODE -ne 0) {
            throw "7-Zip failed to decompress $Archive with exit code $LASTEXITCODE"
        }

        $TarArchive = Get-ChildItem -Path $Stage -File -Filter "*.tar" | Select-Object -First 1
        if (-not $TarArchive) {
            throw "7-Zip did not produce a .tar payload from $Archive"
        }

        & $SevenZip x -y -bb0 "-o$Destination" $TarArchive.FullName | Out-Null
        if ($LASTEXITCODE -ne 0) {
            throw "7-Zip failed to extract $($TarArchive.Name) with exit code $LASTEXITCODE"
        }

        Remove-Item -Recurse -Force $Stage
        return
    }

    $Tar = Get-Command tar.exe -ErrorAction SilentlyContinue
    if (-not $Tar) {
        throw "7-Zip or tar.exe is required to extract micromamba on Windows"
    }

    & $Tar.Source -xjf $Archive -C $Destination | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "tar.exe failed to extract $Archive with exit code $LASTEXITCODE"
    }
}

function Get-Micromamba {
    param([string]$TempDir)

    foreach ($Name in @("micromamba.exe", "micromamba")) {
        $Command = Get-Command $Name -ErrorAction SilentlyContinue
        if ($Command) {
            return $Command.Source
        }
    }

    $Url = "https://micro.mamba.pm/api/micromamba/win-64/$MambaVersion"
    $Archive = Join-Path $TempDir "micromamba.tar.bz2"
    $ExtractDir = Join-Path $TempDir "micromamba"

    Write-Host "Downloading micromamba for win-64"
    Invoke-DownloadFile -Url $Url -Output $Archive
    Expand-TarBz2 -Archive $Archive -Destination $ExtractDir

    $MambaBin = Join-Path $ExtractDir "Library\bin\micromamba.exe"
    if (-not (Test-Path $MambaBin)) {
        throw "micromamba.exe not found at $MambaBin after extraction"
    }

    return $MambaBin
}

function Test-LlvmInstall {
    Repair-LibclangForBindgen

    if (-not (Test-Path $LlvmConfig)) {
        return $false
    }

    if (-not (Test-Path $Libclang)) {
        return $false
    }

    try {
        $FoundVersion = (& $LlvmConfig --version).Trim()
        if (-not $FoundVersion.StartsWith($RequiredVersion)) {
            return $false
        }

        $LibDir = (& $LlvmConfig --libdir).Trim()
        if (-not (Test-Path $LibDir)) {
            return $false
        }

        Repair-LlvmSystemLibsForMsvc

        $StaticLibs = (& $LlvmConfig --libnames --link-static).Trim()
        if ([string]::IsNullOrWhiteSpace($StaticLibs)) {
            return $false
        }

        $SystemLibs = (& $LlvmConfig --system-libs --link-static).Trim()
        foreach ($LibName in @("z.lib", "zstd.dll.lib", "xml2.lib")) {
            if ($SystemLibs -match "(^|\s)$([regex]::Escape($LibName))($|\s)") {
                $ImportLib = Join-Path $LlvmLib $LibName
                if (-not (Test-Path $ImportLib)) {
                    return $false
                }
            }
        }
    }
    catch {
        return $false
    }

    return $true
}

function Repair-LlvmSystemLibsForMsvc {
    if (-not (Test-Path $LlvmLib)) {
        return
    }

    $ZstdDllImportLib = Join-Path $LlvmLib "zstd.dll.lib"
    if (-not (Test-Path $ZstdDllImportLib)) {
        foreach ($CandidateName in @("zstd.lib", "libzstd.lib")) {
            $Candidate = Join-Path $LlvmLib $CandidateName
            if (Test-Path $Candidate) {
                Write-Host "Creating LLVM-compatible zstd.dll.lib from $CandidateName"
                Copy-Item -Force -Path $Candidate -Destination $ZstdDllImportLib
                break
            }
        }
    }
}

function Repair-LibclangForBindgen {
    if (Test-Path $Libclang) {
        return
    }

    if (-not (Test-Path $LlvmBin)) {
        return
    }

    $PackagedLibclang = Get-ChildItem -Path $LlvmBin -File -Filter "libclang-*.dll" |
        Sort-Object Name |
        Select-Object -First 1

    if (-not $PackagedLibclang) {
        return
    }

    Write-Host "Creating bindgen-compatible libclang.dll from $($PackagedLibclang.Name)"
    Copy-Item -Force -Path $PackagedLibclang.FullName -Destination $Libclang
}

function Write-LlvmDiagnostics {
    Write-Host "LLVM prefix: $LlvmPrefix"
    if (Test-Path $LlvmConfig) {
        & $LlvmConfig --version
        & $LlvmConfig --shared-mode
        & $LlvmConfig --libdir
        & $LlvmConfig --libnames --link-static core
        & $LlvmConfig --system-libs --link-static
    }
    else {
        Write-Host "llvm-config.exe not found at $LlvmConfig"
    }

    if (Test-Path $Libclang) {
        Write-Host "Found libclang: $Libclang"
    }
    else {
        Write-Host "libclang.dll not found at $Libclang"
        if (Test-Path $LlvmBin) {
            Get-ChildItem -Path $LlvmBin -File -Filter "*clang*.dll" |
                ForEach-Object { Write-Host "  candidate: $($_.FullName)" }
        }
    }
}

if (Test-LlvmInstall) {
    $FoundVersion = (& $LlvmConfig --version).Trim()
    Write-Host "conda-forge LLVM $FoundVersion already installed at $LlvmPrefix"
    exit 0
}

if (Test-Path $InstallDir) {
    Write-Host "Removing invalid or incompatible LLVM environment at $InstallDir"
    Remove-Item -Recurse -Force $InstallDir
}

New-Item -ItemType Directory -Force -Path (Split-Path -Parent $InstallDir) | Out-Null
New-Item -ItemType Directory -Force -Path $MambaRoot | Out-Null

$TempDir = Join-Path ([System.IO.Path]::GetTempPath()) "pecos-micromamba-$([System.Guid]::NewGuid())"
New-Item -ItemType Directory -Force -Path $TempDir | Out-Null

try {
    $MambaBin = Get-Micromamba -TempDir $TempDir

    Write-Host "Installing conda-forge LLVM $Version, clang $Version, libclang $Version, zlib, and libxml2-devel to $InstallDir"
    $OldMambaRoot = $env:MAMBA_ROOT_PREFIX
    $env:MAMBA_ROOT_PREFIX = $MambaRoot
    try {
        & $MambaBin create `
            -y `
            -p $InstallDir `
            --override-channels `
            -c conda-forge `
            "llvmdev=$Version" `
            "clang=$Version" `
            "libclang=$Version" `
            "zlib" `
            "libxml2-devel"
        if ($LASTEXITCODE -ne 0) {
            throw "micromamba failed to create LLVM environment with exit code $LASTEXITCODE"
        }
    }
    finally {
        if ($null -eq $OldMambaRoot) {
            Remove-Item Env:MAMBA_ROOT_PREFIX -ErrorAction SilentlyContinue
        }
        else {
            $env:MAMBA_ROOT_PREFIX = $OldMambaRoot
        }
    }

    if (-not (Test-LlvmInstall)) {
        Write-LlvmDiagnostics
        throw "conda-forge LLVM install did not provide usable LLVM $RequiredVersion static libraries and libclang"
    }

    Write-LlvmDiagnostics
    Write-Host "Installed conda-forge LLVM $Version to $LlvmPrefix"
}
finally {
    if (Test-Path $TempDir) {
        Remove-Item -Recurse -Force $TempDir
    }
}
