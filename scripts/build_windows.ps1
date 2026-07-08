param(
    [string]$PythonExe = "python",
    [ValidateSet("auto", "cpu", "directml", "torch")]
    [string]$Acceleration = "auto",
    [ValidateSet("onefile", "onedir")]
    [string]$PackageMode = "onefile",
    [switch]$KeepBuildExe
)

$ErrorActionPreference = "Stop"

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)]
        [scriptblock]$Command,
        [Parameter(Mandatory = $true)]
        [string]$Description
    )
    & $Command
    if ($LASTEXITCODE -ne 0) {
        throw "$Description failed with exit code $LASTEXITCODE."
    }
}

function Remove-IntermediateBuildExe {
    $intermediateExe = Join-Path (Get-Location) "build\NVAP\NVAP.exe"
    if (-not (Test-Path -LiteralPath $intermediateExe)) {
        return
    }

    Remove-Item -LiteralPath $intermediateExe -Force
    Write-Host "Removed non-runnable PyInstaller intermediate: build\NVAP\NVAP.exe"
}

$buildMetadataPath = Join-Path (Get-Location) "src\nvap\_build_metadata.py"
$buildMetadataOriginal = $null

function Write-BuildMetadata {
    param([string]$Variant)
    $script:buildMetadataOriginal = Get-Content -LiteralPath $buildMetadataPath -Raw
    $commit = (git rev-parse --short HEAD).Trim()
    $builtAt = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
    @"
"""Build identity used only by the update-check. Auto-generated for this
build by scripts/build_windows.ps1; restored to dev defaults afterward."""

BUILD_COMMIT = "$commit"
BUILD_VARIANT = "$Variant"
BUILT_AT = "$builtAt"
"@ | Set-Content -LiteralPath $buildMetadataPath -Encoding utf8
    Write-Host "Baked build metadata: commit=$commit variant=$Variant built_at=$builtAt"
}

function Restore-BuildMetadata {
    if ($null -ne $script:buildMetadataOriginal) {
        Set-Content -LiteralPath $buildMetadataPath -Value $script:buildMetadataOriginal -Encoding utf8 -NoNewline
        Write-Host "Restored dev build metadata stub."
    }
}

Write-Host "Using Python executable: $PythonExe"
Invoke-Checked { & $PythonExe --version } "Python version check"

Write-Host "Installing NVAP and packaging dependencies..."
Invoke-Checked { & $PythonExe -m pip install --upgrade pip } "pip upgrade"
$extras = "dev"
$resolvedMode = "cpu"
$pyVersion = & $PythonExe -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
$isWindows = [System.Runtime.InteropServices.RuntimeInformation]::IsOSPlatform(
    [System.Runtime.InteropServices.OSPlatform]::Windows
)
if ($Acceleration -eq "directml" -or ($Acceleration -eq "auto" -and $isWindows -and [version]$pyVersion -lt [version]"3.13")) {
    $extras = "dev,directml"
    $resolvedMode = "directml"
    Write-Host "DirectML acceleration will be included when wheels are available for Python $pyVersion."
} elseif ($Acceleration -eq "torch") {
    $extras = "dev,denoise_torch"
    $resolvedMode = "torch"
    Write-Host "Torch acceleration dependencies will be included."
} else {
    Write-Host "Building CPU-safe package. Runtime still auto-detects any bundled GPU backend."
}
Invoke-Checked { & $PythonExe -m pip install -e ".[$extras]" } "NVAP dependency install"

$pyInstallerArgs = @(
    "--noconfirm",
    "--clean",
    "--$PackageMode",
    "--name", "NVAP",
    # UPX-packed, unsigned PyInstaller binaries are a classic false-positive
    # trigger for Windows Defender / SmartScreen and other AV heuristics.
    # Disabling UPX regardless of whether it happens to be on PATH keeps the
    # packaged exe from tripping that heuristic.
    "--noupx",
    "--collect-submodules", "vtkmodules",
    "--collect-submodules", "PySide6",
    "--copy-metadata", "imageio",
    "--copy-metadata", "nvap",
    "--hidden-import", "skimage._shared.geometry",
    "--hidden-import", "imageio.v3"
)

function Test-PythonImport {
    param([string]$ModuleName)
    & $PythonExe -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('$ModuleName') else 1)" *> $null
    return $LASTEXITCODE -eq 0
}

# Bundle sample datasets so they ship inside the executable and register on
# first run. Only added when a samples/ folder is present.
if (Test-Path "samples") {
    Write-Host "Bundling sample datasets from samples\ into the executable."
    $pyInstallerArgs += @("--add-data", "samples;samples")
}

# Only bundle torch / torch-directml when the requested acceleration mode
# actually calls for them. Probing "is it importable" instead of the resolved
# mode would sweep a stray torch install (e.g. left over from a different
# build) into a "CPU-safe" package, silently bloating it and defeating the
# whole point of the CPU-safe build.
if ($resolvedMode -eq "torch" -and (Test-PythonImport "torch")) {
    Write-Host "Including torch modules in executable."
    $pyInstallerArgs += @("--collect-submodules", "torch")
}
if ($resolvedMode -eq "directml" -and (Test-PythonImport "torch_directml")) {
    Write-Host "Including torch-directml modules in executable."
    $pyInstallerArgs += @(
        "--collect-submodules", "torch_directml",
        "--collect-binaries", "torch_directml",
        "--copy-metadata", "torch-directml"
    )
}
if ($resolvedMode -eq "cpu") {
    $pyInstallerArgs += @("--exclude-module", "torch", "--exclude-module", "torch_directml")
}
$pyInstallerArgs += "src\nvap\app.py"

if ($PackageMode -eq "onefile") {
    $packagedExe = "dist\NVAP.exe"
    Write-Host "Building standalone executable with PyInstaller..."
} else {
    $packagedExe = "dist\NVAP\NVAP.exe"
    Write-Host "Building one-folder executable with PyInstaller..."
}
Write-BuildMetadata -Variant $resolvedMode
try {
    Invoke-Checked { & $PythonExe -m PyInstaller @pyInstallerArgs } "PyInstaller build"
}
finally {
    Restore-BuildMetadata
    if (-not $KeepBuildExe) {
        Remove-IntermediateBuildExe
    }
}

Write-Host "Running smoke test on packaged executable..."
if (Test-Path $packagedExe) {
    Invoke-Checked { & $packagedExe --print-runtime-profile } "Runtime profile smoke test"
    if (Test-Path "Input") {
        Invoke-Checked { & $packagedExe --headless-smoke --input Input } "Dataset smoke test"
    } else {
        Write-Host "No Input folder found; skipped dataset smoke test."
    }
} else {
    throw "Build did not produce $packagedExe."
}

Write-Host "Build complete. Click this executable: $packagedExe"
