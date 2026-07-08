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

Write-Host "Using Python executable: $PythonExe"
Invoke-Checked { & $PythonExe --version } "Python version check"

Write-Host "Installing NVAP and packaging dependencies..."
Invoke-Checked { & $PythonExe -m pip install --upgrade pip } "pip upgrade"
$extras = "dev"
$pyVersion = & $PythonExe -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
$isWindows = [System.Runtime.InteropServices.RuntimeInformation]::IsOSPlatform(
    [System.Runtime.InteropServices.OSPlatform]::Windows
)
if ($Acceleration -eq "directml" -or ($Acceleration -eq "auto" -and $isWindows -and [version]$pyVersion -lt [version]"3.13")) {
    $extras = "dev,directml"
    Write-Host "DirectML acceleration will be included when wheels are available for Python $pyVersion."
} elseif ($Acceleration -eq "torch") {
    $extras = "dev,denoise_torch"
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

if (Test-PythonImport "torch") {
    Write-Host "Including torch modules in executable."
    $pyInstallerArgs += @("--collect-submodules", "torch")
}
if (Test-PythonImport "torch_directml") {
    Write-Host "Including torch-directml modules in executable."
    $pyInstallerArgs += @(
        "--collect-submodules", "torch_directml",
        "--collect-binaries", "torch_directml",
        "--copy-metadata", "torch-directml"
    )
}
$pyInstallerArgs += "src\nvap\app.py"

if ($PackageMode -eq "onefile") {
    $packagedExe = "dist\NVAP.exe"
    Write-Host "Building standalone executable with PyInstaller..."
} else {
    $packagedExe = "dist\NVAP\NVAP.exe"
    Write-Host "Building one-folder executable with PyInstaller..."
}
try {
    Invoke-Checked { & $PythonExe -m PyInstaller @pyInstallerArgs } "PyInstaller build"
}
finally {
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
