param(
    [string]$PythonExe = "python"
)

$ErrorActionPreference = "Stop"

Write-Host "Using Python executable: $PythonExe"
& $PythonExe --version

Write-Host "Installing NVAP and packaging dependencies..."
& $PythonExe -m pip install --upgrade pip
& $PythonExe -m pip install .[dev]

Write-Host "Building one-folder executable with PyInstaller..."
& $PythonExe -m PyInstaller `
    --noconfirm `
    --clean `
    --onedir `
    --name NVAP `
    --collect-submodules vtkmodules `
    --collect-submodules PySide6 `
    --hidden-import skimage._shared.geometry `
    --hidden-import imageio.v3 `
    src\nvap\app.py

Write-Host "Running smoke test on packaged executable..."
if (Test-Path "dist\NVAP\NVAP.exe") {
    & "dist\NVAP\NVAP.exe" --headless-smoke --input Input
}

Write-Host "Build complete. Output: dist\NVAP\"
