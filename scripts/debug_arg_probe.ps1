Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Get-Process | Where-Object { $_.ProcessName -eq "fiji-windows-x64" } | ForEach-Object {
    Stop-Process -Id $_.Id -Force -ErrorAction SilentlyContinue
}

$runtime = Resolve-Path ".\MicrogliaMaskingIsolated\FijiRuntime"
$macroPath = Join-Path $runtime "arg_probe.ijm"
$probe = Join-Path $runtime "arg_probe.txt"

$lines = @(
    'macro "ArgProbe" {',
    '  args = getArgument();',
    '  File.append("ARGS=" + args + "\n", "arg_probe.txt");',
    '  items = split(args, ";");',
    '  inDir = ""; outDir = "";',
    '  for (i=0; i<items.length; i++) {',
    '    if (startsWith(items[i], "inputDir=")) inDir = substring(items[i], 9);',
    '    if (startsWith(items[i], "outputDir=")) outDir = substring(items[i], 10);',
    '  }',
    '  File.append("IN=" + inDir + "\n", "arg_probe.txt");',
    '  File.append("OUT=" + outDir + "\n", "arg_probe.txt");',
    '  call("java.lang.System.exit", "0");',
    '}'
)
Set-Content -Path $macroPath -Value $lines -Encoding ASCII
if (Test-Path $probe) {
    Remove-Item $probe -Force
}

$base = Get-ChildItem manual_test_outputs |
    Where-Object { $_.Name -like "microglia_subfunction_test_*" } |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1

$inputDir = (Join-Path $base.FullName "input_slices").Replace("\", "/")
$outDir = (Join-Path $base.FullName "arg_probe_out").Replace("\", "/")
$macroArgs = "inputDir=$inputDir;outputDir=$outDir;pixelSizeUm=1.0;allowUiFallback=false;saveFrameSlices=false;exitOnComplete=true;applyMaskTarget=seg"

$bat = Join-Path $runtime "fiji.bat"
$p = Start-Process -FilePath $bat -WorkingDirectory $runtime -ArgumentList @("--headless", "--console", "-macro", $macroPath.Replace("\", "/"), $macroArgs) -PassThru
$ok = $p.WaitForExit(60000)
if (-not $ok) {
    Write-Output "WAIT_TIMEOUT=true"
    try {
        Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue
    } catch {}
} else {
    Write-Output "EXIT_CODE=$($p.ExitCode)"
}

Write-Output "PROBE_EXISTS=$([bool](Test-Path $probe))"
if (Test-Path $probe) {
    Write-Output "--- arg_probe.txt ---"
    Get-Content $probe
}
