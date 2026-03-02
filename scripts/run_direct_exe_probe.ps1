Set-StrictMode -Version Latest
$ErrorActionPreference='Stop'

$runtime=(Resolve-Path '.\MicrogliaMaskingIsolated\FijiRuntime').Path
$exe=Join-Path $runtime 'fiji-windows-x64.exe'
$macro=(Resolve-Path '.\MicrogliaMaskingIsolated\FijiRuntime\macros\Microglia_Batch_Pipeline.ijm').Path.Replace('\','/')
$base='C:/Users/giaco/Documents/NVAP/NVAP/manual_test_outputs/microglia_subfunction_test_20260301_135130'
$inDir="$base/input_slices"
$outDir="$base/direct_exe_run"

New-Item -ItemType Directory -Path $outDir -Force | Out-Null

$args = 'inputDir=' + $inDir + ';outputDir=' + $outDir + ';pixelSizeUm=1.0;allowUiFallback=false;saveFrameSlices=false;exitOnComplete=true;relativeIntensityThreshold=2.8;minimalMicrogliaSize=500;skeletonMaxLength=450;contactMaxLength=20;applyMaskTarget=seg'

$stdout=Join-Path $outDir 'direct_stdout.log'
$stderr=Join-Path $outDir 'direct_stderr.log'
if (Test-Path $stdout) { Remove-Item $stdout -Force }
if (Test-Path $stderr) { Remove-Item $stderr -Force }

Get-Process | Where-Object { $_.ProcessName -eq 'fiji-windows-x64' } | ForEach-Object {
  Stop-Process -Id $_.Id -Force -ErrorAction SilentlyContinue
}

$p=Start-Process -FilePath $exe -WorkingDirectory $runtime -ArgumentList @('--jaunch-skip-console-check','--headless','--console','-macro',$macro,$args) -PassThru -RedirectStandardOutput $stdout -RedirectStandardError $stderr
$done=$p.WaitForExit(420000)
if(-not $done){
  Write-Output 'WAIT_TIMEOUT=true'
  try{ Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue } catch {}
} else {
  Write-Output ('EXIT_CODE=' + $p.ExitCode)
}

$status=Join-Path $outDir 'pipeline_status.txt'
Write-Output ('STATUS_EXISTS=' + [bool](Test-Path $status))
if(Test-Path $status){
  Write-Output '--- status ---'
  Get-Content $status
}
Write-Output ('STDOUT_EXISTS=' + [bool](Test-Path $stdout))
if(Test-Path $stdout){
  Write-Output '--- stdout tail ---'
  Get-Content $stdout -Tail 120
}
Write-Output ('STDERR_EXISTS=' + [bool](Test-Path $stderr))
if(Test-Path $stderr){
  Write-Output '--- stderr tail ---'
  Get-Content $stderr -Tail 120
}
Write-Output '--- out files ---'
Get-ChildItem -Recurse $outDir | ForEach-Object { $_.FullName }
