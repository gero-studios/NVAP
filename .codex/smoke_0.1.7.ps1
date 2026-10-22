param([Parameter(Mandatory=$true)][string]$Variant)
$ErrorActionPreference = 'Stop'
$root = (Get-Location).Path
$exe = Join-Path $root "dist/NVAP-0.1.7-windows-$Variant.exe"
$state = Join-Path $root ".codex/release-0.1.7-smoke/$Variant"
New-Item -ItemType Directory -Path $state -Force | Out-Null
$previousNvapHome = $env:NVAP_HOME
$env:NVAP_HOME = $state
try {
    & $exe --headless-smoke --input "$root/.codex/release-smoke-input" *> "$state/headless.log"
    if ($LASTEXITCODE -ne 0) { throw "Packaged analysis failed: $LASTEXITCODE" }
    $analysisLog = Get-Content "$state/headless.log" -Raw
    if ($analysisLog -notmatch 'green: voxels=[1-9]' -or $analysisLog -notmatch 'red: voxels=[1-9]') {
        throw 'Packaged analysis did not measure nonempty green and red channels.'
    }
    for ($attempt = 1; $attempt -le 2; $attempt++) {
        $priorLog = if (Test-Path "$state/nvap.log") { (Get-Content -LiteralPath "$state/nvap.log" -Raw).Length } else { 0 }
        $launched = Start-Process -FilePath $exe -WindowStyle Hidden -PassThru -RedirectStandardOutput "$state/gui-$attempt.stdout.log" -RedirectStandardError "$state/gui-$attempt.stderr.log"
        try {
            $deadline = [DateTime]::UtcNow.AddSeconds(90)
            $initialized = $false
            while ([DateTime]::UtcNow -lt $deadline) {
                Start-Sleep -Seconds 2
                $launched.Refresh()
                if ($launched.HasExited) { throw "Packaged GUI exited early: $($launched.ExitCode)" }
                if (Test-Path "$state/nvap.log") {
                    $log = Get-Content -LiteralPath "$state/nvap.log" -Raw
                    if ($log.Length -gt $priorLog -and $log.Substring($priorLog) -match 'NVAP UI initialized') {
                        $initialized = $true
                        break
                    }
                }
            }
            if (-not $initialized) { throw 'Packaged GUI did not initialize within 90 seconds.' }
            $projects = @(Get-Content "$state/recent_projects.json" -Raw | ConvertFrom-Json)
            if ($projects.Count -ne 1) { throw "Expected one sample entry, got $($projects.Count)." }
            $samplePath = [System.IO.Path]::GetFullPath($projects[0].path)
            if (-not $samplePath.StartsWith([System.IO.Path]::GetFullPath($state) + [System.IO.Path]::DirectorySeparatorChar)) { throw 'Sample was saved outside persistent test state.' }
            if (-not (Test-Path -LiteralPath $samplePath -PathType Container)) { throw 'Persisted sample folder is missing.' }
            $imageCount = @(Get-ChildItem -LiteralPath $samplePath -File).Count
            if ($imageCount -lt 2) { throw 'Persisted sample images are missing.' }
            Write-Output "$Variant GUI launch $attempt passed: one persistent sample, $imageCount images."
        } finally {
            $children = @(Get-CimInstance Win32_Process | Where-Object { $_.ParentProcessId -eq $launched.Id })
            foreach ($child in $children) { Stop-Process -Id $child.ProcessId -Force -ErrorAction SilentlyContinue }
            if (-not $launched.WaitForExit(5000)) {
                Stop-Process -Id $launched.Id -Force -ErrorAction SilentlyContinue
            }
        }
    }
    Get-Content "$state/headless.log" | Select-String 'NVAP smoke run OK|green: voxels=|red: voxels=|vasculature:'
} finally {
    if ($null -eq $previousNvapHome) { Remove-Item Env:NVAP_HOME -ErrorAction SilentlyContinue } else { $env:NVAP_HOME = $previousNvapHome }
}



