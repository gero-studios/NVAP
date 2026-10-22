$ErrorActionPreference = 'Stop'
$release = gh release view v0.1.7 --json assets,isDraft,tagName,targetCommitish,url | ConvertFrom-Json
if ($LASTEXITCODE -ne 0) { throw 'Could not read draft release.' }
$expectedCommit = (git rev-parse HEAD).Trim()
if ($release.targetCommitish -ne $expectedCommit -or $release.tagName -ne 'v0.1.7') {
    throw 'Release target does not match the verified source commit.'
}
if ($release.assets.Count -ne 3) { throw 'Expected exactly two executables and one checksum file.' }
$files = @(
    'dist/NVAP-0.1.7-windows-cpu.exe',
    'dist/NVAP-0.1.7-windows-directml.exe',
    'build/release-v0.1.7-assets/SHA256SUMS.txt'
)
foreach ($file in $files) {
    $item = Get-Item -LiteralPath $file
    $asset = @($release.assets | Where-Object { $_.name -eq $item.Name })
    if ($asset.Count -ne 1 -or $asset[0].state -ne 'uploaded' -or $asset[0].size -ne $item.Length) {
        throw "Asset missing or incomplete: $($item.Name)"
    }
    $expectedDigest = 'sha256:' + (Get-FileHash -LiteralPath $file -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($asset[0].digest -ne $expectedDigest) { throw "GitHub checksum mismatch: $($item.Name)" }
    Write-Output "Verified GitHub asset: $($item.Name) ($($item.Length) bytes)."
}
gh release edit v0.1.7 --draft=false --latest
if ($LASTEXITCODE -ne 0) { throw 'Publishing the release failed.' }
$latest = gh api repos/gero-studios/NVAP/releases/latest | ConvertFrom-Json
if ($LASTEXITCODE -ne 0 -or $latest.tag_name -ne 'v0.1.7' -or $latest.draft -or $latest.prerelease) {
    throw 'Latest public release did not report v0.1.7.'
}
$tagCommit = (gh api repos/gero-studios/NVAP/git/ref/tags/v0.1.7 --jq '.object.sha').Trim()
if ($LASTEXITCODE -ne 0 -or $tagCommit -ne $expectedCommit) { throw 'Published tag does not match the verified commit.' }
Write-Output "Published $($latest.html_url) at $tagCommit"
$latest.assets | Select-Object name,size,browser_download_url | ConvertTo-Json


