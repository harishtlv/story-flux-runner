param(
  [string]$Output = "story-flux-runner-colab.zip"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$zipPath = Join-Path $root $Output
$stage = Join-Path $env:TEMP ("story-flux-runner-colab-" + [guid]::NewGuid().ToString("N"))

$excludedDirs = @(".git", "node_modules", ".agents")
$excludedFiles = @(".env", $Output)

New-Item -ItemType Directory -Path $stage | Out-Null

try {
  Get-ChildItem -LiteralPath $root -Force | ForEach-Object {
    if ($_.PSIsContainer -and ($excludedDirs -contains $_.Name)) { return }
    if (-not $_.PSIsContainer -and ($excludedFiles -contains $_.Name)) { return }

    $dest = Join-Path $stage $_.Name
    Copy-Item -LiteralPath $_.FullName -Destination $dest -Recurse -Force
  }

  if (Test-Path -LiteralPath $zipPath) {
    Remove-Item -LiteralPath $zipPath -Force
  }

  Compress-Archive -Path (Join-Path $stage "*") -DestinationPath $zipPath -Force
  Write-Host "Created $zipPath"
  Write-Host "Upload this zip in the first Colab upload cell."
} finally {
  if (Test-Path -LiteralPath $stage) {
    Remove-Item -LiteralPath $stage -Recurse -Force
  }
}
