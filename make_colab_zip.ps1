# make_colab_zip.ps1 — Build a clean .zip of the project for Colab upload.
#
# Includes everything needed to run the experiments end-to-end on Colab,
# excludes the heavy/derived files (.git, .venv, .hdf5, results, caches).
# Uses .NET's ZipFile API directly (NOT Compress-Archive, which writes
# backslashes that break Linux unzip and Python zipfile).
#
# Run from PowerShell, in the project folder:
#     .\make_colab_zip.ps1
#
# Output: ..\Structural-Probes-for-Spanish-Syntax-colab.zip

$ErrorActionPreference = "Stop"
Add-Type -AssemblyName 'System.IO.Compression.FileSystem'

$srcDir   = (Get-Location).Path
$projName = Split-Path -Leaf $srcDir
$zipPath  = Join-Path (Split-Path -Parent $srcDir) "${projName}-colab.zip"

Write-Host "Source : $srcDir"
Write-Host "Output : $zipPath"
Write-Host ""

if (Test-Path $zipPath) { Remove-Item -Force $zipPath }

# Patterns to exclude
$excludeDirs  = @('.git', '.venv', '__pycache__', 'results', 'runs', 'example')
$excludeFiles = @('*.hdf5', '*.zip', '*.pyc')

function Should-Exclude($file, $relPath) {
    foreach ($d in $excludeDirs) {
        if ($relPath -match "(^|[/\\])$d([/\\]|$)") { return $true }
    }
    foreach ($p in $excludeFiles) {
        if ($file.Name -like $p) { return $true }
    }
    return $false
}

$copied = 0
$zip = [System.IO.Compression.ZipFile]::Open($zipPath, [System.IO.Compression.ZipArchiveMode]::Create)
try {
    Get-ChildItem -Path $srcDir -Recurse -File | ForEach-Object {
        $rel = $_.FullName.Substring($srcDir.Length + 1)
        if (Should-Exclude $_ $rel) { return }
        # CRITICAL: force forward-slash separators (ZIP spec / cross-platform)
        $entryName = $rel -replace '\\', '/'
        [System.IO.Compression.ZipFileExtensions]::CreateEntryFromFile(
            $zip, $_.FullName, $entryName,
            [System.IO.Compression.CompressionLevel]::Optimal) | Out-Null
        $copied++
    }
} finally {
    $zip.Dispose()
}

$zipMB = [math]::Round((Get-Item $zipPath).Length / 1MB, 2)
Write-Host ""
Write-Host "==> Done." -ForegroundColor Green
Write-Host "    Zip path : $zipPath"
Write-Host "    Files    : $copied"
Write-Host "    Size     : $zipMB MB"
Write-Host ""
Write-Host "Upload this file to Colab when Cell 1 prompts for it." -ForegroundColor Yellow
