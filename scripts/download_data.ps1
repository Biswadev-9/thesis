# Downloads the datasets required by the study (specification Step 2).
#
#   .\scripts\download_data.ps1                 # primary dataset only
#   .\scripts\download_data.ps1 -IncludeExternal # also the Figshare external set
#
# Requires Kaggle API credentials. Either place kaggle.json in %USERPROFILE%\.kaggle\,
# or set KAGGLE_USERNAME and KAGGLE_KEY in the project's .env file (see .env.example).

[CmdletBinding()]
param(
    [string]$DataDir = "data",
    [switch]$IncludeExternal,
    [switch]$Force
)

$ErrorActionPreference = "Stop"

# Primary dataset: four classes, the main training / validation / internal test source.
$PrimarySlug = "mohamadabouali1/mri-brain-tumor-dataset-4-class-7023-images"
$PrimaryDir = Join-Path $DataDir "raw\bt_mri"

# External dataset A: three classes, used for cross-dataset validation in Step 17.
$ExternalSlug = "ashkhagan/figshare-brain-tumor-dataset"
$ExternalDir = Join-Path $DataDir "raw\figshare"

function Import-DotEnv {
    param([string]$Path = ".env")
    if (-not (Test-Path $Path)) { return }
    foreach ($line in Get-Content $Path) {
        if ($line -match '^\s*#' -or $line -notmatch '=') { continue }
        $name, $value = $line -split '=', 2
        $name = $name.Trim()
        $value = $value.Trim().Trim('"').Trim("'")
        if ($name) { Set-Item -Path "env:$name" -Value $value }
    }
}

function Get-KaggleDataset {
    param([string]$Slug, [string]$Destination)

    if ((Test-Path $Destination) -and -not $Force) {
        $count = (Get-ChildItem -Path $Destination -Recurse -File -ErrorAction SilentlyContinue).Count
        if ($count -gt 0) {
            Write-Host "Already present ($count files): $Destination" -ForegroundColor Green
            return
        }
    }

    Write-Host "Downloading $Slug -> $Destination" -ForegroundColor Cyan
    New-Item -ItemType Directory -Force -Path $Destination | Out-Null
    kaggle datasets download -d $Slug -p $Destination --unzip
    if ($LASTEXITCODE -ne 0) { throw "kaggle download failed for $Slug (exit $LASTEXITCODE)" }

    $count = (Get-ChildItem -Path $Destination -Recurse -File).Count
    Write-Host "Done: $count files in $Destination" -ForegroundColor Green
}

Import-DotEnv

if (-not (Get-Command kaggle -ErrorAction SilentlyContinue)) {
    throw "The 'kaggle' CLI was not found. Install it with: pip install kaggle"
}

Get-KaggleDataset -Slug $PrimarySlug -Destination $PrimaryDir

# The archive nests its content one or two levels deep and the exact layout varies by
# mirror, so report what actually landed rather than assuming.
Write-Host "`nClass folders discovered under $PrimaryDir :" -ForegroundColor Cyan
Get-ChildItem -Path $PrimaryDir -Directory -Recurse -Depth 2 |
    Where-Object { (Get-ChildItem $_.FullName -File -Include *.jpg, *.jpeg, *.png -ErrorAction SilentlyContinue).Count -gt 0 } |
    ForEach-Object {
        $n = (Get-ChildItem $_.FullName -File).Count
        Write-Host ("  {0,-60} {1,6} images" -f $_.FullName.Replace((Resolve-Path $PrimaryDir), '.'), $n)
    }

if ($IncludeExternal) {
    Get-KaggleDataset -Slug $ExternalSlug -Destination $ExternalDir
}

Write-Host "`nIf the class folders sit deeper than $PrimaryDir, point the datamodule at them:" -ForegroundColor Yellow
Write-Host "  python src/analyze.py analysis=step04_audit data.raw_subdir=raw/bt_mri/<subfolder>" -ForegroundColor Yellow
Write-Host "`nNext: python src/analyze.py analysis=step04_audit" -ForegroundColor Green
