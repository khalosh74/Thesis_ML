param(
    [Parameter(Mandatory = $true)]
    [ValidateNotNullOrEmpty()]
    [string]$CampaignTag,

    [string]$OutputsRoot = "outputs",
    [string]$ArchiveRoot = "",
    [switch]$PruneAfterArchive,
    [switch]$Force
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Write-Step {
    param([string]$Message)
    Write-Host ""
    Write-Host "=== $Message ===" -ForegroundColor Cyan
}

function Resolve-FullPath {
    param([Parameter(Mandatory = $true)][string]$PathText)
    if ([System.IO.Path]::IsPathRooted($PathText)) {
        return [System.IO.Path]::GetFullPath($PathText)
    }
    return [System.IO.Path]::GetFullPath((Join-Path (Get-Location) $PathText))
}

function Test-DirectoryHasEntries {
    param([Parameter(Mandatory = $true)][string]$PathText)
    if (-not (Test-Path -LiteralPath $PathText -PathType Container)) {
        return $false
    }
    return $null -ne (Get-ChildItem -LiteralPath $PathText -Force | Select-Object -First 1)
}

function Get-UniqueArchiveSessionPath {
    param([Parameter(Mandatory = $true)][string]$RootPath)
    $stamp = (Get-Date).ToUniversalTime().ToString("yyyyMMddTHHmmssZ")
    $candidate = Join-Path $RootPath $stamp
    if (-not (Test-Path -LiteralPath $candidate)) {
        return $candidate
    }
    $suffix = 1
    while ($true) {
        $withSuffix = "{0}_{1:D2}" -f $candidate, $suffix
        if (-not (Test-Path -LiteralPath $withSuffix)) {
            return $withSuffix
        }
        $suffix += 1
    }
}

$outputsRootPath = Resolve-FullPath $OutputsRoot
if ([string]::IsNullOrWhiteSpace($ArchiveRoot)) {
    $archiveRootPath = Join-Path $outputsRootPath "archive"
} else {
    $archiveRootPath = Resolve-FullPath $ArchiveRoot
}

$campaignRoot = Join-Path (Join-Path $outputsRootPath "campaign") $CampaignTag
$comparisonRoot = Join-Path $campaignRoot "comparison"
$confirmatoryRoot = Join-Path $campaignRoot "confirmatory"
$releaseRoot = Join-Path $campaignRoot "release"
$bundleRoot = Join-Path $campaignRoot "bundle"
$logsRoot = Join-Path $campaignRoot "logs"

Write-Step "Preflight checks"

if (Test-Path -LiteralPath $campaignRoot -PathType Leaf) {
    throw "Campaign root path exists as a file: $campaignRoot"
}
if (Test-DirectoryHasEntries $campaignRoot) {
    if (-not $Force) {
        throw "Campaign root already exists and is non-empty: $campaignRoot. Use -Force to overwrite it."
    }
    Write-Host "Removing existing non-empty campaign root because -Force was set: $campaignRoot" -ForegroundColor Yellow
    Remove-Item -LiteralPath $campaignRoot -Recurse -Force
}

New-Item -ItemType Directory -Path $outputsRootPath -Force | Out-Null

$legacyFolders = @("manual_validation", "reproducibility", "reports")
$foldersToArchive = @()
foreach ($folderName in $legacyFolders) {
    $candidate = Join-Path $outputsRootPath $folderName
    if (Test-Path -LiteralPath $candidate -PathType Container) {
        $foldersToArchive += [PSCustomObject]@{
            name        = $folderName
            source_path = $candidate
        }
    }
}

$archiveSessionPath = $null
$archivedFolders = @()

if ($foldersToArchive.Count -gt 0) {
    Write-Step "Archiving legacy output folders"
    New-Item -ItemType Directory -Path $archiveRootPath -Force | Out-Null
    $archiveSessionPath = Get-UniqueArchiveSessionPath -RootPath $archiveRootPath
    New-Item -ItemType Directory -Path $archiveSessionPath -Force | Out-Null

    foreach ($folder in $foldersToArchive) {
        $destination = Join-Path $archiveSessionPath $folder.name
        Write-Host "Archiving '$($folder.source_path)' -> '$destination' (method: copy)" -ForegroundColor Gray
        Copy-Item -LiteralPath $folder.source_path -Destination $destination -Recurse -Force
        if (-not (Test-Path -LiteralPath $destination -PathType Container)) {
            throw "Archive failed for '$($folder.source_path)': destination missing '$destination'."
        }
        $archivedFolders += [PSCustomObject]@{
            name         = $folder.name
            source_path  = $folder.source_path
            archive_path = $destination
            method       = "copy"
        }
    }
} else {
    Write-Step "Archiving legacy output folders"
    Write-Host "No legacy folders found to archive under '$outputsRootPath'." -ForegroundColor Gray
}

$prunedFolders = @()
if ($PruneAfterArchive -and $archivedFolders.Count -gt 0) {
    Write-Step "Pruning legacy output folders after successful archive"
    foreach ($entry in $archivedFolders) {
        if (-not (Test-Path -LiteralPath $entry.archive_path -PathType Container)) {
            throw "Refusing to prune '$($entry.source_path)' because archived copy is missing."
        }
        Remove-Item -LiteralPath $entry.source_path -Recurse -Force
        $prunedFolders += [PSCustomObject]@{
            name        = $entry.name
            source_path = $entry.source_path
        }
        Write-Host "Pruned '$($entry.source_path)'." -ForegroundColor Yellow
    }
}

Write-Step "Creating fresh campaign root"
$createdFolders = @()
foreach ($pathText in @($campaignRoot, $comparisonRoot, $confirmatoryRoot, $releaseRoot, $bundleRoot, $logsRoot)) {
    if (-not (Test-Path -LiteralPath $pathText -PathType Container)) {
        New-Item -ItemType Directory -Path $pathText -Force | Out-Null
        $createdFolders += $pathText
    }
}

$summary = [ordered]@{
    timestamp_utc        = (Get-Date).ToUniversalTime().ToString("o")
    campaign_tag         = $CampaignTag
    outputs_root         = $outputsRootPath
    archive_root         = $archiveRootPath
    archive_session_path = $archiveSessionPath
    archived_folders     = @($archivedFolders)
    pruned_folders       = @($prunedFolders)
    campaign_root        = $campaignRoot
    created_folders      = @($createdFolders)
    prune_after_archive  = [bool]$PruneAfterArchive
    force                = [bool]$Force
}

$summaryPath = Join-Path $releaseRoot "prep_summary.json"
$summary | ConvertTo-Json -Depth 8 | Set-Content -Encoding utf8 -Path $summaryPath

Write-Step "Preparation summary"
Write-Host "Campaign root: $campaignRoot" -ForegroundColor Green
if ($archiveSessionPath) {
    Write-Host "Archive session: $archiveSessionPath" -ForegroundColor Green
} else {
    Write-Host "Archive session: none (no legacy folders found)." -ForegroundColor Green
}
Write-Host "Archived folders: $($archivedFolders.Count)" -ForegroundColor Green
Write-Host "Pruned folders: $($prunedFolders.Count)" -ForegroundColor Green
Write-Host "Summary JSON: $summaryPath" -ForegroundColor Green
