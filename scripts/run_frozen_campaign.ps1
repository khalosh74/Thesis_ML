param(
    [Parameter(Mandatory = $true)]
    [string]$CampaignTag,

    [Parameter(Mandatory = $true)]
    [string]$IndexCsv,

    [Parameter(Mandatory = $true)]
    [string]$DataRoot,

    [Parameter(Mandatory = $true)]
    [string]$CacheDir,

    [string]$CampaignRoot = "",

    [string]$ConfirmatoryProtocol = "configs/protocols/thesis_confirmatory_v1.json",

    [string[]]$ComparisonSpecs = @(
        "configs/comparisons/model_family_comparison_v1.json",
        "configs/comparisons/model_family_grouped_nested_comparison_v1.json"
    ),

    # Primary comparison bundle target; used by build_publishable_bundle.py
    [string]$PrimaryComparisonSpec = "configs/comparisons/model_family_comparison_v1.json",

    [switch]$SkipGitCleanCheck,
    [switch]$SkipTagCreation
)

$ErrorActionPreference = "Stop"

function Write-Step {
    param([string]$Message)
    Write-Host ""
    Write-Host "=== $Message ===" -ForegroundColor Cyan
}

function Assert-PathExists {
    param([string]$Path, [string]$Label)
    if (-not (Test-Path $Path)) {
        throw "$Label not found: $Path"
    }
}

function Invoke-Checked {
    param(
        [string]$Label,
        [scriptblock]$Command
    )
    Write-Step $Label
    & $Command
    if ($LASTEXITCODE -ne 0) {
        throw "Step failed: $Label (exit code: $LASTEXITCODE)"
    }
}

function Get-SpecName {
    param([string]$SpecPath)
    return [System.IO.Path]::GetFileNameWithoutExtension($SpecPath)
}

function Copy-IfExists {
    param(
        [string]$Source,
        [string]$Destination
    )
    if (Test-Path $Source) {
        Copy-Item $Source $Destination -Force
    }
}

# Resolve campaign root
if ([string]::IsNullOrWhiteSpace($CampaignRoot)) {
    $CampaignRoot = "outputs/campaign/$CampaignTag"
}

$ReleaseRoot      = Join-Path $CampaignRoot "release"
$ComparisonRoot   = Join-Path $CampaignRoot "comparison"
$ConfirmatoryRoot = Join-Path $CampaignRoot "confirmatory"
$BundleRoot       = Join-Path $CampaignRoot "bundle"
$LogRoot          = Join-Path $CampaignRoot "logs"

# Validate inputs
Assert-PathExists $IndexCsv "IndexCsv"
Assert-PathExists $DataRoot "DataRoot"
Assert-PathExists $CacheDir "CacheDir"
Assert-PathExists $ConfirmatoryProtocol "ConfirmatoryProtocol"
foreach ($Spec in $ComparisonSpecs) { Assert-PathExists $Spec "ComparisonSpec" }
Assert-PathExists $PrimaryComparisonSpec "PrimaryComparisonSpec"

# Create dirs
New-Item -ItemType Directory -Force -Path $CampaignRoot      | Out-Null
New-Item -ItemType Directory -Force -Path $ReleaseRoot       | Out-Null
New-Item -ItemType Directory -Force -Path $ComparisonRoot    | Out-Null
New-Item -ItemType Directory -Force -Path $ConfirmatoryRoot  | Out-Null
New-Item -ItemType Directory -Force -Path $BundleRoot        | Out-Null
New-Item -ItemType Directory -Force -Path $LogRoot           | Out-Null

# Export official data env
$env:THESIS_ML_INDEX_CSV = $IndexCsv
$env:THESIS_ML_DATA_ROOT = $DataRoot
$env:THESIS_ML_CACHE_DIR = $CacheDir

Write-Step "Repository state"

if (-not $SkipGitCleanCheck) {
    $gitStatus = git status --porcelain=v1
    $gitStatus | Set-Content (Join-Path $LogRoot "git_status_porcelain.txt")
    if ($gitStatus) {
        throw "Repository is not clean. Commit or stash changes before freezing."
    }
} else {
    "Skipped git clean check." | Set-Content (Join-Path $LogRoot "git_status_porcelain.txt")
}

git rev-parse HEAD        | Tee-Object (Join-Path $ReleaseRoot "commit_sha.txt")
git branch --show-current | Tee-Object (Join-Path $ReleaseRoot "branch.txt")

Invoke-Checked "Release hygiene check" {
    python scripts/release_hygiene_check.py | Tee-Object (Join-Path $LogRoot "release_hygiene.log")
}

Invoke-Checked "Full test suite" {
    python -m pytest -q | Tee-Object (Join-Path $LogRoot "pytest.log")
}

Invoke-Checked "RC1 release gate" {
    python scripts/rc1_release_gate.py `
        --summary-out (Join-Path $ReleaseRoot "rc1_gate_summary.json") `
        --run-ruff `
        --run-performance-smoke
}

Write-Step "Create git tag"
if (-not $SkipTagCreation) {
    git tag $CampaignTag
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to create git tag: $CampaignTag"
    }
    git rev-parse $CampaignTag | Tee-Object (Join-Path $ReleaseRoot "frozen_tag_commit.txt")
} else {
    "Skipped tag creation." | Set-Content (Join-Path $ReleaseRoot "frozen_tag_commit.txt")
}

# Run locked comparisons
foreach ($Spec in $ComparisonSpecs) {
    $SpecName = Get-SpecName $Spec
    $SpecOut  = Join-Path $ComparisonRoot $SpecName

    Invoke-Checked "Run comparison: $SpecName" {
        python -m Thesis_ML.cli.comparison_runner `
            --comparison $Spec `
            --all-variants `
            --reports-root $SpecOut `
            --force
    }

    $ComparisonRunDir = Join-Path $SpecOut "comparison_runs"
    $ResolvedComparisonDir = Get-ChildItem $ComparisonRunDir -Directory | Select-Object -First 1
    if (-not $ResolvedComparisonDir) {
        throw "No comparison run directory found under: $ComparisonRunDir"
    }

    Invoke-Checked "Verify comparison artifacts: $SpecName" {
        python scripts/verify_official_artifacts.py `
            --mode comparison `
            --output-dir $ResolvedComparisonDir.FullName `
            --summary-out (Join-Path $ResolvedComparisonDir.FullName "artifact_verification_summary.json")
    }
}

# Run frozen confirmatory campaign
Invoke-Checked "Run frozen confirmatory protocol" {
    python -m Thesis_ML.cli.protocol_runner `
        --protocol $ConfirmatoryProtocol `
        --all-suites `
        --reports-root $ConfirmatoryRoot `
        --force
}

$ConfirmatoryProtocolRuns = Join-Path $ConfirmatoryRoot "protocol_runs"
$ResolvedConfirmatoryDir = Get-ChildItem $ConfirmatoryProtocolRuns -Directory | Where-Object { $_.Name -like "thesis_confirmatory_v1*" } | Select-Object -First 1
if (-not $ResolvedConfirmatoryDir) {
    $ResolvedConfirmatoryDir = Get-ChildItem $ConfirmatoryProtocolRuns -Directory | Select-Object -First 1
}
if (-not $ResolvedConfirmatoryDir) {
    throw "No confirmatory protocol run directory found under: $ConfirmatoryProtocolRuns"
}

Invoke-Checked "Verify confirmatory artifacts" {
    python scripts/verify_official_artifacts.py `
        --mode confirmatory `
        --output-dir $ResolvedConfirmatoryDir.FullName `
        --summary-out (Join-Path $ConfirmatoryRoot "artifact_verification_summary.json")
}

Invoke-Checked "Verify confirmatory-ready status" {
    python scripts/verify_confirmatory_ready.py `
        --output-dir $ResolvedConfirmatoryDir.FullName `
        --summary-out (Join-Path $ConfirmatoryRoot "confirmatory_ready_summary.json")
}

# Deterministic reproducibility checks
Invoke-Checked "Deterministic comparison reproducibility" {
    python scripts/verify_official_reproducibility.py `
        --mode comparison `
        --config $PrimaryComparisonSpec `
        --index-csv $IndexCsv `
        --data-root $DataRoot `
        --cache-dir $CacheDir `
        --variant ridge `
        --reports-root (Join-Path $ReleaseRoot "determinism_comparison") `
        --summary-out (Join-Path $ReleaseRoot "determinism_comparison_summary.json")
}

Invoke-Checked "Deterministic confirmatory reproducibility" {
    python scripts/verify_official_reproducibility.py `
        --mode protocol `
        --config $ConfirmatoryProtocol `
        --index-csv $IndexCsv `
        --data-root $DataRoot `
        --cache-dir $CacheDir `
        --suite confirmatory_primary_within_subject `
        --reports-root (Join-Path $ReleaseRoot "determinism_confirmatory") `
        --summary-out (Join-Path $ReleaseRoot "determinism_confirmatory_summary.json")
}

# One-command replay / manifest generation
Invoke-Checked "Official replay orchestration" {
    python scripts/replay_official_paths.py `
        --mode both `
        --index-csv $IndexCsv `
        --data-root $DataRoot `
        --cache-dir $CacheDir `
        --reports-root (Join-Path $ReleaseRoot "official_replay") `
        --verify-determinism `
        --summary-out (Join-Path $ReleaseRoot "replay_summary.json") `
        --verification-summary-out (Join-Path $ReleaseRoot "replay_verification_summary.json") `
        --manifest-out (Join-Path $ReleaseRoot "reproducibility_manifest.json")
}

# Resolve primary comparison dir for bundle
$PrimarySpecName = Get-SpecName $PrimaryComparisonSpec
$PrimarySpecRoot = Join-Path $ComparisonRoot $PrimarySpecName
$PrimaryComparisonRunDir = Join-Path $PrimarySpecRoot "comparison_runs"
$ResolvedPrimaryComparisonDir = Get-ChildItem $PrimaryComparisonRunDir -Directory | Select-Object -First 1
if (-not $ResolvedPrimaryComparisonDir) {
    throw "Primary comparison output not found under: $PrimaryComparisonRunDir"
}

# Build and verify bundle
Invoke-Checked "Build publishable bundle" {
    python scripts/build_publishable_bundle.py `
        --output-dir $BundleRoot `
        --comparison-output $ResolvedPrimaryComparisonDir.FullName `
        --confirmatory-output $ResolvedConfirmatoryDir.FullName `
        --confirmatory-ready-summary (Join-Path $ConfirmatoryRoot "confirmatory_ready_summary.json") `
        --replay-summary (Join-Path $ReleaseRoot "replay_summary.json") `
        --replay-verification-summary (Join-Path $ReleaseRoot "replay_verification_summary.json") `
        --repro-manifest (Join-Path $ReleaseRoot "reproducibility_manifest.json")
}

Invoke-Checked "Verify publishable bundle" {
    python scripts/verify_publishable_bundle.py `
        --bundle-dir $BundleRoot `
        --summary-out (Join-Path $BundleRoot "bundle_verification_summary.json")
}

# Campaign record
@"
campaign_tag=$CampaignTag
confirmatory_protocol=$ConfirmatoryProtocol
index_csv=$IndexCsv
data_root=$DataRoot
cache_dir=$CacheDir
primary_comparison_spec=$PrimaryComparisonSpec
"@ | Set-Content (Join-Path $ReleaseRoot "campaign_record.txt")

# Archive high-level summaries into release root
Copy-IfExists (Join-Path $ConfirmatoryRoot "artifact_verification_summary.json") (Join-Path $ReleaseRoot "confirmatory_artifact_verification_summary.json")
Copy-IfExists (Join-Path $ConfirmatoryRoot "confirmatory_ready_summary.json")   (Join-Path $ReleaseRoot "confirmatory_ready_summary.json")
Copy-IfExists (Join-Path $BundleRoot "bundle_verification_summary.json")         (Join-Path $ReleaseRoot "bundle_verification_summary.json")

Write-Step "Final go/no-go checks"

$RequiredFiles = @(
    (Join-Path $ReleaseRoot "rc1_gate_summary.json"),
    (Join-Path $ReleaseRoot "determinism_comparison_summary.json"),
    (Join-Path $ReleaseRoot "determinism_confirmatory_summary.json"),
    (Join-Path $ReleaseRoot "replay_summary.json"),
    (Join-Path $ReleaseRoot "replay_verification_summary.json"),
    (Join-Path $ReleaseRoot "reproducibility_manifest.json"),
    (Join-Path $ReleaseRoot "confirmatory_artifact_verification_summary.json"),
    (Join-Path $ReleaseRoot "confirmatory_ready_summary.json"),
    (Join-Path $ReleaseRoot "bundle_verification_summary.json"),
    (Join-Path $ReleaseRoot "campaign_record.txt")
)

$Missing = @()
foreach ($File in $RequiredFiles) {
    if (-not (Test-Path $File)) { $Missing += $File }
}

if ($Missing.Count -gt 0) {
    Write-Host "Missing required campaign outputs:" -ForegroundColor Red
    $Missing | ForEach-Object { Write-Host " - $_" -ForegroundColor Red }
    throw "Campaign validation failed due to missing required outputs."
}

Write-Host ""
Write-Host "Frozen campaign completed successfully." -ForegroundColor Green
Write-Host "Campaign root: $CampaignRoot" -ForegroundColor Green
Write-Host "Do not modify framework/config/policy code after this point." -ForegroundColor Yellow