
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

    [string]$PrimaryComparisonSpec = "configs/comparisons/model_family_comparison_v1.json",

    [ValidateSet("precheck", "confirmatory", "comparison", "replay", "bundle", "all")]
    [string]$Phase = "all",

    [switch]$SkipGitCleanCheck,
    [switch]$SkipTagCreation
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Write-Step {
    param([string]$Message)
    Write-Host ""
    Write-Host "=== $Message ===" -ForegroundColor Cyan
}

function Assert-PathExists {
    param([string]$Path, [string]$Label)
    if (-not (Test-Path -LiteralPath $Path)) {
        throw "$Label not found: $Path"
    }
}

function Get-SpecName {
    param([string]$SpecPath)
    return [System.IO.Path]::GetFileNameWithoutExtension($SpecPath)
}

function Get-UtcNow {
    return (Get-Date).ToUniversalTime().ToString("o")
}

function Write-JsonFile {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][object]$Payload
    )
    $parent = Split-Path -Path $Path -Parent
    if (-not [string]::IsNullOrWhiteSpace($parent)) {
        New-Item -ItemType Directory -Path $parent -Force | Out-Null
    }
    $Payload | ConvertTo-Json -Depth 32 | Set-Content -Path $Path -Encoding utf8
}

function Read-JsonFile {
    param([Parameter(Mandatory = $true)][string]$Path)
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        return $null
    }
    $raw = Get-Content -LiteralPath $Path -Raw
    if ([string]::IsNullOrWhiteSpace($raw)) {
        return $null
    }
    return ($raw | ConvertFrom-Json)
}

function Convert-CommandPartsToString {
    param([string[]]$CommandParts)
    $tokens = foreach ($part in $CommandParts) {
        if ($part -match "\s") {
            '"{0}"' -f $part.Replace('"', '\"')
        }
        else {
            $part
        }
    }
    return ($tokens -join " ")
}
function New-DefaultPhaseStatusRecord {
    param([string]$PhaseName)
    return [ordered]@{
        phase                   = $PhaseName
        status                  = "pending"
        start_utc               = ""
        end_utc                 = ""
        duration_seconds        = $null
        output_root             = $PhaseDefinitions[$PhaseName].output_root
        dependencies            = @($PhaseDefinitions[$PhaseName].dependencies)
        blocking_classification = @($PhaseDefinitions[$PhaseName].blocking_for)
        error_summary           = ""
        updated_at_utc          = (Get-UtcNow)
    }
}

function Get-PhaseOutputRoot {
    param([string]$PhaseName)
    return [string]$PhaseDefinitions[$PhaseName].output_root
}

function Get-PhaseStatusPath {
    param([string]$PhaseName)
    return (Join-Path (Get-PhaseOutputRoot $PhaseName) "phase_status.json")
}

function Get-PhaseSummaryPath {
    param([string]$PhaseName)
    return (Join-Path (Get-PhaseOutputRoot $PhaseName) "phase_summary.json")
}

function Get-PhaseStatusRecord {
    param([string]$PhaseName)
    $statusPath = Get-PhaseStatusPath -PhaseName $PhaseName
    $record = Read-JsonFile -Path $statusPath
    if ($null -eq $record) {
        return (New-DefaultPhaseStatusRecord -PhaseName $PhaseName)
    }
    return $record
}

function Get-PhaseStatusValue {
    param([string]$PhaseName)
    $record = Get-PhaseStatusRecord -PhaseName $PhaseName
    $value = [string]$record.status
    if ([string]::IsNullOrWhiteSpace($value)) {
        return "pending"
    }
    return $value
}

function Write-PhaseStatus {
    param(
        [Parameter(Mandatory = $true)][string]$PhaseName,
        [Parameter(Mandatory = $true)][string]$Status,
        [string]$StartUtc = "",
        [string]$EndUtc = "",
        [string]$ErrorSummary = ""
    )

    $durationSeconds = $null
    if (-not [string]::IsNullOrWhiteSpace($StartUtc) -and -not [string]::IsNullOrWhiteSpace($EndUtc)) {
        try {
            $start = [DateTime]::Parse($StartUtc)
            $end = [DateTime]::Parse($EndUtc)
            $durationSeconds = [Math]::Round(($end - $start).TotalSeconds, 3)
        }
        catch {
            $durationSeconds = $null
        }
    }

    $statusPayload = [ordered]@{
        phase                   = $PhaseName
        status                  = $Status
        start_utc               = $StartUtc
        end_utc                 = $EndUtc
        duration_seconds        = $durationSeconds
        output_root             = (Get-PhaseOutputRoot -PhaseName $PhaseName)
        dependencies            = @($PhaseDefinitions[$PhaseName].dependencies)
        blocking_classification = @($PhaseDefinitions[$PhaseName].blocking_for)
        error_summary           = $ErrorSummary
        updated_at_utc          = (Get-UtcNow)
    }

    Write-JsonFile -Path (Get-PhaseStatusPath -PhaseName $PhaseName) -Payload $statusPayload
}

function Get-DownstreamReadiness {
    param([string]$PhaseName)

    $result = [ordered]@{}
    foreach ($candidate in $OrderedPhases) {
        $deps = @($PhaseDefinitions[$candidate].dependencies)
        if ($deps -contains $PhaseName) {
            $unmet = @()
            foreach ($dep in $deps) {
                if ((Get-PhaseStatusValue -PhaseName $dep) -ne "passed") {
                    $unmet += $dep
                }
            }
            $result[$candidate] = [ordered]@{
                ready              = ($unmet.Count -eq 0)
                unmet_dependencies = @($unmet)
            }
        }
    }
    return $result
}

function Get-UnmetDependencies {
    param([string]$PhaseName)

    $missing = @()
    foreach ($dep in @($PhaseDefinitions[$PhaseName].dependencies)) {
        $depStatus = Get-PhaseStatusValue -PhaseName $dep
        if ($depStatus -ne "passed") {
            $missing += [ordered]@{
                phase  = $dep
                status = $depStatus
            }
        }
    }
    return @($missing)
}
function New-PhaseContext {
    param([string]$PhaseName)

    $phaseOutputRoot = Get-PhaseOutputRoot -PhaseName $PhaseName
    $phaseLogRoot = Join-Path $LogRoot $PhaseName
    New-Item -ItemType Directory -Path $phaseOutputRoot -Force | Out-Null
    New-Item -ItemType Directory -Path $phaseLogRoot -Force | Out-Null

    return [ordered]@{
        phase                  = $PhaseName
        output_root            = $phaseOutputRoot
        log_root               = $phaseLogRoot
        commands               = @()
        warnings               = @()
        key_outputs            = @()
        verification_summaries = @()
        extra                  = [ordered]@{}
        inputs                 = [ordered]@{
            campaign_tag            = $CampaignTag
            campaign_root           = $CampaignRoot
            index_csv               = $IndexCsv
            data_root               = $DataRoot
            cache_dir               = $CacheDir
            confirmatory_protocol   = $ConfirmatoryProtocol
            comparison_specs        = @($ComparisonSpecs)
            primary_comparison_spec = $PrimaryComparisonSpec
            skip_git_clean_check    = [bool]$SkipGitCleanCheck
            skip_tag_creation       = [bool]$SkipTagCreation
        }
    }
}

function New-LogFileName {
    param(
        $Context,
        [string]$Label
    )

    $index = [int]$Context.commands.Count + 1
    $slug = ($Label.ToLowerInvariant() -replace "[^a-z0-9]+", "_").Trim("_")
    if ([string]::IsNullOrWhiteSpace($slug)) {
        $slug = "step"
    }
    return (Join-Path $Context.log_root ("{0:D2}_{1}.log" -f $index, $slug))
}

function Invoke-PhaseCommand {
    param(
        [Parameter(Mandatory = $true)]$Context,
        [Parameter(Mandatory = $true)][string]$Label,
        [Parameter(Mandatory = $true)][string[]]$CommandParts,
        [string]$LogPath = ""
    )

    if ([string]::IsNullOrWhiteSpace($LogPath)) {
        $LogPath = New-LogFileName -Context $Context -Label $Label
    }

    $commandText = Convert-CommandPartsToString -CommandParts $CommandParts
    Write-Step "$($Context.phase): $Label"

    $started = Get-Date
    $exe = $CommandParts[0]
    $args = @()
    if ($CommandParts.Count -gt 1) {
        $args = $CommandParts[1..($CommandParts.Count - 1)]
    }

    $previousErrorActionPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        $output = & $exe @args 2>&1
    }
    finally {
        $ErrorActionPreference = $previousErrorActionPreference
    }
    @($output | ForEach-Object { $_.ToString() }) | Tee-Object -FilePath $LogPath | Out-Null

    $exitCode = if ($null -eq $LASTEXITCODE) { 0 } else { [int]$LASTEXITCODE }
    $ended = Get-Date

    $record = [ordered]@{
        label            = $Label
        command          = $commandText
        log_path         = $LogPath
        exit_code        = $exitCode
        start_utc        = $started.ToUniversalTime().ToString("o")
        end_utc          = $ended.ToUniversalTime().ToString("o")
        duration_seconds = [Math]::Round(($ended - $started).TotalSeconds, 3)
    }
    $Context.commands += $record

    if ($exitCode -ne 0) {
        throw "Step failed: $Label (exit code: $exitCode). See log: $LogPath"
    }

    return [ordered]@{
        output    = @($output | ForEach-Object { $_.ToString() })
        command   = $commandText
        log_path  = $LogPath
        exit_code = $exitCode
    }
}

function Write-PhaseSummary {
    param(
        [Parameter(Mandatory = $true)]$Context,
        [Parameter(Mandatory = $true)][string]$Status,
        [string]$ErrorSummary = ""
    )

    $phaseName = [string]$Context.phase
    $dependencyStatus = [ordered]@{}
    foreach ($dep in @($PhaseDefinitions[$phaseName].dependencies)) {
        $dependencyStatus[$dep] = Get-PhaseStatusValue -PhaseName $dep
    }

    $summary = [ordered]@{
        phase                       = $phaseName
        status                      = $Status
        output_root                 = $Context.output_root
        inputs                      = $Context.inputs
        dependency_status           = $dependencyStatus
        commands_executed           = @($Context.commands)
        key_outputs_generated       = @($Context.key_outputs)
        verification_summaries_used = @($Context.verification_summaries)
        warnings                    = @($Context.warnings)
        downstream_phase_readiness  = (Get-DownstreamReadiness -PhaseName $phaseName)
        phase_details               = $Context.extra
        error_summary               = $ErrorSummary
        generated_at_utc            = (Get-UtcNow)
    }

    Write-JsonFile -Path (Get-PhaseSummaryPath -PhaseName $phaseName) -Payload $summary
}
function Try-ReadFirstLine {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        return ""
    }
    $line = Get-Content -LiteralPath $Path -TotalCount 1
    if ($null -eq $line) {
        return ""
    }
    return [string]$line
}

function Try-GitValue {
    param([string[]]$Args)
    try {
        $output = & git @Args 2>$null
        if ($LASTEXITCODE -ne 0) {
            return ""
        }
        if ($null -eq $output) {
            return ""
        }
        $first = @($output)[0]
        return [string]$first
    }
    catch {
        return ""
    }
}

function Update-CampaignManifest {
    $existing = Read-JsonFile -Path $CampaignManifestPath
    $createdAt = ""
    if ($null -ne $existing -and -not [string]::IsNullOrWhiteSpace([string]$existing.created_at_utc)) {
        $createdAt = [string]$existing.created_at_utc
    }
    if ([string]::IsNullOrWhiteSpace($createdAt)) {
        $createdAt = Get-UtcNow
    }

    $phaseStatuses = [ordered]@{}
    foreach ($phaseName in $OrderedPhases) {
        $phaseStatuses[$phaseName] = Get-PhaseStatusRecord -PhaseName $phaseName
    }

    $commitSha = Try-ReadFirstLine -Path (Join-Path $PrecheckRoot "commit_sha.txt")
    if ([string]::IsNullOrWhiteSpace($commitSha)) {
        $commitSha = (Try-GitValue -Args @("rev-parse", "HEAD")).Trim()
    }

    $branchName = Try-ReadFirstLine -Path (Join-Path $PrecheckRoot "branch.txt")
    if ([string]::IsNullOrWhiteSpace($branchName)) {
        $branchName = (Try-GitValue -Args @("branch", "--show-current")).Trim()
    }

    $tagCommit = Try-ReadFirstLine -Path (Join-Path $PrecheckRoot "frozen_tag_commit.txt")
    $tagExists = $false
    if (-not [string]::IsNullOrWhiteSpace($tagCommit) -and $tagCommit -notmatch "Skipped") {
        $tagExists = $true
    }
    else {
        $resolvedTag = (Try-GitValue -Args @("rev-parse", "--verify", $CampaignTag)).Trim()
        if (-not [string]::IsNullOrWhiteSpace($resolvedTag)) {
            $tagExists = $true
            $tagCommit = $resolvedTag
        }
    }

    $manifest = [ordered]@{
        schema_version = "frozen-campaign-manifest-v1"
        campaign_tag   = $CampaignTag
        campaign_root  = $CampaignRoot
        created_at_utc = $createdAt
        updated_at_utc = (Get-UtcNow)
        frozen_context = [ordered]@{
            commit_sha = $commitSha
            branch     = $branchName
            tag        = [ordered]@{
                name       = $CampaignTag
                exists     = [bool]$tagExists
                commit_sha = $tagCommit
            }
        }
        inputs = [ordered]@{
            index_csv = $IndexCsv
            data_root = $DataRoot
            cache_dir = $CacheDir
        }
        specs = [ordered]@{
            confirmatory_protocol   = $ConfirmatoryProtocol
            comparison_specs        = @($ComparisonSpecs)
            primary_comparison_spec = $PrimaryComparisonSpec
        }
        phase_definitions = [ordered]@{
            precheck = [ordered]@{ dependencies = @($PhaseDefinitions.precheck.dependencies); output_root = $PhaseDefinitions.precheck.output_root; blocking_for = @($PhaseDefinitions.precheck.blocking_for) }
            confirmatory = [ordered]@{ dependencies = @($PhaseDefinitions.confirmatory.dependencies); output_root = $PhaseDefinitions.confirmatory.output_root; blocking_for = @($PhaseDefinitions.confirmatory.blocking_for) }
            comparison = [ordered]@{ dependencies = @($PhaseDefinitions.comparison.dependencies); output_root = $PhaseDefinitions.comparison.output_root; blocking_for = @($PhaseDefinitions.comparison.blocking_for) }
            replay = [ordered]@{ dependencies = @($PhaseDefinitions.replay.dependencies); output_root = $PhaseDefinitions.replay.output_root; blocking_for = @($PhaseDefinitions.replay.blocking_for) }
            bundle = [ordered]@{ dependencies = @($PhaseDefinitions.bundle.dependencies); output_root = $PhaseDefinitions.bundle.output_root; blocking_for = @($PhaseDefinitions.bundle.blocking_for) }
        }
        phase_statuses = $phaseStatuses
        blocking_model = [ordered]@{
            confirmatory_readiness_blockers = @("precheck", "confirmatory")
            campaign_signoff_blockers       = @($OrderedPhases)
        }
    }

    Write-JsonFile -Path $CampaignManifestPath -Payload $manifest
}

function Ensure-PhaseStatusFiles {
    foreach ($phaseName in $OrderedPhases) {
        $statusPath = Get-PhaseStatusPath -PhaseName $phaseName
        if (-not (Test-Path -LiteralPath $statusPath -PathType Leaf)) {
            Write-JsonFile -Path $statusPath -Payload (New-DefaultPhaseStatusRecord -PhaseName $phaseName)
        }
    }
}

function Get-ConfirmatoryOutputDirFromSummary {
    $summary = Read-JsonFile -Path (Get-PhaseSummaryPath -PhaseName "confirmatory")
    if ($null -eq $summary -or $null -eq $summary.phase_details) {
        throw "Confirmatory phase summary missing. Run -Phase confirmatory first."
    }
    $path = [string]$summary.phase_details.resolved_confirmatory_output_dir
    if ([string]::IsNullOrWhiteSpace($path)) {
        throw "Confirmatory phase summary is missing resolved_confirmatory_output_dir."
    }
    if (-not (Test-Path -LiteralPath $path -PathType Container)) {
        throw "Confirmatory output directory from phase summary does not exist: $path"
    }
    return $path
}

function Get-PrimaryComparisonOutputDirFromSummary {
    $summary = Read-JsonFile -Path (Get-PhaseSummaryPath -PhaseName "comparison")
    if ($null -eq $summary -or $null -eq $summary.phase_details) {
        throw "Comparison phase summary missing. Run -Phase comparison first."
    }
    $path = [string]$summary.phase_details.primary_comparison_output_dir
    if ([string]::IsNullOrWhiteSpace($path)) {
        throw "Comparison phase summary is missing primary_comparison_output_dir."
    }
    if (-not (Test-Path -LiteralPath $path -PathType Container)) {
        throw "Primary comparison output directory from phase summary does not exist: $path"
    }
    return $path
}

function Invoke-Phase {
    param(
        [Parameter(Mandatory = $true)][string]$PhaseName,
        [Parameter(Mandatory = $true)][scriptblock]$Runner
    )

    $context = New-PhaseContext -PhaseName $PhaseName
    $startUtc = Get-UtcNow
    $summaryWritten = $false

    Write-PhaseStatus -PhaseName $PhaseName -Status "running" -StartUtc $startUtc
    Update-CampaignManifest

    try {
        $unmet = @(Get-UnmetDependencies -PhaseName $PhaseName)
        if ($unmet.Count -gt 0) {
            $missingDetails = ($unmet | ForEach-Object { "{0}={1}" -f $_.phase, $_.status }) -join ", "
            throw "Dependencies not satisfied for phase '$PhaseName': $missingDetails"
        }

        & $Runner $context

        Write-PhaseSummary -Context $context -Status "passed"
        $summaryWritten = $true
        Write-PhaseStatus -PhaseName $PhaseName -Status "passed" -StartUtc $startUtc -EndUtc (Get-UtcNow)
        Update-CampaignManifest
    }
    catch {
        $errorSummary = [string]$_.Exception.Message
        if (-not $summaryWritten) {
            Write-PhaseSummary -Context $context -Status "failed" -ErrorSummary $errorSummary
            $summaryWritten = $true
        }
        Write-PhaseStatus -PhaseName $PhaseName -Status "failed" -StartUtc $startUtc -EndUtc (Get-UtcNow) -ErrorSummary $errorSummary
        Update-CampaignManifest
        throw
    }
}
# Resolve campaign root
if ([string]::IsNullOrWhiteSpace($CampaignRoot)) {
    $CampaignRoot = "outputs/campaign/$CampaignTag"
}

$ReleaseRoot = Join-Path $CampaignRoot "release"
$PrecheckRoot = Join-Path $ReleaseRoot "precheck"
$ReplayRoot = Join-Path $ReleaseRoot "replay"
$ComparisonRoot = Join-Path $CampaignRoot "comparison"
$ConfirmatoryRoot = Join-Path $CampaignRoot "confirmatory"
$BundleRoot = Join-Path $CampaignRoot "bundle"
$LogRoot = Join-Path $CampaignRoot "logs"
$CampaignManifestPath = Join-Path $CampaignRoot "campaign_manifest.json"

$OrderedPhases = @("precheck", "confirmatory", "comparison", "replay", "bundle")
$PhaseDefinitions = [ordered]@{
    precheck     = [ordered]@{
        dependencies = @()
        output_root  = $PrecheckRoot
        blocking_for = @("confirmatory_readiness", "campaign_signoff")
    }
    confirmatory = [ordered]@{
        dependencies = @("precheck")
        output_root  = $ConfirmatoryRoot
        blocking_for = @("confirmatory_readiness", "campaign_signoff")
    }
    comparison   = [ordered]@{
        dependencies = @("precheck")
        output_root  = $ComparisonRoot
        blocking_for = @("campaign_signoff")
    }
    replay       = [ordered]@{
        dependencies = @("precheck", "confirmatory", "comparison")
        output_root  = $ReplayRoot
        blocking_for = @("campaign_signoff")
    }
    bundle       = [ordered]@{
        dependencies = @("precheck", "confirmatory", "comparison", "replay")
        output_root  = $BundleRoot
        blocking_for = @("campaign_signoff")
    }
}

# Validate inputs
Assert-PathExists -Path $IndexCsv -Label "IndexCsv"
Assert-PathExists -Path $DataRoot -Label "DataRoot"
Assert-PathExists -Path $CacheDir -Label "CacheDir"
Assert-PathExists -Path $ConfirmatoryProtocol -Label "ConfirmatoryProtocol"
foreach ($spec in $ComparisonSpecs) {
    Assert-PathExists -Path $spec -Label "ComparisonSpec"
}
Assert-PathExists -Path $PrimaryComparisonSpec -Label "PrimaryComparisonSpec"

# Create campaign dirs
foreach ($path in @($CampaignRoot, $ReleaseRoot, $PrecheckRoot, $ReplayRoot, $ComparisonRoot, $ConfirmatoryRoot, $BundleRoot, $LogRoot)) {
    New-Item -ItemType Directory -Path $path -Force | Out-Null
}
foreach ($phaseName in $OrderedPhases) {
    New-Item -ItemType Directory -Path (Join-Path $LogRoot $phaseName) -Force | Out-Null
}

Ensure-PhaseStatusFiles
Update-CampaignManifest

# Export dataset env
$env:THESIS_ML_INDEX_CSV = $IndexCsv
$env:THESIS_ML_DATA_ROOT = $DataRoot
$env:THESIS_ML_CACHE_DIR = $CacheDir

$ResolvedPrimarySpec = (Resolve-Path -LiteralPath $PrimaryComparisonSpec).ProviderPath

$PhaseRunners = [ordered]@{}
$PhaseRunners["precheck"] = {
    param($Context)

    $gitStatusPath = Join-Path $PrecheckRoot "git_status_porcelain.txt"
    if (-not $SkipGitCleanCheck) {
        $statusResult = Invoke-PhaseCommand -Context $Context -Label "Repository clean check" -CommandParts @("git", "status", "--porcelain=v1")
        @($statusResult.output) | Set-Content -Path $gitStatusPath -Encoding utf8
        $Context.key_outputs += $gitStatusPath

        $hasChanges = @($statusResult.output | Where-Object { -not [string]::IsNullOrWhiteSpace($_) }).Count -gt 0
        if ($hasChanges) {
            throw "Repository is not clean. Commit or stash changes before running frozen campaign phases."
        }
    }
    else {
        "Skipped git clean check." | Set-Content -Path $gitStatusPath -Encoding utf8
        $Context.warnings += "Skipped git clean check."
        $Context.key_outputs += $gitStatusPath
    }

    $commitResult = Invoke-PhaseCommand -Context $Context -Label "Capture commit SHA" -CommandParts @("git", "rev-parse", "HEAD")
    $commitSha = [string](@($commitResult.output)[0]).Trim()
    $commitPath = Join-Path $PrecheckRoot "commit_sha.txt"
    $commitSha | Set-Content -Path $commitPath -Encoding utf8
    $Context.key_outputs += $commitPath

    $branchResult = Invoke-PhaseCommand -Context $Context -Label "Capture branch" -CommandParts @("git", "branch", "--show-current")
    $branchName = [string](@($branchResult.output)[0]).Trim()
    $branchPath = Join-Path $PrecheckRoot "branch.txt"
    $branchName | Set-Content -Path $branchPath -Encoding utf8
    $Context.key_outputs += $branchPath

    Invoke-PhaseCommand -Context $Context -Label "Release hygiene check" -CommandParts @("python", "scripts/release_hygiene_check.py") | Out-Null

    Invoke-PhaseCommand -Context $Context -Label "Full test suite" -CommandParts @("python", "-m", "pytest", "-q") | Out-Null

    $rcSummary = Join-Path $PrecheckRoot "rc1_gate_summary.json"
    Invoke-PhaseCommand -Context $Context -Label "RC1 release gate" -CommandParts @(
        "python", "scripts/rc1_release_gate.py",
        "--summary-out", $rcSummary,
        "--run-ruff",
        "--run-performance-smoke"
    ) | Out-Null
    $Context.verification_summaries += $rcSummary
    $Context.key_outputs += $rcSummary

    $tagPath = Join-Path $PrecheckRoot "frozen_tag_commit.txt"
    if (-not $SkipTagCreation) {
        Invoke-PhaseCommand -Context $Context -Label "Create campaign git tag" -CommandParts @("git", "tag", $CampaignTag) | Out-Null
        $tagCommitResult = Invoke-PhaseCommand -Context $Context -Label "Capture tag commit" -CommandParts @("git", "rev-parse", $CampaignTag)
        [string](@($tagCommitResult.output)[0]).Trim() | Set-Content -Path $tagPath -Encoding utf8
    }
    else {
        "Skipped tag creation." | Set-Content -Path $tagPath -Encoding utf8
        $Context.warnings += "Skipped tag creation."
    }
    $Context.key_outputs += $tagPath
}

$PhaseRunners["confirmatory"] = {
    param($Context)

    Invoke-PhaseCommand -Context $Context -Label "Run frozen confirmatory protocol" -CommandParts @(
        "python", "-m", "Thesis_ML.cli.protocol_runner",
        "--protocol", $ConfirmatoryProtocol,
        "--all-suites",
        "--reports-root", $ConfirmatoryRoot,
        "--force"
    ) | Out-Null

    $protocolRunsRoot = Join-Path $ConfirmatoryRoot "protocol_runs"
    if (-not (Test-Path -LiteralPath $protocolRunsRoot -PathType Container)) {
        throw "No protocol_runs directory found under: $protocolRunsRoot"
    }

    $protocolPayload = Read-JsonFile -Path $ConfirmatoryProtocol
    $protocolId = ""
    if ($null -ne $protocolPayload) {
        $protocolId = [string]$protocolPayload.protocol_id
    }

    $resolvedConfirmatoryDir = $null
    $candidateDirs = Get-ChildItem -Path $protocolRunsRoot -Directory | Sort-Object LastWriteTime -Descending
    if (-not [string]::IsNullOrWhiteSpace($protocolId)) {
        $resolvedConfirmatoryDir = $candidateDirs | Where-Object { $_.Name -like "$protocolId*" } | Select-Object -First 1
    }
    if ($null -eq $resolvedConfirmatoryDir) {
        $resolvedConfirmatoryDir = $candidateDirs | Select-Object -First 1
    }
    if ($null -eq $resolvedConfirmatoryDir) {
        throw "No confirmatory protocol output directory found under: $protocolRunsRoot"
    }

    $confirmatoryArtifactSummary = Join-Path $ConfirmatoryRoot "artifact_verification_summary.json"
    Invoke-PhaseCommand -Context $Context -Label "Verify confirmatory artifacts" -CommandParts @(
        "python", "scripts/verify_official_artifacts.py",
        "--mode", "confirmatory",
        "--output-dir", $resolvedConfirmatoryDir.FullName,
        "--summary-out", $confirmatoryArtifactSummary
    ) | Out-Null

    $confirmatoryReadySummary = Join-Path $ConfirmatoryRoot "confirmatory_ready_summary.json"
    Invoke-PhaseCommand -Context $Context -Label "Verify confirmatory-ready status" -CommandParts @(
        "python", "scripts/verify_confirmatory_ready.py",
        "--output-dir", $resolvedConfirmatoryDir.FullName,
        "--summary-out", $confirmatoryReadySummary
    ) | Out-Null

    $Context.extra["resolved_confirmatory_output_dir"] = $resolvedConfirmatoryDir.FullName
    $Context.extra["protocol_runs_root"] = $protocolRunsRoot
    $Context.key_outputs += $resolvedConfirmatoryDir.FullName
    $Context.key_outputs += $confirmatoryArtifactSummary
    $Context.key_outputs += $confirmatoryReadySummary
    $Context.verification_summaries += $confirmatoryArtifactSummary
    $Context.verification_summaries += $confirmatoryReadySummary
}
$PhaseRunners["comparison"] = {
    param($Context)

    $specOutputs = [ordered]@{}
    $primaryOutputDir = ""

    foreach ($spec in $ComparisonSpecs) {
        $specName = Get-SpecName -SpecPath $spec
        $specRoot = Join-Path $ComparisonRoot $specName

        Invoke-PhaseCommand -Context $Context -Label "Run comparison: $specName" -CommandParts @(
            "python", "-m", "Thesis_ML.cli.comparison_runner",
            "--comparison", $spec,
            "--all-variants",
            "--reports-root", $specRoot,
            "--force"
        ) | Out-Null

        $comparisonRunDir = Join-Path $specRoot "comparison_runs"
        if (-not (Test-Path -LiteralPath $comparisonRunDir -PathType Container)) {
            throw "No comparison_runs directory found under: $comparisonRunDir"
        }

        $resolvedComparisonDir = Get-ChildItem -Path $comparisonRunDir -Directory | Sort-Object LastWriteTime -Descending | Select-Object -First 1
        if ($null -eq $resolvedComparisonDir) {
            throw "No comparison output directory found under: $comparisonRunDir"
        }

        $artifactSummaryPath = Join-Path $specRoot "artifact_verification_summary.json"
        Invoke-PhaseCommand -Context $Context -Label "Verify comparison artifacts: $specName" -CommandParts @(
            "python", "scripts/verify_official_artifacts.py",
            "--mode", "comparison",
            "--output-dir", $resolvedComparisonDir.FullName,
            "--summary-out", $artifactSummaryPath
        ) | Out-Null

        $specOutputs[$spec] = [ordered]@{
            spec_name                     = $specName
            reports_root                  = $specRoot
            resolved_output_dir           = $resolvedComparisonDir.FullName
            artifact_verification_summary = $artifactSummaryPath
        }

        $Context.key_outputs += $resolvedComparisonDir.FullName
        $Context.key_outputs += $artifactSummaryPath
        $Context.verification_summaries += $artifactSummaryPath

        $resolvedSpecPath = (Resolve-Path -LiteralPath $spec).ProviderPath
        if ($resolvedSpecPath -eq $ResolvedPrimarySpec) {
            $primaryOutputDir = $resolvedComparisonDir.FullName
        }
    }

    if ([string]::IsNullOrWhiteSpace($primaryOutputDir)) {
        throw "Primary comparison output directory was not resolved for: $PrimaryComparisonSpec"
    }

    $Context.extra["spec_outputs"] = $specOutputs
    $Context.extra["primary_comparison_output_dir"] = $primaryOutputDir
}

$PhaseRunners["replay"] = {
    param($Context)

    $confirmatoryDir = Get-ConfirmatoryOutputDirFromSummary
    $primaryComparisonDir = Get-PrimaryComparisonOutputDirFromSummary

    $determinismComparisonReports = Join-Path $ReplayRoot "determinism_comparison"
    $determinismComparisonSummary = Join-Path $ReplayRoot "determinism_comparison_summary.json"
    Invoke-PhaseCommand -Context $Context -Label "Deterministic comparison reproducibility" -CommandParts @(
        "python", "scripts/verify_official_reproducibility.py",
        "--mode", "comparison",
        "--config", $PrimaryComparisonSpec,
        "--index-csv", $IndexCsv,
        "--data-root", $DataRoot,
        "--cache-dir", $CacheDir,
        "--variant", "ridge",
        "--reports-root", $determinismComparisonReports,
        "--summary-out", $determinismComparisonSummary
    ) | Out-Null

    $determinismConfirmatoryReports = Join-Path $ReplayRoot "determinism_confirmatory"
    $determinismConfirmatorySummary = Join-Path $ReplayRoot "determinism_confirmatory_summary.json"
    Invoke-PhaseCommand -Context $Context -Label "Deterministic confirmatory reproducibility" -CommandParts @(
        "python", "scripts/verify_official_reproducibility.py",
        "--mode", "protocol",
        "--config", $ConfirmatoryProtocol,
        "--index-csv", $IndexCsv,
        "--data-root", $DataRoot,
        "--cache-dir", $CacheDir,
        "--suite", "confirmatory_primary_within_subject",
        "--reports-root", $determinismConfirmatoryReports,
        "--summary-out", $determinismConfirmatorySummary
    ) | Out-Null

    $officialReplayRoot = Join-Path $ReplayRoot "official_replay"
    $replaySummaryPath = Join-Path $ReplayRoot "replay_summary.json"
    $replayVerificationPath = Join-Path $ReplayRoot "replay_verification_summary.json"
    $manifestPath = Join-Path $ReplayRoot "reproducibility_manifest.json"
    Invoke-PhaseCommand -Context $Context -Label "Official replay orchestration" -CommandParts @(
        "python", "scripts/replay_official_paths.py",
        "--mode", "both",
        "--index-csv", $IndexCsv,
        "--data-root", $DataRoot,
        "--cache-dir", $CacheDir,
        "--reports-root", $officialReplayRoot,
        "--verify-determinism",
        "--summary-out", $replaySummaryPath,
        "--verification-summary-out", $replayVerificationPath,
        "--manifest-out", $manifestPath
    ) | Out-Null

    $Context.extra["upstream_confirmatory_output_dir"] = $confirmatoryDir
    $Context.extra["upstream_primary_comparison_output_dir"] = $primaryComparisonDir
    $Context.extra["official_replay_reports_root"] = $officialReplayRoot

    foreach ($path in @(
            $determinismComparisonSummary,
            $determinismConfirmatorySummary,
            $replaySummaryPath,
            $replayVerificationPath,
            $manifestPath
        )) {
        $Context.key_outputs += $path
        $Context.verification_summaries += $path
    }
}
$PhaseRunners["bundle"] = {
    param($Context)

    $confirmatoryOutputDir = Get-ConfirmatoryOutputDirFromSummary
    $primaryComparisonOutputDir = Get-PrimaryComparisonOutputDirFromSummary

    $confirmatoryReadySummary = Join-Path $ConfirmatoryRoot "confirmatory_ready_summary.json"
    $replaySummary = Join-Path $ReplayRoot "replay_summary.json"
    $replayVerificationSummary = Join-Path $ReplayRoot "replay_verification_summary.json"
    $reproManifest = Join-Path $ReplayRoot "reproducibility_manifest.json"

    foreach ($required in @($confirmatoryReadySummary, $replaySummary, $replayVerificationSummary, $reproManifest)) {
        if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
            throw "Bundle phase missing required upstream artifact: $required"
        }
    }

    Invoke-PhaseCommand -Context $Context -Label "Build publishable bundle" -CommandParts @(
        "python", "scripts/build_publishable_bundle.py",
        "--output-dir", $BundleRoot,
        "--comparison-output", $primaryComparisonOutputDir,
        "--confirmatory-output", $confirmatoryOutputDir,
        "--confirmatory-ready-summary", $confirmatoryReadySummary,
        "--replay-summary", $replaySummary,
        "--replay-verification-summary", $replayVerificationSummary,
        "--repro-manifest", $reproManifest
    ) | Out-Null

    $bundleVerificationSummary = Join-Path $BundleRoot "bundle_verification_summary.json"
    Invoke-PhaseCommand -Context $Context -Label "Verify publishable bundle" -CommandParts @(
        "python", "scripts/verify_publishable_bundle.py",
        "--bundle-dir", $BundleRoot,
        "--summary-out", $bundleVerificationSummary
    ) | Out-Null

    $bundleManifest = Join-Path $BundleRoot "bundle_manifest.json"
    $Context.extra["upstream_confirmatory_output_dir"] = $confirmatoryOutputDir
    $Context.extra["upstream_primary_comparison_output_dir"] = $primaryComparisonOutputDir
    $Context.key_outputs += $bundleManifest
    $Context.key_outputs += $bundleVerificationSummary
    $Context.verification_summaries += $bundleVerificationSummary
}

$PhasesToRun = @()
if ($Phase -eq "all") {
    $PhasesToRun = @($OrderedPhases)
}
else {
    $PhasesToRun = @($Phase)
}

Write-Step "Campaign phase execution"
Write-Host "Campaign tag: $CampaignTag"
Write-Host "Campaign root: $CampaignRoot"
Write-Host "Requested phase: $Phase"
Write-Host "Execution order: $($PhasesToRun -join ', ')"

foreach ($phaseName in $PhasesToRun) {
    Invoke-Phase -PhaseName $phaseName -Runner $PhaseRunners[$phaseName]
}

Update-CampaignManifest

Write-Step "Completed"
Write-Host "Campaign phase execution completed successfully." -ForegroundColor Green
Write-Host "Campaign manifest: $CampaignManifestPath" -ForegroundColor Green
