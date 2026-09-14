[CmdletBinding()]
param(
    [switch]$Publish,
    [switch]$WebPreview
)

$ErrorActionPreference = "Stop"

$Repository = "pytorch/pytorch"
$ExpectedTitle = "[FSDP2] Gradient accumulation with fresh or unused parameters raises a mixed-dtype assertion"
$ExpectedReproducerSha256 = "FBD48185F9AA77566037B23C7C4E625B33A87A676819F75D5CF86704258CCF09"
$DraftPath = Join-Path $PSScriptRoot "PYTORCH_ISSUE_DRAFT.md"
$ReproducerPath = Join-Path $PSScriptRoot "reproducer.py"
$ReadyBodyPath = Join-Path $PSScriptRoot "PYTORCH_ISSUE_BODY_READY.md"

if ($Publish -and $WebPreview) {
    throw "Choose either -Publish or -WebPreview, not both."
}

if (-not (Get-Command gh -ErrorAction SilentlyContinue)) {
    throw "GitHub CLI ('gh') was not found in PATH."
}

foreach ($Path in @($DraftPath, $ReproducerPath)) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "Required file not found: $Path"
    }
}

$StrictUtf8 = [System.Text.UTF8Encoding]::new($false, $true)
# Read as strict UTF-8: Windows PowerShell 5.1 otherwise decodes BOM-less files
# with the ANSI code page and corrupts non-ASCII text such as the issue emoji.
$Draft = [System.IO.File]::ReadAllText($DraftPath, $StrictUtf8)
$TitleMatch = [regex]::Match(
    $Draft,
    '(?ms)^# Proposed title\s*\r?\n\s*`([^`\r\n]+)`\s*'
)
$BodyMarker = [regex]::Match($Draft, '(?m)^# Issue body\s*$')

if (-not $TitleMatch.Success -or -not $BodyMarker.Success) {
    throw "The '# Proposed title' or '# Issue body' section was not found."
}

$Title = $TitleMatch.Groups[1].Value.Trim()
$Body = $Draft.Substring($BodyMarker.Index + $BodyMarker.Length).TrimStart("`r", "`n")

if ($Title -cne $ExpectedTitle) {
    throw "Unexpected issue title: $Title"
}
if ([string]::IsNullOrWhiteSpace($Body)) {
    throw "The issue body is empty."
}

$ForbiddenPatterns = @(
    '<!--',
    'TODO',
    'TBD',
    'Draft status',
    'Hold before filing',
    'COLLECT_ENV_OUTPUT',
    'rank 0 waited in reduce-scatter',
    'Co-authored-by:'
)
foreach ($Pattern in $ForbiddenPatterns) {
    if ($Draft -match [regex]::Escape($Pattern)) {
        throw "Unresolved or unwanted text found in the draft: $Pattern"
    }
}

$RequiredPatterns = @(
    '2.15.0.dev20260914+cu130',
    '7d5f0216450b8253e62d88284d49de725d88bd42',
    'last-microbatch',
    'unused-placeholder',
    'rank-divergent-unused',
    'nccl2',
    'Complete collect_env output',
    'AI assistance disclosure'
)
foreach ($Pattern in $RequiredPatterns) {
    if ($Body -notmatch [regex]::Escape($Pattern)) {
        throw "Required evidence is missing from the issue body: $Pattern"
    }
}

$FenceCount = ([regex]::Matches($Body, '(?m)^```')).Count
if (($FenceCount % 2) -ne 0) {
    throw "The issue body contains an unmatched fenced code block."
}

$ActualReproducerSha256 = (Get-FileHash -LiteralPath $ReproducerPath -Algorithm SHA256).Hash
if ($ActualReproducerSha256 -cne $ExpectedReproducerSha256) {
    throw "Unexpected reproducer SHA-256: $ActualReproducerSha256"
}

$EmbeddedMatch = [regex]::Match($Body, '(?s)```python\r?\n(.*?)\r?\n```')
if (-not $EmbeddedMatch.Success) {
    throw "The inline Python reproducer was not found in the issue body."
}
$NormalizeNewlines = {
    param([string]$Text)
    return (($Text -replace "`r`n", "`n") -replace "`r", "`n").TrimEnd("`n")
}
$EmbeddedReproducer = & $NormalizeNewlines $EmbeddedMatch.Groups[1].Value
$FileReproducer = & $NormalizeNewlines ([System.IO.File]::ReadAllText($ReproducerPath, $StrictUtf8))
if ($EmbeddedReproducer -cne $FileReproducer) {
    throw "The inline reproducer does not match reproducer.py byte-for-byte after newline normalization."
}

[System.IO.File]::WriteAllText(
    $ReadyBodyPath,
    $Body,
    [System.Text.UTF8Encoding]::new($false)
)

Write-Host ""
Write-Host "Repository: $Repository"
Write-Host "Title:      $Title"
Write-Host "Body:       $($Body.Length) characters"
Write-Host "Body file:  $ReadyBodyPath"
Write-Host "Reproducer: $ActualReproducerSha256"

if (-not $Publish -and -not $WebPreview) {
    Write-Host ""
    Write-Host "VALIDATION PASSED. No issue was created."
    Write-Host "Open a populated GitHub preview with:"
    Write-Host "  & `"$PSCommandPath`" -WebPreview"
    Write-Host "Publish from the terminal with:"
    Write-Host "  & `"$PSCommandPath`" -Publish"
    return
}

gh auth status --hostname github.com
if ($LASTEXITCODE -ne 0) {
    throw "GitHub CLI is not authenticated. Run 'gh auth login --hostname github.com --web' and retry."
}

if ($WebPreview) {
    gh issue create --repo $Repository --title $Title --body-file $ReadyBodyPath --web
    if ($LASTEXITCODE -ne 0) {
        throw "GitHub web preview failed."
    }
    Write-Host "A prefilled GitHub issue page was opened. Review it there; this script did not submit it."
    return
}

$Confirmation = Read-Host "Type CREATE to publish this issue to $Repository"
if ($Confirmation -cne "CREATE") {
    throw "Publication cancelled."
}

$IssueUrl = gh issue create `
    --repo $Repository `
    --title $Title `
    --body-file $ReadyBodyPath
if ($LASTEXITCODE -ne 0) {
    throw "gh issue create failed."
}

Write-Host ""
Write-Host "Created: $IssueUrl"
