[CmdletBinding()]
param(
    [switch]$Publish
)

$ErrorActionPreference = "Stop"

$Repository = "pytorch/pytorch"
$DraftPath = Join-Path $PSScriptRoot "PYTORCH_ISSUE_DRAFT.md"

if (-not (Get-Command gh -ErrorAction SilentlyContinue)) {
    throw "GitHub CLI ('gh') was not found in PATH."
}

if (-not (Test-Path -LiteralPath $DraftPath -PathType Leaf)) {
    throw "Issue draft not found: $DraftPath"
}

$Draft = Get-Content -Raw -LiteralPath $DraftPath
$TitleMatch = [regex]::Match($Draft, '(?m)^#\s+(.+?)\s*$')
if (-not $TitleMatch.Success) {
    throw "The first Markdown H1 title was not found in the issue draft."
}

$Title = $TitleMatch.Groups[1].Value.Trim()
$Body = $Draft.Remove($TitleMatch.Index, $TitleMatch.Length).TrimStart("`r", "`n")

if ([string]::IsNullOrWhiteSpace($Body)) {
    throw "The issue body is empty."
}

$ForbiddenPatterns = @(
    '<!--',
    'TODO',
    'TBD',
    'Draft status',
    'COLLECT_ENV_OUTPUT',
    'Codex',
    'Claude',
    'Co-authored-by:'
)

foreach ($Pattern in $ForbiddenPatterns) {
    if ($Draft -match [regex]::Escape($Pattern)) {
        throw "Unresolved or unwanted text found in the draft: $Pattern"
    }
}

$BodyFile = Join-Path ([System.IO.Path]::GetTempPath()) (
    "pytorch-issue-body-{0}.md" -f [guid]::NewGuid().ToString("N")
)

try {
    [System.IO.File]::WriteAllText(
        $BodyFile,
        $Body,
        [System.Text.UTF8Encoding]::new($false)
    )

    Write-Host ""
    Write-Host "Repository: $Repository"
    Write-Host "Title:      $Title"
    Write-Host "Body:       $($Body.Length) characters"
    Write-Host "Draft:      $DraftPath"

    if (-not $Publish) {
        Write-Host ""
        Write-Host "PREVIEW ONLY: no issue was created."
        Write-Host "Review the draft, then publish with:"
        Write-Host "  & `"$PSCommandPath`" -Publish"
        return
    }

    gh auth status --hostname github.com
    if ($LASTEXITCODE -ne 0) {
        throw "GitHub CLI is not authenticated for github.com. Run 'gh auth login --hostname github.com --web' and retry."
    }

    $Confirmation = Read-Host "Type CREATE to publish this issue to $Repository"
    if ($Confirmation -cne "CREATE") {
        throw "Publication cancelled."
    }

    $IssueUrl = gh issue create `
        --repo $Repository `
        --title $Title `
        --body-file $BodyFile

    if ($LASTEXITCODE -ne 0) {
        throw "gh issue create failed."
    }

    Write-Host ""
    Write-Host "Created: $IssueUrl"
    Write-Host ""
    Write-Host "Optional follow-up comment:"
    Write-Host 'cc @fduwjj - this appears adjacent to #169943, but here the affected legacy ProcessGroupNCCL rank stops polling for the dump signal before communicator destruction completes.'
}
finally {
    Remove-Item -LiteralPath $BodyFile -Force -ErrorAction SilentlyContinue
}
