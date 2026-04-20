[CmdletBinding()]
param(
    [Parameter(Mandatory = $true, Position = 0)]
    [ValidatePattern('^\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?$')]
    [string]$Version,

    [string]$Remote = "origin",

    [switch]$SkipPublish,

    [switch]$SkipPush
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Invoke-External {
    param(
        [Parameter(Mandatory = $true)]
        [string]$FilePath,

        [Parameter(Mandatory = $true)]
        [string[]]$ArgumentList
    )

    Write-Host ">> $FilePath $($ArgumentList -join ' ')"
    & $FilePath @ArgumentList
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed: $FilePath $($ArgumentList -join ' ')"
    }
}

function Get-ExternalOutput {
    param(
        [Parameter(Mandatory = $true)]
        [string]$FilePath,

        [Parameter(Mandatory = $true)]
        [string[]]$ArgumentList
    )

    $output = & $FilePath @ArgumentList 2>&1
    if ($LASTEXITCODE -ne 0) {
        $message = ($output | Out-String).Trim()
        throw "Command failed: $FilePath $($ArgumentList -join ' ')`n$message"
    }

    return ($output | Out-String).Trim()
}

function Get-ManifestPath {
    $repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
    return (Join-Path $repoRoot "Cargo.toml")
}

function Get-ChangelogPath {
    $repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
    return (Join-Path $repoRoot "docs/CHANGELOG.md")
}

function Get-PackageVersion {
    param(
        [Parameter(Mandatory = $true)]
        [string]$ManifestPath
    )

    $manifest = Get-Content $ManifestPath -Raw
    $match = [regex]::Match(
        $manifest,
        '(?ms)^\[package\]\s*$.*?^version\s*=\s*"([^"]+)"\s*$'
    )

    if (-not $match.Success) {
        throw "Could not find [package].version in Cargo.toml."
    }

    return $match.Groups[1].Value
}

function Set-PackageVersion {
    param(
        [Parameter(Mandatory = $true)]
        [string]$ManifestPath,

        [Parameter(Mandatory = $true)]
        [string]$NewVersion
    )

    $manifest = Get-Content $ManifestPath -Raw
    $updated = [regex]::Replace(
        $manifest,
        '(?ms)(^\[package\]\s*$.*?^version\s*=\s*")([^"]+)(".*$)',
        ('${1}' + $NewVersion + '${3}'),
        1
    )

    if ($updated -eq $manifest) {
        throw "Failed to update [package].version in Cargo.toml."
    }

    $encoding = [System.Text.UTF8Encoding]::new($false)
    [System.IO.File]::WriteAllText($ManifestPath, $updated, $encoding)
}

function Update-ChangelogForRelease {
    param(
        [Parameter(Mandatory = $true)]
        [string]$ChangelogPath,

        [Parameter(Mandatory = $true)]
        [string]$Version
    )

    if (-not (Test-Path -LiteralPath $ChangelogPath)) {
        throw "CHANGELOG.md not found at $ChangelogPath."
    }

    $content = Get-Content $ChangelogPath -Raw
    $newline = if ($content.Contains("`r`n")) { "`r`n" } else { "`n" }
    $normalized = $content -replace "`r`n", "`n"

    $unreleasedMatch = [regex]::Match($normalized, '(?m)^## \[Unreleased\]\s*$')
    if (-not $unreleasedMatch.Success) {
        throw "docs/CHANGELOG.md is missing the ## [Unreleased] section."
    }

    $unreleasedBodyStart = $unreleasedMatch.Index + $unreleasedMatch.Length
    $afterUnreleased = $normalized.Substring($unreleasedBodyStart)
    $nextSectionMatch = [regex]::Match($afterUnreleased, '(?m)^## \[')

    if ($nextSectionMatch.Success) {
        $unreleasedBody = $afterUnreleased.Substring(0, $nextSectionMatch.Index)
        $remainingSections = $afterUnreleased.Substring($nextSectionMatch.Index)
    }
    else {
        $unreleasedBody = $afterUnreleased
        $remainingSections = ""
    }

    $categories = @("Added", "Changed", "Deprecated", "Removed", "Fixed", "Security")
    $categoryPattern = '(?ms)^### (?<name>Added|Changed|Deprecated|Removed|Fixed|Security)[ \t]*$\n?(?<body>.*?)(?=^### (?:Added|Changed|Deprecated|Removed|Fixed|Security)[ \t]*$|\z)'
    $categoryBodies = @{}
    foreach ($category in $categories) {
        $categoryBodies[$category] = ""
    }

    $trimmedUnreleasedBody = $unreleasedBody.Trim("`n")
    foreach ($match in [regex]::Matches($trimmedUnreleasedBody, $categoryPattern)) {
        $categoryBodies[$match.Groups["name"].Value] = $match.Groups["body"].Value.Trim()
    }

    $unsupportedContent = [regex]::Replace($trimmedUnreleasedBody, $categoryPattern, "").Trim()
    if (-not [string]::IsNullOrWhiteSpace($unsupportedContent)) {
        throw "docs/CHANGELOG.md contains unsupported content under Unreleased. Only standard category sections are supported."
    }

    $releaseBlocks = New-Object System.Collections.Generic.List[string]
    foreach ($category in $categories) {
        $body = $categoryBodies[$category]
        if (-not [string]::IsNullOrWhiteSpace($body)) {
            $releaseBlocks.Add("### $category`n`n$body")
        }
    }

    if ($releaseBlocks.Count -eq 0) {
        throw "docs/CHANGELOG.md does not contain any Unreleased entries to release."
    }

    $releaseDate = Get-Date -Format "yyyy-MM-dd"
    $emptyUnreleased = @(
        "## [Unreleased]",
        "",
        "### Added",
        "",
        "### Changed",
        "",
        "### Deprecated",
        "",
        "### Removed",
        "",
        "### Fixed",
        "",
        "### Security"
    ) -join "`n"

    $releaseSection = @(
        "## [$Version] - $releaseDate",
        "",
        ($releaseBlocks -join "`n`n")
    ) -join "`n"

    $prefix = $normalized.Substring(0, $unreleasedMatch.Index)
    $updated = @(
        $prefix.TrimEnd("`n"),
        "",
        $emptyUnreleased,
        "",
        $releaseSection
    ) -join "`n"

    if (-not [string]::IsNullOrWhiteSpace($remainingSections)) {
        $updated = @(
            $updated.TrimEnd("`n"),
            "",
            $remainingSections.TrimStart("`n")
        ) -join "`n"
    }

    if ($content.EndsWith("`r`n")) {
        $updated = $updated.TrimEnd("`n") + "`n"
    }
    elseif ($content.EndsWith("`n")) {
        $updated = $updated.TrimEnd("`n") + "`n"
    }

    $encoding = [System.Text.UTF8Encoding]::new($false)
    [System.IO.File]::WriteAllText($ChangelogPath, ($updated -replace "`n", $newline), $encoding)
}

function Assert-CleanWorktree {
    $status = Get-ExternalOutput git @("status", "--short")
    if ($status) {
        throw "Git worktree is not clean. Commit or stash changes before running the release script."
    }
}

function Assert-RemoteExists {
    param(
        [Parameter(Mandatory = $true)]
        [string]$RemoteName
    )

    $null = Get-ExternalOutput git @("remote", "get-url", $RemoteName)
}

function Assert-TagDoesNotExist {
    param(
        [Parameter(Mandatory = $true)]
        [string]$RemoteName,

        [Parameter(Mandatory = $true)]
        [string]$TagName
    )

    & git rev-parse --verify --quiet "refs/tags/$TagName" *> $null
    if ($LASTEXITCODE -eq 0) {
        throw "Tag $TagName already exists locally."
    }

    $remoteTag = Get-ExternalOutput git @("ls-remote", "--tags", $RemoteName, "refs/tags/$TagName")
    if ($remoteTag) {
        throw "Tag $TagName already exists on remote $RemoteName."
    }
}

function Get-CurrentBranch {
    $branch = Get-ExternalOutput git @("branch", "--show-current")
    if (-not $branch) {
        throw "Detached HEAD is not supported for releases."
    }

    return $branch
}

if ($SkipPublish -and -not $SkipPush) {
    throw "-SkipPublish requires -SkipPush. Pushing a release tag without publishing the crate is not supported."
}

$manifestPath = Get-ManifestPath
$changelogPath = Get-ChangelogPath
$repoRoot = Split-Path $manifestPath -Parent
Set-Location $repoRoot

Assert-CleanWorktree
Assert-RemoteExists -RemoteName $Remote

$currentVersion = Get-PackageVersion -ManifestPath $manifestPath
if ($currentVersion -eq $Version) {
    throw "Cargo.toml is already at version $Version."
}

$tagName = "v$Version"
Assert-TagDoesNotExist -RemoteName $Remote -TagName $tagName

$branch = Get-CurrentBranch

Write-Host "Updating Cargo.toml version: $currentVersion -> $Version"
Set-PackageVersion -ManifestPath $manifestPath -NewVersion $Version
Write-Host "Updating docs/CHANGELOG.md for release $Version"
Update-ChangelogForRelease -ChangelogPath $changelogPath -Version $Version

Invoke-External cargo @("test")
Invoke-External cargo @("publish", "--dry-run", "--locked", "--allow-dirty")

Invoke-External git @("add", "Cargo.toml", "Cargo.lock", "docs/CHANGELOG.md")
Invoke-External git @("commit", "-m", $Version)

if (-not $SkipPublish) {
    Invoke-External cargo @("publish", "--locked")
}

Invoke-External git @("tag", $tagName)

if (-not $SkipPush) {
    Invoke-External git @("push", $Remote, $branch)
    Invoke-External git @("push", $Remote, $tagName)
}

Write-Host "Release $Version completed."
