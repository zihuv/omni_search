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

Invoke-External cargo @("test")
Invoke-External cargo @("publish", "--dry-run", "--locked")

Invoke-External git @("add", "Cargo.toml")
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
