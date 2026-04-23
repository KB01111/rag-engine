[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$version = "34.1"

function Get-MatchingProtocPath {
    param(
        [string]$Candidate
    )

    if (-not $Candidate -or -not (Test-Path $Candidate)) {
        return $null
    }

    try {
        $versionOutput = & $Candidate --version 2>$null
    } catch {
        return $null
    }

    if ($versionOutput -match "libprotoc\\s+$([regex]::Escape($version))") {
        return $Candidate
    }

    return $null
}

$resolved = Get-MatchingProtocPath $env:PROTOC
if ($resolved) {
    Write-Output $resolved
    exit 0
}

$existing = Get-Command protoc -ErrorAction SilentlyContinue
if ($existing) {
    $resolved = Get-MatchingProtocPath $existing.Source
    if ($resolved) {
        Write-Output $resolved
        exit 0
    }
}

$engineRoot = Split-Path -Parent $PSScriptRoot
$toolDir = Join-Path $engineRoot ".tools\protoc\$version\win64"
$binPath = Join-Path $toolDir "bin\protoc.exe"

if (-not (Test-Path $binPath)) {
    New-Item -ItemType Directory -Force -Path $toolDir | Out-Null
    $zipPath = Join-Path $toolDir "protoc-$version-win64.zip"
    $url = "https://github.com/protocolbuffers/protobuf/releases/download/v$version/protoc-$version-win64.zip"

    Write-Host "Downloading protoc $version for Windows..."
    Invoke-WebRequest -Uri $url -OutFile $zipPath
    Expand-Archive -Path $zipPath -DestinationPath $toolDir -Force
}

Write-Output $binPath
