[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"

if ($env:PROTOC -and (Test-Path $env:PROTOC)) {
    Write-Output $env:PROTOC
    exit 0
}

$existing = Get-Command protoc -ErrorAction SilentlyContinue
if ($existing) {
    Write-Output $existing.Source
    exit 0
}

$engineRoot = Split-Path -Parent $PSScriptRoot
$version = "34.1"
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