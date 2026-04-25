param(
    [string]$BundleRoot = "",
    [Parameter(Mandatory = $true)]
    [string]$ModelPath,
    [string]$RuntimeBackend = "mistralrs",
    [string]$EmbeddingCacheDir = "",
    [bool]$EmbeddingAllowDownload = $true,
    [int[]]$Ports = @()
)

$ErrorActionPreference = "Stop"
$errors = New-Object System.Collections.Generic.List[string]

function Add-DoctorError {
    param([string]$Message)
    $null = $errors.Add($Message)
    Write-Error $Message -ErrorAction Continue
}

function Test-PortAvailable {
    param([int]$Port)

    $listener = $null
    try {
        $listener = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, $Port)
        $listener.Start()
        return $true
    } catch {
        Add-DoctorError "loopback port $Port is not available: $($_.Exception.Message)"
        return $false
    } finally {
        if ($listener) {
            $listener.Stop()
        }
    }
}

if ([string]::IsNullOrWhiteSpace($BundleRoot)) {
    $scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
    if (Test-Path (Join-Path $scriptRoot "bin\server.exe")) {
        $BundleRoot = $scriptRoot
    } else {
        $BundleRoot = Join-Path $scriptRoot "dist" | Join-Path -ChildPath "windows-backend"
    }
}

$bundleBin = Join-Path $BundleRoot "bin"
foreach ($binary in @("server.exe", "client.exe", "ai_engine_daemon.exe", "context_server.exe")) {
    $path = Join-Path $bundleBin $binary
    if (-not (Test-Path $path)) {
        Add-DoctorError "missing bundled binary: $path"
    }
}

if ($RuntimeBackend.Trim().ToLowerInvariant() -ne "mistralrs") {
    Add-DoctorError "runtime backend must be mistralrs for real-model smoke, got: $RuntimeBackend"
} else {
    Write-Host "runtime backend: mistralrs"
}

try {
    $modelItem = Get-Item -LiteralPath $ModelPath -ErrorAction Stop
    if ($modelItem.PSIsContainer) {
        Add-DoctorError "model path is a directory, expected file: $ModelPath"
    } elseif ($modelItem.Extension -ne ".gguf") {
        Write-Warning "model file does not use .gguf extension: $($modelItem.FullName)"
    } else {
        $stream = [System.IO.File]::Open($modelItem.FullName, [System.IO.FileMode]::Open, [System.IO.FileAccess]::Read, [System.IO.FileShare]::Read)
        $stream.Dispose()
        Write-Host "model readable: $($modelItem.FullName)"
    }
} catch {
    Add-DoctorError "model path is not readable: $ModelPath ($($_.Exception.Message))"
}

if (-not [string]::IsNullOrWhiteSpace($EmbeddingCacheDir)) {
    if (Test-Path $EmbeddingCacheDir) {
        Write-Host "embedding cache directory present: $EmbeddingCacheDir"
    } elseif ($EmbeddingAllowDownload) {
        Write-Host "embedding cache directory will be created/populated by FastEmbed: $EmbeddingCacheDir"
    } else {
        Add-DoctorError "embedding downloads are disabled and cache directory is missing: $EmbeddingCacheDir"
    }
}

foreach ($port in $Ports) {
    if ($port -le 0 -or $port -gt 65535) {
        Add-DoctorError "invalid port: $port"
        continue
    }
    $null = Test-PortAvailable -Port $port
}

if ($errors.Count -gt 0) {
    Write-Error "AI Engine doctor found $($errors.Count) issue(s)."
    exit 1
}

Write-Host "AI Engine doctor passed."
