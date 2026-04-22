param(
    [switch]$SkipBuild
)

$ErrorActionPreference = "Stop"

function Get-FreePort {
    $listener = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, 0)
    $listener.Start()
    try {
        return $listener.LocalEndpoint.Port
    } finally {
        $listener.Stop()
    }
}

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$goDir = Join-Path $root "go"
$rustDaemonBin = Join-Path $root "rust\\target\\release\\ai_engine_daemon.exe"
$buildScript = Join-Path $root "build.bat"
$serverBin = Join-Path $goDir "bin\\server.exe"
$clientBin = Join-Path $goDir "bin\\client.exe"
$daemonBin = Join-Path $goDir "bin\\ai_engine_daemon.exe"

Get-Process ai_engine_daemon -ErrorAction SilentlyContinue | Stop-Process -Force

if (-not $SkipBuild) {
    Write-Host "Building engine binaries..."
    & $buildScript
    if ($LASTEXITCODE -ne 0) {
        throw "build failed with exit code $LASTEXITCODE"
    }
}

if (-not (Test-Path $serverBin)) { throw "missing server binary: $serverBin" }
if (-not (Test-Path $clientBin)) { throw "missing client binary: $clientBin" }
if (-not (Test-Path $daemonBin)) {
    $daemonBin = $rustDaemonBin
}
if (-not (Test-Path $daemonBin)) { throw "missing daemon binary: $daemonBin" }

$workspace = Join-Path ([System.IO.Path]::GetTempPath()) ("ai-engine-smoke-" + [guid]::NewGuid().ToString("N"))
$null = New-Item -ItemType Directory -Path $workspace -Force
$modelsDir = Join-Path $workspace "models"
$trainingDir = Join-Path $workspace "training"
$lancedbDir = Join-Path $workspace "lancedb"
$null = New-Item -ItemType Directory -Path $modelsDir, $trainingDir, $lancedbDir -Force
$null = Set-Content -Path (Join-Path $modelsDir "demo.gguf") -Value "demo-model"

$httpPort = Get-FreePort
$grpcPort = Get-FreePort
$daemonPort = Get-FreePort
$configPath = Join-Path $workspace "config.yaml"
$serverStdout = Join-Path $workspace "server.stdout.log"
$serverStderr = Join-Path $workspace "server.stderr.log"

@"
server:
  host: "127.0.0.1"
  port: $httpPort
  grpc:
    host: "127.0.0.1"
    port: $grpcPort

daemon:
  host: "127.0.0.1"
  port: $daemonPort
  command: "$($daemonBin -replace '\\','/')"
  args: []
  startup_timeout: 15s
  restart_backoff: 3s
  ready_timeout: 10s
  llama_cli: "llama-cli"
  training_cli: "llama-train"

storage:
  lancedb_uri: "$($lancedbDir -replace '\\','/')"
  enable_fts: true
  enable_hybrid_search: true

runtime:
  models_path: "$($modelsDir -replace '\\','/')"
  max_memory_mb: 8192
  providers: []

rag:
  storage_path: "$($lancedbDir -replace '\\','/')"
  embedding_model: "sentence-transformers/all-MiniLM-L6-v2"
  chunk_size: 512
  chunk_overlap: 50
  top_k: 10

training:
  working_dir: "$($trainingDir -replace '\\','/')"
  max_concurrent_jobs: 2

mcp:
  timeout: 30s
  retries: 3

logging:
  level: "info"
  format: "json"
"@ | Set-Content -Path $configPath

Write-Host "Starting server..."
$proc = Start-Process -FilePath $serverBin `
    -ArgumentList @("--config", $configPath) `
    -WorkingDirectory $goDir `
    -RedirectStandardOutput $serverStdout `
    -RedirectStandardError $serverStderr `
    -PassThru

try {
    $healthUrl = "http://127.0.0.1:$httpPort/health"
    $ready = $false
    for ($i = 0; $i -lt 60; $i++) {
        Start-Sleep -Milliseconds 500
        try {
            $health = Invoke-RestMethod -Uri $healthUrl -TimeoutSec 2
            if ($health.status -eq "ok") {
                $ready = $true
                break
            }
        } catch {
        }
        if ($proc.HasExited) {
            throw "server exited early; see $serverStdout and $serverStderr"
        }
    }

    if (-not $ready) {
        throw "server did not become healthy; see $serverStdout and $serverStderr"
    }

    Write-Host "Running gRPC smoke client..."
    $clientOutput = & $clientBin -addr ("127.0.0.1:{0}" -f $grpcPort) 2>&1
    $clientText = ($clientOutput | Out-String)
    $clientText | Write-Host

    foreach ($needle in @(
        "Version:",
        "Available models:",
        "Upserted document:",
        "Search results:",
        "Started training run:",
        "MCP connected:",
        "Available tools:"
    )) {
        if ($clientText -notmatch [regex]::Escape($needle)) {
            throw "missing smoke output marker '$needle'"
        }
    }

    $status = Invoke-RestMethod -Uri ("http://127.0.0.1:{0}/api/v1/status" -f $httpPort) -TimeoutSec 2
    if (-not $status.running) {
        throw "expected running status from HTTP endpoint"
    }

    Write-Host "Smoke test passed."
} finally {
    $daemonChildren = @()
    if ($proc) {
        $daemonChildren = Get-CimInstance Win32_Process -Filter ("ParentProcessId = {0}" -f $proc.Id) -ErrorAction SilentlyContinue
    }
    if ($proc -and -not $proc.HasExited) {
        Stop-Process -Id $proc.Id -Force
    }
    foreach ($child in $daemonChildren) {
        Stop-Process -Id $child.ProcessId -Force -ErrorAction SilentlyContinue
    }
    Get-Process ai_engine_daemon -ErrorAction SilentlyContinue | Stop-Process -Force
}
