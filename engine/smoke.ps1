param(
    [switch]$SkipBuild,
    [switch]$Force
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
$bundleRoot = Join-Path $root "dist" | Join-Path -ChildPath "windows-backend"
$bundleBin = Join-Path $bundleRoot "bin"
$buildScript = Join-Path $root "build.bat"
$serverBin = Join-Path $bundleBin "server.exe"
$clientBin = Join-Path $bundleBin "client.exe"
$daemonBin = Join-Path $bundleBin "ai_engine_daemon.exe"
$contextServerBin = Join-Path $bundleBin "context_server.exe"

# Only stop ai_engine_daemon processes if -Force is specified
$runningDaemons = Get-Process ai_engine_daemon -ErrorAction SilentlyContinue
if ($runningDaemons) {
    if ($Force) {
        $runningDaemons | Where-Object {
            try {
                $_.MainModule.FileName -like "*ai_engine_daemon*"
            } catch {
                $false
            }
        } | Stop-Process -Force
    } else {
        Write-Warning "Found running ai_engine_daemon process(es). Use -Force to stop them automatically."
        exit 1
    }
}

if (-not $SkipBuild) {
    Write-Host "Building engine binaries..."
    & $buildScript
    if ($LASTEXITCODE -ne 0) {
        throw "build failed with exit code $LASTEXITCODE"
    }
}

if (-not (Test-Path $serverBin)) { throw "missing server binary: $serverBin" }
if (-not (Test-Path $clientBin)) { throw "missing client binary: $clientBin" }
if (-not (Test-Path $daemonBin)) { throw "missing daemon binary: $daemonBin" }
if (-not (Test-Path $contextServerBin)) { throw "missing context_server binary: $contextServerBin" }

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
$contextPort = Get-FreePort
$configPath = Join-Path $workspace "config.yaml"
$serverStdout = Join-Path $workspace "server.stdout.log"
$serverStderr = Join-Path $workspace "server.stderr.log"
$fakeLlamaCli = Join-Path $workspace "fake-llama.cmd"

@"
@echo off
echo local-backend:STREAM_OK
"@ | Set-Content -Path $fakeLlamaCli -Encoding ASCII

@"
server:
  host: "127.0.0.1"
  port: $httpPort
  mode: "production"
  grpc:
    host: "127.0.0.1"
    port: $grpcPort

daemon:
  host: "127.0.0.1"
  port: $daemonPort
  required: true
  command: "$($daemonBin -replace '\\','/')"
  args: []
  startup_timeout: 15s
  restart_backoff: 3s
  ready_timeout: 10s
  llama_cli: "$($fakeLlamaCli -replace '\\','/')"
  training_cli: "llama-train"

context:
  enabled: true
  service_url: "http://127.0.0.1:$contextPort"
  # The following fields (binary_path, data_dir, auto_start, managed_roots) are present for
  # bundle validation only and are not used when daemon.required is true, as the daemon
  # acts as context client rather than launching the context service directly.
  binary_path: "$($contextServerBin -replace '\\','/')"
  data_dir: "$((Join-Path $workspace 'context') -replace '\\','/')"
  auto_start: true
  startup_timeout: 20s
  managed_roots:
    - "workspace=$($workspace -replace '\\','/')"
  openviking:
    url: ""
    api_key: ""

services:
  enable_training: false
  enable_mcp: false

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

function Start-EngineServer {
    param(
        [string]$Binary,
        [string]$Config,
        [string]$WorkingDirectory,
        [string]$StdoutPath,
        [string]$StderrPath
    )

    Start-Process -FilePath $Binary `
        -ArgumentList @("--config", $Config) `
        -WorkingDirectory $WorkingDirectory `
        -RedirectStandardOutput $StdoutPath `
        -RedirectStandardError $StderrPath `
        -PassThru
}

function Wait-ForEngine {
    param(
        [string]$HealthUrl,
        $Process
    )

    for ($i = 0; $i -lt 60; $i++) {
        Start-Sleep -Milliseconds 500
        try {
            $health = Invoke-RestMethod -Uri $HealthUrl -TimeoutSec 2
            if ($health.ready -and $health.status -eq "ok") {
                return
            }
        } catch {
            Write-Verbose "Health check attempt $i failed: $($_.Exception.Message)"
        }
        if ($Process.HasExited) {
            throw "server exited early; see $serverStdout and $serverStderr"
        }
    }

    throw "server did not become healthy; see $serverStdout and $serverStderr"
}

function Stop-Engine {
    param(
        $Process
    )

    $daemonChildren = @()
    if ($Process) {
        $daemonChildren = Get-CimInstance Win32_Process -Filter ("ParentProcessId = {0}" -f $Process.Id) -ErrorAction SilentlyContinue
    }
    if ($Process -and -not $Process.HasExited) {
        Stop-Process -Id $Process.Id -Force -ErrorAction SilentlyContinue
        Wait-Process -Id $Process.Id -Timeout 5 -ErrorAction SilentlyContinue
    }
    foreach ($child in $daemonChildren) {
        Stop-Process -Id $child.ProcessId -Force -ErrorAction SilentlyContinue
        Wait-Process -Id $child.ProcessId -Timeout 5 -ErrorAction SilentlyContinue
    }
}

Write-Host "Starting server..."
$proc = Start-EngineServer -Binary $serverBin -Config $configPath -WorkingDirectory $bundleRoot -StdoutPath $serverStdout -StderrPath $serverStderr

try {
    $healthUrl = "http://127.0.0.1:$httpPort/health"
    Wait-ForEngine -HealthUrl $healthUrl -Process $proc

    Write-Host "Running gRPC smoke client..."
    $docId = "doc-smoke"
    $clientOutput = & $clientBin `
        -addr ("127.0.0.1:{0}" -f $grpcPort) `
        -doc-id $docId `
        -query "retrieval local models" `
        -prompt "Say hello from the bundled WinUI runtime path." 2>&1
    $clientText = ($clientOutput | Out-String)
    $clientText | Write-Host

    foreach ($needle in @(
        "Version:",
        "Available models:",
        "Loaded model:",
        "Inference output: local-backend:STREAM_OK",
        "Upserted document:",
        "Documents:",
        "Document present: True",
        "Search results:",
        "Demo Complete"
    )) {
        if ($clientText -notmatch [regex]::Escape($needle)) {
            throw "missing smoke output marker '$needle'"
        }
    }

    $status = Invoke-RestMethod -Uri ("http://127.0.0.1:{0}/api/v1/status" -f $httpPort) -TimeoutSec 2
    if (-not $status.running) {
        throw "expected running status from HTTP endpoint"
    }

    Write-Host "Restarting server to verify persistence..."
    Stop-Engine -Process $proc
    Start-Sleep -Seconds 1
    $proc = Start-EngineServer -Binary $serverBin -Config $configPath -WorkingDirectory $bundleRoot -StdoutPath $serverStdout -StderrPath $serverStderr
    Wait-ForEngine -HealthUrl $healthUrl -Process $proc

    $restartOutput = & $clientBin `
        -addr ("127.0.0.1:{0}" -f $grpcPort) `
        -doc-id $docId `
        -query "retrieval local models" `
        -prompt "Verify the restarted runtime path is alive." `
        -skip-upsert 2>&1
    $restartText = ($restartOutput | Out-String)
    $restartText | Write-Host

    foreach ($needle in @(
        "Loaded model:",
        "Inference output: local-backend:STREAM_OK",
        "Documents:",
        "Document present: True",
        "Search results:"
    )) {
        if ($restartText -notmatch [regex]::Escape($needle)) {
            throw "missing restart smoke output marker '$needle'"
        }
    }

    Write-Host "Smoke test passed."
} finally {
    Stop-Engine -Process $proc
}
