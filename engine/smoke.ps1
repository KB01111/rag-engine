param(
    [Parameter(Mandatory = $true)]
    [string]$ModelPath,
    [switch]$SkipBuild,
    [switch]$Force,
    [switch]$ForceCpu,
    [string]$EmbeddingCacheDir = "",
    [switch]$DisableEmbeddingDownload,
    [int]$ClientTimeoutSeconds = 180,
    [int]$StartupTimeoutSeconds = 180
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

function Resolve-BundleRoot {
    param([string]$ScriptRoot)

    if (Test-Path (Join-Path $ScriptRoot "bin\server.exe")) {
        return $ScriptRoot
    }

    return (Join-Path $ScriptRoot "dist" | Join-Path -ChildPath "windows-backend")
}

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
        $Process,
        [int]$TimeoutSeconds
    )

    for ($i = 0; $i -lt ($TimeoutSeconds * 2); $i++) {
        Start-Sleep -Milliseconds 500
        try {
            $health = Invoke-RestMethod -Uri $HealthUrl -TimeoutSec 2
            if ($health.ready -and $health.status -eq "ok" -and $health.execution_mode -eq "daemon") {
                return
            }
        } catch {
            Write-Verbose "Health check attempt $i failed: $($_.Exception.Message)"
        }
        if ($Process.HasExited) {
            throw "server exited early; see $serverStdout and $serverStderr"
        }
    }

    throw "server did not become healthy in daemon mode; see $serverStdout and $serverStderr"
}

function Stop-Engine {
    param($Process)

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

function Assert-OutputContains {
    param(
        [string]$Text,
        [string[]]$Needles
    )

    foreach ($needle in $Needles) {
        if ($Text -notmatch [regex]::Escape($needle)) {
            throw "missing smoke output marker '$needle'"
        }
    }
}

function Assert-InferenceOutput {
    param([string]$Text)

    $match = [regex]::Match($Text, "Inference output:\s*(?<output>.+)")
    if (-not $match.Success) {
        throw "missing inference output line"
    }
    $output = $match.Groups["output"].Value.Trim()
    if ([string]::IsNullOrWhiteSpace($output)) {
        throw "real inference output was empty"
    }
}

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$bundleRoot = Resolve-BundleRoot -ScriptRoot $root
$bundleBin = Join-Path $bundleRoot "bin"
$buildScript = Join-Path $root "build.bat"
$doctorScript = Join-Path $root "doctor.ps1"
if (-not (Test-Path $doctorScript)) {
    $doctorScript = Join-Path $bundleRoot "doctor.ps1"
}

$serverBin = Join-Path $bundleBin "server.exe"
$clientBin = Join-Path $bundleBin "client.exe"
$daemonBin = Join-Path $bundleBin "ai_engine_daemon.exe"
$contextServerBin = Join-Path $bundleBin "context_server.exe"

$modelItem = Get-Item -LiteralPath $ModelPath -ErrorAction Stop
if ($modelItem.PSIsContainer) {
    throw "ModelPath must point to a model file, got directory: $ModelPath"
}

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
if (-not (Test-Path $doctorScript)) { throw "missing doctor script: $doctorScript" }

$workspace = Join-Path ([System.IO.Path]::GetTempPath()) ("ai-engine-smoke-" + [guid]::NewGuid().ToString("N"))
$null = New-Item -ItemType Directory -Path $workspace -Force
$modelsDir = Join-Path $workspace "models"
$trainingDir = Join-Path $workspace "training"
$lancedbDir = Join-Path $workspace "lancedb"
$contextDir = Join-Path $workspace "context"
$embeddingCache = if ($EmbeddingCacheDir) { $EmbeddingCacheDir } else { Join-Path $workspace "embedding-cache" }
$null = New-Item -ItemType Directory -Path $modelsDir, $trainingDir, $lancedbDir, $contextDir, $embeddingCache -Force

$modelCopy = Join-Path $modelsDir $modelItem.Name
Copy-Item -LiteralPath $modelItem.FullName -Destination $modelCopy -Force

$httpPort = Get-FreePort
$grpcPort = Get-FreePort
$daemonPort = Get-FreePort
$contextPort = Get-FreePort
$configPath = Join-Path $workspace "config.yaml"
$serverStdout = Join-Path $workspace "server.stdout.log"
$serverStderr = Join-Path $workspace "server.stderr.log"
$forceCpuValue = if ($ForceCpu) { "true" } else { "false" }
$embeddingAllowDownload = -not $DisableEmbeddingDownload.IsPresent
$embeddingAllowDownloadValue = if ($embeddingAllowDownload) { "true" } else { "false" }

& $doctorScript `
    -BundleRoot $bundleRoot `
    -ModelPath $modelItem.FullName `
    -RuntimeBackend "mistralrs" `
    -EmbeddingCacheDir $embeddingCache `
    -EmbeddingAllowDownload:$embeddingAllowDownload `
    -Ports @($httpPort, $grpcPort, $daemonPort, $contextPort)
if ($LASTEXITCODE -ne 0) {
    throw "doctor failed with exit code $LASTEXITCODE"
}

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
  startup_timeout: ${StartupTimeoutSeconds}s
  restart_backoff: 3s
  ready_timeout: 60s
  llama_cli: "llama-cli"
  training_cli: "llama-train"

context:
  enabled: true
  service_url: "http://127.0.0.1:$contextPort"
  binary_path: "$($contextServerBin -replace '\\','/')"
  data_dir: "$($contextDir -replace '\\','/')"
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
  backend: "mistralrs"
  mistralrs:
    force_cpu: $forceCpuValue
    max_num_seqs: 32
    auto_isq: ""
  providers: []

rag:
  storage_path: "$($lancedbDir -replace '\\','/')"
  embedding_provider: "fastembed"
  embedding_model: "sentence-transformers/all-MiniLM-L6-v2"
  embedding_cache_dir: "$($embeddingCache -replace '\\','/')"
  embedding_allow_download: $embeddingAllowDownloadValue
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
$proc = Start-EngineServer -Binary $serverBin -Config $configPath -WorkingDirectory $bundleRoot -StdoutPath $serverStdout -StderrPath $serverStderr

try {
    $healthUrl = "http://127.0.0.1:$httpPort/health"
    Wait-ForEngine -HealthUrl $healthUrl -Process $proc -TimeoutSeconds $StartupTimeoutSeconds

    Write-Host "Running gRPC smoke client..."
    $docId = "doc-smoke"
    $clientOutput = & $clientBin `
        -addr ("127.0.0.1:{0}" -f $grpcPort) `
        -timeout ("{0}s" -f $ClientTimeoutSeconds) `
        -model-id $modelItem.Name `
        -doc-id $docId `
        -query "retrieval local models" `
        -prompt "Say hello from the bundled WinUI runtime path." 2>&1
    $clientText = ($clientOutput | Out-String)
    $clientText | Write-Host

    Assert-OutputContains -Text $clientText -Needles @(
        "Version:",
        "Available models:",
        "Loaded model:",
        "Inference output:",
        "Upserted document:",
        "Documents:",
        "Document present: true",
        "Search results:",
        "Demo Complete"
    )
    Assert-InferenceOutput -Text $clientText

    $status = Invoke-RestMethod -Uri ("http://127.0.0.1:{0}/api/v1/status" -f $httpPort) -TimeoutSec 2
    if (-not $status.running -or $status.execution_mode -ne "daemon") {
        throw "expected running daemon status from HTTP endpoint"
    }

    $ragStatus = Invoke-RestMethod -Uri ("http://127.0.0.1:{0}/api/v1/rag/status" -f $httpPort) -TimeoutSec 10
    if ($ragStatus.status.embedding_provider -ne "fastembed") {
        throw "expected RAG embedding_provider=fastembed, got $($ragStatus.status.embedding_provider)"
    }
    if ($ragStatus.status.requires_reindex) {
        throw "expected requires_reindex=false, got reasons: $($ragStatus.status.reindex_reasons -join ', ')"
    }

    Write-Host "Restarting server to verify persistence..."
    Stop-Engine -Process $proc
    Start-Sleep -Seconds 1
    $proc = Start-EngineServer -Binary $serverBin -Config $configPath -WorkingDirectory $bundleRoot -StdoutPath $serverStdout -StderrPath $serverStderr
    Wait-ForEngine -HealthUrl $healthUrl -Process $proc -TimeoutSeconds $StartupTimeoutSeconds

    $restartOutput = & $clientBin `
        -addr ("127.0.0.1:{0}" -f $grpcPort) `
        -timeout ("{0}s" -f $ClientTimeoutSeconds) `
        -model-id $modelItem.Name `
        -doc-id $docId `
        -query "retrieval local models" `
        -prompt "Verify the restarted runtime path is alive." `
        -skip-upsert 2>&1
    $restartText = ($restartOutput | Out-String)
    $restartText | Write-Host

    Assert-OutputContains -Text $restartText -Needles @(
        "Loaded model:",
        "Inference output:",
        "Documents:",
        "Document present: true",
        "Search results:"
    )
    Assert-InferenceOutput -Text $restartText

    $restartRagStatus = Invoke-RestMethod -Uri ("http://127.0.0.1:{0}/api/v1/rag/status" -f $httpPort) -TimeoutSec 10
    if ($restartRagStatus.status.embedding_provider -ne "fastembed") {
        throw "expected restart RAG embedding_provider=fastembed, got $($restartRagStatus.status.embedding_provider)"
    }
    if ($restartRagStatus.status.requires_reindex) {
        throw "expected restart requires_reindex=false, got reasons: $($restartRagStatus.status.reindex_reasons -join ', ')"
    }

    Write-Host "Smoke test passed."
} finally {
    Stop-Engine -Process $proc
}
