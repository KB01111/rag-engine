@echo off
setlocal

echo Building AI Engine...
set "ROOT=%~dp0"
set "GO_DIR=%ROOT%go"
set "RUST_DIR=%ROOT%rust"

REM PROTOC can be set via environment variable or must be in PATH

REM Setup Go modules
cd /d "%GO_DIR%"
call go mod download
if errorlevel 1 (
    echo Failed to download Go dependencies
    exit /b 1
)
for /f "delims=" %%i in ('go env GOPATH') do set "GOPATH_DIR=%%i"
set "PATH=%GOPATH_DIR%\bin;%PATH%"

echo Installing Go proto plugins...
call go install google.golang.org/protobuf/cmd/protoc-gen-go@latest
if errorlevel 1 (
    echo Failed to install protoc-gen-go
    exit /b 1
)
call go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@latest
if errorlevel 1 (
    echo Failed to install protoc-gen-go-grpc
    exit /b 1
)

REM Generate proto files (requires protoc)
if exist "..\proto\engine.proto" (
    echo Generating proto files...
    pushd "%ROOT%proto"
    if defined PROTOC (
        "%PROTOC%" -I . ^
                  --go_out=. --go_opt=paths=source_relative ^
                  --go-grpc_out=. --go-grpc_opt=paths=source_relative ^
                  engine.proto
    ) else (
        protoc -I . ^
               --go_out=. --go_opt=paths=source_relative ^
               --go-grpc_out=. --go-grpc_opt=paths=source_relative ^
               engine.proto
    )
    popd
    if errorlevel 1 (
        echo Warning: protoc not found or failed. Install protoc and run manually.
    )
)

if not exist "bin" mkdir "bin"

REM Build Rust daemon
echo Building Rust daemon...
cd /d "%RUST_DIR%"
cargo build --release -p ai_engine_daemon
if errorlevel 1 (
    echo Failed to build Rust daemon
    exit /b 1
)
copy /Y "target\release\ai_engine_daemon.exe" "%GO_DIR%\bin\ai_engine_daemon.exe" >nul
if errorlevel 1 (
    echo Warning: failed to copy Rust daemon binary into go\bin. Using rust\target\release directly is still supported.
)

cd /d "%GO_DIR%"

REM Build server
echo Building server...
go build -o bin\server.exe .\cmd\server
if errorlevel 1 (
    echo Failed to build server
    exit /b 1
)

REM Build client
echo Building client...
go build -o bin\client.exe .\cmd\client
if errorlevel 1 (
    echo Failed to build client
    exit /b 1
)

echo Build complete!
echo Run: bin\server.exe --config ..\config.example.yaml