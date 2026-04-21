@echo off
setlocal

echo Building AI Engine...

REM Setup Go modules
cd /d "%~dp0go"
call go mod download
if errorlevel 1 (
    echo Failed to download Go dependencies
    exit /b 1
)

REM Generate proto files (requires protoc)
if exist "..\proto\engine.proto" (
    echo Generating proto files...
    protoc --go_out=. --go_opt=paths=source_relative ^
           --go-grpc_out=. --go-grpc_opt=paths=source_relative ^
           ..\proto\engine.proto
    if errorlevel 1 (
        echo Warning: protoc not found or failed. Install protoc and run manually.
    )
)

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
echo Run: bin\server.exe