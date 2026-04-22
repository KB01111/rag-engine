@echo off
setlocal

echo Building AI Engine...

set "GOEXE=go"
if exist "%ProgramFiles%\Go\bin\go.exe" set "GOEXE=%ProgramFiles%\Go\bin\go.exe"

set "CARGOEXE=cargo"
if exist "%USERPROFILE%\.cargo\bin\cargo.exe" set "CARGOEXE=%USERPROFILE%\.cargo\bin\cargo.exe"

set "PROTOCEXE=protoc"
if exist "%~dp0..\.tools\protoc-34.1-win64\bin\protoc.exe" set "PROTOCEXE=%~dp0..\.tools\protoc-34.1-win64\bin\protoc.exe"

set "PROTOC_GEN_GO=protoc-gen-go"
if exist "%USERPROFILE%\go\bin\protoc-gen-go.exe" set "PROTOC_GEN_GO=%USERPROFILE%\go\bin\protoc-gen-go.exe"

set "PROTOC_GEN_GO_GRPC=protoc-gen-go-grpc"
if exist "%USERPROFILE%\go\bin\protoc-gen-go-grpc.exe" set "PROTOC_GEN_GO_GRPC=%USERPROFILE%\go\bin\protoc-gen-go-grpc.exe"

REM Setup Go modules
cd /d "%~dp0go"
call "%GOEXE%" mod download
if errorlevel 1 (
    echo Failed to download Go dependencies
    exit /b 1
)

REM Generate proto files (requires protoc)
if exist "..\proto\engine.proto" (
    echo Generating proto files...
    cd /d "%~dp0proto"
    call "%PROTOCEXE%" -I . ^
                      --plugin=protoc-gen-go="%PROTOC_GEN_GO%" ^
                      --plugin=protoc-gen-go-grpc="%PROTOC_GEN_GO_GRPC%" ^
                      --go_out=. --go_opt=paths=source_relative ^
                      --go-grpc_out=. --go-grpc_opt=paths=source_relative ^
                      engine.proto
    if errorlevel 1 (
        echo Warning: failed to generate root proto package.
    )
    call "%PROTOCEXE%" -I . ^
                      --plugin=protoc-gen-go="%PROTOC_GEN_GO%" ^
                      --plugin=protoc-gen-go-grpc="%PROTOC_GEN_GO_GRPC%" ^
                      --go_out=go --go_opt=paths=source_relative ^
                      --go-grpc_out=go --go-grpc_opt=paths=source_relative ^
                      engine.proto
    if errorlevel 1 (
        echo Warning: protoc not found or failed. Install protoc and run manually.
    )
    cd /d "%~dp0go"
)

REM Build server
echo Building server...
call "%GOEXE%" build -o bin\server.exe .\cmd\server
if errorlevel 1 (
    echo Failed to build server
    exit /b 1
)

REM Build client
echo Building client...
call "%GOEXE%" build -o bin\client.exe .\cmd\client
if errorlevel 1 (
    echo Failed to build client
    exit /b 1
)

REM Build Rust context service
echo Building Rust context service...
cd /d "%~dp0rust"
set "PROTOC=%~dp0..\.tools\protoc-34.1-win64\bin\protoc.exe"
call "%CARGOEXE%" build --release -p context_server
if errorlevel 1 (
    echo Failed to build Rust context service
    exit /b 1
)

echo Build complete!
echo Run: go\bin\server.exe
