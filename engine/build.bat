@echo off
setlocal EnableExtensions

echo Building AI Engine...
set "ROOT=%~dp0"
set "GO_DIR=%ROOT%go"
set "RUST_DIR=%ROOT%rust"
set "PROTO_DIR=%ROOT%proto"
set "CACHE_DIR=%ROOT%.cache"
set "PROTOC_SCRIPT=%ROOT%scripts\ensure_protoc.ps1"
set "BUNDLE_DIR=%ROOT%dist\windows-backend"
set "BUNDLE_BIN=%BUNDLE_DIR%\bin"

if not exist "%CACHE_DIR%" mkdir "%CACHE_DIR%"
set "GOCACHE=%CACHE_DIR%\go-build"
if not exist "%GOCACHE%" mkdir "%GOCACHE%"
if not defined CARGO_HOME set "CARGO_HOME=%CACHE_DIR%\cargo-home"
if not exist "%CARGO_HOME%" mkdir "%CARGO_HOME%"

echo Using GOCACHE=%GOCACHE%
echo Using CARGO_HOME=%CARGO_HOME%

set "GOEXE=go"
if exist "%ProgramFiles%\Go\bin\go.exe" set "GOEXE=%ProgramFiles%\Go\bin\go.exe"

set "CARGOEXE=cargo"
if exist "%USERPROFILE%\.cargo\bin\cargo.exe" set "CARGOEXE=%USERPROFILE%\.cargo\bin\cargo.exe"

REM Ensure protoc is available locally for both Go codegen and Rust builds.
for /f "usebackq delims=" %%i in (`powershell -NoProfile -ExecutionPolicy Bypass -File "%PROTOC_SCRIPT%"`) do set "PROTOC=%%i"
if not exist "%PROTOC%" (
    echo Failed to provision protoc
    exit /b 1
)

set "PROTOCEXE=%PROTOC%"

set "PROTOC_GEN_GO=protoc-gen-go"
if exist "%USERPROFILE%\go\bin\protoc-gen-go.exe" set "PROTOC_GEN_GO=%USERPROFILE%\go\bin\protoc-gen-go.exe"

set "PROTOC_GEN_GO_GRPC=protoc-gen-go-grpc"
if exist "%USERPROFILE%\go\bin\protoc-gen-go-grpc.exe" set "PROTOC_GEN_GO_GRPC=%USERPROFILE%\go\bin\protoc-gen-go-grpc.exe"

cd /d "%GO_DIR%"
call "%GOEXE%" mod download
if errorlevel 1 (
    echo Failed to download Go dependencies
    exit /b 1
)

for /f "delims=" %%i in ('"%GOEXE%" env GOPATH') do set "GOPATH_DIR=%%i"
if defined GOPATH_DIR set "PATH=%GOPATH_DIR%\bin;%PATH%"

echo Installing Go proto plugins...
call "%GOEXE%" install google.golang.org/protobuf/cmd/protoc-gen-go@latest
if errorlevel 1 (
    echo Failed to install protoc-gen-go
    exit /b 1
)
call "%GOEXE%" install google.golang.org/grpc/cmd/protoc-gen-go-grpc@latest
if errorlevel 1 (
    echo Failed to install protoc-gen-go-grpc
    exit /b 1
)

if exist "%PROTO_DIR%\engine.proto" (
    echo Generating proto files...
    pushd "%PROTO_DIR%"
    "%PROTOCEXE%" -I . ^
        --plugin=protoc-gen-go="%PROTOC_GEN_GO%" ^
        --plugin=protoc-gen-go-grpc="%PROTOC_GEN_GO_GRPC%" ^
        --go_out=. --go_opt=paths=source_relative ^
        --go-grpc_out=. --go-grpc_opt=paths=source_relative ^
        engine.proto
    if errorlevel 1 (
        popd
        echo Failed to generate root proto package
        exit /b 1
    )

    "%PROTOCEXE%" -I . ^
        --plugin=protoc-gen-go="%PROTOC_GEN_GO%" ^
        --plugin=protoc-gen-go-grpc="%PROTOC_GEN_GO_GRPC%" ^
        --go_out=go --go_opt=paths=source_relative ^
        --go-grpc_out=go --go-grpc_opt=paths=source_relative ^
        engine.proto
    if errorlevel 1 (
        popd
        echo Failed to generate Go module proto package
        exit /b 1
    )
    popd
)

if not exist "%GO_DIR%\bin" mkdir "%GO_DIR%\bin"

echo Building Rust daemon...
cd /d "%RUST_DIR%"
set "PROTOC=%PROTOCEXE%"
call "%CARGOEXE%" build --release -p ai_engine_daemon
if errorlevel 1 (
    echo Failed to build Rust daemon
    exit /b 1
)
copy /Y "target\release\ai_engine_daemon.exe" "%GO_DIR%\bin\ai_engine_daemon.exe" >nul
if errorlevel 1 (
    echo Failed to copy ai_engine_daemon into go\bin
    exit /b 1
)

echo Building Rust context service...
call "%CARGOEXE%" build --release -p context_server
if errorlevel 1 (
    echo Failed to build Rust context service
    exit /b 1
)
copy /Y "target\release\context_server.exe" "%GO_DIR%\bin\context_server.exe" >nul
if errorlevel 1 (
    echo Failed to copy context_server into go\bin
    exit /b 1
)

cd /d "%GO_DIR%"

echo Building server...
call "%GOEXE%" build -o bin\server.exe .\cmd\server
if errorlevel 1 (
    echo Failed to build server
    exit /b 1
)

echo Building client...
call "%GOEXE%" build -o bin\client.exe .\cmd\client
if errorlevel 1 (
    echo Failed to build client
    exit /b 1
)

call :prepare_bundle
if errorlevel 1 exit /b 1

echo Build complete!
echo Bundle: %BUNDLE_DIR%
echo Run: %BUNDLE_BIN%\server.exe --config %BUNDLE_DIR%\config.example.yaml
exit /b 0

:prepare_bundle
if defined BUNDLE_DIR (
    if not "%BUNDLE_DIR:~1,2%"==":\" (
        if exist "%BUNDLE_DIR%" rmdir /s /q "%BUNDLE_DIR%"
    ) else if not "%BUNDLE_DIR%"=="%BUNDLE_DIR:~0,3%" (
        if exist "%BUNDLE_DIR%" rmdir /s /q "%BUNDLE_DIR%"
    )
)
mkdir "%BUNDLE_BIN%"
if errorlevel 1 (
    echo Failed to create bundle directory
    exit /b 1
)

copy /Y "%GO_DIR%\bin\server.exe" "%BUNDLE_BIN%\server.exe" >nul
if errorlevel 1 (
    echo Failed to copy server.exe from %GO_DIR%\bin to %BUNDLE_BIN%
    exit /b 1
)
copy /Y "%GO_DIR%\bin\client.exe" "%BUNDLE_BIN%\client.exe" >nul
if errorlevel 1 (
    echo Failed to copy client.exe from %GO_DIR%\bin to %BUNDLE_BIN%
    exit /b 1
)
copy /Y "%GO_DIR%\bin\ai_engine_daemon.exe" "%BUNDLE_BIN%\ai_engine_daemon.exe" >nul
if errorlevel 1 (
    echo Failed to copy ai_engine_daemon.exe from %GO_DIR%\bin to %BUNDLE_BIN%
    exit /b 1
)
copy /Y "%GO_DIR%\bin\context_server.exe" "%BUNDLE_BIN%\context_server.exe" >nul
if errorlevel 1 (
    echo Failed to copy context_server.exe from %GO_DIR%\bin to %BUNDLE_BIN%
    exit /b 1
)
copy /Y "%ROOT%config.example.yaml" "%BUNDLE_DIR%\config.example.yaml" >nul
if errorlevel 1 (
    echo Failed to copy config.example.yaml from %ROOT% to %BUNDLE_DIR%
    exit /b 1
)
copy /Y "%ROOT%smoke.ps1" "%BUNDLE_DIR%\smoke.ps1" >nul
if errorlevel 1 (
    echo Failed to copy smoke.ps1 from %ROOT% to %BUNDLE_DIR%
    exit /b 1
)
copy /Y "%ROOT%doctor.ps1" "%BUNDLE_DIR%\doctor.ps1" >nul
if errorlevel 1 (
    echo Failed to copy doctor.ps1 from %ROOT% to %BUNDLE_DIR%
    exit /b 1
)
copy /Y "%ROOT%README.md" "%BUNDLE_DIR%\README.md" >nul
if errorlevel 1 (
    echo Failed to copy README.md from %ROOT% to %BUNDLE_DIR%
    exit /b 1
)
goto :eof
