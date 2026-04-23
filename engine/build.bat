@echo off
setlocal

echo Building AI Engine...
set "ROOT=%~dp0"
set "GO_DIR=%ROOT%go"
set "RUST_DIR=%ROOT%rust"
set "PROTO_DIR=%ROOT%proto"
set "CACHE_DIR=%ROOT%.cache"

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

set "PROTOCEXE=protoc"
if exist "%ROOT%..\.tools\protoc-34.1-win64\bin\protoc.exe" set "PROTOCEXE=%ROOT%..\.tools\protoc-34.1-win64\bin\protoc.exe"
if defined PROTOC set "PROTOCEXE=%PROTOC%"

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
    call "%PROTOCEXE%" -I . ^
                      --plugin=protoc-gen-go="%PROTOC_GEN_GO%" ^
                      --plugin=protoc-gen-go-grpc="%PROTOC_GEN_GO_GRPC%" ^
                      --go_out=. --go_opt=paths=source_relative ^
                      --go-grpc_out=. --go-grpc_opt=paths=source_relative ^
                      engine.proto
    if errorlevel 1 (
        echo Warning: failed to generate root proto package.
        exit /b 1
    ) else (
        call "%PROTOCEXE%" -I . ^
                          --plugin=protoc-gen-go="%PROTOC_GEN_GO%" ^
                          --plugin=protoc-gen-go-grpc="%PROTOC_GEN_GO_GRPC%" ^
                          --go_out=go --go_opt=paths=source_relative ^
                          --go-grpc_out=go --go-grpc_opt=paths=source_relative ^
                          engine.proto
        if errorlevel 1 (
            echo Warning: failed to generate Go module proto package.
        )
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
    echo Warning: failed to copy Rust daemon binary into go\bin. Using rust\target\release directly is still supported.
)

echo Building Rust context service...
call "%CARGOEXE%" build --release -p context_server
if errorlevel 1 (
    echo Failed to build Rust context service
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

echo Build complete!
echo Run: go\bin\server.exe --config config.example.yaml
