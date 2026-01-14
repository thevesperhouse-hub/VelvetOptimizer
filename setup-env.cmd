@echo off
REM Setup MSVC environment for Rust + CUDA compilation
REM Run this ONCE per terminal session before cargo commands

echo Configuring MSVC environment...
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"

echo.
echo Environment ready!
echo Now you can run:
echo   cargo build --release
echo   cargo check --workspace --no-default-features
echo.
