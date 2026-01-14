# Setup MSVC environment for PowerShell
# Run this in the SAME PowerShell where you run cargo

Write-Host "Configuring MSVC environment for PowerShell..." -ForegroundColor Cyan

$vcvarsPath = "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"

if (-not (Test-Path $vcvarsPath)) {
    Write-Host "ERROR: vcvars64.bat not found at: $vcvarsPath" -ForegroundColor Red
    exit 1
}

# Import environment from vcvars64.bat into current PowerShell session
cmd /c "`"$vcvarsPath`" && set" | ForEach-Object {
    if ($_ -match '^([^=]+)=(.*)') {
        $name = $matches[1]
        $value = $matches[2]
        [System.Environment]::SetEnvironmentVariable($name, $value, 'Process')
    }
}

Write-Host ""
Write-Host "Environment ready!" -ForegroundColor Green
Write-Host "LIB path configured: $env:LIB" -ForegroundColor Yellow
Write-Host ""
Write-Host "Now you can run:" -ForegroundColor Cyan
Write-Host "  cargo build --release" -ForegroundColor White
Write-Host "  cargo check --workspace --no-default-features" -ForegroundColor White
Write-Host ""
