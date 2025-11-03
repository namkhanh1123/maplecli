# OpenAI CLI Installation Script for PowerShell
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "OpenAI CLI Installation" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if Python is available
try {
    $pythonVersion = python --version 2>&1
    Write-Host "[INFO] Found Python: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "[ERROR] Python is not installed or not in PATH" -ForegroundColor Red
    Write-Host "Please install Python from https://www.python.org/" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

# Install package in editable mode
Write-Host ""
Write-Host "[1/3] Installing Python dependencies..." -ForegroundColor Yellow
pip install -e .

if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Failed to install dependencies" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host ""
Write-Host "[2/3] Installation complete!" -ForegroundColor Green

# Test if command works
Write-Host ""
Write-Host "[3/3] Testing installation..." -ForegroundColor Yellow

$testResult = openaicli --help 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "[WARNING] The command 'openaicli' is not working yet." -ForegroundColor Yellow
    Write-Host "You may need to add Python Scripts to your PATH." -ForegroundColor Yellow
    Write-Host ""
    
    # Try to find Scripts directory
    $scriptsPath = python -c "import os, sys; print(os.path.join(sys.prefix, 'Scripts'))" 2>$null
    
    if ($scriptsPath) {
        Write-Host "Detected Scripts path: $scriptsPath" -ForegroundColor Cyan
        Write-Host ""
        Write-Host "To add to PATH, run this in PowerShell as Administrator:" -ForegroundColor Yellow
        Write-Host "[Environment]::SetEnvironmentVariable('Path', `$env:Path + ';$scriptsPath', 'User')" -ForegroundColor White
        Write-Host ""
        Write-Host "Or for current session only:" -ForegroundColor Yellow
        Write-Host "`$env:Path += ';$scriptsPath'" -ForegroundColor White
        Write-Host ""
        
        # Ask if user wants to add to PATH for current session
        $addPath = Read-Host "Add to PATH for current session? (y/n)"
        if ($addPath -eq 'y' -or $addPath -eq 'Y') {
            $env:Path += ";$scriptsPath"
            Write-Host "PATH updated for current session!" -ForegroundColor Green
            
            # Test again
            openaicli --help 2>&1 | Out-Null
            if ($LASTEXITCODE -eq 0) {
                Write-Host ""
                Write-Host "========================================" -ForegroundColor Green
                Write-Host "SUCCESS! Installation complete!" -ForegroundColor Green
                Write-Host "========================================" -ForegroundColor Green
                Write-Host ""
                Write-Host "You can now use 'openaicli' in this session!" -ForegroundColor Cyan
                Write-Host ""
                Write-Host "Try it: openaicli chat" -ForegroundColor White
                Write-Host ""
                Write-Host "Note: To make this permanent, add the PATH manually or" -ForegroundColor Yellow
                Write-Host "      rerun this script as Administrator." -ForegroundColor Yellow
            }
        }
    }
    
    Write-Host ""
    Write-Host "Alternative: Use it directly with: python main.py chat" -ForegroundColor Cyan
} else {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Green
    Write-Host "SUCCESS! Installation complete!" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "You can now use 'openaicli' from anywhere!" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Try it: openaicli chat" -ForegroundColor White
    Write-Host ""
}

Write-Host ""
Read-Host "Press Enter to exit"
