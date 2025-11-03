@echo off
echo ========================================
echo OpenAI CLI Installation
echo ========================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python from https://www.python.org/
    pause
    exit /b 1
)

echo [1/3] Installing Python dependencies...
pip install -e .
if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)

echo.
echo [2/3] Installation complete!
echo.
echo [3/3] Testing installation...
openaicli --help
if errorlevel 1 (
    echo.
    echo WARNING: The command 'openaicli' is not working yet.
    echo You may need to add Python Scripts to your PATH.
    echo.
    echo Try running this in PowerShell as Administrator:
    echo [Environment]::SetEnvironmentVariable("Path", $env:Path + ";$env:APPDATA\Python\Python311\Scripts", "User")
    echo.
    echo Or use it directly: python main.py chat
) else (
    echo.
    echo ========================================
    echo SUCCESS! Installation complete!
    echo ========================================
    echo.
    echo You can now use 'openaicli' from anywhere!
    echo.
    echo Try it: openaicli chat
    echo.
)

pause
