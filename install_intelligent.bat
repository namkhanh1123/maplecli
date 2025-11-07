@echo off
REM Installation script for MapleCLI with intelligent features (Windows)

echo ========================================
echo MapleCLI Intelligent Features Installer
echo ========================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Please install Python first.
    pause
    exit /b 1
)

echo [OK] Python is available
echo.

REM Ask user what to install
echo Choose installation option:
echo 1) Basic installation (no intelligent features)
echo 2) Full installation (with intelligent features - recommended)
echo 3) Full installation with GPU support (requires CUDA)
echo.
set /p choice="Enter choice [1-3]: "

if "%choice%"=="1" (
    echo.
    echo Installing basic MapleCLI...
    pip install -e .
    goto :done
)

if "%choice%"=="2" (
    echo.
    echo Installing MapleCLI with intelligent features...
    echo This will download ~500MB of dependencies
    echo.
    set /p confirm="Continue? [y/N]: "
    if /i "%confirm%"=="y" (
        pip install -e ".[intelligent]"
    ) else (
        echo Installation cancelled.
        pause
        exit /b 0
    )
    goto :done
)

if "%choice%"=="3" (
    echo.
    echo Installing MapleCLI with GPU support...
    echo This requires CUDA to be installed on your system.
    echo.
    set /p confirm="Continue? [y/N]: "
    if /i "%confirm%"=="y" (
        pip install -e ".[intelligent]"
        pip uninstall -y faiss-cpu
        pip install faiss-gpu
    ) else (
        echo Installation cancelled.
        pause
        exit /b 0
    )
    goto :done
)

echo Invalid choice. Exiting.
pause
exit /b 1

:done
echo.
echo ========================================
echo Installation complete!
echo ========================================
echo.
echo Quick Start:
echo   1. Run: maplecli chat
echo   2. Enable YOLO mode: :yolo
echo   3. Switch to project: :cd C:\path\to\project
echo   4. Analyze: :analyze
echo   5. Ask questions!
echo.
echo Documentation:
echo   - Quick Start: QUICKSTART_INTELLIGENT.md
echo   - Full Guide: INTELLIGENT_FEATURES.md
echo   - Examples: YOLO_EXAMPLES.md
echo.
echo Happy coding!
echo.
pause

