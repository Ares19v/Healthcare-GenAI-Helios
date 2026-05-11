@echo off
setlocal EnableDelayedExpansion

title Helios GenAI — Install

echo.
echo  =====================================================
echo   Healthcare-GenAI-Helios  ^|  install.bat
echo   Setting up Python environment for utility scripts
echo  =====================================================
echo.

set "PROJECT_DIR=%~dp0"
set "VENV_DIR=%PROJECT_DIR%.venv"

:: ── Check Python ─────────────────────────────────────────────────────────
python --version >nul 2>&1
if errorlevel 1 (
    echo  [ERROR] Python not found. Please install Python 3.11+ and add it to PATH.
    pause
    exit /b 1
)

for /f "tokens=2 delims= " %%V in ('python --version 2^>^&1') do set "PY_VERSION=%%V"
echo  [OK] Found Python %PY_VERSION%

:: ── Create venv ───────────────────────────────────────────────────────────
if exist "%VENV_DIR%" (
    echo  [INFO] Virtual environment already exists — skipping creation.
) else (
    echo  [1/2] Creating virtual environment...
    python -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo  [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo        Created: .venv\
)

:: ── Install dependencies ──────────────────────────────────────────────────
echo  [2/2] Installing dependencies from requirements.txt...
"%VENV_DIR%\Scripts\pip.exe" install --upgrade pip --quiet
"%VENV_DIR%\Scripts\pip.exe" install -r "%PROJECT_DIR%requirements.txt" --quiet

if errorlevel 1 (
    echo  [ERROR] pip install failed. Check your internet connection.
    pause
    exit /b 1
)

echo.
echo  =====================================================
echo   Installation complete!
echo.
echo   Usage examples:
echo     .venv\Scripts\python.exe scripts\auto_caption.py
echo     .venv\Scripts\python.exe scripts\batch_generate.py
echo  =====================================================
echo.
pause
endlocal
