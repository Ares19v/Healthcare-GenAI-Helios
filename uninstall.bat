@echo off
setlocal EnableDelayedExpansion

title Helios GenAI — Uninstall

echo.
echo  =====================================================
echo   Healthcare-GenAI-Helios  ^|  uninstall.bat
echo  =====================================================
echo.
echo  This will remove the Python virtual environment (.venv).
echo  Your dataset, models, workflows, and outputs are NOT affected.
echo.
set /p CONFIRM="  Type YES to confirm and continue: "

if /i not "%CONFIRM%"=="YES" (
    echo.
    echo  Cancelled. Nothing was removed.
    pause
    exit /b 0
)

set "PROJECT_DIR=%~dp0"
set "VENV_DIR=%PROJECT_DIR%.venv"

:: ── Remove venv ───────────────────────────────────────────────────────────
if exist "%VENV_DIR%" (
    echo.
    echo  Removing .venv...
    rmdir /s /q "%VENV_DIR%"
    echo  Done.
) else (
    echo  No .venv directory found — nothing to remove.
)

echo.
echo  =====================================================
echo   Uninstall complete. Run install.bat to reinstall.
echo  =====================================================
echo.
pause
endlocal
