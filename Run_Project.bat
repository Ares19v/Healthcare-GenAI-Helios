@echo off
setlocal EnableDelayedExpansion

title Helios GenAI — Launch Studio

echo.
echo  =====================================================
echo   Healthcare-GenAI-Helios  ^|  Run_Project.bat
echo  =====================================================
echo.

:: ── Paths ────────────────────────────────────────────────────────────────
set "PROJECT_DIR=%~dp0"
set "COMFYUI_DIR=C:\Users\Devansh Tyagi\Documents\ComfyUI"
set "LORA_SRC=%PROJECT_DIR%models\Helios_OrthoJoint_v1.safetensors"
set "LORA_DST=%COMFYUI_DIR%\models\loras\Helios_OrthoJoint_v1.safetensors"
set "WORKFLOW_SRC=%PROJECT_DIR%workflows"
set "WORKFLOW_DST=%COMFYUI_DIR%\user\default\workflows"
set "COMFYUI_PYTHON=%COMFYUI_DIR%\python_embeds\python.exe"

:: ── Validate ComfyUI exists ───────────────────────────────────────────────
if not exist "%COMFYUI_DIR%" (
    echo  [ERROR] ComfyUI not found at: %COMFYUI_DIR%
    echo          Please update COMFYUI_DIR in this script.
    pause
    exit /b 1
)

:: ── Sync LoRA ─────────────────────────────────────────────────────────────
if exist "%LORA_SRC%" (
    echo  [1/3] Copying LoRA to ComfyUI...
    copy /Y "%LORA_SRC%" "%LORA_DST%" >nul
    echo        Done: Helios_OrthoJoint_v1.safetensors
) else (
    echo  [1/3] LoRA not found yet — skipping copy.
    echo        Train the model first, then re-run this script.
)

:: ── Sync Workflows ────────────────────────────────────────────────────────
echo  [2/3] Copying workflows to ComfyUI...
if not exist "%WORKFLOW_DST%" mkdir "%WORKFLOW_DST%"
for %%F in ("%WORKFLOW_SRC%\*.json") do (
    copy /Y "%%F" "%WORKFLOW_DST%\" >nul
    echo        Synced: %%~nxF
)

:: ── Launch ComfyUI ────────────────────────────────────────────────────────
echo  [3/3] Starting ComfyUI engine...
echo.

:: Try embedded Python first, fall back to system Python
if exist "%COMFYUI_PYTHON%" (
    start "" "%COMFYUI_PYTHON%" "%COMFYUI_DIR%\main.py" --auto-launch
) else (
    start "" python "%COMFYUI_DIR%\main.py" --auto-launch
)

echo  ComfyUI is starting...
echo  It will open automatically at: http://127.0.0.1:8188
echo.
echo  Load a workflow:  Menu ^> Load Workflow ^> Helios_Surgeon_v1 or Helios_Clinic_v1
echo.
pause
endlocal
