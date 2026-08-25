@echo off
setlocal EnableDelayedExpansion
title Helios GenAI - Studio Launcher

echo.
echo  ========================================================================
echo    Healthcare-GenAI-Helios  ^|  Clinical & Orthopedic GenAI Pipeline
echo  ========================================================================
echo.

:: -----------------------------------------------------------------------------
:: Paths
:: -----------------------------------------------------------------------------
set "PROJECT_DIR=%~dp0"
set "COMFYUI_APP=%LOCALAPPDATA%\Programs\ComfyUI\ComfyUI.exe"
set "COMFYUI_USER_DIR=%USERPROFILE%\Documents\ComfyUI"
set "COMFY_LORA_DIR=%COMFYUI_USER_DIR%\models\loras"
set "COMFY_WORKFLOW_DIR=%COMFYUI_USER_DIR%\user\default\workflows"

:: -----------------------------------------------------------------------------
:: Ensure target directories exist
:: -----------------------------------------------------------------------------
if not exist "%COMFY_LORA_DIR%" mkdir "%COMFY_LORA_DIR%" >nul 2>&1
if not exist "%COMFY_WORKFLOW_DIR%" mkdir "%COMFY_WORKFLOW_DIR%" >nul 2>&1

:: -----------------------------------------------------------------------------
:: 1. Deploy LoRA Weights
:: -----------------------------------------------------------------------------
echo  [1/3] Syncing LoRA weights...
if exist "%PROJECT_DIR%models\Helios_OrthoJoint_v1.safetensors" (
    copy /Y "%PROJECT_DIR%models\Helios_OrthoJoint_v1.safetensors" "%COMFY_LORA_DIR%\Helios_OrthoJoint_v1.safetensors" >nul
    echo        [OK] Helios_OrthoJoint_v1.safetensors is ready
) else if exist "%COMFY_LORA_DIR%\Helios_OrthoJoint_v1.safetensors" (
    echo        [OK] Helios_OrthoJoint_v1.safetensors verified in ComfyUI
) else (
    echo        [!] Warning: LoRA not found in models\ folder
)

:: -----------------------------------------------------------------------------
:: 2. Deploy Workflows
:: -----------------------------------------------------------------------------
echo  [2/3] Syncing ComfyUI workflows...
set "WF_COUNT=0"
for %%F in ("%PROJECT_DIR%workflows\*.json") do (
    copy /Y "%%F" "%COMFY_WORKFLOW_DIR%\" >nul
    echo        [OK] Loaded workflow: %%~nxF
    set /a WF_COUNT+=1
)

:: -----------------------------------------------------------------------------
:: 3. Check / Start ComfyUI Server
:: -----------------------------------------------------------------------------
echo  [3/3] Checking ComfyUI Server status...
powershell -NoProfile -Command "(New-Object System.Net.Sockets.TcpClient).Connect('127.0.0.1', 8188)" >nul 2>&1
if %ERRORLEVEL% equ 0 (
    echo        [OK] ComfyUI server is ALREADY running at http://127.0.0.1:8188
    goto OPEN_STUDIO
)

echo        Starting ComfyUI Desktop App...
if exist "%COMFYUI_APP%" (
    start "" "%COMFYUI_APP%"
) else (
    echo  [ERROR] ComfyUI Desktop not found at: %COMFYUI_APP%
    echo          Please install ComfyUI Desktop or start ComfyUI manually.
    pause
    exit /b 1
)

echo        Waiting for ComfyUI server to be ready...
set "WAIT_ATTEMPTS=0"

:WAIT_LOOP
timeout /t 2 /nobreak >nul
set /a WAIT_ATTEMPTS+=1
powershell -NoProfile -Command "(New-Object System.Net.Sockets.TcpClient).Connect('127.0.0.1', 8188)" >nul 2>&1
if %ERRORLEVEL% equ 0 (
    echo        [OK] ComfyUI server is ONLINE!
    goto OPEN_STUDIO
)

if %WAIT_ATTEMPTS% lss 15 (
    goto WAIT_LOOP
)

:OPEN_STUDIO
echo.
echo  ========================================================================
echo   Studio is ready! Opening in your default browser...
echo   URL: http://127.0.0.1:8188
echo.
echo   Prompt Trigger Cheat-Sheet:
echo     - Clinic & Facility Environment : HeliosClinic
echo     - Surgical & Specialist Shots   : HeliosSurgeon
echo     - Orthopedic & Joint Concept    : HeliosOrtho
echo.
echo   Workflows available in ComfyUI (Load / Browse):
echo     - Helios_Clinic_v1
echo     - Helios_Surgeon_v1
echo     - Helios_AnimateDiff_Txt2Vid_v1
echo  ========================================================================
echo.

start "" "http://127.0.0.1:8188"
timeout /t 3 >nul
exit /b 0
