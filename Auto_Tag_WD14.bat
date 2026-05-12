@echo off
setlocal EnableDelayedExpansion

title Helios GenAI — Auto Tagger (WD14)

echo.
echo  =====================================================
echo   Healthcare-GenAI-Helios  ^|  Auto_Tag_WD14.bat
echo   Automated WD14 tagging via Kohya_ss engine
echo  =====================================================
echo.

:: ── Configuration ────────────────────────────────────────────────────────────
set "PROJECT_DIR=%~dp0"
set "KOHYA_DIR=C:\kohya_ss"
set "KOHYA_PYTHON=%KOHYA_DIR%\venv\Scripts\python.exe"
set "TAGGER_SCRIPT=%KOHYA_DIR%\sd-scripts\finetune\tag_images_by_wd14_tagger.py"
set "PYTHONPATH=%KOHYA_DIR%\sd-scripts"

:: WD14 model — SwinV2 ships with ONNX weights (required for --onnx mode)
set "MODEL_REPO=SmilingWolf/wd-v1-4-swinv2-tagger-v2"
set "MODEL_CACHE=%KOHYA_DIR%\wd14_tagger_model"

:: Confidence threshold — 0.35 is the industry standard
:: Lower = more tags (noisier). Higher = fewer tags (may miss detail).
set "THRESHOLD=0.35"

:: ── Validate Kohya exists ─────────────────────────────────────────────────────
if not exist "%KOHYA_PYTHON%" (
    echo  [ERROR] Kohya_ss Python not found at: %KOHYA_PYTHON%
    echo.
    echo  Make sure Kohya_ss is installed at C:\kohya_ss
    echo  If it's installed elsewhere, update KOHYA_DIR in this script.
    pause
    exit /b 1
)

if not exist "%TAGGER_SCRIPT%" (
    echo  [ERROR] WD14 tagger script not found at: %TAGGER_SCRIPT%
    echo.
    echo  Expected path: %TAGGER_SCRIPT%
    echo  Your Kohya_ss installation may be an older version.
    echo  Try running this from Kohya GUI instead: Utilities ^> WD14 Captioning
    pause
    exit /b 1
)

:: ── Check images exist ────────────────────────────────────────────────────────
set "SURGEON_DIR=%PROJECT_DIR%dataset\20_HeliosSurgeon"
set "CLINIC_DIR=%PROJECT_DIR%dataset\20_HeliosClinic"

set "SURGEON_HAS_IMAGES=0"
set "CLINIC_HAS_IMAGES=0"

for %%F in ("%SURGEON_DIR%\*.jpg" "%SURGEON_DIR%\*.jpeg" "%SURGEON_DIR%\*.png" "%SURGEON_DIR%\*.webp") do (
    set "SURGEON_HAS_IMAGES=1"
)
for %%F in ("%CLINIC_DIR%\*.jpg" "%CLINIC_DIR%\*.jpeg" "%CLINIC_DIR%\*.png" "%CLINIC_DIR%\*.webp") do (
    set "CLINIC_HAS_IMAGES=1"
)

if "%SURGEON_HAS_IMAGES%"=="0" if "%CLINIC_HAS_IMAGES%"=="0" (
    echo  [ERROR] No images found in either dataset folder.
    echo.
    echo  Please drop your photos in:
    echo    %SURGEON_DIR%
    echo    %CLINIC_DIR%
    echo  Then re-run this script.
    pause
    exit /b 1
)

echo  [CHECK] Found images in dataset folders. Starting tagging pipeline...
echo.

:: ════════════════════════════════════════════════════════════════════════════
:: STEP 1 — Run WD14 tagger on SURGEON folder
:: ════════════════════════════════════════════════════════════════════════════
if "%SURGEON_HAS_IMAGES%"=="1" (
    echo  [1/3] Running WD14 tagger on 20_HeliosSurgeon...
    echo        Model: %MODEL_REPO%
    echo        Threshold: %THRESHOLD%
    echo.

    "%KOHYA_PYTHON%" "%TAGGER_SCRIPT%" ^
        --repo_id "%MODEL_REPO%" ^
        --model_dir "%MODEL_CACHE%" ^
        --caption_extension ".txt" ^
        --thresh %THRESHOLD% ^
        --batch_size 4 ^
        --caption_separator ", " ^
        --append_tags ^
        --onnx ^
        "%SURGEON_DIR%"

    if errorlevel 1 (
        echo.
        echo  [ERROR] WD14 tagger failed on surgeon folder.
        echo  Check the error above. Common cause: missing internet on first run
        echo  ^(model downloads ~600MB on first use^).
        pause
        exit /b 1
    )
    echo.
    echo  [OK] Surgeon folder tagged.
    echo.
) else (
    echo  [SKIP] No images in 20_HeliosSurgeon — skipping.
)

:: ════════════════════════════════════════════════════════════════════════════
:: STEP 2 — Run WD14 tagger on CLINIC folder
:: ════════════════════════════════════════════════════════════════════════════
if "%CLINIC_HAS_IMAGES%"=="1" (
    echo  [2/3] Running WD14 tagger on 20_HeliosClinic...
    echo.

    "%KOHYA_PYTHON%" "%TAGGER_SCRIPT%" ^
        --repo_id "%MODEL_REPO%" ^
        --model_dir "%MODEL_CACHE%" ^
        --caption_extension ".txt" ^
        --thresh %THRESHOLD% ^
        --batch_size 4 ^
        --caption_separator ", " ^
        --append_tags ^
        --onnx ^
        "%CLINIC_DIR%"

    if errorlevel 1 (
        echo.
        echo  [ERROR] WD14 tagger failed on clinic folder.
        pause
        exit /b 1
    )
    echo.
    echo  [OK] Clinic folder tagged.
    echo.
) else (
    echo  [SKIP] No images in 20_HeliosClinic — skipping.
)

:: ════════════════════════════════════════════════════════════════════════════
:: STEP 3 — Run auto_caption.py to prepend trigger words
:: WD14 writes tags but does NOT know about our trigger words.
:: auto_caption.py checks every .txt and safely prepends the
:: trigger word at the front without overwriting anything.
:: ════════════════════════════════════════════════════════════════════════════
echo  [3/3] Prepending trigger words (HeliosSurgeon / HeliosClinic)...
echo.

:: Use project venv if available, else fall back to system Python
if exist "%PROJECT_DIR%.venv\Scripts\python.exe" (
    set "PROJ_PYTHON=%PROJECT_DIR%.venv\Scripts\python.exe"
) else (
    set "PROJ_PYTHON=python"
)

"%PROJ_PYTHON%" "%PROJECT_DIR%scripts\auto_caption.py"

if errorlevel 1 (
    echo.
    echo  [ERROR] auto_caption.py failed.
    echo  Run 'install.bat' first to set up the Python environment.
    pause
    exit /b 1
)

:: ════════════════════════════════════════════════════════════════════════════
echo.
echo  =====================================================
echo   DONE! All images have been tagged and trigger
echo   words have been prepended to every caption file.
echo.
echo   Next step: Run training using config_lora.toml
echo   Or open Kohya GUI to review captions manually.
echo  =====================================================
echo.
pause
endlocal
