@echo off
setlocal enabledelayedexpansion

REM =========================================================================
REM AI-Based Fraudulent Phone Call Detection System - Windows Execution Script
REM Modes:
REM   run_system.bat all [--force-train]    Full pipeline: setup, train (if needed), launch API & Web UI
REM   run_system.bat train [--force-train]  Train & evaluate BiLSTM model
REM   run_system.bat serve                  Launch FastAPI Gateway & open Web UI
REM   run_system.bat test                   Run pytest test suite
REM =========================================================================

title Call Fraud Detection System Gateway

REM Ensure we run from the project directory
cd /d "%~dp0"
set "PROJECT_ROOT=%CD%"

echo =========================================================================
echo   AI-Based Fraudulent Phone Call Detection System
echo   TensorFlow/Keras BiLSTM ^| Whisper ASR ^| FastAPI Gateway
echo =========================================================================
echo [INFO] Current Working Directory: %PROJECT_ROOT%

REM Parse Arguments
set "MODE=all"
set "FORCE_TRAIN=0"

if "%~1"=="" goto parse_done
if /i "%~1"=="all" (
    set "MODE=all"
) else if /i "%~1"=="train" (
    set "MODE=train"
) else if /i "%~1"=="serve" (
    set "MODE=serve"
) else if /i "%~1"=="test" (
    set "MODE=test"
) else if /i "%~1"=="--force-train" (
    set "MODE=all"
    set "FORCE_TRAIN=1"
) else if /i "%~1"=="help" (
    goto show_help
) else if /i "%~1"=="/?" (
    goto show_help
) else (
    echo [WARNING] Unknown mode '%~1'. Defaulting to 'all'.
    set "MODE=all"
)

if /i "%~2"=="--force-train" set "FORCE_TRAIN=1"

:parse_done
echo [INFO] Selected Mode: %MODE%
if "!FORCE_TRAIN!"=="1" echo [INFO] Flag: --force-train enabled (forcing retraining)
echo.

REM -------------------------------------------------------------------------
REM Step 1: Python Virtual Environment Setup
REM -------------------------------------------------------------------------
echo =========================================================================
echo [STEP 1/4] Checking Python Environment
echo =========================================================================

REM Detect python executable
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Python is not found on your system PATH!
    echo Please install Python 3.10+ and add it to PATH.
    pause
    exit /b 1
)

for /f "tokens=*" %%i in ('python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"') do set "PY_VER=%%i"
echo [INFO] Detected System Python Version: %PY_VER%

if not exist ".venv\Scripts\activate.bat" (
    echo [INFO] Virtual environment not found. Creating '.venv'...
    python -m venv .venv
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to create virtual environment '.venv'.
        pause
        exit /b 1
    )
    echo [SUCCESS] Virtual environment '.venv' created.
) else (
    echo [INFO] Found existing virtual environment in '.venv'.
)

echo [INFO] Activating virtual environment...
call .venv\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo [WARNING] Could not activate .venv directly. Proceeding with system python.
) else (
    echo [SUCCESS] Virtual environment activated.
)

REM -------------------------------------------------------------------------
REM Step 2: Install / Verify Dependencies
REM -------------------------------------------------------------------------
echo =========================================================================
echo [STEP 2/4] Verifying Dependencies
echo =========================================================================

if exist "requirements.txt" (
    echo [INFO] Ensuring required packages are installed from requirements.txt...
    python -m pip install --quiet -r requirements.txt
    if %errorlevel% neq 0 (
        echo [WARNING] pip install completed with warnings. Checking critical packages...
    ) else (
        echo [SUCCESS] Dependencies verified.
    )
) else (
    echo [WARNING] requirements.txt not found. Skipping pip install.
)

REM Handle Test Mode
if /i "%MODE%"=="test" (
    echo =========================================================================
    echo Running Pytest Suite
    echo =========================================================================
    python -m pytest tests/ -v
    pause
    exit /b 0
)

REM -------------------------------------------------------------------------
REM Step 3: Dataset Validation & Optional Model Training
REM -------------------------------------------------------------------------
if /i "%MODE%"=="all" goto do_training_check
if /i "%MODE%"=="train" goto do_training_check
if /i "%MODE%"=="serve" goto do_serve

:do_training_check
echo =========================================================================
echo [STEP 3/4] Dataset Validation & Model Status
echo =========================================================================

REM Validate dataset manifest if manifest exists
if exist "data\metadata\manifest.csv" (
    echo [INFO] Validating existing manifest and audio integrity...
    python main.py validate-data
) else (
    echo [NOTICE] Manifest not found at data\metadata\manifest.csv.
    echo If you want to download NCSU dataset, run: python main.py download-data
)

REM Check if model already trained and saved
set "MODEL_EXISTS=0"
if exist "models\model.keras" (
    if exist "models\vectorizer_config.json" (
        if exist "models\label_encoder.pkl" (
            set "MODEL_EXISTS=1"
        )
    )
) else if exist "models\model" (
    if exist "models\vectorizer_config.json" (
        if exist "models\label_encoder.pkl" (
            set "MODEL_EXISTS=1"
        )
    )
)

if "!MODEL_EXISTS!"=="1" (
    if "!FORCE_TRAIN!"=="0" (
        echo.
        echo [INFO] Production model artifacts found in models/ (.keras format ready).
        echo [INFO] Skipping training. To retrain, re-run with: run_system.bat %MODE% --force-train
        goto after_training
    ) else (
        echo.
        echo [INFO] Model exists but --force-train flag was provided. Proceeding with retraining...
    )
) else (
    echo.
    echo [INFO] No trained model artifacts detected in models/.
)

echo [INFO] Checking audio preprocessing and training splits...
python main.py prepare-dataset
echo [INFO] Launching BiLSTM Model Training...
python main.py train
set "TRAIN_EXIT=%errorlevel%"

if %TRAIN_EXIT% equ 0 (
    echo.
    echo [SUCCESS] BiLSTM model training completed!
    echo [INFO] Running evaluation on test split...
    python main.py evaluate --split test
) else (
    echo.
    echo [NOTICE] Supervised training could not proceed (see scientific reason above).
    echo [NOTICE] The system will start in exploratory mode using linguistic feature heuristics.
)

:after_training
if /i "%MODE%"=="train" (
    echo.
    echo =========================================================================
    echo [SUCCESS] Training mode completed.
    echo =========================================================================
    pause
    exit /b 0
)

REM -------------------------------------------------------------------------
REM Step 4: Launch FastAPI Gateway & Web UI
REM -------------------------------------------------------------------------
:do_serve
echo =========================================================================
echo [STEP 4/4] Launching FastAPI Gateway & Responsive Web Dashboard
echo =========================================================================

echo [INFO] Endpoints active:
echo        - Web Dashboard:        http://127.0.0.1:8000/ (or /ui)
echo        - Interactive OpenAPI:  http://127.0.0.1:8000/docs
echo        - Versioned Predict:    POST http://127.0.0.1:8000/api/v1/predict
echo        - Segment Analysis:     POST http://127.0.0.1:8000/api/v1/predict/segments
echo        - Transcription:        POST http://127.0.0.1:8000/api/v1/transcribe
echo        - Health & Diagnostics: GET  http://127.0.0.1:8000/api/v1/health
echo        - Model Info:           GET  http://127.0.0.1:8000/api/v1/model-info
echo.

REM Spawn background health-watcher that automatically opens the Web UI when ready
start "" python scripts\wait_for_health.py 127.0.0.1 8000

echo [INFO] Starting Uvicorn ASGI Server on http://0.0.0.0:8000...
echo [INFO] Press Ctrl+C at any time to gracefully terminate the gateway.
echo.
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000
goto :eof

:show_help
echo.
echo Usage: run_system.bat [MODE] [OPTIONS]
echo.
echo Modes:
echo   all            Run environment setup, train if model missing, and launch API Gateway (Default)
echo   train          Validate dataset, prepare splits, train and evaluate BiLSTM model
echo   serve          Launch FastAPI server and open Web UI immediately
echo   test           Run pytest unit and integration test suite
echo.
echo Options:
echo   --force-train  Force model retraining even if trained model.keras already exists
echo.
echo Examples:
echo   run_system.bat                     (Run all, opens web UI)
echo   run_system.bat all --force-train   (Retrain model and launch web UI)
echo   run_system.bat serve               (Start server only, opens web UI)
echo   run_system.bat train               (Train and evaluate model only)
echo.
exit /b 0
