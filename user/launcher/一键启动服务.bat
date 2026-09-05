@echo off
setlocal enabledelayedexpansion

cd /d "%~dp0\..\.."

set "SERVER_SCRIPT=implementation\server\inference\toothseg_server.py"
set "SERVER_URL=http://127.0.0.1:8000/api/v1"
set "PYTHON_EXE="

echo ============================================
echo   CBCT ToothSeg Semantic Inference Server
echo ============================================
echo.

rem Step 1: User-defined environment variables
if defined CBCT_PYTHON_EXE if exist "%CBCT_PYTHON_EXE%" set "PYTHON_EXE=%CBCT_PYTHON_EXE%"
if not defined PYTHON_EXE if defined NNINTERACTIVE_PYTHON if exist "%NNINTERACTIVE_PYTHON%" set "PYTHON_EXE=%NNINTERACTIVE_PYTHON%"

rem Step 2: Currently active Conda environment
if not defined PYTHON_EXE if defined CONDA_PREFIX if exist "%CONDA_PREFIX%\python.exe" set "PYTHON_EXE=%CONDA_PREFIX%\python.exe"

rem Step 3: Conda environments registry (~/.conda/environments.txt)
if not defined PYTHON_EXE if exist "%USERPROFILE%\.conda\environments.txt" (
    for /f "tokens=* delims=" %%a in ('findstr /i "nninteractive" "%USERPROFILE%\.conda\environments.txt" 2^>nul') do (
        if not defined PYTHON_EXE if exist "%%a\python.exe" set "PYTHON_EXE=%%a\python.exe"
    )
)

rem Step 4: Common installation directories across machines
if not defined PYTHON_EXE if exist "D:\person_download_old\conda\miniconda3\envs\nninteractive\python.exe" set "PYTHON_EXE=D:\person_download_old\conda\miniconda3\envs\nninteractive\python.exe"
if not defined PYTHON_EXE if exist "E:\miniconda3\envs\nninteractive\python.exe" set "PYTHON_EXE=E:\miniconda3\envs\nninteractive\python.exe"
if not defined PYTHON_EXE if exist "%USERPROFILE%\miniconda3\envs\nninteractive\python.exe" set "PYTHON_EXE=%USERPROFILE%\miniconda3\envs\nninteractive\python.exe"
if not defined PYTHON_EXE if exist "%USERPROFILE%\anaconda3\envs\nninteractive\python.exe" set "PYTHON_EXE=%USERPROFILE%\anaconda3\envs\nninteractive\python.exe"
if not defined PYTHON_EXE if exist "C:\miniconda3\envs\nninteractive\python.exe" set "PYTHON_EXE=C:\miniconda3\envs\nninteractive\python.exe"
if not defined PYTHON_EXE if exist "C:\anaconda3\envs\nninteractive\python.exe" set "PYTHON_EXE=C:\anaconda3\envs\nninteractive\python.exe"
if not defined PYTHON_EXE if exist "D:\miniconda3\envs\nninteractive\python.exe" set "PYTHON_EXE=D:\miniconda3\envs\nninteractive\python.exe"
if not defined PYTHON_EXE if exist "D:\anaconda3\envs\nninteractive\python.exe" set "PYTHON_EXE=D:\anaconda3\envs\nninteractive\python.exe"
if not defined PYTHON_EXE if exist "E:\anaconda3\envs\nninteractive\python.exe" set "PYTHON_EXE=E:\anaconda3\envs\nninteractive\python.exe"

rem Step 5: Fallback to system PATH python
if not defined PYTHON_EXE (
    where python >nul 2>nul
    if not errorlevel 1 set "PYTHON_EXE=python"
)

if not defined PYTHON_EXE (
    echo [ERROR] Could not automatically locate the nninteractive Python environment.
    echo.
    echo Suggestions:
    echo   1. Run 'conda activate nninteractive' before executing this script.
    echo   2. Or set environment variable: set "CBCT_PYTHON_EXE=path\to\python.exe"
    echo.
    pause
    exit /b 1
)

rem Prepend Python and its Scripts directory to PATH for nnUNetv2_predict
if not "%PYTHON_EXE%"=="python" (
    for %%i in ("%PYTHON_EXE%") do set "PY_DIR=%%~dpi"
    if "!PY_DIR:~-1!"=="\" set "PY_DIR=!PY_DIR:~0,-1!"
    set "PATH=!PY_DIR!\Scripts;!PY_DIR!;%PATH%"
)

echo Server URL:
echo   %SERVER_URL%
echo.
echo Python Environment:
echo   %PYTHON_EXE%
echo.
echo nnUNet_results:
if defined nnUNet_results (
  echo   %nnUNet_results%
) else (
  echo   auto-detect in server
)
echo.
echo Close this window or press Ctrl+C to stop the server.
echo.

if not exist "%SERVER_SCRIPT%" (
  echo [ERROR] Server script not found:
  echo   %CD%\%SERVER_SCRIPT%
  echo.
  pause
  exit /b 1
)

"%PYTHON_EXE%" "%SERVER_SCRIPT%"

if errorlevel 1 (
  echo.
  echo [ERROR] Server exited with an error. Please check the logs above.
)

echo.
pause
