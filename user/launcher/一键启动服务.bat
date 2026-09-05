@echo off
setlocal

cd /d "%~dp0\..\.."

set "PYTHON_EXE=E:\miniconda3\envs\nninteractive\python.exe"
set "PATH=E:\miniconda3\envs\nninteractive\Scripts;E:\miniconda3\envs\nninteractive;%PATH%"
set "SERVER_SCRIPT=implementation\server\inference\toothseg_server.py"
set "SERVER_URL=http://127.0.0.1:8000/api/v1"

echo ============================================
echo   CBCT ToothSeg Semantic Inference Server
echo ============================================
echo.
echo Server URL:
echo   %SERVER_URL%
echo.
echo Python:
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

if not exist "%PYTHON_EXE%" (
  echo [ERROR] Python environment not found:
  echo   %PYTHON_EXE%
  echo.
  echo Please check whether the nnInteractive conda environment exists.
  echo.
  pause
  exit /b 1
)

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
  echo [ERROR] Server exited with an error.
  echo Please check the messages above.
)

echo.
pause
