@echo off
chcp 65001 >nul
rem 定位到项目根目录（本脚本位于项目根/user/launcher 下）
cd /d "%~dp0\..\.."
echo ============================================
echo    CBCT 本地推理服务端 (ToothSeg 真实推理)
echo ============================================
echo.
set "PY=E:\miniconda3\envs\nninteractive\python.exe"
if not exist "%PY%" (
  echo [警告] 未找到 nninteractive 环境，回退到系统 python（可能缺少 torch/依赖）
  set "PY=python"
)
echo 使用 Python: %PY%
echo 启动后地址: http://127.0.0.1:8000/api/v1
echo 注意: /predict 完整推理约 30-90 分钟，期间请勿关闭本窗口
echo （关闭本窗口 = 停止服务）
echo.
"%PY%" implementation\server\inference\toothseg_server.py
if errorlevel 1 (
  echo.
  echo 启动失败：请确认使用 nninteractive 环境（含 fastapi/uvicorn/torch）。
  echo 详细说明见 user\launcher\环境说明.md
)
echo.
pause
