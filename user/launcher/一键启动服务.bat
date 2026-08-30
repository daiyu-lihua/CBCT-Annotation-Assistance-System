@echo off
chcp 65001 >nul
rem 定位到项目根目录（本脚本位于项目根/user/launcher 下）
cd /d "%~dp0\..\.."
echo ============================================
echo    CBCT 本地推理服务端 (mock) 启动
echo ============================================
echo.
echo 启动后地址: http://127.0.0.1:8000/api/v1
echo （关闭本窗口 = 停止服务）
echo.
python implementation\server\inference\mock_server.py
if errorlevel 1 (
  echo.
  echo 启动失败：请确认已安装依赖，且在当前正确的 Python 环境中运行。
  echo 详细说明见 user\launcher\环境说明.md
)
echo.
pause