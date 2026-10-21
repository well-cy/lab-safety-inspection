@echo off
REM 实验室安全智能巡检系统 - 启动脚本
REM 首次运行前请先执行: venv\Scripts\pip install -r requirements.txt
chcp 65001 >nul
cd /d "%~dp0"
echo 正在启动 实验室安全智能巡检与违规预警系统 ...
echo 启动后请访问 http://127.0.0.1:8000
REM 虚拟环境兼容探测：优先 .venv，否则 venv（两者只用其一即可）
set "PYEXE="
if exist ".venv\Scripts\python.exe" set "PYEXE=.venv\Scripts\python.exe"
if not defined PYEXE if exist "venv\Scripts\python.exe" set "PYEXE=venv\Scripts\python.exe"
if not defined PYEXE (
    echo 未找到虚拟环境，请先在仓库根目录执行: venv\Scripts\pip install -r requirements.txt
    pause
    exit /b 1
)
%PYEXE% -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
pause
