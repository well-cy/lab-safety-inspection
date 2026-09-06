@echo off
REM 实验室安全智能巡检系统 - 启动脚本
REM 首次运行前请先执行: venv\Scripts\pip install -r requirements.txt
chcp 65001 >nul
cd /d "%~dp0"
echo 正在启动 实验室安全智能巡检与违规预警系统 ...
echo 启动后请访问 http://127.0.0.1:8000
venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
pause
