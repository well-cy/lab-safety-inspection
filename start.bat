@echo off
REM 实验室安全智能巡检系统 - 启动脚本
REM 首次运行前请先执行: venv\Scripts\pip install -r requirements.txt
chcp 65001 >nul
cd /d "%~dp0"
echo 正在启动 实验室安全智能巡检与违规预警系统 ...
echo 启动后请访问 http://127.0.0.1:8000
REM ---- 1. 虚拟环境探测：项目约定为 venv，兼容 .venv（两者只用其一即可）----
set "PYEXE="
if exist "venv\Scripts\python.exe" set "PYEXE=venv\Scripts\python.exe"
if not defined PYEXE if exist ".venv\Scripts\python.exe" set "PYEXE=.venv\Scripts\python.exe"
if not defined PYEXE (
    echo [错误] 未找到虚拟环境，请先在仓库根目录执行:
    echo        python -m venv venv
    echo        venv\Scripts\pip install -r requirements.txt
    pause
    exit /b 1
)
echo 使用解释器: %PYEXE%
REM ---- 2. 模型文件检查（权重不随 GitHub 分发，需组长单独提供）----
if not exist "ai\model\labppe_v8s.pt" (
    if not exist "ai\model\sh17_yolov8s.pt" (
        echo [警告] 未找到模型文件 ai\model\labppe_v8s.pt
        echo        正式权重不随 GitHub 仓库分发，请联系组长单独获取后放入 ai\model        echo        （也可用环境变量 LABSAFETY_MODEL_PATH 指向任意路径）
        echo        继续启动将以无模型状态运行，检测功能不可用。
        pause
    )
)
%PYEXE% -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
pause
