@echo off
REM Toggle between SH17 general model and E2 custom model
cd /d "%~dp0.."
if exist "ai\model\labppe_v8s.pt" (
  move /Y "ai\model\labppe_v8s.pt" "ai\model\backup\labppe_v8s.pt" >nul
  echo [OK] Now using SH17 general model (better for random web images).
) else if exist "ai\model\backup\labppe_v8s.pt" (
  move /Y "ai\model\backup\labppe_v8s.pt" "ai\model\labppe_v8s.pt" >nul
  echo [OK] Now using E2 custom 4-class model (best with lab dataset images).
) else (
  echo [ERROR] labppe_v8s.pt not found in ai\model or ai\model\backup
)
echo Close and re-run start.bat to apply the change.
pause
