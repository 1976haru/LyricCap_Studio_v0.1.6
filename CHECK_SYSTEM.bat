@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title LyricCap Studio System Check
echo ================================================
echo LyricCap Studio - System Check
echo ================================================
echo.
echo [Windows]
ver
echo.
echo [Python launcher]
where py 2>nul
py -0p 2>nul
echo.
echo [Python command]
where python 2>nul
python --version 2>nul
echo.
echo [WinGet]
where winget 2>nul
winget --version 2>nul
echo.
echo [FFmpeg]
where ffmpeg 2>nul
ffmpeg -version 2>nul | findstr /b "ffmpeg version"
echo.
echo [LyricCap venv]
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" --version
  echo.
  echo [Python packages]
  ".venv\Scripts\python.exe" -c "import stable_whisper; print('stable-ts: OK')" 2^>nul || echo stable-ts: MISSING
  ".venv\Scripts\python.exe" -c "import demucs; print('demucs: OK')" 2^>nul || echo demucs: MISSING
  ".venv\Scripts\python.exe" -c "import torch; print('torch:', torch.__version__, 'CUDA:', torch.cuda.is_available())" 2^>nul || echo torch: MISSING
) else (
  echo NOT INSTALLED YET
)
echo.
echo ================================================
echo This window will stay open.
echo Type EXIT and press ENTER when finished.
echo ================================================
cmd /k
