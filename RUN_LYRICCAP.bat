@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title LyricCap Studio
if not exist ".venv\Scripts\python.exe" (
  echo LyricCap has not been installed yet.
  echo Starting the one-click installer...
  powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup_and_run.ps1"
  if errorlevel 1 cmd /k
  exit /b
)
set "PYTHONPATH=%~dp0src"
".venv\Scripts\python.exe" "%~dp0app.py"
if errorlevel 1 (
  echo.
  echo LyricCap stopped with an error. This window will remain open.
  cmd /k
)
