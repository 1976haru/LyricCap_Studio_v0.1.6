@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title LyricCap Studio - Start Here
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup_and_run.ps1"
if errorlevel 1 (
  echo.
  echo LyricCap could not start. See the message above and setup_log.txt.
  echo.
  cmd /k
)
