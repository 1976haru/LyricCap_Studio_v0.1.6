@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title LyricCap Studio Setup
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup_and_run.ps1" -InstallOnly
if errorlevel 1 (
  echo.
  echo Setup did not complete. See the message above and setup_log.txt.
  echo.
  cmd /k
)
