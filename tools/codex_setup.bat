@echo off
setlocal

rem Ignore a stale Conda executable inherited from the parent process.
set "CONDA_EXE="

call conda create --yes --prefix .conda python=3.12
if errorlevel 1 exit /b %errorlevel%

call conda activate .\.conda
if errorlevel 1 exit /b %errorlevel%

call setup.bat
exit /b %errorlevel%
