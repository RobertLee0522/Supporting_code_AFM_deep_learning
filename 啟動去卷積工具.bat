@echo off
REM Launch the zero-training deconvolution GUI (double-click).
REM ASCII only: cmd.exe reads .bat with the system codepage, so UTF-8 Chinese comments get garbled.
cd /d "%~dp0"
REM Prefer the project venv (has matplotlib/scipy/vtk); fall back to the py launcher.
if exist "afm\Scripts\python.exe" (
    "afm\Scripts\python.exe" zerotrain\blind_deconvolution.py
) else (
    py zerotrain\blind_deconvolution.py
)
if errorlevel 1 pause
