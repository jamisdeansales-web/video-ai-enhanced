@echo off
REM Video Upscaler Pro - Start Script for Windows
setlocal enabledelayedexpansion

echo ========================================
echo Video Upscaler Pro
echo ========================================
echo.

cd /d "%~dp0"

REM Run the venv interpreter directly instead of activating. activate.bat only
REM prepends to PATH, so if it fails the app would silently run on the system
REM Python and report confusing missing-dependency errors.
set "VENV_PY=%CD%\venv\Scripts\python.exe"

if not exist "!VENV_PY!" (
    echo ERROR: no virtual environment found at venv\Scripts\python.exe
    echo.
    echo Please run install.bat first.
    echo.
    pause
    exit /b 1
)

if not exist app.py (
    echo ERROR: app.py not found in %CD%
    echo Run start.bat from the project folder.
    echo.
    pause
    exit /b 1
)

REM A venv whose dependencies never installed is the common broken state, and
REM "ModuleNotFoundError: gradio" is not a helpful thing to show for it.
"!VENV_PY!" -c "import gradio, torch, cv2, numpy" >nul 2>&1
if errorlevel 1 (
    echo ERROR: the virtual environment is missing its dependencies.
    echo.
    echo The installation did not finish. Please run install.bat again.
    echo For a full report, run:
    echo   venv\Scripts\python.exe scripts\verify_install.py
    echo.
    pause
    exit /b 1
)

echo Starting Video Upscaler Pro...
echo.
echo The application will open in your browser automatically.
echo If it does not, visit http://localhost:7860
echo.
echo Press Ctrl+C in this window to stop it.
echo.

"!VENV_PY!" app.py
set "EXIT_CODE=!errorlevel!"

if not "!EXIT_CODE!"=="0" (
    echo.
    echo ERROR: the application exited with code !EXIT_CODE!
    echo.
    REM A negative or very large code means the process was killed
    REM rather than raising a Python error, so nothing was printed
    REM about why. The log holds the run context up to that point.
    echo The run log is at:
    echo   %LOCALAPPDATA%\video-upscaler-pro\logs\app.log
    echo.
    echo If the app died without a Python error ^(a negative exit code^),
    echo that log and the last lines above are what identify the cause.
    echo Re-running with --verbose adds detail:
    echo   venv\Scripts\python.exe app.py --verbose
    echo.
    pause
)

exit /b !EXIT_CODE!
