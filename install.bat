@echo off
REM Video Upscaler Pro - Installation Script for Windows
REM Sets up the Python virtual environment and installs dependencies.
setlocal enabledelayedexpansion

echo ========================================
echo Video Upscaler Pro - Installation
echo ========================================
echo.

cd /d "%~dp0"

echo [1/7] Looking for a usable Python interpreter...
echo.

REM Probe interpreters newest-first and keep the first one that passes
REM scripts\check_python.py. Probing matters here: "py -3.12" and a "python" on
REM PATH can both exist and still be unusable (a pre-release build, or an
REM MSYS2 python that cannot install the torch wheels).
set "PYTHON_CMD="
for %%V in (3.12 3.11 3.10) do (
    if not defined PYTHON_CMD call :probe_python "py -%%V"
)
if not defined PYTHON_CMD call :probe_python "python"
if not defined PYTHON_CMD call :probe_python "python3"

if not defined PYTHON_CMD (
    echo.
    echo ERROR: no usable Python 3.10 - 3.12 installation was found.
    echo.
    echo This project needs a final-release CPython 3.10, 3.11 or 3.12 with pip.
    echo Python 3.13+ is not supported yet: basicsr requires numpy ^< 2, which has
    echo no wheels for 3.13.
    echo.
    echo Install Python 3.12 from:
    echo   https://www.python.org/downloads/release/python-31210/
    echo and tick "Add Python to PATH" in the installer.
    echo.
    pause
    exit /b 1
)

echo.
echo Using: !PYTHON_CMD!
!PYTHON_CMD! --version

echo.
echo [2/7] Selecting the PyTorch build for your GPU...
set "CUDA_TAG="
for /f "delims=" %%t in ('!PYTHON_CMD! scripts\detect_cuda.py') do set "CUDA_TAG=%%t"
if not defined CUDA_TAG set "CUDA_TAG=cpu"

if "!CUDA_TAG!"=="cpu" (
    echo No NVIDIA GPU detected - installing the CPU build of PyTorch.
    echo Upscaling will work but will be very slow.
) else (
    echo NVIDIA GPU detected - installing the !CUDA_TAG! build of PyTorch.
)

echo.
echo [3/7] Creating virtual environment...
if exist venv (
    echo Removing the existing venv...
    rmdir /s /q venv
)

!PYTHON_CMD! -m venv venv
if errorlevel 1 (
    echo ERROR: failed to create the virtual environment
    pause
    exit /b 1
)

REM Call the venv interpreter by path rather than activating. Activation only
REM edits PATH, so a failure to activate would quietly install into the system
REM Python instead of the venv.
set "VENV_PY=%CD%\venv\Scripts\python.exe"
if not exist "!VENV_PY!" (
    echo ERROR: the virtual environment has no python.exe
    pause
    exit /b 1
)

echo.
echo [4/7] Upgrading pip...
"!VENV_PY!" -m pip install --upgrade pip setuptools wheel
if errorlevel 1 (
    echo ERROR: failed to upgrade pip
    pause
    exit /b 1
)

echo.
echo [5/7] Installing PyTorch ^(!CUDA_TAG!^)...
echo This downloads around 3 GB and takes several minutes.
if "!CUDA_TAG!"=="cpu" (
    "!VENV_PY!" -m pip install torch torchvision
) else (
    "!VENV_PY!" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/!CUDA_TAG!
)
if errorlevel 1 (
    echo.
    echo ERROR: failed to install PyTorch
    echo Check your internet connection and try again.
    pause
    exit /b 1
)

echo.
echo [6/7] Installing the remaining dependencies...
"!VENV_PY!" -m pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo ERROR: failed to install dependencies from requirements.txt
    pause
    exit /b 1
)

REM basicsr ships only an sdist whose setup.py imports torch, so it has to be
REM built against the venv rather than pip's isolated build environment.
echo.
echo Installing basicsr and realesrgan...
"!VENV_PY!" -m pip install --no-build-isolation basicsr realesrgan
if errorlevel 1 (
    echo.
    echo WARNING: basicsr / realesrgan could not be installed.
    echo The app will still start and fall back to interpolation-based upscaling,
    echo but Real-ESRGAN will be unavailable.
    echo.
) else (
    "!VENV_PY!" scripts\patch_basicsr.py
    if errorlevel 1 (
        echo WARNING: basicsr still does not import - Real-ESRGAN will be unavailable.
    )
)

echo.
echo [7/7] Verifying the installation...
"!VENV_PY!" scripts\verify_install.py
if errorlevel 1 (
    echo.
    echo ERROR: the installation is incomplete - see the report above.
    pause
    exit /b 1
)

echo.
echo ========================================
echo Installation completed successfully!
echo ========================================
echo.
echo To start the application, run:
echo   start.bat
echo.
pause
exit /b 0


:probe_python
REM %~1 = interpreter command to test. Sets PYTHON_CMD on success.
REM errorlevel is read with "if errorlevel" rather than "%errorlevel%" because a
REM parenthesised block expands %errorlevel% once, when the block is parsed.
%~1 --version >nul 2>&1
if errorlevel 1 goto :eof
for /f "delims=" %%r in ('%~1 scripts\check_python.py 2^>^&1') do set "PROBE_MSG=%%r"
%~1 scripts\check_python.py >nul 2>&1
if errorlevel 1 (
    echo   %~1 - skipped: !PROBE_MSG!
    goto :eof
)
echo   %~1 - !PROBE_MSG!
set "PYTHON_CMD=%~1"
goto :eof
