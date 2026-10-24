#!/usr/bin/env bash
# Video Upscaler Pro - Installation Script for Linux/macOS
# Sets up the Python virtual environment and installs dependencies.
set -euo pipefail

cd "$(dirname "$0")"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "========================================"
echo "Video Upscaler Pro - Installation"
echo "========================================"
echo ""

echo "[1/7] Looking for a usable Python interpreter..."
echo ""

# Probe interpreters newest-first and keep the first that passes
# scripts/check_python.py. A "python3" on PATH can exist and still be unusable
# (a pre-release build, or one without pip or venv).
PYTHON_CMD=""
for candidate in python3.12 python3.11 python3.10 python3 python; do
    command -v "$candidate" >/dev/null 2>&1 || continue
    if msg=$("$candidate" scripts/check_python.py 2>&1); then
        echo -e "  ${GREEN}$candidate${NC} - $msg"
        PYTHON_CMD="$candidate"
        break
    fi
    echo -e "  ${YELLOW}$candidate${NC} - skipped: $msg"
done

if [ -z "$PYTHON_CMD" ]; then
    echo ""
    echo -e "${RED}ERROR: no usable Python 3.10 - 3.12 installation was found.${NC}"
    echo ""
    echo "This project needs a final-release CPython 3.10, 3.11 or 3.12 with pip"
    echo "and venv. Python 3.13+ is not supported yet: basicsr requires numpy < 2,"
    echo "which has no wheels for 3.13."
    echo ""
    echo "On Ubuntu/Debian: sudo apt install python3.12 python3.12-venv python3-pip"
    echo "On macOS:         brew install python@3.12"
    exit 1
fi

echo ""
echo "[2/7] Selecting the PyTorch build for your GPU..."
CUDA_TAG=$("$PYTHON_CMD" scripts/detect_cuda.py || echo cpu)
if [ "$CUDA_TAG" = "cpu" ]; then
    echo -e "${YELLOW}No NVIDIA GPU detected - installing the CPU build of PyTorch.${NC}"
    echo "Upscaling will work but will be very slow."
else
    echo -e "${GREEN}NVIDIA GPU detected - installing the $CUDA_TAG build of PyTorch.${NC}"
fi

echo ""
echo "[3/7] Creating virtual environment..."
if [ -d venv ]; then
    echo -e "${YELLOW}Removing the existing venv...${NC}"
    rm -rf venv
fi
"$PYTHON_CMD" -m venv venv

# Call the venv interpreter by path rather than sourcing activate, so a failure
# cannot quietly install into the system Python instead.
VENV_PY="$PWD/venv/bin/python"
if [ ! -x "$VENV_PY" ]; then
    echo -e "${RED}ERROR: the virtual environment has no python executable${NC}"
    exit 1
fi
echo -e "${GREEN}OK - virtual environment created${NC}"

echo ""
echo "[4/7] Upgrading pip..."
"$VENV_PY" -m pip install --upgrade pip setuptools wheel

echo ""
echo "[5/7] Installing PyTorch ($CUDA_TAG)..."
echo "This downloads around 3 GB and takes several minutes."
if [ "$CUDA_TAG" = "cpu" ]; then
    "$VENV_PY" -m pip install torch torchvision
else
    "$VENV_PY" -m pip install torch torchvision \
        --index-url "https://download.pytorch.org/whl/$CUDA_TAG"
fi

echo ""
echo "[6/7] Installing the remaining dependencies..."
"$VENV_PY" -m pip install -r requirements.txt

# basicsr ships only an sdist whose setup.py imports torch, so it has to be
# built against the venv rather than pip's isolated build environment.
echo ""
echo "Installing basicsr and realesrgan..."
if "$VENV_PY" -m pip install --no-build-isolation basicsr realesrgan; then
    if ! "$VENV_PY" scripts/patch_basicsr.py; then
        echo -e "${YELLOW}WARNING: basicsr still does not import - Real-ESRGAN will be unavailable.${NC}"
    fi
else
    echo ""
    echo -e "${YELLOW}WARNING: basicsr / realesrgan could not be installed.${NC}"
    echo "The app will still start and fall back to interpolation-based upscaling,"
    echo "but Real-ESRGAN will be unavailable."
fi

echo ""
echo "[7/7] Verifying the installation..."
if ! "$VENV_PY" scripts/verify_install.py; then
    echo ""
    echo -e "${RED}ERROR: the installation is incomplete - see the report above.${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}========================================"
echo "Installation completed successfully!"
echo -e "========================================${NC}"
echo ""
echo "To start the application, run:"
echo "  ./start.sh"
echo ""
