#!/usr/bin/env bash
# Video Upscaler Pro - Start Script for Linux/macOS
set -uo pipefail

cd "$(dirname "$0")"

RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}========================================"
echo "Video Upscaler Pro"
echo -e "========================================${NC}"
echo ""

# Run the venv interpreter directly instead of sourcing activate, so a failure
# cannot silently fall back to the system Python and report confusing
# missing-dependency errors.
VENV_PY="$PWD/venv/bin/python"

if [ ! -x "$VENV_PY" ]; then
    echo -e "${RED}ERROR: no virtual environment found at venv/bin/python${NC}"
    echo "Please run ./install.sh first."
    exit 1
fi

if [ ! -f app.py ]; then
    echo -e "${RED}ERROR: app.py not found in $PWD${NC}"
    echo "Run ./start.sh from the project folder."
    exit 1
fi

# A venv whose dependencies never installed is the common broken state, and
# "ModuleNotFoundError: gradio" is not a helpful thing to show for it.
if ! "$VENV_PY" -c "import gradio, torch, cv2, numpy" >/dev/null 2>&1; then
    echo -e "${RED}ERROR: the virtual environment is missing its dependencies.${NC}"
    echo ""
    echo "The installation did not finish. Please run ./install.sh again."
    echo "For a full report, run:"
    echo "  venv/bin/python scripts/verify_install.py"
    exit 1
fi

echo -e "${GREEN}Starting Video Upscaler Pro...${NC}"
echo ""
echo "The application will open in your browser automatically."
echo "If it does not, visit http://localhost:7860"
echo ""
echo "Press Ctrl+C to stop it."
echo ""

"$VENV_PY" app.py
EXIT_CODE=$?

if [ "$EXIT_CODE" -ne 0 ]; then
    echo ""
    echo -e "${RED}ERROR: the application exited with code $EXIT_CODE${NC}"
fi

exit "$EXIT_CODE"
