#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status.

SCRIPT_DIR=$(dirname "$0")

export NO_CMAKE_BUILD=1

echo Installing development python packages...
pip install -r "$SCRIPT_DIR/../requirements-dev.txt"

echo "Installing slangpy (editable, no CMake build)..."
pip install --editable "$SCRIPT_DIR/../external/slangpy"

echo "Installing falcor2 (editable, no CMake build)..."
pip install --editable "$SCRIPT_DIR/.."

echo "Building extensions..."
python "$SCRIPT_DIR/build.py"

echo "Installing slangpy_torch (if torch is available)..."
python "$SCRIPT_DIR/install_slangpy_torch.py"

unset NO_CMAKE_BUILD
