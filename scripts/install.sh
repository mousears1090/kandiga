#!/bin/bash
set -e

echo ""
echo "  Kandiga — Run 35B AI models in 1.5GB of RAM"
echo ""

# Check for Apple Silicon
if [[ $(uname -m) != "arm64" ]]; then
    echo "Error: Kandiga requires Apple Silicon (M1/M2/M3/M4)."
    echo "       x86 Macs are not supported (no unified memory / NEON)."
    exit 1
fi

# Check for macOS
if [[ $(uname -s) != "Darwin" ]]; then
    echo "Error: Kandiga requires macOS with Apple Silicon."
    exit 1
fi

# Check for Python 3.10+
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3.10+ required."
    echo "       Install from https://python.org or via Homebrew: brew install python"
    exit 1
fi

PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PYTHON_MAJOR=$(echo "$PYTHON_VERSION" | cut -d. -f1)
PYTHON_MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f2)

if [[ "$PYTHON_MAJOR" -lt 3 ]] || [[ "$PYTHON_MAJOR" -eq 3 && "$PYTHON_MINOR" -lt 10 ]]; then
    echo "Error: Python 3.10+ required (found $PYTHON_VERSION)."
    exit 1
fi

echo "  Python $PYTHON_VERSION detected"
echo ""

# Install kandiga
echo "  Installing kandiga..."
pip3 install kandiga

echo ""
echo "  Done! Kandiga installed."
echo ""
echo "  Next steps:"
echo "    1. kandiga setup       # Download model (~20GB, one-time)"
echo "    2. kandiga chat        # Start chatting"
echo "    3. kandiga chat --fast # 2x speed mode"
echo ""
