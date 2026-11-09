#!/bin/bash
# Pro Dragon — macOS/Linux install script
# Run once to set up the Python environment

echo ""
echo " Pro Dragon — Dependency Installer"
echo " ==================================="
echo ""

if ! command -v python3 &>/dev/null; then
    echo "ERROR: python3 not found. Install Python 3.10+ first."
    exit 1
fi

echo "Creating virtual environment..."
python3 -m venv venv

echo "Activating virtual environment..."
source venv/bin/activate

echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo " Done! To run Pro Dragon:"
echo "   source venv/bin/activate"
echo "   python main.py"
echo ""
