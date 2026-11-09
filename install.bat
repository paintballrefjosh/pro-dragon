@echo off
REM Pro Dragon — Windows install script
REM Run this once to set up the Python environment

echo.
echo  Pro Dragon — Dependency Installer
echo  ===================================
echo.

python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Install Python 3.10+ from https://python.org
    pause
    exit /b 1
)

echo Creating virtual environment...
python -m venv venv

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Installing dependencies...
pip install --upgrade pip
pip install -r requirements.txt

echo.
echo  Done! To run Pro Dragon:
echo    venv\Scripts\activate
echo    python main.py
echo.
pause
