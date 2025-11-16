@echo off
REM Quick setup script for ML CI/CD Showcase (Windows)

echo =========================================
echo ML CI/CD Showcase - Quick Setup
echo =========================================
echo.

REM Check Python
echo Checking Python version...
python --version
if errorlevel 1 (
    echo ERROR: Python not found. Please install Python 3.10+
    pause
    exit /b 1
)
echo [OK] Python found

REM Create virtual environment
echo.
echo Creating virtual environment...
python -m venv venv
echo [OK] Virtual environment created

REM Activate virtual environment
echo.
echo Activating virtual environment...
call venv\Scripts\activate.bat
echo [OK] Virtual environment activated

REM Upgrade pip
echo.
echo Upgrading pip...
python -m pip install --upgrade pip --quiet
echo [OK] pip upgraded

REM Install dependencies
echo.
echo Installing dependencies...
pip install -r requirements.txt --quiet
echo [OK] Dependencies installed

REM Create .env file
if not exist .env (
    echo.
    echo Creating .env file...
    copy .env.example .env
    echo [OK] .env file created
    echo.
    echo WARNING: Please edit .env and add your ANTHROPIC_API_KEY
) else (
    echo.
    echo [OK] .env file already exists
)

REM Run quick tests
echo.
echo Running quick tests...
pytest tests/test_cnn.py -v -m "not slow" --maxfail=1 -q
if errorlevel 1 (
    echo.
    echo WARNING: Some tests failed, but setup is complete
) else (
    echo [OK] Quick tests passed
)

echo.
echo =========================================
echo Setup Complete!
echo =========================================
echo.
echo Next steps:
echo   1. Edit .env and add your ANTHROPIC_API_KEY
echo   2. Activate environment: venv\Scripts\activate
echo   3. Run tests: pytest tests/ -v
echo   4. Train models: python train.py
echo.
echo For more information, see README.md
echo.
pause
