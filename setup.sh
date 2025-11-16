#!/bin/bash
# Quick setup script for ML CI/CD Showcase

set -e

echo "========================================="
echo "ML CI/CD Showcase - Quick Setup"
echo "========================================="
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Found Python $python_version"

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python3 -m venv venv
echo "✓ Virtual environment created"

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate
echo "✓ Virtual environment activated"

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip --quiet
echo "✓ pip upgraded"

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt --quiet
echo "✓ Dependencies installed"

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo ""
    echo "Creating .env file..."
    cp .env.example .env
    echo "✓ .env file created"
    echo ""
    echo "⚠️  Please edit .env and add your ANTHROPIC_API_KEY"
else
    echo ""
    echo "✓ .env file already exists"
fi

# Run quick tests
echo ""
echo "Running quick tests..."
pytest tests/test_cnn.py -v -m "not slow" --maxfail=1 -q
echo "✓ Quick tests passed"

echo ""
echo "========================================="
echo "Setup Complete! 🎉"
echo "========================================="
echo ""
echo "Next steps:"
echo "  1. Edit .env and add your ANTHROPIC_API_KEY"
echo "  2. Activate environment: source venv/bin/activate"
echo "  3. Run tests: pytest tests/ -v"
echo "  4. Train models: python train.py"
echo ""
echo "For more information, see README.md"
echo ""
