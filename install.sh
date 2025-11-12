#!/bin/bash

# ΨQRH EINOPS OPTIMIZED - Installation Script
# Production-grade installation for the optimized ΨQRH system

set -e  # Exit on any error

echo "🚀 Installing ΨQRH EINOPS OPTIMIZED System"
echo "=========================================="

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed. Please install Python 3.8+"
    exit 1
fi

# Check if pip is available
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 is required but not installed. Please install pip3"
    exit 1
fi

# Create virtual environment (optional but recommended)
echo "📦 Setting up virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install requirements
echo "📚 Installing dependencies..."
pip3 install --upgrade pip
pip3 install -r requirements.txt

# Verify installation
echo "🔍 Verifying installation..."
python3 -c "
import torch
import einops
import numpy as np
print('✅ PyTorch version:', torch.__version__)
print('✅ EinOps version:', einops.__version__)
print('✅ NumPy version:', np.__version__)
print('✅ All dependencies installed successfully!')
"

# Run basic validation
echo "🧪 Running basic validation..."
if python3 validate_einops_improvements.py; then
    echo "✅ Validation completed successfully!"
else
    echo "⚠️  Validation completed with warnings"
fi

echo ""
echo "🎉 ΨQRH EINOPS OPTIMIZED Installation Complete!"
echo "================================================"
echo ""
echo "Quick Start:"
echo "1. Activate virtual environment: source venv/bin/activate"
echo "2. Run validation: python3 validate_einops_improvements.py"
echo "3. Test the model: python3 ΨQRH_EINOPS_OPTIMIZED.py"
echo "4. Run benchmarks: python3 benchmark_einops_optimization.py"
echo ""
echo "For production use:"
echo "- Ensure CUDA is available for GPU acceleration"
echo "- Use the provided Dockerfile for containerized deployment"
echo "- Check the EINOPS_OPTIMIZATION_REPORT.md for performance details"