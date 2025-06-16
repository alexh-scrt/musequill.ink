#!/bin/bash
# MuseQuill.ink Conda Environment Activation Script
# Usage: source activate_musequill.sh

echo "🚀 Activating MuseQuill conda environment..."
conda activate musequill

if [ $? -eq 0 ]; then
    echo "✅ Environment 'musequill' activated successfully"
    echo "🔧 Python: $(python --version)"
    echo "📦 Pip: $(pip --version)"
    echo ""
    echo "📋 Quick commands:"
    echo "  python main.py --mode config    # Test configuration"
    echo "  python main.py --mode api       # Start API server"
    echo "  make test                       # Run tests"
    echo "  make run-api                    # Development server"
    echo ""
    echo "To deactivate: conda deactivate"
else
    echo "❌ Failed to activate environment 'musequill'"
fi
