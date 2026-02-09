#!/bin/bash
set -e

echo "============================================"
echo "  GPU Tester - Environment Setup"
echo "============================================"

# Check if conda/python is available
if command -v python3 &> /dev/null; then
    PYTHON=python3
elif command -v python &> /dev/null; then
    PYTHON=python
else
    echo "ERROR: Python not found. Please install Python 3.10+."
    exit 1
fi

echo "Using Python: $($PYTHON --version)"

# Create and activate venv
VENV_DIR=".venv"
if [ ! -d "$VENV_DIR" ]; then
    echo ""
    echo "Creating Python virtual environment in $VENV_DIR..."
    $PYTHON -m venv "$VENV_DIR"
fi

echo "Activating virtual environment..."
source "$VENV_DIR/bin/activate"
echo "Using venv Python: $(python --version) at $(which python)"

# Check CUDA availability
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}'); [print(f'  GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]" 2>/dev/null || echo "PyTorch not installed yet."

# Install requirements
echo ""
echo "Installing requirements..."
pip install -r requirements.txt

echo ""
echo "============================================"
echo "  Setup complete! GPU stress test:"
echo "============================================"
echo ""
echo "  python train_vlm.py --device 0"
echo "  python train_vlm.py --device 0 --use_4bit"
echo "  python train_vlm.py --device 0 --max_steps 50 --img_size 512"
echo ""
echo "  Uses synthetic data — no downloads needed."
echo "============================================"
