#!/usr/bin/env bash
# ============================================================
# clean_and_reinstall.sh
# Complete environment setup for torch 2.9.1 + CUDA 13.0
#
# USAGE:
#   1. Open a terminal with conda available
#   2. Activate your env: conda activate <your_env_name>
#   3. Run: chmod +x clean_and_reinstall.sh && ./clean_and_reinstall.sh
# ============================================================

set -e  # Stop on any error

CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo ""
echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN}  Step 1: Uninstalling all pip packages${NC}"
echo -e "${CYAN}========================================${NC}"

packages=$(python -m pip freeze | grep -v "^-e")
if [ -n "$packages" ]; then
    echo "$packages" | while read -r line; do
        pkg_name=$(echo "$line" | cut -d'=' -f1)
        python -m pip uninstall -y "$pkg_name" 2>/dev/null || true
    done
else
    echo -e "${YELLOW}No packages to uninstall.${NC}"
fi

echo ""
echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN}  Step 2: Clearing pip cache${NC}"
echo -e "${CYAN}========================================${NC}"
python -m pip cache purge

echo ""
echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN}  Step 3: Upgrading pip${NC}"
echo -e "${CYAN}========================================${NC}"
python -m pip install --upgrade pip

echo ""
echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN}  Step 4: Installing NumPy FIRST${NC}"
echo -e "${CYAN}  (must be done before everything else)${NC}"
echo -e "${CYAN}========================================${NC}"
python -m pip install "numpy==2.1.0" --no-cache-dir

echo ""
echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN}  Step 5: Installing PyTorch stack${NC}"
echo -e "${CYAN}  (torch 2.9.1 + CUDA 13.0)${NC}"
echo -e "${CYAN}========================================${NC}"
python -m pip install \
    torch==2.9.1 \
    torchvision==0.24.1 \
    torchaudio==2.9.1 \
    --index-url https://download.pytorch.org/whl/cu130 \
    --no-cache-dir

echo ""
echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN}  Step 6: Verifying GPU is visible${NC}"
echo -e "${CYAN}========================================${NC}"
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda)"

echo ""
echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN}  Step 7: Pinning NumPy back to 2.1.0${NC}"
echo -e "${CYAN}  (torch install may have upgraded it)${NC}"
echo -e "${CYAN}========================================${NC}"
python -m pip install "numpy==2.1.0" --force-reinstall --no-cache-dir

echo ""
echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN}  Step 8: Installing scipy + scikit-learn${NC}"
echo -e "${CYAN}  (must be built against numpy 2.1.0)${NC}"
echo -e "${CYAN}========================================${NC}"
python -m pip install scipy scikit-learn --force-reinstall --no-cache-dir

echo ""
echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN}  Step 9: Installing pandas${NC}"
echo -e "${CYAN}  (pinned to 2.2.3 - last version${NC}"
echo -e "${CYAN}   compatible with numpy 2.1.0)${NC}"
echo -e "${CYAN}========================================${NC}"
python -m pip install "pandas==2.2.3" --no-cache-dir

echo ""
echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN}  Step 10: Installing Pillow${NC}"
echo -e "${CYAN}  (pinned to 9.5.0 for transformers${NC}"
echo -e "${CYAN}   PIL.Image compatibility)${NC}"
echo -e "${CYAN}========================================${NC}"
python -m pip install "Pillow==9.5.0" --no-cache-dir

echo ""
echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN}  Step 11: Installing HuggingFace stack${NC}"
echo -e "${CYAN}========================================${NC}"
python -m pip install \
    "transformers==4.57.6" \
    "peft==0.18.0" \
    "datasets==3.6.0" \
    "sentence-transformers==3.4.1" \
    "fsspec==2025.3.0" \
    --no-cache-dir

echo ""
echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN}  Step 12: Installing HuggingFace${NC}"
echo -e "${CYAN}  support libraries${NC}"
echo -e "${CYAN}========================================${NC}"
python -m pip install \
    accelerate \
    safetensors \
    tokenizers \
    huggingface-hub \
    --no-cache-dir

echo ""
echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN}  Step 13: Installing ML utilities${NC}"
echo -e "${CYAN}========================================${NC}"
python -m pip install \
    hdbscan \
    matplotlib \
    seaborn \
    --no-cache-dir

echo ""
echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN}  Step 14: Installing test framework${NC}"
echo -e "${CYAN}========================================${NC}"
python -m pip install pytest --no-cache-dir

echo ""
echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN}  Step 15: Final numpy pin${NC}"
echo -e "${CYAN}  (insurance against any upgrade${NC}"
echo -e "${CYAN}   that snuck in above)${NC}"
echo -e "${CYAN}========================================${NC}"
python -m pip install "numpy==2.1.0" --force-reinstall --no-cache-dir

echo ""
echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN}  Step 16: Pinning numpy in conda${NC}"
echo -e "${CYAN}  (prevents future upgrades)${NC}"
echo -e "${CYAN}========================================${NC}"
conda config --add pinned_packages numpy=2.1.0

echo ""
echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN}  Step 17: Final verification${NC}"
echo -e "${CYAN}========================================${NC}"
python -c "
import torch
import transformers
import peft
import datasets
import sentence_transformers
import sklearn
import scipy
import hdbscan
import pandas
import numpy
import matplotlib
import seaborn
import PIL
import pytest
import fsspec

print('============ VERSION REPORT ============')
print('torch:               ', torch.__version__)
print('transformers:        ', transformers.__version__)
print('peft:                ', peft.__version__)
print('datasets:            ', datasets.__version__)
print('sentence-transformers:', sentence_transformers.__version__)
print('scikit-learn:        ', sklearn.__version__)
print('scipy:               ', scipy.__version__)
print('numpy:               ', numpy.__version__)
print('pandas:              ', pandas.__version__)
print('Pillow:              ', PIL.__version__)
print('fsspec:              ', fsspec.__version__)
print('========================================')
print('CUDA available:      ', torch.cuda.is_available())
print('CUDA version:        ', torch.version.cuda)
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print('GPU name:            ', props.name)
    print('GPU memory (GB):     ', props.total_memory / 1e9)
print('========================================')
print('All packages installed successfully!')
"

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Setup complete!${NC}"
echo -e "${GREEN}  IMPORTANT REMINDERS:${NC}"
echo -e "${YELLOW}  - Always use 'python -m pip install'${NC}"
echo -e "${YELLOW}    NOT bare 'pip install'${NC}"
echo -e "${YELLOW}  - If numpy gets upgraded again, run:${NC}"
echo -e "${YELLOW}    python -m pip install numpy==2.1.0${NC}"
echo -e "${YELLOW}    --force-reinstall --no-cache-dir${NC}"
echo -e "${YELLOW}  - Then reinstall scipy + scikit-learn${NC}"
echo -e "${YELLOW}    to rebuild against correct numpy${NC}"
echo -e "${GREEN}========================================${NC}"