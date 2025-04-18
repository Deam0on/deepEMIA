#!/bin/bash
set -e

# 1. Update and install dependencies
sudo apt update
sudo apt install -y g++-9 gcc-9 ninja-build wget

# 2. Set compilers to compatible versions
echo "export CC=gcc-9" >> ~/.bashrc
echo "export CXX=g++-9" >> ~/.bashrc
export CC=gcc-9
export CXX=g++-9

# 3. Install specific PyTorch version with CUDA 11.3
pip uninstall -y torch torchvision torchaudio
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 \
  --extra-index-url https://download.pytorch.org/whl/cu113

# 4. Clean temp and pip cache
rm -rf /tmp/pip-req-build-*
rm -rf ~/.cache/pip

# 5. Install Python dependencies from requirements.txt
pip install -r /home/hladekf/uw-com-vision/requirements.txt

# 6. Install Detectron2 from source
pip install 'git+https://github.com/facebookresearch/detectron2.git@v0.6'

echo "✅ Setup complete. Please restart your terminal or source ~/.bashrc."
