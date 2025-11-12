#!/bin/bash

# Install SuperPoint and LightGlue for advanced feature matching
echo "Installing SuperPoint and LightGlue..."

pip install torch torchvision --upgrade
pip install kornia
pip install git+https://github.com/cvg/LightGlue.git

echo "Installation complete!"
