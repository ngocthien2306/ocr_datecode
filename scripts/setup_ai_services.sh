#!/bin/bash
# ============================================
# AI Services Setup Script for OCR Datecode
# Optimized for Jetson Orin with CUDA
# ============================================
set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

print_success() { echo -e "${GREEN}✓ $1${NC}"; }
print_info() { echo -e "${BLUE}ℹ $1${NC}"; }
print_warning() { echo -e "${YELLOW}⚠ $1${NC}"; }
print_error() { echo -e "${RED}✗ $1${NC}"; }

AI_SERVICES_DIR="$HOME/Source/ocr_datecode/ai_services"

echo "========================================"
echo "  AI Services Setup for OCR Datecode"
echo "  (Jetson Orin with CUDA)"
echo "========================================"
echo ""

# Check if ai_services directory exists
if [ ! -d "$AI_SERVICES_DIR" ]; then
    print_error "AI Services directory not found: $AI_SERVICES_DIR"
    print_info "Please run setup_system.sh first"
    exit 1
fi

cd "$AI_SERVICES_DIR"
print_info "Working directory: $AI_SERVICES_DIR"

# Setup CUDA environment
print_info "Setting up CUDA environment..."
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Add to bashrc if not exists
if ! grep -q "CUDA_HOME" ~/.bashrc; then
    print_info "Adding CUDA environment to ~/.bashrc..."
    cat >> ~/.bashrc << 'EOF'

# CUDA Environment
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
EOF
    print_success "CUDA environment added to ~/.bashrc"
fi

# Upgrade pip tools
print_info "Upgrading pip tools..."
pip3 install --upgrade pip setuptools wheel packaging

# Install ONNX Runtime GPU for Jetson
print_info "Installing ONNX Runtime GPU for Jetson..."
pip install onnxruntime-gpu==1.23.0 --extra-index-url https://pypi.jetson-ai-lab.io/jp6/cu126 || {
    print_warning "Jetson ONNX Runtime not available, trying standard version..."
    pip install onnxruntime-gpu
}

# Install PyCUDA
print_info "Installing PyCUDA..."
pip3 install pycuda --user || print_warning "PyCUDA installation may require manual setup"

# Install Basler camera SDK (pypylon)
print_info "Installing pypylon (Basler camera SDK)..."
pip install pypylon

print_success "AI Services packages installed"

# Check CUDA availability
echo ""
print_info "Checking CUDA availability..."
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | cut -d',' -f1)
    print_success "CUDA Version: $CUDA_VERSION"
else
    print_warning "nvcc not found. CUDA may not be properly installed."
fi

# Check if nvidia-smi works
if command -v nvidia-smi &> /dev/null; then
    print_success "NVIDIA GPU detected"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
else
    # On Jetson, use tegrastats or jtop instead
    if command -v tegrastats &> /dev/null; then
        print_success "Running on Jetson platform"
    else
        print_warning "GPU tools not found"
    fi
fi

echo ""
echo "========================================"
print_success "AI Services Setup Complete!"
echo "========================================"
echo ""
echo "To run AI service manually:"
echo "  cd $AI_SERVICES_DIR"
echo "  python3 camera_management_service.py"
echo ""
echo "Note: Run 'source ~/.bashrc' to load CUDA environment"
echo ""
