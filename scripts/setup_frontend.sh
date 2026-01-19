#!/bin/bash
# ============================================
# Frontend Setup Script for OCR Datecode
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

FRONTEND_DIR="$HOME/Source/ocr_datecode/frontend-ts"

echo "========================================"
echo "  Frontend Setup for OCR Datecode"
echo "========================================"
echo ""

# Check if frontend directory exists
if [ ! -d "$FRONTEND_DIR" ]; then
    print_error "Frontend directory not found: $FRONTEND_DIR"
    print_info "Please run setup_system.sh first"
    exit 1
fi

# Install Node.js if not installed
if ! command -v node &> /dev/null; then
    print_info "Installing Node.js 20.x..."
    curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
    sudo apt install -y nodejs
    print_success "Node.js installed"
else
    NODE_VERSION=$(node --version)
    print_success "Node.js already installed: $NODE_VERSION"
fi

# Install Yarn if not installed
if ! command -v yarn &> /dev/null; then
    print_info "Installing Yarn..."
    sudo npm install -g yarn
    print_success "Yarn installed"
else
    YARN_VERSION=$(yarn --version)
    print_success "Yarn already installed: $YARN_VERSION"
fi

# Install frontend dependencies
cd "$FRONTEND_DIR"
print_info "Working directory: $FRONTEND_DIR"

print_info "Installing frontend dependencies..."
yarn install

print_success "Frontend dependencies installed"

echo ""
echo "========================================"
print_success "Frontend Setup Complete!"
echo "========================================"
echo ""
echo "To run frontend manually:"
echo "  cd $FRONTEND_DIR"
echo "  yarn dev --port 5000 --host 0.0.0.0"
echo ""
echo "To build for production:"
echo "  yarn build"
echo ""
