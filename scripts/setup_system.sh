#!/bin/bash
# ============================================
# System Setup Script for OCR Datecode
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

echo "========================================"
echo "  System Setup for OCR Datecode"
echo "========================================"
echo ""

# Update system
print_info "Updating system packages..."
sudo apt update

# Install essential packages
print_info "Installing essential packages..."
sudo apt install -y \
    python3-dev \
    python3-pip \
    python3-venv \
    build-essential \
    git \
    curl \
    wget \
    htop \
    nano \
    vim

print_success "Essential packages installed"

# Setup Source directory
SOURCE_DIR="$HOME/Source"
if [ ! -d "$SOURCE_DIR" ]; then
    print_info "Creating $SOURCE_DIR directory..."
    mkdir -p "$SOURCE_DIR"
    print_success "Created $SOURCE_DIR"
else
    print_success "$SOURCE_DIR already exists"
fi

# Check if ocr_datecode exists
if [ ! -d "$SOURCE_DIR/ocr_datecode" ]; then
    print_info "Cloning ocr_datecode repository..."
    cd "$SOURCE_DIR"
    git clone https://github.com/ngocthien2306/ocr_datecode.git
    cd ocr_datecode
    git checkout release_v1
    print_success "Repository cloned and switched to release_v1"
else
    print_success "ocr_datecode already exists"
    cd "$SOURCE_DIR/ocr_datecode"

    # Check current branch
    CURRENT_BRANCH=$(git branch --show-current)
    if [ "$CURRENT_BRANCH" != "release_v1" ]; then
        print_warning "Current branch is $CURRENT_BRANCH, switching to release_v1..."
        git checkout release_v1
    fi
fi

# Make scripts executable
print_info "Making scripts executable..."
chmod +x "$SOURCE_DIR/ocr_datecode/scripts/"*.sh 2>/dev/null || true
chmod +x "$SOURCE_DIR/ocr_datecode/setup_mongodb.sh" 2>/dev/null || true

print_success "Scripts are now executable"

echo ""
echo "========================================"
print_success "System Setup Complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo "  1. Run: ./scripts/setup_dio_sudoers.sh  (if using hardware trigger)"
echo "  2. Run: ./setup_mongodb.sh"
echo "  3. Run: ./scripts/setup_backend.sh"
echo "  4. Run: ./scripts/setup_ai_services.sh"
echo "  5. Run: ./scripts/setup_frontend.sh"
echo "  6. Run: ./scripts/setup_ngrok.sh"
echo "  7. Run: ./scripts/install_services.sh"
echo ""
