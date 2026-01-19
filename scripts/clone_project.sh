#!/bin/bash
# ============================================
# Clone OCR Datecode Project
# Run this first on a fresh machine
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

REPO_URL="https://github.com/ngocthien2306/ocr_datecode.git"
BRANCH="release_v1"
SOURCE_DIR="$HOME/Source"
PROJECT_DIR="$SOURCE_DIR/ocr_datecode"

echo "========================================"
echo "  Clone OCR Datecode Project"
echo "========================================"
echo ""

# Install git if not installed
if ! command -v git &> /dev/null; then
    print_info "Installing git..."
    sudo apt update && sudo apt install -y git
    print_success "Git installed"
fi

# Create Source directory
print_info "Creating $SOURCE_DIR..."
mkdir -p "$SOURCE_DIR"
cd "$SOURCE_DIR"
print_success "Directory created: $SOURCE_DIR"

# Clone or update repository
if [ -d "$PROJECT_DIR" ]; then
    print_warning "Project already exists: $PROJECT_DIR"
    read -p "Do you want to pull latest changes? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        cd "$PROJECT_DIR"
        git fetch origin
        git checkout "$BRANCH"
        git pull origin "$BRANCH"
        print_success "Updated to latest $BRANCH"
    fi
else
    print_info "Cloning repository..."
    git clone "$REPO_URL"
    cd "$PROJECT_DIR"
    git checkout "$BRANCH"
    print_success "Cloned and switched to $BRANCH"
fi

# Make scripts executable
print_info "Making scripts executable..."
chmod +x "$PROJECT_DIR/scripts/"*.sh 2>/dev/null || true
chmod +x "$PROJECT_DIR/setup_mongodb.sh" 2>/dev/null || true

echo ""
echo "========================================"
print_success "Project Ready!"
echo "========================================"
echo ""
echo "Location: $PROJECT_DIR"
echo "Branch:   $BRANCH"
echo ""
echo "Next step:"
echo "  cd $PROJECT_DIR/scripts"
echo "  ./setup_all.sh"
echo ""
