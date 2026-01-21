#!/bin/bash
# ╔══════════════════════════════════════════════════════════════════╗
# ║           OCR Datecode - One-Click Installer                     ║
# ║                  For Jetson Orin Platform                        ║
# ╚══════════════════════════════════════════════════════════════════╝
#
# Usage:
#   ./install.sh              # Full installation
#   ./install.sh --help       # Show help
#   ./install.sh --skip-clone # Skip git clone (if source already exists)
#
set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m'
BOLD='\033[1m'

print_success() { echo -e "${GREEN}✓ $1${NC}"; }
print_info() { echo -e "${BLUE}ℹ $1${NC}"; }
print_warning() { echo -e "${YELLOW}⚠ $1${NC}"; }
print_error() { echo -e "${RED}✗ $1${NC}"; }
print_step() { echo -e "\n${CYAN}${BOLD}▶ $1${NC}\n"; }

# Configuration
REPO_URL="https://github.com/ngocthien2306/ocr_datecode.git"
BRANCH="release_v1"

# Get real user (not root if using sudo)
if [ -n "$SUDO_USER" ]; then
    INSTALL_USER="$SUDO_USER"
    INSTALL_HOME=$(getent passwd "$SUDO_USER" | cut -d: -f6)
else
    INSTALL_USER="$USER"
    INSTALL_HOME="$HOME"
fi

SOURCE_DIR="$INSTALL_HOME/Source"
PROJECT_DIR="$SOURCE_DIR/ocr_datecode"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKIP_CLONE=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-clone)
            SKIP_CLONE=true
            shift
            ;;
        --help|-h)
            echo "OCR Datecode Installer"
            echo ""
            echo "Usage: ./install.sh [options]"
            echo ""
            echo "Options:"
            echo "  --skip-clone    Skip git clone (use existing source)"
            echo "  --help, -h      Show this help"
            echo ""
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Banner
clear
echo ""
echo -e "${CYAN}"
echo "  ╔══════════════════════════════════════════════════════════╗"
echo "  ║                                                          ║"
echo "  ║           OCR DATECODE INSTALLER                         ║"
echo "  ║           For Jetson Orin Platform                       ║"
echo "  ║                                                          ║"
echo "  ╚══════════════════════════════════════════════════════════╝"
echo -e "${NC}"
echo ""
echo "  This installer will setup:"
echo "    • System packages & dependencies"
echo "    • MongoDB database (Docker)"
echo "    • Backend API (Python/FastAPI)"
echo "    • AI Services (CUDA/ONNX Runtime)"
echo "    • Frontend (Node.js/React)"
echo "    • Ngrok tunneling (optional)"
echo "    • Systemd services"
echo ""
echo "  Estimated time: 15-30 minutes"
echo ""
echo -e "  ${YELLOW}Note: Some steps require sudo password${NC}"
echo ""

read -p "  Press ENTER to start installation or Ctrl+C to cancel..."

# ============================================
# Step 1: System Prerequisites
# ============================================
print_step "Step 1/8: Installing System Prerequisites"

sudo apt update
sudo apt install -y \
    python3-dev \
    python3-pip \
    python3-venv \
    build-essential \
    git \
    curl \
    wget

print_success "System prerequisites installed"

# ============================================
# Step 2: Clone Project
# ============================================
print_step "Step 2/8: Setting up Project"

mkdir -p "$SOURCE_DIR"

if [ "$SKIP_CLONE" = true ]; then
    print_info "Skipping git clone (--skip-clone)"
    if [ ! -d "$PROJECT_DIR" ]; then
        print_error "Project directory not found: $PROJECT_DIR"
        exit 1
    fi
elif [ -d "$PROJECT_DIR" ]; then
    print_info "Project already exists, updating..."
    cd "$PROJECT_DIR"
    git fetch origin
    git checkout "$BRANCH"
    git pull origin "$BRANCH" || true
else
    print_info "Cloning repository..."
    cd "$SOURCE_DIR"
    git clone "$REPO_URL"
    cd "$PROJECT_DIR"
    git checkout "$BRANCH"
fi

# Update SCRIPT_DIR to project scripts
SCRIPT_DIR="$PROJECT_DIR/scripts"
chmod +x "$SCRIPT_DIR"/*.sh 2>/dev/null || true

print_success "Project ready at $PROJECT_DIR"

# ============================================
# Step 3: DIO Setup (Optional)
# ============================================
print_step "Step 3/8: DIO Setup (Hardware Trigger)"

echo "  DIO is required only for hardware trigger with cameras."
read -p "  Setup DIO? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    if [ -f "$SCRIPT_DIR/setup_dio_sudoers.sh" ]; then
        bash "$SCRIPT_DIR/setup_dio_sudoers.sh"
        print_success "DIO configured"
    fi
else
    print_info "Skipped DIO setup"
fi

# ============================================
# Step 4: MongoDB Setup
# ============================================
print_step "Step 4/8: Installing MongoDB"

if [ -f "$SCRIPT_DIR/setup_mongodb.sh" ]; then
    bash "$SCRIPT_DIR/setup_mongodb.sh" "$HOME/mongodb"
    print_success "MongoDB installed"
else
    print_warning "setup_mongodb.sh not found, skipping..."
fi


# ============================================
# Step 5: Backend Setup
# ============================================
print_step "Step 5/8: Setting up Backend"

cd "$PROJECT_DIR/backend"

print_info "Installing Python packages..."
pip install -r requirements.txt --upgrade
pip install langgraph --upgrade
pip install langchain --upgrade
pip install langchain-openai --upgrade
pip install langchain-core --upgrade
pip install langchain-mongodb --upgrade
pip install langchain-community --upgrade
sudo pip3 install -U jetson-stats 2>/dev/null || true
pip install numpy==1.26.4 --upgrade

# Create .env if not exists
if [ ! -f ".env" ]; then
    print_info "Creating .env file..."
    cat > .env << 'EOF'
MONGODB_URL=mongodb://admin:password@localhost:27017/
DATABASE_NAME=ocr_datecode_db
REDIS_URL=redis://localhost:6379
REDIS_CACHE_TTL_RECENT=60
REDIS_CACHE_TTL_HISTORICAL=300
SECRET_KEY=change-this-secret-key-in-production
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=43200
CORS_ORIGINS=["http://localhost:3000","http://localhost:5173","https://suntech-vision.ngrok.app"]
APP_NAME="OCR Datecode API"
APP_VERSION="1.0.0"
DEBUG=True
API_BASE_URL=http://localhost:8000
TIMEZONE=Asia/Ho_Chi_Minh
OPENAI_API_KEY=sk-your-api-key-here
EOF
    print_warning "Created .env - Please update OPENAI_API_KEY later!"
fi

print_success "Backend configured"

# ============================================
# Step 5.1: Data Migration (Optional)
# ============================================
echo ""
echo "  If you have exported data from another machine, you can import it now."
read -p "  Do you have data to migrate? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    read -p "  Enter path to exports directory (e.g., ./exports): " EXPORTS_DIR
    if [ -d "$EXPORTS_DIR" ]; then
        print_info "Migrating data from $EXPORTS_DIR..."
        cd "$PROJECT_DIR"
        python3 tests/migration_data.py --input-dir "$EXPORTS_DIR"
        print_success "Data migration completed"
    else
        print_warning "Directory not found: $EXPORTS_DIR, skipping migration..."
    fi
else
    print_info "Skipped data migration"
fi

# ============================================
# Step 6: AI Services Setup
# ============================================
print_step "Step 6/8: Setting up AI Services"

# CUDA environment
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Add to bashrc
if ! grep -q "CUDA_HOME" ~/.bashrc; then
    cat >> ~/.bashrc << 'EOF'

# CUDA Environment
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
EOF
fi

pip3 install --upgrade pip setuptools wheel packaging
pip install onnxruntime-gpu==1.23.0 --extra-index-url https://pypi.jetson-ai-lab.io/jp6/cu126 2>/dev/null || pip install onnxruntime-gpu || true
pip3 install pycuda --user 2>/dev/null || true
pip install pypylon

print_success "AI Services configured"

# ============================================
# Step 7: Frontend Setup
# ============================================
print_step "Step 7/8: Setting up Frontend"

# Install Node.js
if ! command -v node &> /dev/null; then
    print_info "Installing Node.js 20.x..."
    curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
    sudo apt install -y nodejs
fi

# Install Yarn
if ! command -v yarn &> /dev/null; then
    sudo npm install -g yarn
fi

cd "$PROJECT_DIR/frontend-ts"
print_info "Installing frontend dependencies..."
yarn install

print_success "Frontend configured"

# ============================================
# Step 8: Install Services
# ============================================
print_step "Step 8/8: Installing Systemd Services"

if [ -d "$SCRIPT_DIR/services" ]; then
    # Replace variables in service file
    sed -e "s|\${INSTALL_USER}|$INSTALL_USER|g" \
        -e "s|\${PROJECT_DIR}|$PROJECT_DIR|g" \
        "$SCRIPT_DIR/services/ocr-all.service" | sudo tee /etc/systemd/system/ocr-all.service > /dev/null

    sudo systemctl daemon-reload
    sudo systemctl enable ocr-all 2>/dev/null || true
    sudo systemctl start ocr-all 2>/dev/null || true
    print_success "OCR service installed, enabled and started (User: $INSTALL_USER)"
fi

# ============================================
# Ngrok Setup (Optional)
# ============================================
echo ""
read -p "  Setup Ngrok for external access? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    if [ -f "$SCRIPT_DIR/setup_ngrok.sh" ]; then
        bash "$SCRIPT_DIR/setup_ngrok.sh"
    fi
fi

# ============================================
# Final Summary
# ============================================
echo ""
echo -e "${GREEN}"
echo "  ╔══════════════════════════════════════════════════════════╗"
echo "  ║                                                          ║"
echo "  ║           INSTALLATION COMPLETE!                         ║"
echo "  ║                                                          ║"
echo "  ╚══════════════════════════════════════════════════════════╝"
echo -e "${NC}"
echo ""
echo "  Access URLs:"
echo "    • Frontend:      http://localhost:5000"
echo "    • Backend API:   http://localhost:8000"
echo "    • API Docs:      http://localhost:8000/docs"
echo "    • MongoDB:       mongodb://admin:password@localhost:27017"
echo "    • Mongo Express: http://localhost:8081 (admin/admin123)"
echo ""
echo "  Start Services:"
echo "    sudo systemctl start ocr-backend"
echo "    sudo systemctl start ocr-frontend"
echo "    sudo systemctl start ocr-ai-services"
echo ""
echo "  Or start all at once:"
echo "    sudo systemctl start ocr-backend ocr-frontend ocr-ai-services"
echo ""
echo "  View Logs:"
echo "    journalctl -u ocr-backend -f"
echo ""
echo -e "  ${YELLOW}Important: Update OPENAI_API_KEY in $PROJECT_DIR/backend/.env${NC}"
echo ""

# ============================================
# Reboot Option
# ============================================
echo ""
read -p "  Reboot system now to apply all changes? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "🔄 System will reboot in 5 seconds..."
    echo "   Press Ctrl+C to cancel"
    sleep 5
    sudo reboot
else
    echo ""
    echo "⚠️  Please reboot manually later to ensure all services start correctly"
    echo "   Run: sudo reboot"
fi
echo ""
echo "  Documentation: $SCRIPT_DIR/README.md"
echo ""
