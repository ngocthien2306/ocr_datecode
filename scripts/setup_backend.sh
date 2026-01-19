#!/bin/bash
# ============================================
# Backend Setup Script for OCR Datecode
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

BACKEND_DIR="$HOME/Source/ocr_datecode/backend"

echo "========================================"
echo "  Backend Setup for OCR Datecode"
echo "========================================"
echo ""

# Check if backend directory exists
if [ ! -d "$BACKEND_DIR" ]; then
    print_error "Backend directory not found: $BACKEND_DIR"
    print_info "Please run setup_system.sh first"
    exit 1
fi

cd "$BACKEND_DIR"
print_info "Working directory: $BACKEND_DIR"

# Install Python requirements
print_info "Installing Python requirements..."
pip install -r requirements.txt

# Install LangChain packages
print_info "Installing LangChain packages..."
pip install langgraph
pip install langchain
pip install langchain-openai
pip install langchain-core
pip install langchain-mongodb
pip install langchain-community

# Install Jetson stats (for monitoring)
print_info "Installing Jetson stats..."
sudo pip3 install -U jetson-stats || print_warning "jetson-stats may not be available on non-Jetson systems"

# Fix numpy version
print_info "Fixing numpy version..."
pip install numpy==1.26.4 --upgrade

print_success "Python packages installed"

# Create .env file if not exists
ENV_FILE="$BACKEND_DIR/.env"
if [ ! -f "$ENV_FILE" ]; then
    print_info "Creating .env file..."
    cat > "$ENV_FILE" << 'EOF'
# ============================================
# MongoDB Configuration
# ============================================
MONGODB_URL=mongodb://admin:password@localhost:27017/
DATABASE_NAME=ocr_datecode_db

# ============================================
# Redis Configuration
# ============================================
REDIS_URL=redis://localhost:6379
REDIS_CACHE_TTL_RECENT=60
REDIS_CACHE_TTL_HISTORICAL=300

# ============================================
# JWT Configuration
# ============================================
SECRET_KEY=your-secret-key-here-change-this-in-production
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=43200

# ============================================
# CORS Configuration
# ============================================
CORS_ORIGINS=["http://localhost:3000","http://localhost:5173","https://suntech-vision.ngrok.app"]

# ============================================
# Application Configuration
# ============================================
APP_NAME="OCR Datecode API"
APP_VERSION="1.0.0"
DEBUG=True
API_BASE_URL=http://localhost:8000
TIMEZONE=Asia/Ho_Chi_Minh

# ============================================
# OpenAI Configuration (optional)
# ============================================
OPENAI_API_KEY=sk-xxxx-your-api-key-here
EOF
    print_success ".env file created"
    print_warning "Please update OPENAI_API_KEY in $ENV_FILE"
else
    print_success ".env file already exists"
fi

echo ""
echo "========================================"
print_success "Backend Setup Complete!"
echo "========================================"
echo ""
echo "To run backend manually:"
echo "  cd $BACKEND_DIR"
echo "  python3 -m uvicorn app.main:app --reload --port 8000 --host 0.0.0.0"
echo ""
