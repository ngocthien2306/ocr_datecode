#!/bin/bash
# ============================================
# Master Setup Script for OCR Datecode
# Run this on a fresh machine to setup everything
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
print_header() {
    echo ""
    echo "========================================"
    echo "  $1"
    echo "========================================"
    echo ""
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║       OCR Datecode - Complete Setup Script               ║"
echo "║              For Jetson Orin Platform                    ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
echo "This script will setup:"
echo "  1. System packages"
echo "  2. MongoDB (Docker)"
echo "  3. Backend (Python/FastAPI)"
echo "  4. AI Services (CUDA/ONNX)"
echo "  5. Frontend (Node.js/React)"
echo "  6. Ngrok (tunneling)"
echo "  7. Systemd services"
echo ""

read -p "Continue with setup? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Setup cancelled."
    exit 0
fi

# Step 1: System Setup
print_header "Step 1/7: System Setup"
bash "$SCRIPT_DIR/setup_system.sh"
print_success "System setup completed"

# Step 2: DIO Setup (optional)
print_header "Step 2/7: DIO Setup (Hardware Trigger)"
read -p "Setup DIO for hardware trigger? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    bash "$SCRIPT_DIR/setup_dio_sudoers.sh"
    print_success "DIO setup completed"
else
    print_info "Skipping DIO setup"
fi

# Step 3: MongoDB Setup
print_header "Step 3/7: MongoDB Setup"
if [ -f "$PROJECT_DIR/setup_mongodb.sh" ]; then
    bash "$PROJECT_DIR/setup_mongodb.sh" "$HOME/mongodb"
    print_success "MongoDB setup completed"
else
    print_warning "setup_mongodb.sh not found, skipping..."
fi

# Step 4: Backend Setup
print_header "Step 4/7: Backend Setup"
bash "$SCRIPT_DIR/setup_backend.sh"
print_success "Backend setup completed"

# Step 5: AI Services Setup
print_header "Step 5/7: AI Services Setup"
bash "$SCRIPT_DIR/setup_ai_services.sh"
print_success "AI Services setup completed"

# Step 6: Frontend Setup
print_header "Step 6/7: Frontend Setup"
bash "$SCRIPT_DIR/setup_frontend.sh"
print_success "Frontend setup completed"

# Step 7: Ngrok Setup
print_header "Step 7/7: Ngrok Setup"
read -p "Setup Ngrok for external access? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    bash "$SCRIPT_DIR/setup_ngrok.sh"
    print_success "Ngrok setup completed"
else
    print_info "Skipping Ngrok setup"
fi

# Install Services
print_header "Installing Systemd Services"
bash "$SCRIPT_DIR/install_services.sh"
print_success "Services installed"

# Data Migration (optional)
print_header "Data Migration"
read -p "Do you have exported data to migrate? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    read -p "Enter path to exports directory: " EXPORTS_DIR
    if [ -d "$EXPORTS_DIR" ]; then
        cd "$PROJECT_DIR"
        python3 tests/migration_data.py --input-dir "$EXPORTS_DIR"
        print_success "Data migration completed"
    else
        print_error "Directory not found: $EXPORTS_DIR"
    fi
else
    print_info "Skipping data migration"
fi

# Final Summary
echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║             Setup Complete!                              ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
echo "Services:"
echo "  Backend:     http://localhost:8000"
echo "  Frontend:    http://localhost:5000"
echo "  MongoDB:     mongodb://admin:password@localhost:27017"
echo "  Mongo UI:    http://localhost:8081"
echo ""
echo "Start services:"
echo "  sudo systemctl start ocr-backend"
echo "  sudo systemctl start ocr-frontend"
echo "  sudo systemctl start ocr-ai-services"
echo "  sudo systemctl start ocr-ngrok  (optional)"
echo ""
echo "Or start all at once:"
echo "  sudo systemctl start ocr-backend ocr-frontend ocr-ai-services"
echo ""
echo "View logs:"
echo "  journalctl -u ocr-backend -f"
echo "  journalctl -u ocr-frontend -f"
echo "  journalctl -u ocr-ai-services -f"
echo ""
print_warning "Don't forget to update OPENAI_API_KEY in backend/.env"
echo ""
