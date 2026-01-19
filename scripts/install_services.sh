#!/bin/bash
# ============================================
# Install Systemd Services for OCR Datecode
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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVICES_DIR="$SCRIPT_DIR/services"

echo "========================================"
echo "  Install OCR Datecode Services"
echo "========================================"
echo ""

# Check if running as root or with sudo
if [ "$EUID" -ne 0 ]; then
    print_info "This script requires sudo privileges"
fi

# List of services to install
SERVICES=(
    "ocr-backend.service"
    "ocr-frontend.service"
    "ocr-ai-services.service"
    "ocr-ngrok.service"
)

# Copy service files
print_info "Installing systemd service files..."
for service in "${SERVICES[@]}"; do
    if [ -f "$SERVICES_DIR/$service" ]; then
        sudo cp "$SERVICES_DIR/$service" /etc/systemd/system/
        print_success "Installed $service"
    else
        print_warning "Service file not found: $service"
    fi
done

# Reload systemd daemon
print_info "Reloading systemd daemon..."
sudo systemctl daemon-reload
print_success "Systemd daemon reloaded"

# Enable services
print_info "Enabling services..."
for service in "${SERVICES[@]}"; do
    sudo systemctl enable "${service%.service}" 2>/dev/null || true
    print_success "Enabled ${service%.service}"
done

echo ""
echo "========================================"
print_success "Services Installation Complete!"
echo "========================================"
echo ""
echo "Available services:"
echo "  - ocr-backend      (API server on port 8000)"
echo "  - ocr-frontend     (Web UI on port 5000)"
echo "  - ocr-ai-services  (Camera management)"
echo "  - ocr-ngrok        (Ngrok tunnels)"
echo ""
echo "Commands:"
echo "  Start all:    sudo systemctl start ocr-backend ocr-frontend ocr-ai-services"
echo "  Stop all:     sudo systemctl stop ocr-backend ocr-frontend ocr-ai-services ocr-ngrok"
echo "  Status:       sudo systemctl status ocr-backend"
echo "  Logs:         journalctl -u ocr-backend -f"
echo ""
echo "To start everything on boot:"
echo "  Services are already enabled and will start automatically"
echo ""
echo "To start now:"
echo "  sudo systemctl start ocr-backend"
echo "  sudo systemctl start ocr-frontend"
echo "  sudo systemctl start ocr-ai-services"
echo "  sudo systemctl start ocr-ngrok  (optional, for external access)"
echo ""
