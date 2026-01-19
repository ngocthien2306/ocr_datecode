#!/bin/bash
# ============================================
# Ngrok Setup Script for OCR Datecode
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

# Ngrok configuration
NGROK_AUTHTOKEN="33yUJysiQWcjRrgGr3deROsRSW2_2YJcWaoSo8nEDvzoYSJRq"
NGROK_DOMAIN_FRONTEND="suntech-vision.ngrok.app"
NGROK_DOMAIN_API="suntech-vision-api.ngrok.app"
NGROK_DOMAIN_DB="suntech-vision-db.ngrok.app"

echo "========================================"
echo "  Ngrok Setup for OCR Datecode"
echo "========================================"
echo ""

# Install ngrok if not installed
if ! command -v ngrok &> /dev/null; then
    print_info "Installing Ngrok..."

    # Add ngrok repository
    curl -sSL https://ngrok-agent.s3.amazonaws.com/ngrok.asc \
        | sudo tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null

    echo "deb https://ngrok-agent.s3.amazonaws.com bookworm main" \
        | sudo tee /etc/apt/sources.list.d/ngrok.list

    sudo apt update
    sudo apt install -y ngrok

    print_success "Ngrok installed"
else
    NGROK_VERSION=$(ngrok --version)
    print_success "Ngrok already installed: $NGROK_VERSION"
fi

# Configure ngrok authtoken
print_info "Configuring Ngrok authtoken..."
ngrok config add-authtoken "$NGROK_AUTHTOKEN"
print_success "Ngrok authtoken configured"

# Create ngrok configuration file
NGROK_CONFIG_DIR="$HOME/.config/ngrok"
mkdir -p "$NGROK_CONFIG_DIR"

print_info "Creating Ngrok multi-tunnel configuration..."
cat > "$NGROK_CONFIG_DIR/ngrok-ocr.yml" << EOF
version: "2"
authtoken: $NGROK_AUTHTOKEN
tunnels:
  frontend:
    proto: http
    addr: 5000
    hostname: $NGROK_DOMAIN_FRONTEND
  api:
    proto: http
    addr: 8000
    hostname: $NGROK_DOMAIN_API
  mongo-express:
    proto: http
    addr: 8081
    hostname: $NGROK_DOMAIN_DB
EOF

print_success "Ngrok configuration created: $NGROK_CONFIG_DIR/ngrok-ocr.yml"

echo ""
echo "========================================"
print_success "Ngrok Setup Complete!"
echo "========================================"
echo ""
echo "Configured tunnels:"
echo "  Frontend:      https://$NGROK_DOMAIN_FRONTEND -> localhost:5000"
echo "  API:           https://$NGROK_DOMAIN_API -> localhost:8000"
echo "  Mongo Express: https://$NGROK_DOMAIN_DB -> localhost:8081"
echo ""
echo "To start all tunnels:"
echo "  ngrok start --config $NGROK_CONFIG_DIR/ngrok-ocr.yml --all"
echo ""
echo "To start individual tunnels:"
echo "  ngrok http --url=$NGROK_DOMAIN_FRONTEND 5000"
echo "  ngrok http --url=$NGROK_DOMAIN_API 8000"
echo "  ngrok http --url=$NGROK_DOMAIN_DB 8081"
echo ""
echo "For SSH tunnel:"
echo "  ngrok tcp --region=ap --remote-addr=1.tcp.ap.ngrok.io:23838 22"
echo ""
