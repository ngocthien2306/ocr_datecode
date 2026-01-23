#!/bin/bash
# MongoDB Setup Script for Jetson Orin - Auto Install & Configure
set -e

GREEN='\033[0;32m'; BLUE='\033[0;34m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
print_success() { echo -e "${GREEN}✓ $1${NC}"; }
print_info() { echo -e "${BLUE}ℹ $1${NC}"; }
print_warning() { echo -e "${YELLOW}⚠ $1${NC}"; }
print_error() { echo -e "${RED}✗ $1${NC}"; }

echo "🚀 MongoDB Setup for Jetson Orin"
echo "================================"
echo ""

# Install Docker if needed
if ! command -v docker &> /dev/null; then
    print_info "Installing Docker..."
    curl -fsSL https://get.docker.com -o get-docker.sh && sudo sh get-docker.sh
    sudo usermod -aG docker $USER
    sudo apt install -y docker-compose
    print_success "Docker installed"
    print_warning "Logout and login, then re-run: ./$(basename $0)"
    exit 0
fi
print_success "Docker installed"

# Configure Docker for Jetson (fix iptables + GPU)
print_info "Configuring Docker daemon..."
sudo tee /etc/docker/daemon.json > /dev/null <<'EOF'
{
    "runtimes": {
        "nvidia": {
            "path": "nvidia-container-runtime",
            "runtimeArgs": []
        }
    },
    "default-runtime": "nvidia",
    "iptables": false
}
EOF
sudo systemctl restart docker && sleep 2
print_success "Docker configured"

# Check Docker permissions
if ! docker ps &> /dev/null; then
    print_warning "Docker permission issue - adding user to docker group"
    sudo usermod -aG docker $USER
    print_warning "Applying new group permissions..."
    exec sg docker "$0 $@"
fi

# Setup MongoDB
MONGODB_DIR="${1:-$HOME/mongodb}"
print_info "Installing MongoDB in: $MONGODB_DIR"
mkdir -p "$MONGODB_DIR/mongodb_data" && cd "$MONGODB_DIR"

# Create docker-compose.yml
cat > docker-compose.yml <<'EOF'
services:
  mongodb:
    image: mongo:latest
    container_name: mongodb
    restart: unless-stopped
    ports:
      - "27017:27017"
    environment:
      MONGO_INITDB_ROOT_USERNAME: admin
      MONGO_INITDB_ROOT_PASSWORD: password
    volumes:
      - ./mongodb_data:/data/db
    healthcheck:
      test: mongosh --eval 'db.runCommand("ping").ok' --quiet
      interval: 10s
      timeout: 5s
      retries: 3

  mongo-express:
    image: mongo-express:latest
    container_name: mongo-express
    restart: unless-stopped
    ports:
      - "8081:8081"
    environment:
      ME_CONFIG_MONGODB_ADMINUSERNAME: admin
      ME_CONFIG_MONGODB_ADMINPASSWORD: password
      ME_CONFIG_MONGODB_URL: mongodb://admin:password@mongodb:27017/
      ME_CONFIG_BASICAUTH_USERNAME: admin
      ME_CONFIG_BASICAUTH_PASSWORD: admin123
    depends_on:
      mongodb:
        condition: service_healthy
EOF

print_success "docker-compose.yml created"

# Start services
print_info "Starting MongoDB..."
docker compose pull && docker compose up -d
sleep 3

# Verify
if docker compose ps | grep -q "mongodb.*Up"; then
    print_success "MongoDB running"
else
    print_error "Failed to start. Check: docker compose logs"
    exit 1
fi

# Summary
echo ""
echo "========================================"
print_success "MongoDB Setup Complete! 🎉"
echo "========================================"
echo ""
echo "🔗 Access:"
echo "   MongoDB:       mongodb://admin:password@localhost:27017"
echo "   Mongo Express: http://localhost:8081 (admin/admin123)"
echo ""
echo "📝 Commands:"
echo "   Start:  cd $MONGODB_DIR && docker compose up -d"
echo "   Stop:   cd $MONGODB_DIR && docker compose down"
echo "   Logs:   docker compose logs -f"
echo "   Shell:  docker exec -it mongodb mongosh -u admin -p password"
echo ""

# Add to bashrc
read -p "Add MONGODB_URL to ~/.bashrc? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]] && ! grep -q "MONGODB_URL" ~/.bashrc; then
    echo -e "\n# MongoDB\nexport MONGODB_URL='mongodb://admin:password@localhost:27017'" >> ~/.bashrc
    print_success "Added to ~/.bashrc - run: source ~/.bashrc"
fi

echo ""
print_info "Test: docker exec -it mongodb mongosh -u admin -p password"