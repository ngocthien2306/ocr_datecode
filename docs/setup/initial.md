# Jetson Orin Setup Guide - Complete Installation

Hướng dẫn cài đặt môi trường phát triển đầy đủ trên NVIDIA Jetson Orin 16GB cho Computer Vision & Backend Development.

---

## 📋 Table of Contents

1. [Docker Installation](#1-docker-installation)
2. [MongoDB Setup](#2-mongodb-setup)
3. [Node.js & Yarn Installation](#3-nodejs--yarn-installation)
4. [Python Environment Setup](#4-python-environment-setup)
5. [Troubleshooting](#5-troubleshooting)

---

## 1. Docker Installation

### Kiểm tra Docker đã cài chưa

```bash
docker --version
docker compose version
```

### Cài đặt Docker (nếu chưa có)

```bash
# Update system
sudo apt update
sudo apt install -y curl

# Cài Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Add user vào docker group (không cần sudo mỗi lần)
sudo usermod -aG docker $USER
newgrp docker

# Verify installation
docker --version
docker run hello-world
```

### Cài Docker Compose

```bash
# Method 1: apt (recommended)
sudo apt install -y docker-compose

# Method 2: pip (nếu cần version mới hơn)
sudo pip3 install docker-compose

# Verify
docker compose version
```

### Configure Docker cho Jetson

```bash
# Tạo/sửa Docker daemon config
sudo nano /etc/docker/daemon.json
```

Thêm nội dung sau:

```json
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
```

```bash
# Restart Docker
sudo systemctl restart docker

# Test GPU access
docker run --runtime nvidia --rm nvidia/cuda:11.4.0-base-ubuntu20.04 nvidia-smi
```

### Tăng Swap (Optional - cho build image lớn)

```bash
sudo systemctl disable nvzramconfig
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab

# Verify
free -h
```

---

## 2. MongoDB Setup

### Tạo thư mục project

```bash
mkdir -p ~/mongodb
cd ~/mongodb
```

### Tạo docker-compose.yml

```bash
nano docker-compose.yml
```

Nội dung file:

```yaml
version: '3.8'

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
    networks:
      - mongo-network

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
      - mongodb
    networks:
      - mongo-network

networks:
  mongo-network:
    driver: bridge
```

### Tạo thư mục data

```bash
mkdir -p mongodb_data
```

### Chạy MongoDB

```bash
# Start containers
docker compose up -d

# Check status
docker compose ps
docker compose logs -f mongodb

# Stop containers
docker compose down
```

### Truy cập MongoDB

- **MongoDB**: `mongodb://admin:password@localhost:27017`
- **Mongo Express (Web UI)**: `http://localhost:8081` (login: admin/admin123)

### Connection từ Python

```python
# Trong code Python
MONGODB_URL = "mongodb://admin:password@localhost:27017"

# Hoặc dùng environment variable
import os
MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://admin:password@localhost:27017")
```

```bash
# Set environment variable
export MONGODB_URL="mongodb://admin:password@localhost:27017"
```

### Backup & Restore

```bash
# Backup
docker compose exec mongodb mongodump --out /data/backup

# Restore
docker compose exec mongodb mongorestore /data/backup
```

---

## 3. Node.js & Yarn Installation

### Cài Cairo dependencies (cho node-canvas)

```bash
# Cài các thư viện cần thiết
sudo apt update
sudo apt install -y \
    build-essential \
    libcairo2-dev \
    libpango1.0-dev \
    libjpeg-dev \
    libgif-dev \
    librsvg2-dev \
    pkg-config
```

### Cài Node.js (LTS version 20.x)

```bash
# Thêm NodeSource repository
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -

# Cài Node.js
sudo apt install -y nodejs

# Verify
node --version
npm --version
```

### Cài Yarn

```bash
# Method 1: Via npm (recommended)
sudo npm install -g yarn

# Verify
yarn --version
```

### Fix npm permission issues

```bash
# Tạo npm directory cho user
mkdir ~/.npm-global
npm config set prefix '~/.npm-global'

# Add vào PATH
echo 'export PATH=~/.npm-global/bin:$PATH' >> ~/.bashrc
source ~/.bashrc

# Giờ cài global packages không cần sudo
npm install -g yarn
```

### Test installation

```bash
node --version
npm --version
yarn --version

# Test tạo project
mkdir ~/test-node && cd ~/test-node
yarn init -y
yarn add express
node -e "console.log('Node.js works on Jetson!')"
```

### Setup Frontend Project

```bash
cd ~/Source/ocr_datecode/frontend-ts

# Xóa node_modules và cache cũ (nếu có lỗi)
rm -rf node_modules yarn.lock

# Install dependencies
yarn install

# Nếu canvas vẫn lỗi, skip optional dependencies
yarn install --ignore-optional

# Run development server
yarn dev
```

**Note:** Nếu gặp lỗi với `canvas` package, đảm bảo đã cài đủ Cairo dependencies ở bước trước.

---

## 4. Python Environment Setup

### Check Python version

```bash
python3 --version
pip3 --version
```

### Cài pip (nếu chưa có)

```bash
sudo apt update
sudo apt install -y python3-pip
```

### Setup Python environment cho FastAPI project

```bash
cd ~/Source/ocr_datecode/backend

# Cài dependencies
pip3 install -r requirements.txt

# Hoặc cài riêng lẻ
pip3 install fastapi uvicorn pymongo python-dotenv

# Add pip bin to PATH (nếu uvicorn không chạy được)
echo 'export PATH=$HOME/.local/bin:$PATH' >> ~/.bashrc
source ~/.bashrc
```

### Initialize Database

```bash
# Connect MongoDB từ Python code
export MONGODB_URL="mongodb://admin:password@localhost:27017"
python3 init_db.py
```

### Chạy FastAPI server

```bash
# Development mode với auto-reload
uvicorn app.main:app --reload

# Production mode (expose ra ngoài)
uvicorn app.main:app --host 0.0.0.0 --port 8000

# Hoặc chạy qua Python module
python3 -m uvicorn app.main:app --reload
```

### Truy cập API

- API: `http://localhost:8000`
- Swagger Docs: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

---

## 5. Troubleshooting

### Canvas build error (node-gyp)

```bash
# Lỗi: "Package cairo was not found in the pkg-config search path"
# Fix: Cài Cairo dependencies
sudo apt update
sudo apt install -y \
    build-essential \
    libcairo2-dev \
    libpango1.0-dev \
    libjpeg-dev \
    libgif-dev \
    librsvg2-dev \
    pkg-config

# Sau đó install lại
cd ~/Source/ocr_datecode/frontend-ts
rm -rf node_modules yarn.lock
yarn install
```

### Docker permission denied

```bash
sudo usermod -aG docker $USER
newgrp docker
# Hoặc logout/login lại
```

### Docker iptables error

```bash
# Sửa /etc/docker/daemon.json
sudo nano /etc/docker/daemon.json

# Thêm:
{
    "iptables": false
}

# Restart
sudo systemctl restart docker
```

### MongoDB connection từ Python bị lỗi "mongodb:27017"

```bash
# Sửa connection string từ "mongodb:27017" thành "localhost:27017"
# Hoặc add vào /etc/hosts:
echo "127.0.0.1   mongodb" | sudo tee -a /etc/hosts
```

### uvicorn: command not found

```bash
# Add pip bin to PATH
echo 'export PATH=$HOME/.local/bin:$PATH' >> ~/.bashrc
source ~/.bashrc

# Hoặc chạy qua Python
python3 -m uvicorn app.main:app --reload
```

### Node.js version cũ

```bash
# Gỡ version cũ
sudo apt remove nodejs npm

# Cài lại từ NodeSource
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt install -y nodejs
```

### MongoDB Compass connection

```bash
# Ensure MongoDB exposed port
docker compose ps

# Connection string từ máy khác:
mongodb://admin:password@<JETSON_IP>:27017/?authSource=admin
```

---

## 📦 Complete Setup Script

Tạo file `setup.sh` để chạy tất cả một lần:

```bash
#!/bin/bash

echo "🚀 Starting Jetson Orin Setup..."

# 1. Docker
echo "📦 Installing Docker..."
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER
sudo apt install -y docker-compose

# 2. Docker config
echo "⚙️ Configuring Docker..."
sudo tee /etc/docker/daemon.json > /dev/null <<EOF
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
sudo systemctl restart docker

# 3. Cairo dependencies for node-canvas
echo "🎨 Installing Cairo dependencies..."
sudo apt install -y \
    build-essential \
    libcairo2-dev \
    libpango1.0-dev \
    libjpeg-dev \
    libgif-dev \
    librsvg2-dev \
    pkg-config

# 4. Node.js
echo "📗 Installing Node.js..."
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt install -y nodejs
sudo npm install -g yarn

# 5. Python tools
echo "🐍 Setting up Python..."
sudo apt install -y python3-pip
pip3 install --upgrade pip

# 6. PATH updates
echo "🔧 Updating PATH..."
echo 'export PATH=$HOME/.local/bin:$PATH' >> ~/.bashrc
echo 'export PATH=~/.npm-global/bin:$PATH' >> ~/.bashrc

echo "✅ Setup complete! Please run: source ~/.bashrc"
echo "📝 Next steps:"
echo "   1. Setup MongoDB: cd ~/mongodb && docker compose up -d"
echo "   2. Install Python deps: pip3 install -r requirements.txt"
echo "   3. Install Frontend deps: cd frontend-ts && yarn install"
echo "   4. Run FastAPI: uvicorn app.main:app --reload"
echo "   5. Run Frontend: cd frontend-ts && yarn dev"
```

Chạy script:

```bash
chmod +x setup.sh
./setup.sh
source ~/.bashrc
```

---

## 🎯 Quick Reference

### Docker Commands

```bash
docker compose up -d          # Start services
docker compose down           # Stop services
docker compose ps             # List services
docker compose logs -f        # View logs
docker stats                  # Resource usage
```

### MongoDB Commands

```bash
# Access MongoDB shell
docker exec -it mongodb mongosh -u admin -p password

# Backup
docker compose exec mongodb mongodump --out /data/backup

# Restore
docker compose exec mongodb mongorestore /data/backup
```

### Development Workflow

```bash
# Terminal 1: MongoDB
cd ~/mongodb
docker compose up

# Terminal 2: FastAPI Backend
cd ~/Source/ocr_datecode/backend
export MONGODB_URL="mongodb://admin:password@localhost:27017"
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Terminal 3: Frontend
cd ~/Source/ocr_datecode/frontend-ts
yarn dev
```

**Access points:**
- Frontend: `http://localhost:3000` (or port specified in package.json)
- Backend API: `http://localhost:8000`
- API Docs: `http://localhost:8000/docs`
- MongoDB Express: `http://localhost:8081`

---

## 📚 Resources

- Docker: https://docs.docker.com/
- MongoDB: https://docs.mongodb.com/
- FastAPI: https://fastapi.tiangolo.com/
- Node.js: https://nodejs.org/
- NVIDIA Jetson: https://developer.nvidia.com/embedded/jetson

---

**Author:** Nguyen Ngoc Thien  
**Updated:** December 2024  
**Platform:** NVIDIA Jetson Orin 16GB