# 🚀 Quick Start Guide

Hướng dẫn khởi động nhanh OCR Datecode System với Docker.

## ⚡ TL;DR - Chạy ngay

```bash
# 1. Clone repo (nếu chưa có)
git clone <repository-url>
cd ocr_datecode

# 2. Copy environment file
cp .env.docker .env

# 3. Build và start (Jetson)
docker compose -f docker-compose.yml -f docker-compose.jetson.yml up -d --build

# 4. Check logs
docker compose logs -f backend

# 5. Access
# - Frontend: http://localhost:80
# - Backend API: http://localhost:8000
# - API Docs: http://localhost:8000/docs
```

---

## 📋 Prerequisites

✅ Docker Engine 20.10+
✅ Docker Compose V2 (plugin v5.0+)
✅ 4GB RAM minimum
✅ 10GB disk space
✅ **Jetson only**: jetson-stats cài trên host

---

## 🎯 Step-by-Step (Jetson Device)

### 1️⃣ Install jetson-stats trên host

```bash
# QUAN TRỌNG: Cài trên Jetson HOST, không phải trong container
sudo pip3 install -U jetson-stats
sudo systemctl restart jtop.service

# Verify
jtop  # Press 'q' to exit
ls -la /run/jtop.sock  # Should exist
```

### 2️⃣ Prepare environment

```bash
cd /home/demo/Source/ocr_datecode

# Copy environment template
cp .env.docker .env

# Edit if needed (optional)
nano .env
```

### 3️⃣ Build images

```bash
# Build tất cả (backend + frontend)
docker compose -f docker-compose.yml -f docker-compose.jetson.yml build
docker compose -f docker-compose.yml -f docker-compose.jetson.yml build backend

# Hoặc build không dùng cache (clean build)
docker compose -f docker-compose.yml -f docker-compose.jetson.yml build --no-cache
```

### 4️⃣ Start services

```bash
# Start all services
docker compose -f docker-compose.yml -f docker-compose.jetson.yml up -d

# Check status
docker compose ps
```

### 5️⃣ Verify

```bash
# Check backend logs
docker compose logs -f backend

# Should see:
# ✅ jtop initialized successfully
# ✅ Application startup complete

# Check API
curl http://localhost:8000/health

# Check Jetson monitoring
curl http://localhost:8000/api/jetson-monitoring/status
```

### 6️⃣ Access application

- **Frontend**: http://localhost:80
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Mongo Express**: http://localhost:8081 (username: admin, password: admin123)

---

## 🖥️ For Non-Jetson Devices

Nếu chạy trên máy thường (không có Jetson):

```bash
# Use standard docker-compose only
docker compose up -d --build

# App vẫn chạy bình thường
# Monitoring sẽ fallback sang psutil (không có GPU metrics)
```

---

## 🔄 Common Operations

### View logs

```bash
# All services
docker compose logs -f

# Specific service
docker compose logs -f backend
docker compose logs -f frontend
```

### Restart service

```bash
docker compose restart backend
docker compose restart frontend
```

### Stop services

```bash
# Stop (keep data)
docker compose down

# Stop and remove data (⚠️ CAUTION)
docker compose down -v
```

### Update code

```bash
# Pull latest
git pull

# Rebuild and restart
docker compose -f docker-compose.yml -f docker-compose.jetson.yml up -d --build
```

---

## 🐛 Troubleshooting

### ❌ Backend không start

```bash
# Check logs
docker compose logs backend

# Check MongoDB connection
docker compose exec backend python -c "from app.db.mongodb import connect_to_mongo"

# Restart
docker compose restart backend
```

### ❌ Frontend không hiển thị

```bash
# Check nginx logs
docker compose logs frontend

# Rebuild frontend
docker compose build --no-cache frontend
docker compose up -d frontend
```

### ❌ MongoDB connection failed

```bash
# Check MongoDB running
docker compose ps mongodb

# Test connection
docker compose exec mongodb mongosh -u admin -p password --eval "db.adminCommand('ping')"

# Restart MongoDB
docker compose restart mongodb
```

### ❌ jtop not available (Jetson only)

```bash
# Check socket exists
ls -la /run/jtop.sock

# Restart jtop service on host
sudo systemctl restart jtop.service

# Rebuild with jetson compose
docker compose -f docker-compose.yml -f docker-compose.jetson.yml build --no-cache backend
```

### ❌ Port already in use

```bash
# Check what's using port
sudo lsof -i :8000
sudo lsof -i :80

# Change ports in .env
# BACKEND_PORT=8001
# FRONTEND_PORT=8080

# Restart
docker compose down
docker compose up -d
```

---

## 🧹 Cleanup

```bash
# Stop and remove containers
docker compose down

# Remove with volumes (⚠️ deletes data)
docker compose down -v

# Clean Docker system
docker system prune -a
```

---

## 📚 Full Documentation

Chi tiết đầy đủ xem:
- **Docker Commands**: [DOCKER_COMMANDS.md](DOCKER_COMMANDS.md)
- **Jetson Guide**: [JETSON_DOCKER_GUIDE.md](JETSON_DOCKER_GUIDE.md)
- **General Docker**: [README_DOCKER.md](README_DOCKER.md)

---

## 💡 Tips

1. **Alias cho lệnh dài**: Thêm vào `~/.bashrc`
   ```bash
   alias dc-jetson='docker compose -f docker-compose.yml -f docker-compose.jetson.yml'
   alias dc-logs='docker compose logs -f'
   ```

   Dùng:
   ```bash
   dc-jetson up -d
   dc-logs backend
   ```

2. **Monitor resources**:
   ```bash
   docker stats
   ```

3. **Quick rebuild backend**:
   ```bash
   docker compose build backend && docker compose up -d backend
   ```

4. **Export logs**:
   ```bash
   docker compose logs backend > backend.log
   ```

---

**🎉 That's it! Your OCR Datecode System should be running now!**

Questions? Check [DOCKER_COMMANDS.md](DOCKER_COMMANDS.md) for detailed reference.
