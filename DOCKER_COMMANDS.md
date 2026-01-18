# 🐳 Docker Commands Reference

Tài liệu đầy đủ các lệnh Docker để quản lý OCR Datecode System.

## 📋 Table of Contents
- [Build Images](#build-images)
- [Start/Stop Services](#startstop-services)
- [View Logs](#view-logs)
- [Container Management](#container-management)
- [Database Operations](#database-operations)
- [Health Checks](#health-checks)
- [Troubleshooting](#troubleshooting)
- [Cleanup](#cleanup)
- [Update & Rebuild](#update--rebuild)

---

## 🔨 Build Images

### Build tất cả services

```bash
# Standard deployment (máy thường)
docker compose -f docker-compose.yml build

# Jetson deployment
docker compose -f docker-compose.yml -f docker-compose.jetson.yml build

# Build without cache (clean build)
docker compose -f docker-compose.yml -f docker-compose.jetson.yml build --no-cache

# Build song song (parallel - nhanh hơn)
docker compose -f docker-compose.yml -f docker-compose.jetson.yml build --parallel
```

### Build từng service riêng

```bash
# Build backend only
docker compose -f docker-compose.yml -f docker-compose.jetson.yml build backend

# Build frontend only
docker compose -f docker-compose.yml -f docker-compose.jetson.yml build frontend

# Build MongoDB (thường không cần vì dùng pre-built image)
docker compose -f docker-compose.yml -f docker-compose.jetson.yml build mongodb
```

### Build với specific tag

```bash
# Tag image với version
docker compose -f docker-compose.yml -f docker-compose.jetson.yml build --tag ocr_datecode:v1.0.0
```

---

## 🚀 Start/Stop Services

### Start services

```bash
# Start tất cả services (detached mode)
docker compose -f docker-compose.yml -f docker-compose.jetson.yml up -d

# Start và build lại nếu cần
docker compose -f docker-compose.yml -f docker-compose.jetson.yml up -d --build

# Start trong foreground (xem logs trực tiếp)
docker compose -f docker-compose.yml -f docker-compose.jetson.yml up

# Start service cụ thể
docker compose -f docker-compose.yml -f docker-compose.jetson.yml up -d backend
docker compose -f docker-compose.yml -f docker-compose.jetson.yml up -d frontend

# Start với Mongo Express (admin UI)
docker compose -f docker-compose.yml --profile admin up -d
```

### Stop services

```bash
# Stop tất cả services (giữ containers)
docker compose -f docker-compose.yml -f docker-compose.jetson.yml stop

# Stop service cụ thể
docker compose -f docker-compose.yml -f docker-compose.jetson.yml stop backend
docker compose -f docker-compose.yml -f docker-compose.jetson.yml stop frontend

# Stop và xóa containers (GIỮ volumes/data)
docker compose -f docker-compose.yml -f docker-compose.jetson.yml down

# Stop, xóa containers VÀ volumes (⚠️ MẤT DỮ LIỆU)
docker compose -f docker-compose.yml -f docker-compose.jetson.yml down -v
```

### Restart services

```bash
# Restart tất cả
docker compose -f docker-compose.yml -f docker-compose.jetson.yml restart

# Restart service cụ thể
docker compose -f docker-compose.yml -f docker-compose.jetson.yml restart backend
docker compose -f docker-compose.yml -f docker-compose.jetson.yml restart frontend
docker compose -f docker-compose.yml -f docker-compose.jetson.yml restart mongodb
```

### Pause/Unpause

```bash
# Pause tất cả containers (freeze processes)
docker compose -f docker-compose.yml -f docker-compose.jetson.yml pause

# Unpause
docker compose -f docker-compose.yml -f docker-compose.jetson.yml unpause
```

---

## 📋 View Logs

### View logs

```bash
# Tất cả services (follow mode)
docker compose -f docker-compose.yml -f docker-compose.jetson.yml logs -f

# Service cụ thể
docker compose logs -f backend
docker compose logs -f frontend
docker compose logs -f mongodb
docker compose logs -f redis

# Logs với timestamps
docker compose logs -f --timestamps backend

# Giới hạn số dòng (100 dòng cuối)
docker compose logs --tail=100 backend

# Logs từ thời điểm cụ thể
docker compose logs --since="2026-01-19T10:00:00" backend
docker compose logs --since="30m" backend  # 30 phút trước
docker compose logs --since="2h" backend   # 2 giờ trước

# Export logs ra file
docker compose logs backend > backend.log
docker compose logs frontend > frontend.log
```

### View logs từ container trực tiếp

```bash
# Using container name
docker logs -f ocr-backend
docker logs -f ocr-frontend
docker logs -f ocr-mongodb

# Theo dõi real-time
docker logs -f --tail=50 ocr-backend
```

---

## 🔧 Container Management

### Kiểm tra trạng thái

```bash
# List tất cả containers
docker compose ps

# List tất cả containers (bao gồm stopped)
docker compose ps -a

# Kiểm tra status
docker ps | grep ocr

# Chi tiết container
docker inspect ocr-backend
docker inspect ocr-frontend
```

### Shell access

```bash
# Backend shell
docker compose exec backend bash

# Frontend shell (nginx container)
docker compose exec frontend sh

# MongoDB shell
docker compose exec mongodb mongosh -u admin -p password

# Redis CLI
docker compose exec redis redis-cli

# Root access (nếu cần)
docker compose exec -u root backend bash
```

### Execute commands

```bash
# Run Python trong backend
docker compose exec backend python --version
docker compose exec backend python -c "import cv2; print(cv2.__version__)"

# Check packages
docker compose exec backend pip list
docker compose exec backend pip show opencv-python

# Run Django/FastAPI commands
docker compose exec backend python -m pytest
docker compose exec backend python init_db.py

# File operations
docker compose exec backend ls -la /app
docker compose exec backend cat /app/.env
```

### Copy files

```bash
# Copy FROM container TO host
docker cp ocr-backend:/app/logs ./backend_logs
docker cp ocr-backend:/app/uploads ./backend_uploads

# Copy FROM host TO container
docker cp ./config.yaml ocr-backend:/app/config.yaml
docker cp ./new_env ocr-backend:/app/.env
```

### Resource monitoring

```bash
# Real-time resource usage
docker stats

# Specific containers
docker stats ocr-backend ocr-frontend ocr-mongodb

# One-time snapshot
docker stats --no-stream
```

---

## 🗄️ Database Operations

### MongoDB

```bash
# Access MongoDB shell
docker compose exec mongodb mongosh -u admin -p password

# Backup database
docker compose exec mongodb mongodump \
  --username=admin \
  --password=password \
  --authenticationDatabase=admin \
  --db=ocr_datecode_db \
  --out=/data/backup

# Copy backup to host
docker cp ocr-mongodb:/data/backup ./mongodb_backup_$(date +%Y%m%d)

# Restore database
docker cp ./mongodb_backup ocr-mongodb:/data/restore
docker compose exec mongodb mongorestore \
  --username=admin \
  --password=password \
  --authenticationDatabase=admin \
  --db=ocr_datecode_db \
  /data/restore/ocr_datecode_db

# Drop database
docker compose exec mongodb mongosh -u admin -p password --eval "use ocr_datecode_db; db.dropDatabase()"

# List databases
docker compose exec mongodb mongosh -u admin -p password --eval "show dbs"

# List collections
docker compose exec mongodb mongosh -u admin -p password --eval "use ocr_datecode_db; show collections"

# Count documents
docker compose exec mongodb mongosh -u admin -p password --eval "use ocr_datecode_db; db.users.countDocuments()"
```

### Redis

```bash
# Access Redis CLI
docker compose exec redis redis-cli

# Inside Redis CLI:
# - PING
# - KEYS *
# - GET key_name
# - FLUSHALL (⚠️ clear all data)

# From shell
docker compose exec redis redis-cli PING
docker compose exec redis redis-cli KEYS '*'
docker compose exec redis redis-cli INFO
docker compose exec redis redis-cli FLUSHDB  # Clear current DB
```

---

## 🏥 Health Checks

### Check service health

```bash
# Backend API health
curl http://localhost:8000/health
curl http://localhost:8000/docs  # API documentation

# Frontend
curl http://localhost:80

# MongoDB
docker compose exec mongodb mongosh --eval "db.adminCommand('ping')"

# Redis
docker compose exec redis redis-cli PING

# Jetson monitoring (if on Jetson)
curl http://localhost:8000/api/jetson-monitoring/status
curl http://localhost:8000/api/jetson-monitoring/metrics
```

### Docker health status

```bash
# Check health status from Docker
docker inspect --format='{{.State.Health.Status}}' ocr-backend
docker inspect --format='{{.State.Health.Status}}' ocr-mongodb

# View health check logs
docker inspect --format='{{range .State.Health.Log}}{{.Output}}{{end}}' ocr-backend
```

---

## 🐛 Troubleshooting

### Backend issues

```bash
# View backend logs
docker compose logs -f --tail=100 backend

# Check if backend is running
docker compose ps backend

# Check environment variables
docker compose exec backend env | grep -E "MONGODB|REDIS|DATABASE"

# Test MongoDB connection
docker compose exec backend python -c "from app.db.mongodb import connect_to_mongo; import asyncio; asyncio.run(connect_to_mongo())"

# Check Python packages
docker compose exec backend pip list | grep -E "fastapi|opencv|numpy"

# Restart backend
docker compose restart backend

# Rebuild backend
docker compose build --no-cache backend
docker compose up -d backend
```

### Frontend issues

```bash
# View frontend logs
docker compose logs -f frontend

# Check nginx config
docker compose exec frontend cat /etc/nginx/conf.d/default.conf

# Check built files
docker compose exec frontend ls -la /usr/share/nginx/html

# Rebuild frontend
docker compose build --no-cache frontend
docker compose up -d frontend
```

### MongoDB issues

```bash
# Check MongoDB logs
docker compose logs -f mongodb

# Test connection
docker compose exec mongodb mongosh -u admin -p password --eval "db.adminCommand('ping')"

# Check disk space
docker compose exec mongodb df -h

# Check MongoDB status
docker compose exec mongodb mongosh -u admin -p password --eval "db.serverStatus()"

# Restart MongoDB
docker compose restart mongodb
```

### Network issues

```bash
# List networks
docker network ls

# Inspect network
docker network inspect ocr_datecode_ocr-network

# Check container IPs
docker compose ps -q | xargs docker inspect -f '{{.Name}} - {{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}'

# Test connectivity between containers
docker compose exec backend ping mongodb
docker compose exec backend ping redis
docker compose exec backend curl http://frontend:80
```

### Port conflicts

```bash
# Check which process uses port
sudo lsof -i :8000
sudo lsof -i :80
sudo lsof -i :27017

# Change ports in .env
# BACKEND_PORT=8001
# FRONTEND_PORT=8080
# MONGODB_PORT=27018

# Restart with new ports
docker compose down
docker compose up -d
```

### Disk space issues

```bash
# Check Docker disk usage
docker system df

# Check detailed usage
docker system df -v

# Check volumes
docker volume ls
docker volume inspect ocr_datecode_mongodb_data
```

---

## 🧹 Cleanup

### Remove stopped containers

```bash
# Remove all stopped containers
docker compose down

# Remove with volumes (⚠️ deletes data)
docker compose down -v

# Remove with images
docker compose down --rmi all
```

### Clean Docker system

```bash
# Remove unused data
docker system prune

# Remove everything unused (⚠️ aggressive)
docker system prune -a

# Remove unused volumes
docker volume prune

# Remove unused networks
docker network prune

# Remove unused images
docker image prune -a

# Clean build cache
docker builder prune
docker builder prune -a  # Remove all cache
```

### Remove specific containers/images

```bash
# Stop and remove containers
docker stop ocr-backend ocr-frontend ocr-mongodb ocr-redis
docker rm ocr-backend ocr-frontend ocr-mongodb ocr-redis

# Remove images
docker rmi ocr_datecode-backend
docker rmi ocr_datecode-frontend

# Force remove
docker rmi -f ocr_datecode-backend
```

### Clean logs

```bash
# Truncate logs (giữ container chạy)
truncate -s 0 $(docker inspect --format='{{.LogPath}}' ocr-backend)
truncate -s 0 $(docker inspect --format='{{.LogPath}}' ocr-frontend)

# Or remove logs by recreating container
docker compose up -d --force-recreate backend
```

---

## 🔄 Update & Rebuild

### Update code và rebuild

```bash
# Pull latest code
git pull origin main

# Rebuild và restart (quick)
docker compose -f docker-compose.yml -f docker-compose.jetson.yml up -d --build

# Full rebuild (no cache)
docker compose -f docker-compose.yml -f docker-compose.jetson.yml build --no-cache
docker compose -f docker-compose.yml -f docker-compose.jetson.yml up -d
```

### Update environment variables

```bash
# Edit .env file
nano .env

# Recreate containers with new env
docker compose -f docker-compose.yml -f docker-compose.jetson.yml up -d --force-recreate

# Or restart
docker compose -f docker-compose.yml -f docker-compose.jetson.yml restart
```

### Update single service

```bash
# Update backend only
git pull
docker compose build backend
docker compose up -d backend

# Update frontend only
git pull
docker compose build --no-cache frontend
docker compose up -d frontend
```

### Rolling update (zero downtime)

```bash
# Scale to 2 instances
docker compose up -d --scale backend=2

# Update image
docker compose build backend

# Restart with new image
docker compose up -d --no-deps backend

# Scale back
docker compose up -d --scale backend=1
```

---

## 📊 Advanced Commands

### Export/Import images

```bash
# Export image to tar
docker save -o ocr_backend.tar ocr_datecode-backend
docker save -o ocr_frontend.tar ocr_datecode-frontend

# Import image from tar
docker load -i ocr_backend.tar
docker load -i ocr_frontend.tar

# Transfer to another machine
scp ocr_backend.tar user@remote:/path/
ssh user@remote "docker load -i /path/ocr_backend.tar"
```

### Create backup script

```bash
#!/bin/bash
# backup.sh

BACKUP_DIR=/backups/ocr_datecode
DATE=$(date +%Y%m%d_%H%M%S)

# Backup MongoDB
docker compose exec -T mongodb mongodump \
  --username=admin \
  --password=$MONGODB_PASSWORD \
  --authenticationDatabase=admin \
  --archive > $BACKUP_DIR/mongodb_$DATE.archive

# Backup volumes
docker run --rm \
  -v ocr_datecode_mongodb_data:/data \
  -v $BACKUP_DIR:/backup \
  ubuntu tar czf /backup/mongodb_volume_$DATE.tar.gz /data

# Backup uploads
tar czf $BACKUP_DIR/uploads_$DATE.tar.gz backend/uploads

# Keep only last 7 days
find $BACKUP_DIR -name "*.archive" -mtime +7 -delete
find $BACKUP_DIR -name "*.tar.gz" -mtime +7 -delete
```

### Watch logs in real-time (multiple services)

```bash
# Install multitail
sudo apt install multitail

# Watch multiple logs
multitail \
  -l "docker compose logs -f backend" \
  -l "docker compose logs -f frontend" \
  -l "docker compose logs -f mongodb"
```

---

## 🎯 Quick Reference

### Daily operations

```bash
# Start
docker compose -f docker-compose.yml -f docker-compose.jetson.yml up -d

# View logs
docker compose logs -f backend

# Restart service
docker compose restart backend

# Stop
docker compose down
```

### Maintenance

```bash
# Update
git pull && docker compose up -d --build

# Backup
docker compose exec mongodb mongodump --archive > backup.archive

# Clean
docker system prune -a
```

### Debugging

```bash
# Logs
docker compose logs -f --tail=100 backend

# Shell
docker compose exec backend bash

# Stats
docker stats ocr-backend
```

---

## 📚 Additional Resources

- **Docker Compose Reference**: https://docs.docker.com/compose/reference/
- **Docker CLI Reference**: https://docs.docker.com/engine/reference/commandline/cli/
- **Best Practices**: https://docs.docker.com/develop/dev-best-practices/

---

## ⚠️ Important Notes

1. **Always use Docker Compose V2** syntax: `docker compose` (không có dấu gạch ngang)
2. **Backup trước khi update**: Luôn backup database trước khi update production
3. **Test trong dev trước**: Test changes trong development environment trước
4. **Monitor resources**: Theo dõi disk space và memory usage thường xuyên
5. **Keep logs clean**: Truncate hoặc rotate logs định kỳ
6. **Use .env for secrets**: Không commit sensitive data vào git
7. **Network security**: Chỉ expose ports cần thiết ra ngoài

---

**Version**: 1.0.0
**Last Updated**: 2026-01-19
**Platform**: Ubuntu 22.04 (Docker) + NVIDIA Jetson (optional)
