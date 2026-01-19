# OCR Datecode - Hướng dẫn Cài đặt

Hướng dẫn cài đặt hệ thống OCR Datecode trên Jetson Orin.

## Mục lục

- [Yêu cầu hệ thống](#yêu-cầu-hệ-thống)
- [Cài đặt nhanh](#cài-đặt-nhanh)
- [Cài đặt từng bước](#cài-đặt-từng-bước)
- [Quản lý Services](#quản-lý-services)
- [Export/Import Data](#exportimport-data)
- [Cấu hình](#cấu-hình)
- [Troubleshooting](#troubleshooting)

---

## Yêu cầu hệ thống

| Thành phần | Yêu cầu |
|------------|---------|
| Platform | NVIDIA Jetson Orin |
| OS | Ubuntu 20.04+ / JetPack 6.x |
| RAM | 8GB+ |
| Storage | 32GB+ |
| CUDA | 12.x |
| Docker | 20.10+ |

## Cài đặt nhanh

Chạy master script để cài đặt tất cả tự động:

```bash
# Clone repository (nếu chưa có)
mkdir -p ~/Source && cd ~/Source
git clone https://github.com/ngocthien2306/ocr_datecode.git
cd ocr_datecode
git checkout release_v1

# Chạy master setup script
chmod +x scripts/*.sh
./scripts/setup_all.sh
```

Script sẽ hướng dẫn bạn qua từng bước.

---

## Cài đặt từng bước

### Bước 1: Cài đặt System Packages

```bash
cd ~/Source/ocr_datecode/scripts
./setup_system.sh
```

Script này sẽ:
- Cập nhật apt packages
- Cài đặt python3-dev, pip, git, curl...
- Tạo thư mục ~/Source
- Clone repository (nếu chưa có)

### Bước 2: Setup DIO (Hardware Trigger) - Tùy chọn

Chỉ cần nếu sử dụng hardware trigger cho camera:

```bash
./setup_dio_sudoers.sh
```

### Bước 3: Cài đặt MongoDB

```bash
cd ~/Source/ocr_datecode
./setup_mongodb.sh
```

MongoDB sẽ được cài đặt qua Docker với:
- **MongoDB**: `mongodb://admin:password@localhost:27017`
- **Mongo Express UI**: `http://localhost:8081` (admin/admin123)
- **Data directory**: `~/mongodb/mongodb_data`

Kiểm tra MongoDB:
```bash
docker exec -it mongodb mongosh -u admin -p password
```

### Bước 4: Cài đặt Backend

```bash
cd ~/Source/ocr_datecode/scripts
./setup_backend.sh
```

Script này sẽ:
- Cài đặt Python packages từ requirements.txt
- Cài đặt LangChain packages
- Cài đặt jetson-stats
- Tạo file `.env` mẫu

**Quan trọng**: Cập nhật file `backend/.env`:
```bash
nano ~/Source/ocr_datecode/backend/.env
```

Thay đổi các giá trị:
- `SECRET_KEY`: Đổi thành key bảo mật riêng
- `OPENAI_API_KEY`: Thêm API key của bạn (nếu sử dụng AI features)

### Bước 5: Cài đặt AI Services

```bash
./setup_ai_services.sh
```

Script này sẽ:
- Cấu hình CUDA environment
- Cài đặt ONNX Runtime GPU cho Jetson
- Cài đặt PyCUDA
- Cài đặt pypylon (Basler camera SDK)

### Bước 6: Cài đặt Frontend

```bash
./setup_frontend.sh
```

Script này sẽ:
- Cài đặt Node.js 20.x
- Cài đặt Yarn
- Chạy `yarn install`

### Bước 7: Cài đặt Ngrok (Tùy chọn)

Chỉ cần nếu muốn truy cập từ bên ngoài:

```bash
./setup_ngrok.sh
```

Các tunnel được cấu hình:
| Service | Local | Public URL |
|---------|-------|------------|
| Frontend | :5000 | suntech-vision.ngrok.app |
| API | :8000 | suntech-vision-api.ngrok.app |
| Mongo Express | :8081 | suntech-vision-db.ngrok.app |

### Bước 8: Cài đặt Systemd Services

```bash
./install_services.sh
```

---

## Quản lý Services

### Danh sách Services

| Service | Mô tả | Port |
|---------|-------|------|
| `ocr-backend` | FastAPI Backend | 8000 |
| `ocr-frontend` | React Frontend | 5000 |
| `ocr-ai-services` | Camera Management | - |
| `ocr-ngrok` | Ngrok Tunnels | - |

### Lệnh quản lý

```bash
# Khởi động service
sudo systemctl start ocr-backend
sudo systemctl start ocr-frontend
sudo systemctl start ocr-ai-services
sudo systemctl start ocr-ngrok

# Khởi động tất cả (trừ ngrok)
sudo systemctl start ocr-backend ocr-frontend ocr-ai-services

# Dừng service
sudo systemctl stop ocr-backend

# Dừng tất cả
sudo systemctl stop ocr-backend ocr-frontend ocr-ai-services ocr-ngrok

# Kiểm tra trạng thái
sudo systemctl status ocr-backend
sudo systemctl status ocr-frontend
sudo systemctl status ocr-ai-services

# Xem logs (real-time)
journalctl -u ocr-backend -f
journalctl -u ocr-frontend -f
journalctl -u ocr-ai-services -f

# Xem logs (last 100 lines)
journalctl -u ocr-backend -n 100

# Restart service
sudo systemctl restart ocr-backend

# Enable/Disable auto-start on boot
sudo systemctl enable ocr-backend
sudo systemctl disable ocr-backend
```

### Chạy thủ công (Development)

```bash
# Terminal 1: Backend
cd ~/Source/ocr_datecode/backend
python3 -m uvicorn app.main:app --reload --port 8000 --host 0.0.0.0

# Terminal 2: Frontend
cd ~/Source/ocr_datecode/frontend-ts
yarn dev --port 5000 --host 0.0.0.0

# Terminal 3: AI Services
cd ~/Source/ocr_datecode/ai_services
python3 camera_management_service.py

# Terminal 4: Ngrok (optional)
ngrok start --config ~/.config/ngrok/ngrok-ocr.yml --all
```

---

## Export/Import Data

### Export Data (Máy cũ)

```bash
cd ~/Source/ocr_datecode/tests
python3 export_data.py --output-dir ./exports
```

Output:
```
exports/
├── recipes_YYYYMMDD_HHMMSS.json
├── users_YYYYMMDD_HHMMSS.json
├── cameras_YYYYMMDD_HHMMSS.json
└── manifest.json
```

### Import Data (Máy mới)

```bash
# Copy thư mục exports sang máy mới
scp -r exports/ user@new-machine:~/

# Trên máy mới
cd ~/Source/ocr_datecode/tests
python3 migration_data.py --input-dir ~/exports

# Hoặc xóa data cũ trước khi import
python3 migration_data.py --input-dir ~/exports --drop-existing
```

### Options

**export_data.py**:
```bash
python3 export_data.py \
    --output-dir ./exports \
    --mongodb-url "mongodb://admin:password@localhost:27017/" \
    --database "ocr_datecode_db"
```

**migration_data.py**:
```bash
python3 migration_data.py \
    --input-dir ./exports \
    --mongodb-url "mongodb://admin:newpass@localhost:27017/" \
    --database "ocr_datecode_db" \
    --drop-existing \
    --no-manifest
```

---

## Cấu hình

### Backend (.env)

File: `~/Source/ocr_datecode/backend/.env`

```ini
# MongoDB
MONGODB_URL=mongodb://admin:password@localhost:27017/
DATABASE_NAME=ocr_datecode_db

# Redis
REDIS_URL=redis://localhost:6379
REDIS_CACHE_TTL_RECENT=60
REDIS_CACHE_TTL_HISTORICAL=300

# JWT
SECRET_KEY=your-secret-key-here
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=43200

# CORS
CORS_ORIGINS=["http://localhost:3000","http://localhost:5173"]

# App
APP_NAME="OCR Datecode API"
DEBUG=True
API_BASE_URL=http://localhost:8000
TIMEZONE=Asia/Ho_Chi_Minh

# OpenAI (optional)
OPENAI_API_KEY=sk-xxxx
```

### Ngrok Config

File: `~/.config/ngrok/ngrok-ocr.yml`

```yaml
version: "2"
authtoken: YOUR_AUTHTOKEN
tunnels:
  frontend:
    proto: http
    addr: 5000
    hostname: your-domain.ngrok.app
  api:
    proto: http
    addr: 8000
    hostname: your-api.ngrok.app
```

---

## Troubleshooting

### MongoDB không khởi động

```bash
# Kiểm tra Docker
docker ps -a

# Xem logs MongoDB
cd ~/mongodb
docker compose logs mongodb

# Restart MongoDB
docker compose down && docker compose up -d
```

### Service không start

```bash
# Kiểm tra logs
journalctl -u ocr-backend -n 50

# Kiểm tra syntax service file
sudo systemd-analyze verify /etc/systemd/system/ocr-backend.service

# Reload daemon sau khi sửa service file
sudo systemctl daemon-reload
```

### CUDA không nhận

```bash
# Kiểm tra CUDA
nvcc --version

# Kiểm tra environment
echo $CUDA_HOME
echo $LD_LIBRARY_PATH

# Load lại environment
source ~/.bashrc
```

### Frontend build lỗi

```bash
# Clear cache và reinstall
cd ~/Source/ocr_datecode/frontend-ts
rm -rf node_modules yarn.lock
yarn install
```

### Permission denied khi chạy script

```bash
chmod +x ~/Source/ocr_datecode/scripts/*.sh
```

### Port đang bị sử dụng

```bash
# Tìm process đang dùng port
sudo lsof -i :8000
sudo lsof -i :5000

# Kill process
sudo kill -9 <PID>
```

---

## Cấu trúc thư mục

```
~/Source/ocr_datecode/
├── backend/                 # FastAPI Backend
│   ├── app/
│   ├── .env                # Configuration
│   └── requirements.txt
├── frontend-ts/            # React Frontend
│   ├── src/
│   └── package.json
├── ai_services/            # Camera & AI Services
│   └── camera_management_service.py
├── tests/
│   ├── export_data.py      # Export MongoDB data
│   └── migration_data.py   # Import MongoDB data
├── scripts/
│   ├── setup_system.sh
│   ├── setup_backend.sh
│   ├── setup_ai_services.sh
│   ├── setup_frontend.sh
│   ├── setup_ngrok.sh
│   ├── setup_dio_sudoers.sh
│   ├── install_services.sh
│   ├── setup_all.sh        # Master script
│   └── services/           # Systemd service files
├── setup_mongodb.sh
└── docs/
    └── SETUP_GUIDE.md      # This document
```

---

## Liên hệ hỗ trợ

- Repository: https://github.com/ngocthien2306/ocr_datecode
- Branch: release_v1

---

*Cập nhật lần cuối: Tháng 1, 2026*
