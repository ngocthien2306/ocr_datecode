# OCR Datecode - Hướng dẫn Cài đặt

## Cài đặt Nhanh (1 lệnh)

```bash
chmod +x install.sh
./install.sh
```

Script sẽ tự động cài đặt tất cả và hướng dẫn bạn qua từng bước.

---

## Yêu cầu Hệ thống

| Thành phần | Yêu cầu |
|------------|---------|
| Platform | NVIDIA Jetson Orin |
| OS | Ubuntu 20.04+ / JetPack 6.x |
| RAM | 8GB+ |
| Storage | 32GB+ |

---

## Cấu trúc Thư mục

```
scripts/
├── install.sh              ← CHẠY FILE NÀY ĐỂ CÀI ĐẶT
├── README.md               ← Bạn đang đọc file này
│
├── setup_system.sh         # Cài đặt system packages
├── setup_mongodb.sh        # Cài đặt MongoDB (Docker)
├── setup_backend.sh        # Cài đặt Backend Python
├── setup_ai_services.sh    # Cài đặt AI/CUDA
├── setup_frontend.sh       # Cài đặt Frontend Node.js
├── setup_ngrok.sh          # Cài đặt Ngrok (tùy chọn)
├── setup_dio_sudoers.sh    # Setup DIO (hardware trigger)
├── install_services.sh     # Cài đặt systemd services
│
└── services/               # Systemd service files
    ├── ocr-backend.service
    ├── ocr-frontend.service
    ├── ocr-ai-services.service
    └── ocr-ngrok.service
```

---

## Sau khi Cài đặt

### Khởi động Services

```bash
# Khởi động từng service
sudo systemctl start ocr-backend
sudo systemctl start ocr-frontend
sudo systemctl start ocr-ai-services

# Hoặc khởi động tất cả
sudo systemctl start ocr-backend ocr-frontend ocr-ai-services
```

### Truy cập Ứng dụng

| Service | URL |
|---------|-----|
| Frontend | http://localhost:5000 |
| Backend API | http://localhost:8000 |
| API Docs | http://localhost:8000/docs |
| Mongo Express | http://localhost:8081 |

### Xem Logs

```bash
# Xem log real-time
journalctl -u ocr-backend -f
journalctl -u ocr-frontend -f
journalctl -u ocr-ai-services -f

# Xem 100 dòng cuối
journalctl -u ocr-backend -n 100
```

### Dừng Services

```bash
sudo systemctl stop ocr-backend ocr-frontend ocr-ai-services
```

---

## Cấu hình

### Backend (.env)

File: `~/Source/ocr_datecode/backend/.env`

**Quan trọng**: Cập nhật các giá trị sau:
- `SECRET_KEY`: Đổi thành key bảo mật riêng
- `OPENAI_API_KEY`: API key của OpenAI (nếu dùng AI features)

```bash
nano ~/Source/ocr_datecode/backend/.env
```

### MongoDB

- URL: `mongodb://admin:password@localhost:27017`
- Data: `~/mongodb/mongodb_data`

Quản lý MongoDB:
```bash
# Start/Stop
cd ~/mongodb
docker compose up -d
docker compose down

# Shell
docker exec -it mongodb mongosh -u admin -p password
```

---

## Export/Import Data

### Export (Máy cũ)
```bash
cd ~/Source/ocr_datecode/tests
python3 export_data.py --output-dir ./exports
```

### Import (Máy mới)
```bash
python3 migration_data.py --input-dir ./exports
```

---

## Troubleshooting

### Service không start
```bash
# Xem lỗi
journalctl -u ocr-backend -n 50

# Restart
sudo systemctl restart ocr-backend
```

### MongoDB không chạy
```bash
cd ~/mongodb
docker compose logs mongodb
docker compose down && docker compose up -d
```

### Port đang bị sử dụng
```bash
sudo lsof -i :8000
sudo kill -9 <PID>
```

---

## Liên hệ

- Repository: https://github.com/ngocthien2306/ocr_datecode
- Branch: release_v1
