# Setup SSH Remote Jetson qua Ngrok

## 1. Cài đặt ngrok trên Jetson

### Cách 1: Qua APT (Khuyến nghị)
```bash
curl -sSL https://ngrok-agent.s3.amazonaws.com/ngrok.asc \
  | sudo tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null \
  && echo "deb https://ngrok-agent.s3.amazonaws.com bookworm main" \
  | sudo tee /etc/apt/sources.list.d/ngrok.list \
  && sudo apt update \
  && sudo apt install ngrok

# Authenticate
ngrok config add-authtoken YOUR_TOKEN
```

### Cách 2: Download binary (Alternative)
```bash
wget https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-arm64.tgz
tar -xvzf ngrok-v3-stable-linux-arm64.tgz
sudo mv ngrok /usr/local/bin/
ngrok config add-authtoken YOUR_TOKEN
```

## 2. Tạo Reserved TCP Address

- Vào https://dashboard.ngrok.com/cloud-edge/tcp-addresses
- Click "New TCP Address"
- Chọn region: **ap** (Asia Pacific)
- Lưu address, ví dụ: `1.tcp.ap.ngrok.io:23769`

## 3. Cài SSH Server trên Jetson
```bash
sudo apt update
sudo apt install openssh-server
sudo systemctl enable ssh
sudo systemctl start ssh
```

## 4. Tạo SSH Key trên Mac
```bash
# Tạo key riêng cho Jetson
ssh-keygen -t ed25519 -C "thien@mac" -f ~/.ssh/id_ed25519_jetson

# Hoặc dùng key có sẵn (bỏ qua bước này)
```

## 5. Test SSH qua ngrok
```bash
# Trên Jetson: start ngrok
ngrok tcp 22 --region=ap --remote-addr=1.tcp.ap.ngrok.io:23769

# Trên Mac: test connect
ssh -p 23769 demo@1.tcp.ap.ngrok.io
```

## 6. Copy SSH Key
```bash
# Trên Mac
ssh-copy-id -i ~/.ssh/id_ed25519_jetson.pub -p 23769 demo@1.tcp.ap.ngrok.io

# Hoặc dùng key mặc định
ssh-copy-id -p 23769 demo@1.tcp.ap.ngrok.io
```

## 7. Setup SSH Config trên Mac
```bash
nano ~/.ssh/config
```
```
Host jetson
    HostName 1.tcp.ap.ngrok.io
    Port 23769
    User demo
    IdentityFile ~/.ssh/id_ed25519_jetson
```

Giờ chỉ cần: `ssh jetson`

## 8. Tạo ngrok systemd service trên Jetson
```bash
sudo nano /etc/systemd/system/ngrok-ssh.service
```
```ini
[Unit]
Description=Ngrok SSH Tunnel
After=network-online.target
Wants=network-online.target

[Service]
ExecStart=/usr/bin/ngrok tcp 22 --region=ap --remote-addr=1.tcp.ap.ngrok.io:23769
Restart=always
RestartSec=10
User=demo
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```
```bash
sudo systemctl daemon-reload
sudo systemctl enable ngrok-ssh
sudo systemctl start ngrok-ssh
sudo systemctl status ngrok-ssh
```

## Commands hữu ích
```bash
# Check status
sudo systemctl status ngrok-ssh

# Restart
sudo systemctl restart ngrok-ssh

# View logs
sudo journalctl -u ngrok-ssh -f
```

## Done!

Giờ Jetson tự động có SSH tunnel khi khởi động. Connect từ bất kỳ đâu:
```bash
ssh jetson
```