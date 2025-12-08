# Build EXE từ Python Script

## Cách 1: Dùng script tự động

### Windows:
```cmd
build_exe.bat
```

### Mac/Linux:
```bash
chmod +x build_exe.sh
./build_exe.sh
```

## Cách 2: Chạy PyInstaller trực tiếp

### Build thành 1 file EXE duy nhất:
```bash
pyinstaller --onefile --windowed --name "PLC_Camera_Trigger" plc_camera_gui_tk.py
```

### Các options quan trọng:
- `--onefile`: Build thành 1 file duy nhất (không có folder)
- `--windowed`: Không hiện console window (chỉ hiện GUI)
- `--name "..."`: Tên file exe output
- `--icon=icon.ico`: Thêm icon cho exe (optional)

### Build với thêm file/folder:
```bash
pyinstaller --onefile --windowed --name "PLC_Camera_Trigger" --add-data "images:images" plc_camera_gui_tk.py
```

## Output

File exe sẽ nằm trong folder:
```
dist/
  └── PLC_Camera_Trigger.exe    (Windows)
  └── PLC_Camera_Trigger         (Mac/Linux)
```

## Lưu ý

1. **Dependencies**: PyInstaller sẽ tự động đóng gói:
   - pypylon
   - pymodbus
   - opencv-python
   - Pillow
   - tkinter (built-in)

2. **File size**: File exe sẽ lớn (~100-200MB) vì chứa tất cả dependencies

3. **Test trước khi deploy**:
   - Chạy thử file exe trên máy clean (không có Python)
   - Test kết nối PLC và camera

4. **Antivirus**: Một số antivirus có thể cảnh báo file exe mới build
   - Thêm exception nếu cần
   - Hoặc code sign file exe

## Giảm file size (optional)

Nếu muốn file nhỏ hơn:
```bash
pyinstaller --onefile --windowed --name "PLC_Camera_Trigger" --exclude-module matplotlib --exclude-module scipy plc_camera_gui_tk.py
```

## Build spec file (advanced)

Tạo file .spec để customize:
```bash
pyi-makespec --onefile --windowed --name "PLC_Camera_Trigger" plc_camera_gui_tk.py
```

Sau đó edit file `PLC_Camera_Trigger.spec` và build:
```bash
pyinstaller PLC_Camera_Trigger.spec
```
