# Basler Camera Test Application - Build Instructions

## Quick Build

### Windows:

```bash
# Double click build.bat
# OR run in command prompt:
build.bat
```

### Linux/Mac:

```bash
chmod +x build.sh
./build.sh
```

## Manual Build Steps

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Build Executable

**Option A: Simple one-file build**

```bash
pyinstaller --onefile --windowed --name="BaslerCameraTest" basler_camera_test.py
```

**Option B: Using spec file (recommended)**

```bash
pyinstaller basler_camera_test.spec
```

### 3. Find Your Executable

- Windows: `dist/BaslerCameraTest.exe`
- Linux/Mac: `dist/BaslerCameraTest`

## Build Options Explained

- `--onefile`: Single executable file (no dependencies folder)
- `--windowed` or `-w`: No console window (GUI only)
- `--name="BaslerCameraTest"`: Output filename
- `--icon=icon.ico`: Add custom icon (optional)
- `--add-data "src;dest"`: Include additional files (optional)

## Advanced: Custom Icon

1. Create or download an `.ico` file
2. Build with icon:

```bash
pyinstaller --onefile --windowed --icon=app_icon.ico --name="BaslerCameraTest" basler_camera_test.py
```

## Troubleshooting

### Issue: "Module not found" errors

**Solution**: Add hidden imports in spec file:

```python
hiddenimports=['pypylon', 'cv2', 'numpy', 'PyQt5']
```

### Issue: Large file size

**Solutions**:

1. Use UPX compression: `--upx-dir=/path/to/upx`
2. Exclude unnecessary modules: `--exclude-module=matplotlib`

### Issue: Antivirus blocking

**Solution**: Add exception in antivirus or sign the executable

### Issue: DLL not found

**Solution**:

1. Install Pylon Camera Software Suite
2. Add pylon DLLs to build:

```python
datas=[('C:/Program Files/Basler/pylon 7/Runtime/x64/*.dll', '.')]
```

## File Structure After Build

```
dist/
  └── BaslerCameraTest.exe    # Your executable!
build/                        # Temporary build files (can delete)
basler_camera_test.spec       # Build configuration
```

## Distribution

### Package for distribution:

1. Copy `BaslerCameraTest.exe` from `dist/` folder
2. Optionally create installer with NSIS or Inno Setup
3. Users also need to install Pylon Camera Software Suite

## Important Notes

⚠️ **Pylon SDK Required**:
Users must install Basler Pylon Camera Software Suite on their machine for the camera to work.

⚠️ **File Size**:
Expect 100-200MB executable due to included libraries.

⚠️ **Test Before Distribution**:
Always test the .exe on a clean machine without Python installed.

## Clean Build

Remove old build files:

```bash
# Windows
rmdir /s /q build dist
del /q *.spec

# Linux/Mac
rm -rf build dist *.spec
```

Then rebuild.
