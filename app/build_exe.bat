@echo off
echo Building PLC Camera GUI to EXE...
echo.

pyinstaller --onefile --windowed --name "PLC_Camera_Trigger" --icon=NONE plc_camera_gui_tk.py

echo.
echo Done! Check dist\PLC_Camera_Trigger.exe
pause
