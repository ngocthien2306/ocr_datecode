#!/bin/bash
echo "Building PLC Camera GUI to EXE..."
echo

pyinstaller --onefile --windowed --name "PLC_Camera_Trigger" plc_camera_gui_tk.py

echo
echo "Done! Check dist/PLC_Camera_Trigger"
