"""
Basler Camera Test Application with PyQt5
Features:
- Camera connection and disconnection
- Live view (continuous grabbing)
- Software trigger
- Hardware trigger configuration
- Camera parameter adjustment (Exposure, Gain, Width, Height)
- Image saving
- Trigger status monitoring
"""

import sys
import cv2
import numpy as np
from datetime import datetime
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QLineEdit, 
                             QComboBox, QGroupBox, QSpinBox, QDoubleSpinBox,
                             QTextEdit, QFileDialog, QMessageBox, QCheckBox,
                             QGridLayout, QTabWidget)
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from pypylon import pylon
from pypylon import genicam


class CameraThread(QThread):
    """Thread to handle image grabbing from camera"""
    image_ready = pyqtSignal(np.ndarray)
    error_occurred = pyqtSignal(str)
    frame_info = pyqtSignal(dict)
    
    def __init__(self):
        super().__init__()
        self.camera = None
        self.running = False
        self.converter = pylon.ImageFormatConverter()
        self.converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        self.converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
        
    def set_camera(self, camera):
        self.camera = camera
        
    def run(self):
        """Run continuous image grabbing loop"""
        try:
            if not self.camera or not self.camera.IsOpen():
                self.error_occurred.emit("Camera not connected")
                return
                
            self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
            self.running = True
            frame_count = 0
            
            while self.running and self.camera.IsGrabbing():
                grabResult = self.camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
                
                if grabResult.GrabSucceeded():
                    frame_count += 1
                    
                    # Convert to BGR format
                    image = self.converter.Convert(grabResult)
                    img_array = image.GetArray()
                    
                    # Emit image
                    self.image_ready.emit(img_array)
                    
                    # Emit frame info
                    info = {
                        'frame_count': frame_count,
                        'width': grabResult.Width,
                        'height': grabResult.Height,
                        'timestamp': grabResult.TimeStamp if hasattr(grabResult, 'TimeStamp') else 0
                    }
                    self.frame_info.emit(info)
                else:
                    self.error_occurred.emit(f"Grab failed: {grabResult.ErrorDescription}")
                    
                grabResult.Release()
                
        except Exception as e:
            self.error_occurred.emit(f"Error in CameraThread: {str(e)}")
        finally:
            if self.camera and self.camera.IsGrabbing():
                self.camera.StopGrabbing()
                
    def stop(self):
        """Stop thread"""
        self.running = False
        self.wait()


class BaslerCameraTestApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.camera = None
        self.camera_thread = None
        self.current_image = None
        self.initUI()
        
    def initUI(self):
        """Initialize user interface"""
        self.setWindowTitle('Basler Camera Test Application')
        self.setGeometry(100, 100, 1400, 900)
        
        # Main widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # Left side: Image display
        left_layout = QVBoxLayout()
        
        # Image display label
        self.image_label = QLabel()
        self.image_label.setMinimumSize(800, 600)
        self.image_label.setStyleSheet("border: 2px solid #4A90E2; background-color: #F5F5F5;")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setText("No Image")
        left_layout.addWidget(self.image_label)
        
        # Frame info
        self.frame_info_label = QLabel("Frame: 0 | Size: 0x0 | Timestamp: 0")
        self.frame_info_label.setStyleSheet("background-color: #4A90E2; color: white; padding: 5px; font-weight: bold;")
        left_layout.addWidget(self.frame_info_label)
        
        main_layout.addLayout(left_layout, 2)
        
        # Right side: Controls
        right_layout = QVBoxLayout()
        
        # Tab widget for functions
        tab_widget = QTabWidget()
        
        # IMPORTANT: Create log tab FIRST before other tabs that use log_message
        log_tab = self.create_log_tab()
        tab_widget.addTab(log_tab, "Log")
        
        # Tab 1: Camera Connection
        connection_tab = self.create_connection_tab()
        tab_widget.addTab(connection_tab, "Connection")
        
        # Tab 2: Camera Settings
        settings_tab = self.create_settings_tab()
        tab_widget.addTab(settings_tab, "Settings")
        
        # Tab 3: Trigger
        trigger_tab = self.create_trigger_tab()
        tab_widget.addTab(trigger_tab, "Trigger")
        
        right_layout.addWidget(tab_widget)
        main_layout.addLayout(right_layout, 1)
        
    def create_connection_tab(self):
        """Camera connection tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Camera selection
        camera_group = QGroupBox("Select Camera")
        camera_layout = QVBoxLayout()
        
        self.camera_combo = QComboBox()
        camera_layout.addWidget(QLabel("Camera List:"))
        camera_layout.addWidget(self.camera_combo)
        
        refresh_btn = QPushButton("Refresh List")
        refresh_btn.clicked.connect(self.refresh_camera_list)
        camera_layout.addWidget(refresh_btn)
        
        camera_group.setLayout(camera_layout)
        layout.addWidget(camera_group)
        
        # Connection buttons
        connection_group = QGroupBox("Connection")
        connection_layout = QVBoxLayout()
        
        self.connect_btn = QPushButton("Connect Camera")
        self.connect_btn.clicked.connect(self.connect_camera)
        self.connect_btn.setStyleSheet("background-color: #4A90E2; color: white; padding: 10px; font-weight: bold; border-radius: 5px;")
        connection_layout.addWidget(self.connect_btn)

        self.disconnect_btn = QPushButton("Disconnect")
        self.disconnect_btn.clicked.connect(self.disconnect_camera)
        self.disconnect_btn.setEnabled(False)
        self.disconnect_btn.setStyleSheet("background-color: #E74C3C; color: white; padding: 10px; font-weight: bold; border-radius: 5px;")
        connection_layout.addWidget(self.disconnect_btn)
        
        connection_group.setLayout(connection_layout)
        layout.addWidget(connection_group)
        
        # Grabbing controls
        grabbing_group = QGroupBox("Image Acquisition")
        grabbing_layout = QVBoxLayout()
        
        self.start_live_btn = QPushButton("Start Live View")
        self.start_live_btn.clicked.connect(self.start_live_view)
        self.start_live_btn.setEnabled(False)
        grabbing_layout.addWidget(self.start_live_btn)
        
        self.stop_live_btn = QPushButton("Stop Live View")
        self.stop_live_btn.clicked.connect(self.stop_live_view)
        self.stop_live_btn.setEnabled(False)
        grabbing_layout.addWidget(self.stop_live_btn)
        
        self.grab_one_btn = QPushButton("Grab Single Image")
        self.grab_one_btn.clicked.connect(self.grab_single_image)
        self.grab_one_btn.setEnabled(False)
        grabbing_layout.addWidget(self.grab_one_btn)
        
        self.save_btn = QPushButton("Save Current Image")
        self.save_btn.clicked.connect(self.save_current_image)
        self.save_btn.setEnabled(False)
        grabbing_layout.addWidget(self.save_btn)
        
        grabbing_group.setLayout(grabbing_layout)
        layout.addWidget(grabbing_group)
        
        layout.addStretch()
        
        # Load camera list
        self.refresh_camera_list()
        
        return widget
        
    def create_settings_tab(self):
        """Camera settings tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Exposure settings
        exposure_group = QGroupBox("Exposure Time (µs)")
        exposure_layout = QHBoxLayout()
        
        self.exposure_spin = QDoubleSpinBox()
        self.exposure_spin.setRange(100, 1000000)
        self.exposure_spin.setValue(10000)
        self.exposure_spin.setSingleStep(1000)
        exposure_layout.addWidget(QLabel("Exposure:"))
        exposure_layout.addWidget(self.exposure_spin)
        
        self.exposure_set_btn = QPushButton("Apply")
        self.exposure_set_btn.clicked.connect(self.set_exposure)
        self.exposure_set_btn.setEnabled(False)
        exposure_layout.addWidget(self.exposure_set_btn)
        
        exposure_group.setLayout(exposure_layout)
        layout.addWidget(exposure_group)
        
        # Gain settings
        gain_group = QGroupBox("Gain")
        gain_layout = QHBoxLayout()
        
        self.gain_spin = QDoubleSpinBox()
        self.gain_spin.setRange(0, 30)
        self.gain_spin.setValue(0)
        self.gain_spin.setSingleStep(0.5)
        gain_layout.addWidget(QLabel("Gain (dB):"))
        gain_layout.addWidget(self.gain_spin)
        
        self.gain_set_btn = QPushButton("Apply")
        self.gain_set_btn.clicked.connect(self.set_gain)
        self.gain_set_btn.setEnabled(False)
        gain_layout.addWidget(self.gain_set_btn)
        
        gain_group.setLayout(gain_layout)
        layout.addWidget(gain_group)
        
        # ROI settings
        roi_group = QGroupBox("ROI (Region of Interest)")
        roi_layout = QGridLayout()
        
        self.width_spin = QSpinBox()
        self.width_spin.setRange(64, 4096)
        self.width_spin.setValue(1920)
        self.width_spin.setSingleStep(8)
        roi_layout.addWidget(QLabel("Width:"), 0, 0)
        roi_layout.addWidget(self.width_spin, 0, 1)
        
        self.height_spin = QSpinBox()
        self.height_spin.setRange(64, 4096)
        self.height_spin.setValue(1200)
        self.height_spin.setSingleStep(8)
        roi_layout.addWidget(QLabel("Height:"), 1, 0)
        roi_layout.addWidget(self.height_spin, 1, 1)
        
        self.offset_x_spin = QSpinBox()
        self.offset_x_spin.setRange(0, 4096)
        self.offset_x_spin.setValue(0)
        roi_layout.addWidget(QLabel("Offset X:"), 2, 0)
        roi_layout.addWidget(self.offset_x_spin, 2, 1)
        
        self.offset_y_spin = QSpinBox()
        self.offset_y_spin.setRange(0, 4096)
        self.offset_y_spin.setValue(0)
        roi_layout.addWidget(QLabel("Offset Y:"), 3, 0)
        roi_layout.addWidget(self.offset_y_spin, 3, 1)
        
        self.roi_set_btn = QPushButton("Apply ROI")
        self.roi_set_btn.clicked.connect(self.set_roi)
        self.roi_set_btn.setEnabled(False)
        roi_layout.addWidget(self.roi_set_btn, 4, 0, 1, 2)
        
        roi_group.setLayout(roi_layout)
        layout.addWidget(roi_group)
        
        # Get current settings button
        self.get_settings_btn = QPushButton("Read Current Settings")
        self.get_settings_btn.clicked.connect(self.get_current_settings)
        self.get_settings_btn.setEnabled(False)
        layout.addWidget(self.get_settings_btn)
        
        layout.addStretch()
        return widget
        
    def create_trigger_tab(self):
        """Trigger settings tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Trigger Mode
        trigger_mode_group = QGroupBox("Trigger Mode")
        trigger_mode_layout = QVBoxLayout()
        
        self.trigger_off_radio = QCheckBox("Trigger OFF (Free Running)")
        self.trigger_off_radio.setChecked(True)
        trigger_mode_layout.addWidget(self.trigger_off_radio)
        
        self.software_trigger_radio = QCheckBox("Software Trigger")
        trigger_mode_layout.addWidget(self.software_trigger_radio)
        
        self.hardware_trigger_radio = QCheckBox("Hardware Trigger")
        trigger_mode_layout.addWidget(self.hardware_trigger_radio)
        
        self.apply_trigger_mode_btn = QPushButton("Apply Trigger Mode")
        self.apply_trigger_mode_btn.clicked.connect(self.apply_trigger_mode)
        self.apply_trigger_mode_btn.setEnabled(False)
        trigger_mode_layout.addWidget(self.apply_trigger_mode_btn)
        
        trigger_mode_group.setLayout(trigger_mode_layout)
        layout.addWidget(trigger_mode_group)
        
        # Hardware Trigger Settings
        hw_trigger_group = QGroupBox("Hardware Trigger Settings")
        hw_trigger_layout = QGridLayout()
        
        hw_trigger_layout.addWidget(QLabel("Trigger Source:"), 0, 0)
        self.trigger_source_combo = QComboBox()
        self.trigger_source_combo.addItems(['Line1', 'Line2', 'Line3'])
        hw_trigger_layout.addWidget(self.trigger_source_combo, 0, 1)
        
        hw_trigger_layout.addWidget(QLabel("Trigger Activation:"), 1, 0)
        self.trigger_activation_combo = QComboBox()
        self.trigger_activation_combo.addItems(['RisingEdge', 'FallingEdge', 'AnyEdge', 'LevelHigh', 'LevelLow'])
        hw_trigger_layout.addWidget(self.trigger_activation_combo, 1, 1)
        
        self.apply_hw_trigger_btn = QPushButton("Apply Hardware Trigger")
        self.apply_hw_trigger_btn.clicked.connect(self.apply_hardware_trigger_settings)
        self.apply_hw_trigger_btn.setEnabled(False)
        hw_trigger_layout.addWidget(self.apply_hw_trigger_btn, 2, 0, 1, 2)
        
        hw_trigger_group.setLayout(hw_trigger_layout)
        layout.addWidget(hw_trigger_group)
        
        # Software Trigger Control
        sw_trigger_group = QGroupBox("Software Trigger Control")
        sw_trigger_layout = QVBoxLayout()
        
        self.execute_sw_trigger_btn = QPushButton("Execute Software Trigger")
        self.execute_sw_trigger_btn.clicked.connect(self.execute_software_trigger)
        self.execute_sw_trigger_btn.setEnabled(False)
        sw_trigger_layout.addWidget(self.execute_sw_trigger_btn)
        
        # Auto trigger
        auto_trigger_layout = QHBoxLayout()
        self.auto_trigger_checkbox = QCheckBox("Auto Trigger")
        auto_trigger_layout.addWidget(self.auto_trigger_checkbox)
        
        auto_trigger_layout.addWidget(QLabel("Interval (ms):"))
        self.auto_trigger_interval = QSpinBox()
        self.auto_trigger_interval.setRange(100, 10000)
        self.auto_trigger_interval.setValue(1000)
        auto_trigger_layout.addWidget(self.auto_trigger_interval)
        
        sw_trigger_layout.addLayout(auto_trigger_layout)
        sw_trigger_group.setLayout(sw_trigger_layout)
        layout.addWidget(sw_trigger_group)
        
        # Line Status Monitor
        line_status_group = QGroupBox("Line Status Monitor")
        line_status_layout = QVBoxLayout()
        
        self.line_status_label = QLabel("Line Status: N/A")
        self.line_status_label.setStyleSheet("background-color: #E8F4F8; color: #2C3E50; padding: 5px; border: 1px solid #4A90E2;")
        line_status_layout.addWidget(self.line_status_label)
        
        self.monitor_line_checkbox = QCheckBox("Monitor Line Status")
        self.monitor_line_checkbox.stateChanged.connect(self.toggle_line_monitor)
        line_status_layout.addWidget(self.monitor_line_checkbox)
        
        line_status_group.setLayout(line_status_layout)
        layout.addWidget(line_status_group)
        
        layout.addStretch()
        
        # Timer for auto trigger
        self.auto_trigger_timer = QTimer()
        self.auto_trigger_timer.timeout.connect(self.execute_software_trigger)
        self.auto_trigger_checkbox.stateChanged.connect(self.toggle_auto_trigger)
        
        # Timer for line status monitor
        self.line_monitor_timer = QTimer()
        self.line_monitor_timer.timeout.connect(self.update_line_status)
        
        return widget
        
    def create_log_tab(self):
        """Log information tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        layout.addWidget(QLabel("System Log:"))
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet("background-color: #F8FBFD; color: #2C3E50; font-family: monospace; border: 1px solid #4A90E2;")
        layout.addWidget(self.log_text)
        
        clear_log_btn = QPushButton("Clear Log")
        clear_log_btn.clicked.connect(self.log_text.clear)
        layout.addWidget(clear_log_btn)
        
        return widget
        
    def log_message(self, message):
        """Write log message"""
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        log_entry = f"[{timestamp}] {message}"
        self.log_text.append(log_entry)
        print(log_entry)
        
    def refresh_camera_list(self):
        """Refresh camera list"""
        try:
            self.camera_combo.clear()
            tlFactory = pylon.TlFactory.GetInstance()
            devices = tlFactory.EnumerateDevices()
            
            if len(devices) == 0:
                self.camera_combo.addItem("No camera found")
                self.log_message("No camera found")
            else:
                for i, device in enumerate(devices):
                    camera_name = f"{device.GetModelName()} ({device.GetSerialNumber()})"
                    self.camera_combo.addItem(camera_name)
                self.log_message(f"Found {len(devices)} camera(s)")
        except Exception as e:
            self.log_message(f"Error refreshing camera list: {str(e)}")
            
    def connect_camera(self):
        """Connect to camera"""
        try:
            tlFactory = pylon.TlFactory.GetInstance()
            devices = tlFactory.EnumerateDevices()
            
            if len(devices) == 0:
                QMessageBox.warning(self, "Error", "No camera found")
                return
                
            # Get first camera
            self.camera = pylon.InstantCamera(tlFactory.CreateFirstDevice())
            self.camera.Open()
            
            # Get camera info
            device_info = self.camera.GetDeviceInfo()
            self.log_message(f"Connected: {device_info.GetModelName()}")
            self.log_message(f"Serial Number: {device_info.GetSerialNumber()}")
            
            # Enable buttons
            self.connect_btn.setEnabled(False)
            self.disconnect_btn.setEnabled(True)
            self.start_live_btn.setEnabled(True)
            self.grab_one_btn.setEnabled(True)
            self.exposure_set_btn.setEnabled(True)
            self.gain_set_btn.setEnabled(True)
            self.roi_set_btn.setEnabled(True)
            self.get_settings_btn.setEnabled(True)
            self.apply_trigger_mode_btn.setEnabled(True)
            
            # Read current settings
            self.get_current_settings()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Cannot connect camera: {str(e)}")
            self.log_message(f"Connection error: {str(e)}")
            
    def disconnect_camera(self):
        """Disconnect camera"""
        try:
            # Stop live view if running
            self.stop_live_view()
            
            if self.camera and self.camera.IsOpen():
                self.camera.Close()
                self.log_message("Camera disconnected")
                
            self.camera = None
            
            # Disable buttons
            self.connect_btn.setEnabled(True)
            self.disconnect_btn.setEnabled(False)
            self.start_live_btn.setEnabled(False)
            self.stop_live_btn.setEnabled(False)
            self.grab_one_btn.setEnabled(False)
            self.save_btn.setEnabled(False)
            self.exposure_set_btn.setEnabled(False)
            self.gain_set_btn.setEnabled(False)
            self.roi_set_btn.setEnabled(False)
            self.get_settings_btn.setEnabled(False)
            self.apply_trigger_mode_btn.setEnabled(False)
            self.execute_sw_trigger_btn.setEnabled(False)
            self.apply_hw_trigger_btn.setEnabled(False)
            
            # Clear image
            self.image_label.clear()
            self.image_label.setText("No Image")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Disconnection error: {str(e)}")
            self.log_message(f"Disconnection error: {str(e)}")
            
    def start_live_view(self):
        """Start live view"""
        try:
            if not self.camera or not self.camera.IsOpen():
                QMessageBox.warning(self, "Error", "Camera not connected")
                return
                
            # Create and start camera thread
            self.camera_thread = CameraThread()
            self.camera_thread.set_camera(self.camera)
            self.camera_thread.image_ready.connect(self.update_image)
            self.camera_thread.error_occurred.connect(self.handle_camera_error)
            self.camera_thread.frame_info.connect(self.update_frame_info)
            self.camera_thread.start()
            
            self.start_live_btn.setEnabled(False)
            self.stop_live_btn.setEnabled(True)
            self.save_btn.setEnabled(True)
            
            self.log_message("Live View started")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Cannot start live view: {str(e)}")
            self.log_message(f"Live view error: {str(e)}")
            
    def stop_live_view(self):
        """Stop live view"""
        try:
            if self.camera_thread:
                self.camera_thread.stop()
                self.camera_thread = None
                
            self.start_live_btn.setEnabled(True)
            self.stop_live_btn.setEnabled(False)
            
            self.log_message("Live View stopped")
            
        except Exception as e:
            self.log_message(f"Error stopping live view: {str(e)}")
            
    def grab_single_image(self):
        """Grab a single image"""
        try:
            if not self.camera or not self.camera.IsOpen():
                QMessageBox.warning(self, "Error", "Camera not connected")
                return
                
            # If live view is running, stop it
            was_grabbing = self.camera_thread is not None
            if was_grabbing:
                self.stop_live_view()
                
            # Grab one image
            self.camera.StartGrabbingMax(1)
            grabResult = self.camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
            
            if grabResult.GrabSucceeded():
                converter = pylon.ImageFormatConverter()
                converter.OutputPixelFormat = pylon.PixelType_BGR8packed
                converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
                
                image = converter.Convert(grabResult)
                img_array = image.GetArray()
                
                self.update_image(img_array)
                self.save_btn.setEnabled(True)
                
                self.log_message("Single image grabbed")
            else:
                self.log_message(f"Grab failed: {grabResult.ErrorDescription}")
                
            grabResult.Release()
            self.camera.StopGrabbing()
            
            # If previously in live view, restart it
            if was_grabbing:
                self.start_live_view()
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Cannot grab image: {str(e)}")
            self.log_message(f"Grab error: {str(e)}")
            
    def update_image(self, img_array):
        """Update image display"""
        try:
            self.current_image = img_array.copy()
            
            # Convert to QImage for display
            height, width, channel = img_array.shape
            bytes_per_line = 3 * width
            q_image = QImage(img_array.data, width, height, bytes_per_line, QImage.Format_BGR888)
            
            # Scale image to fit label
            pixmap = QPixmap.fromImage(q_image)
            scaled_pixmap = pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.image_label.setPixmap(scaled_pixmap)
            
        except Exception as e:
            self.log_message(f"Error updating image: {str(e)}")
            
    def update_frame_info(self, info):
        """Update frame information"""
        info_text = f"Frame: {info['frame_count']} | Size: {info['width']}x{info['height']} | Timestamp: {info['timestamp']}"
        self.frame_info_label.setText(info_text)
        
    def handle_camera_error(self, error_msg):
        """Handle camera error"""
        self.log_message(f"Camera Error: {error_msg}")
        
    def save_current_image(self):
        """Save current image"""
        try:
            if self.current_image is None:
                QMessageBox.warning(self, "Error", "No image to save")
                return
                
            # Create filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            default_filename = f"basler_image_{timestamp}.png"
            
            filename, _ = QFileDialog.getSaveFileName(
                self, 
                "Save Image", 
                default_filename,
                "PNG Files (*.png);;JPEG Files (*.jpg);;All Files (*)"
            )
            
            if filename:
                cv2.imwrite(filename, self.current_image)
                self.log_message(f"Image saved: {filename}")
                QMessageBox.information(self, "Success", f"Image saved: {filename}")
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Cannot save image: {str(e)}")
            self.log_message(f"Save error: {str(e)}")
            
    def set_exposure(self):
        """Set exposure time"""
        try:
            if not self.camera or not self.camera.IsOpen():
                return
                
            exposure_value = self.exposure_spin.value()
            self.camera.ExposureTime.SetValue(exposure_value)
            
            actual_value = self.camera.ExposureTime.GetValue()
            self.log_message(f"Exposure time set: {actual_value} µs")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Cannot set exposure: {str(e)}")
            self.log_message(f"Exposure error: {str(e)}")
            
    def set_gain(self):
        """Set gain"""
        try:
            if not self.camera or not self.camera.IsOpen():
                return
                
            gain_value = self.gain_spin.value()
            
            # Check if camera supports Gain
            if genicam.IsAvailable(self.camera.Gain):
                self.camera.Gain.SetValue(gain_value)
                actual_value = self.camera.Gain.GetValue()
                self.log_message(f"Gain set: {actual_value} dB")
            else:
                self.log_message("Camera does not support Gain parameter")
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Cannot set gain: {str(e)}")
            self.log_message(f"Gain error: {str(e)}")
            
    def set_roi(self):
        """Set ROI"""
        try:
            if not self.camera or not self.camera.IsOpen():
                return
                
            # Stop grabbing if running
            was_grabbing = self.camera_thread is not None
            if was_grabbing:
                self.stop_live_view()
                
            width = self.width_spin.value()
            height = self.height_spin.value()
            offset_x = self.offset_x_spin.value()
            offset_y = self.offset_y_spin.value()
            
            # Set ROI parameters
            self.camera.Width.SetValue(width)
            self.camera.Height.SetValue(height)
            self.camera.OffsetX.SetValue(offset_x)
            self.camera.OffsetY.SetValue(offset_y)
            
            self.log_message(f"ROI set: {width}x{height} at ({offset_x}, {offset_y})")
            
            # Restart if previously grabbing
            if was_grabbing:
                self.start_live_view()
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Cannot set ROI: {str(e)}")
            self.log_message(f"ROI error: {str(e)}")
            
    def get_current_settings(self):
        """Read current settings from camera"""
        try:
            if not self.camera or not self.camera.IsOpen():
                return
                
            # Read and update UI
            if genicam.IsReadable(self.camera.ExposureTime):
                exposure = self.camera.ExposureTime.GetValue()
                self.exposure_spin.setValue(exposure)
                
            if genicam.IsReadable(self.camera.Gain):
                gain = self.camera.Gain.GetValue()
                self.gain_spin.setValue(gain)
                
            if genicam.IsReadable(self.camera.Width):
                width = self.camera.Width.GetValue()
                self.width_spin.setValue(width)
                
            if genicam.IsReadable(self.camera.Height):
                height = self.camera.Height.GetValue()
                self.height_spin.setValue(height)
                
            if genicam.IsReadable(self.camera.OffsetX):
                offset_x = self.camera.OffsetX.GetValue()
                self.offset_x_spin.setValue(offset_x)
                
            if genicam.IsReadable(self.camera.OffsetY):
                offset_y = self.camera.OffsetY.GetValue()
                self.offset_y_spin.setValue(offset_y)
                
            self.log_message("Current settings read from camera")
            
        except Exception as e:
            self.log_message(f"Error reading settings: {str(e)}")
            
    def apply_trigger_mode(self):
        """Apply trigger mode"""
        try:
            if not self.camera or not self.camera.IsOpen():
                return
                
            # Stop grabbing if running
            was_grabbing = self.camera_thread is not None
            if was_grabbing:
                self.stop_live_view()
                
            if self.trigger_off_radio.isChecked():
                # Free running mode
                self.camera.TriggerMode.SetValue('Off')
                self.log_message("Trigger Mode: OFF (Free Running)")
                self.execute_sw_trigger_btn.setEnabled(False)
                self.apply_hw_trigger_btn.setEnabled(False)
                
            elif self.software_trigger_radio.isChecked():
                # Software trigger mode
                self.camera.TriggerSelector.SetValue('FrameStart')
                self.camera.TriggerMode.SetValue('On')
                self.camera.TriggerSource.SetValue('Software')
                self.log_message("Trigger Mode: Software Trigger")
                self.execute_sw_trigger_btn.setEnabled(True)
                self.apply_hw_trigger_btn.setEnabled(False)
                
            elif self.hardware_trigger_radio.isChecked():
                # Hardware trigger mode
                self.apply_hardware_trigger_settings()
                self.execute_sw_trigger_btn.setEnabled(False)
                self.apply_hw_trigger_btn.setEnabled(True)
                
            # Restart if previously grabbing
            if was_grabbing:
                self.start_live_view()
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Cannot apply trigger mode: {str(e)}")
            self.log_message(f"Trigger mode error: {str(e)}")
            
    def apply_hardware_trigger_settings(self):
        """Apply hardware trigger settings"""
        try:
            if not self.camera or not self.camera.IsOpen():
                return
                
            trigger_source = self.trigger_source_combo.currentText()
            trigger_activation = self.trigger_activation_combo.currentText()
            
            # Configure hardware trigger
            self.camera.TriggerSelector.SetValue('FrameStart')
            self.camera.TriggerMode.SetValue('On')
            self.camera.TriggerSource.SetValue(trigger_source)
            self.camera.TriggerActivation.SetValue(trigger_activation)
            
            # Configure line as input
            self.camera.LineSelector.SetValue(trigger_source)
            self.camera.LineMode.SetValue('Input')
            
            self.log_message(f"Hardware Trigger: Source={trigger_source}, Activation={trigger_activation}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Cannot configure hardware trigger: {str(e)}")
            self.log_message(f"Hardware trigger error: {str(e)}")
            
    def execute_software_trigger(self):
        """Execute software trigger"""
        try:
            if not self.camera or not self.camera.IsOpen():
                return
                
            if self.camera.WaitForFrameTriggerReady(1000, pylon.TimeoutHandling_Return):
                self.camera.ExecuteSoftwareTrigger()
                self.log_message("Software trigger executed")
            else:
                self.log_message("Camera not ready for trigger")
                
        except Exception as e:
            self.log_message(f"Execute trigger error: {str(e)}")
            
    def toggle_auto_trigger(self, state):
        """Toggle auto trigger"""
        if state == Qt.Checked:
            interval = self.auto_trigger_interval.value()
            self.auto_trigger_timer.start(interval)
            self.log_message(f"Auto trigger started (interval: {interval}ms)")
        else:
            self.auto_trigger_timer.stop()
            self.log_message("Auto trigger stopped")
            
    def toggle_line_monitor(self, state):
        """Toggle line status monitor"""
        if state == Qt.Checked:
            self.line_monitor_timer.start(100)  # Update every 100ms
            self.log_message("Line status monitor started")
        else:
            self.line_monitor_timer.stop()
            self.line_status_label.setText("Line Status: N/A")
            self.log_message("Line status monitor stopped")
            
    def update_line_status(self):
        """Update line status"""
        try:
            if not self.camera or not self.camera.IsOpen():
                return
                
            trigger_source = self.trigger_source_combo.currentText()
            
            # Read line status
            self.camera.LineSelector.SetValue(trigger_source)
            line_status = self.camera.LineStatus.GetValue()
            
            status_text = "HIGH" if line_status else "LOW"
            color = "#27AE60" if line_status else "#E74C3C"

            self.line_status_label.setText(f"Line Status: {status_text}")
            self.line_status_label.setStyleSheet(f"background-color: {color}; color: white; padding: 5px; font-weight: bold; border: 2px solid #2C3E50;")
            
        except Exception as e:
            self.line_status_label.setText(f"Error: {str(e)}")
            
    def closeEvent(self, event):
        """Handle application close"""
        try:
            # Stop all timers
            self.auto_trigger_timer.stop()
            self.line_monitor_timer.stop()
            
            # Stop live view
            self.stop_live_view()
            
            # Disconnect camera
            if self.camera and self.camera.IsOpen():
                self.camera.Close()
                
        except Exception as e:
            print(f"Error during cleanup: {str(e)}")
            
        event.accept()


def main():
    app = QApplication(sys.argv)

    # Set style
    app.setStyle('Fusion')

    # Set application-wide stylesheet with white and blue theme
    app.setStyleSheet("""
        QMainWindow {
            background-color: white;
        }
        QWidget {
            background-color: white;
            color: #2C3E50;
        }
        QGroupBox {
            font-weight: bold;
            border: 2px solid #4A90E2;
            border-radius: 5px;
            margin-top: 10px;
            padding-top: 10px;
            background-color: #F8FBFD;
        }
        QGroupBox::title {
            color: #4A90E2;
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px;
        }
        QPushButton {
            background-color: #4A90E2;
            color: white;
            border: none;
            padding: 8px;
            border-radius: 4px;
            font-weight: bold;
            min-width: 80px;
        }
        QPushButton:hover {
            background-color: #357ABD;
        }
        QPushButton:pressed {
            background-color: #2E6DA4;
        }
        QPushButton:disabled {
            background-color: #BDC3C7;
            color: #7F8C8D;
        }
        QLabel {
            color: #2C3E50;
        }
        QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
            border: 1px solid #4A90E2;
            border-radius: 3px;
            padding: 5px;
            background-color: white;
            selection-background-color: #4A90E2;
        }
        QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
            border: 2px solid #4A90E2;
        }
        QTabWidget::pane {
            border: 2px solid #4A90E2;
            border-radius: 5px;
            background-color: white;
        }
        QTabBar::tab {
            background-color: #E8F4F8;
            color: #2C3E50;
            border: 1px solid #4A90E2;
            padding: 8px 16px;
            margin-right: 2px;
        }
        QTabBar::tab:selected {
            background-color: #4A90E2;
            color: white;
            font-weight: bold;
        }
        QTabBar::tab:hover {
            background-color: #6FB3F5;
            color: white;
        }
        QCheckBox {
            color: #2C3E50;
            spacing: 5px;
        }
        QCheckBox::indicator {
            width: 18px;
            height: 18px;
            border: 2px solid #4A90E2;
            border-radius: 3px;
            background-color: white;
        }
        QCheckBox::indicator:checked {
            background-color: #4A90E2;
            image: url(none);
        }
        QCheckBox::indicator:hover {
            border: 2px solid #357ABD;
        }
    """)

    # Create and show window
    window = BaslerCameraTestApp()
    window.show()

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()