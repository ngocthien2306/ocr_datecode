#!/usr/bin/env python3
"""
GPIO Camera Trigger System
Combines Jetson GPIO event detection with Basler camera software trigger and reject output control.
"""

import os
import sys
import time
import threading
from datetime import datetime

# Import PyQt5
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QGroupBox, QLabel, QPushButton, QSpinBox, QDoubleSpinBox,
                             QComboBox, QRadioButton, QCheckBox, QLineEdit, QTextEdit,
                             QButtonGroup, QFileDialog, QMessageBox, QStatusBar)
from PyQt5.QtCore import QObject, pyqtSignal, QThread, Qt
from PyQt5.QtGui import QPixmap, QImage, QFont

# Use PIL instead of cv2 to avoid Qt plugin conflicts
from PIL import Image
import numpy as np

# Other imports
from pypylon import pylon
import Jetson.GPIO as GPIO


# ============================================================================
# SIGNALS CLASS
# ============================================================================

class GPIOSignals(QObject):
    """Thread-safe signals for cross-thread communication"""
    # Logging
    log_signal = pyqtSignal(str, str)  # (message, level)

    # GPIO events
    trigger_detected_signal = pyqtSignal(str)  # edge_type: "RISING"/"FALLING"

    # Camera events
    image_captured_signal = pyqtSignal(object, int, int)  # (numpy_array, current_idx, total_count)
    capture_complete_signal = pyqtSignal(int, list)  # (total_images, file_paths)
    capture_error_signal = pyqtSignal(str)  # error_message

    # Reject system
    reject_triggered_signal = pyqtSignal()
    reject_complete_signal = pyqtSignal()

    # Status updates
    status_update_signal = pyqtSignal(str, str)  # (component, status)
    capture_count_signal = pyqtSignal(int)  # total_captures


# ============================================================================
# GPIO CONTROLLER
# ============================================================================

class GPIOController:
    """Manages GPIO pin setup, event detection, and output control"""

    def __init__(self, signals):
        self.signals = signals
        GPIO.setmode(GPIO.BOARD)
        GPIO.setwarnings(False)

        self.input_pin = None
        self.output_pin = None
        self.edge_type = GPIO.RISING
        self.event_active = False

        # Available pins for Jetson Orin NX
        self.available_pins = {
            7: "MCLK05 (Audio Master Clock)",
            11: "UART1_RTS",
            12: "I2S2_CLK",
            19: "SPI1_MOSI",
            21: "SPI1_MISO",
            23: "SPI1_SCK",
            26: "SPI1_CS1_N",
            32: "GPIO8",
            36: "UART1_CTS"
        }

    def setup_trigger_input(self, pin, edge_type, debounce_ms=200):
        """
        Setup GPIO input pin for trigger detection

        Args:
            pin: GPIO pin number (BOARD numbering)
            edge_type: GPIO.RISING, GPIO.FALLING, or GPIO.BOTH
            debounce_ms: Debounce time in milliseconds
        """
        try:
            # Remove existing event detection if any
            if self.event_active and self.input_pin is not None:
                try:
                    GPIO.remove_event_detect(self.input_pin)
                except:
                    pass

            # Setup pin as input
            GPIO.setup(pin, GPIO.IN)

            # Add event detection with callback
            GPIO.add_event_detect(
                pin,
                edge_type,
                callback=self._trigger_callback,
                bouncetime=debounce_ms
            )

            self.input_pin = pin
            self.edge_type = edge_type
            self.event_active = True

            edge_name = self._edge_type_name(edge_type)
            self.signals.log_signal.emit(
                f"GPIO trigger setup: Pin {pin}, Edge: {edge_name}, Debounce: {debounce_ms}ms",
                "SUCCESS"
            )
            return True

        except Exception as e:
            self.signals.log_signal.emit(f"Error setting up GPIO trigger: {e}", "ERROR")
            return False

    def setup_reject_output(self, pin):
        """
        Setup GPIO output pin for reject system

        Args:
            pin: GPIO pin number (BOARD numbering)
        """
        try:
            # Setup pin as output, initial LOW
            GPIO.setup(pin, GPIO.OUT, initial=GPIO.LOW)
            self.output_pin = pin

            self.signals.log_signal.emit(
                f"Reject output setup: Pin {pin} (Initial: LOW)",
                "SUCCESS"
            )
            return True

        except Exception as e:
            self.signals.log_signal.emit(f"Error setting up reject output: {e}", "ERROR")
            return False

    def _trigger_callback(self, channel):
        """
        GPIO callback (runs in separate thread!)
        CRITICAL: Only emit signals here, no direct GUI updates!
        """
        # Determine edge type from current state
        current_state = GPIO.input(channel)

        if self.edge_type == GPIO.RISING:
            edge_name = "RISING"
        elif self.edge_type == GPIO.FALLING:
            edge_name = "FALLING"
        else:  # GPIO.BOTH
            edge_name = "RISING" if current_state == 1 else "FALLING"

        # Emit signal to main thread (thread-safe)
        self.signals.trigger_detected_signal.emit(edge_name)

    def kick_reject_pulse(self, pulse_width_ms, delay_ms):
        """
        Trigger reject output pulse (runs in separate thread)

        Args:
            pulse_width_ms: Duration of HIGH pulse in milliseconds
            delay_ms: Delay before pulse in milliseconds
        """
        def pulse_thread():
            try:
                if self.output_pin is None:
                    self.signals.log_signal.emit("Reject output not configured", "ERROR")
                    return

                # Wait delay before reject
                if delay_ms > 0:
                    time.sleep(delay_ms / 1000.0)

                # Kick HIGH
                GPIO.output(self.output_pin, GPIO.HIGH)
                self.signals.reject_triggered_signal.emit()
                self.signals.log_signal.emit(f"Reject OUTPUT HIGH (Pin {self.output_pin})", "EVENT")

                # Wait pulse width
                time.sleep(pulse_width_ms / 1000.0)

                # Back to LOW
                GPIO.output(self.output_pin, GPIO.LOW)
                self.signals.reject_complete_signal.emit()
                self.signals.log_signal.emit(
                    f"Reject pulse complete ({pulse_width_ms}ms)",
                    "SUCCESS"
                )

            except Exception as e:
                self.signals.log_signal.emit(f"Reject pulse error: {e}", "ERROR")

        # Run in thread to avoid blocking
        thread = threading.Thread(target=pulse_thread, daemon=True)
        thread.start()

    def remove_event_detection(self):
        """Remove GPIO event detection"""
        if self.event_active and self.input_pin is not None:
            try:
                GPIO.remove_event_detect(self.input_pin)
                self.event_active = False
                self.signals.log_signal.emit("GPIO event detection removed", "INFO")
            except Exception as e:
                self.signals.log_signal.emit(f"Error removing event detection: {e}", "ERROR")

    def cleanup(self):
        """Cleanup GPIO on exit"""
        try:
            self.remove_event_detection()
            GPIO.cleanup()
            self.signals.log_signal.emit("GPIO cleanup complete", "INFO")
        except Exception as e:
            self.signals.log_signal.emit(f"GPIO cleanup error: {e}", "ERROR")

    def _edge_type_name(self, edge_type):
        """Convert edge type constant to string name"""
        if edge_type == GPIO.RISING:
            return "RISING (0→1)"
        elif edge_type == GPIO.FALLING:
            return "FALLING (1→0)"
        else:
            return "BOTH"


# ============================================================================
# CAMERA CONTROLLER
# ============================================================================

class CameraController:
    """Manages Basler camera initialization, configuration, and capture"""

    def __init__(self, signals):
        self.signals = signals
        self.camera = None
        self.converter = None
        self.connected = False

    def connect(self, exposure_us=10000, gain=0.0):
        """
        Connect to Basler camera and configure software trigger

        Args:
            exposure_us: Exposure time in microseconds
            gain: Gain value
        """
        try:
            self.signals.log_signal.emit("Connecting to camera...", "INFO")

            # Initialize pypylon
            tlFactory = pylon.TlFactory.GetInstance()
            devices = tlFactory.EnumerateDevices()

            if len(devices) == 0:
                self.signals.log_signal.emit("No camera found", "ERROR")
                return False

            # Create and open camera
            self.camera = pylon.InstantCamera(tlFactory.CreateFirstDevice())
            self.camera.Open()

            camera_name = self.camera.GetDeviceInfo().GetModelName()
            self.signals.log_signal.emit(f"Camera found: {camera_name}", "INFO")

            # Configure software trigger mode
            self.camera.TriggerSelector.SetValue('FrameStart')
            self.camera.TriggerMode.SetValue('On')
            self.camera.TriggerSource.SetValue('Software')
            self.signals.log_signal.emit("Software trigger mode configured", "SUCCESS")

            # Apply initial settings
            self.apply_settings(exposure_us, gain)

            # Start grabbing
            self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

            # Initialize converter
            self.converter = pylon.ImageFormatConverter()
            self.converter.OutputPixelFormat = pylon.PixelType_BGR8packed

            self.connected = True
            self.signals.status_update_signal.emit("camera", "connected")
            self.signals.log_signal.emit(f"Camera connected: {camera_name}", "SUCCESS")

            return True

        except Exception as e:
            self.signals.log_signal.emit(f"Camera connection error: {e}", "ERROR")
            if self.camera:
                try:
                    self.camera.Close()
                except:
                    pass
                self.camera = None
            return False

    def disconnect(self):
        """Disconnect from camera"""
        try:
            if self.camera:
                if self.camera.IsGrabbing():
                    self.camera.StopGrabbing()
                self.camera.Close()
                self.camera = None

            self.connected = False
            self.converter = None
            self.signals.status_update_signal.emit("camera", "disconnected")
            self.signals.log_signal.emit("Camera disconnected", "INFO")

        except Exception as e:
            self.signals.log_signal.emit(f"Camera disconnect error: {e}", "ERROR")

    def apply_settings(self, exposure_us, gain):
        """
        Apply exposure and gain settings to camera

        Args:
            exposure_us: Exposure time in microseconds
            gain: Gain value
        """
        if not self.connected or not self.camera:
            return

        try:
            # Set exposure (try both old and new API)
            try:
                self.camera.ExposureTime.SetValue(exposure_us)
                self.signals.log_signal.emit(f"Exposure set: {exposure_us} µs", "INFO")
            except:
                try:
                    self.camera.ExposureTimeAbs.SetValue(exposure_us)
                    self.signals.log_signal.emit(f"Exposure set: {exposure_us} µs", "INFO")
                except Exception as e:
                    self.signals.log_signal.emit(f"Could not set exposure: {e}", "WARN")

            # Set gain (try both old and new API)
            try:
                self.camera.Gain.SetValue(gain)
                self.signals.log_signal.emit(f"Gain set: {gain}", "INFO")
            except:
                try:
                    self.camera.GainRaw.SetValue(int(gain))
                    self.signals.log_signal.emit(f"Gain set: {gain}", "INFO")
                except Exception as e:
                    self.signals.log_signal.emit(f"Could not set gain: {e}", "WARN")

        except Exception as e:
            self.signals.log_signal.emit(f"Settings error: {e}", "ERROR")

    def capture_image(self):
        """
        Capture single image using software trigger

        Returns:
            numpy array of captured image, or None on error
        """
        if not self.connected or not self.camera:
            return None

        try:
            # Execute software trigger
            self.camera.TriggerSoftware.Execute()

            # Retrieve result
            grabResult = self.camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

            if grabResult.GrabSucceeded():
                # Convert to BGR8 numpy array
                image = self.converter.Convert(grabResult)
                img_array = image.GetArray()
                grabResult.Release()
                return img_array
            else:
                grabResult.Release()
                return None

        except Exception as e:
            self.signals.log_signal.emit(f"Capture error: {e}", "ERROR")
            return None


# ============================================================================
# CAPTURE WORKER THREAD
# ============================================================================

class CaptureWorker(QThread):
    """Background thread for capture sequence"""

    def __init__(self, camera_controller, config, signals, capture_number):
        super().__init__()
        self.camera = camera_controller
        self.config = config
        self.signals = signals
        self.capture_number = capture_number

    def run(self):
        """Execute capture sequence in background"""
        try:
            # Extract config
            num_images = self.config['num_images']
            delay_between_ms = self.config['delay_between_ms']
            first_delay_ms = self.config['first_delay_ms']
            save_folder = self.config['save_folder']

            # Create save directory
            os.makedirs(save_folder, exist_ok=True)

            # Generate timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Create subfolder for multiple images
            if num_images > 1:
                subfolder = os.path.join(
                    save_folder,
                    f"trigger_{self.capture_number:03d}_{timestamp}"
                )
                os.makedirs(subfolder, exist_ok=True)
                self.signals.log_signal.emit(f"Created subfolder: {os.path.basename(subfolder)}", "INFO")
            else:
                subfolder = save_folder

            # Wait first delay
            if first_delay_ms > 0:
                self.signals.log_signal.emit(f"Waiting {first_delay_ms}ms before capture...", "INFO")
                time.sleep(first_delay_ms / 1000.0)

            # Capture loop
            captured_images = []
            file_paths = []

            for i in range(num_images):
                # Capture image
                img = self.camera.capture_image()

                if img is not None:
                    # Generate filename
                    if num_images > 1:
                        filename = os.path.join(subfolder, f"image_{i+1:02d}.png")
                    else:
                        filename = os.path.join(subfolder, f"trigger_{self.capture_number:03d}_{timestamp}.png")

                    # Save image (convert BGR to RGB for PIL)
                    img_rgb = img[:, :, ::-1]  # BGR to RGB
                    pil_img = Image.fromarray(img_rgb)
                    pil_img.save(filename)
                    captured_images.append(img)
                    file_paths.append(filename)

                    # Emit progress
                    self.signals.image_captured_signal.emit(img, i+1, num_images)
                    self.signals.log_signal.emit(
                        f"Captured {i+1}/{num_images}: {os.path.basename(filename)} ({img.shape[1]}x{img.shape[0]}px)",
                        "SUCCESS"
                    )

                    # Delay between captures
                    if i < num_images - 1:
                        time.sleep(delay_between_ms / 1000.0)
                else:
                    self.signals.log_signal.emit(f"Capture {i+1}/{num_images} failed", "ERROR")

            # Emit completion
            if captured_images:
                self.signals.capture_complete_signal.emit(len(captured_images), file_paths)
                self.signals.log_signal.emit(
                    f"Capture complete: {len(captured_images)}/{num_images} images saved",
                    "SUCCESS"
                )
            else:
                self.signals.capture_error_signal.emit("All captures failed")

        except Exception as e:
            self.signals.capture_error_signal.emit(str(e))
            self.signals.log_signal.emit(f"Capture worker error: {e}", "ERROR")


# ============================================================================
# MAIN WINDOW
# ============================================================================

class MainWindow(QMainWindow):
    """Main application window"""

    def __init__(self):
        super().__init__()

        self.setWindowTitle("GPIO Camera Trigger System")
        self.setGeometry(100, 100, 1400, 900)

        # Initialize signals
        self.signals = GPIOSignals()

        # Initialize controllers
        self.gpio_controller = GPIOController(self.signals)
        self.camera_controller = CameraController(self.signals)

        # State
        self.monitoring_active = False
        self.capture_counter = 0
        self.capture_worker = None

        # Setup UI
        self.setup_ui()

        # Connect signals
        self.connect_signals()

        # Initial log
        self.add_log("GPIO Camera Trigger System initialized", "INFO")
        self.add_log("Configure settings and connect camera to start", "INFO")

    def setup_ui(self):
        """Setup main user interface"""
        # Set dark theme
        self.set_dark_theme()

        # Central widget
        central = QWidget()
        self.setCentralWidget(central)

        main_layout = QHBoxLayout()
        central.setLayout(main_layout)

        # Left panel (configuration)
        left_panel = self.create_left_panel()
        main_layout.addWidget(left_panel, stretch=1)

        # Right panel (display and log)
        right_panel = self.create_right_panel()
        main_layout.addWidget(right_panel, stretch=2)

        # Status bar
        self.create_status_bar()

    def create_left_panel(self):
        """Create left configuration panel"""
        panel = QWidget()
        panel.setMaximumWidth(500)
        panel.setMinimumWidth(450)
        layout = QVBoxLayout()
        panel.setLayout(layout)

        # GPIO Trigger Input Group
        gpio_group = QGroupBox("GPIO Trigger Input")
        gpio_layout = QVBoxLayout()

        # Pin selection
        pin_layout = QHBoxLayout()
        pin_layout.addWidget(QLabel("Trigger Pin:"))
        self.trigger_pin_combo = QComboBox()
        for pin, desc in self.gpio_controller.available_pins.items():
            self.trigger_pin_combo.addItem(f"Pin {pin} - {desc}", pin)
        self.trigger_pin_combo.setCurrentIndex(7)  # Default Pin 32
        pin_layout.addWidget(self.trigger_pin_combo)
        gpio_layout.addLayout(pin_layout)

        # Edge type
        edge_label = QLabel("Edge Detection:")
        gpio_layout.addWidget(edge_label)

        edge_layout = QHBoxLayout()
        self.edge_button_group = QButtonGroup()

        self.rising_radio = QRadioButton("RISING (0→1)")
        self.rising_radio.setChecked(True)
        self.edge_button_group.addButton(self.rising_radio, 0)
        edge_layout.addWidget(self.rising_radio)

        self.falling_radio = QRadioButton("FALLING (1→0)")
        self.edge_button_group.addButton(self.falling_radio, 1)
        edge_layout.addWidget(self.falling_radio)

        self.both_radio = QRadioButton("BOTH")
        self.edge_button_group.addButton(self.both_radio, 2)
        edge_layout.addWidget(self.both_radio)

        gpio_layout.addLayout(edge_layout)

        # Debounce
        debounce_layout = QHBoxLayout()
        debounce_layout.addWidget(QLabel("Debounce (ms):"))
        self.debounce_spin = QSpinBox()
        self.debounce_spin.setRange(0, 1000)
        self.debounce_spin.setValue(200)
        self.debounce_spin.setSingleStep(50)
        debounce_layout.addWidget(self.debounce_spin)
        debounce_layout.addStretch()
        gpio_layout.addLayout(debounce_layout)

        gpio_group.setLayout(gpio_layout)
        layout.addWidget(gpio_group)

        # Camera Settings Group
        camera_group = QGroupBox("Camera Settings")
        camera_layout = QVBoxLayout()

        # Exposure
        exp_layout = QHBoxLayout()
        exp_layout.addWidget(QLabel("Exposure (µs):"))
        self.exposure_spin = QSpinBox()
        self.exposure_spin.setRange(100, 1000000)
        self.exposure_spin.setValue(10000)
        self.exposure_spin.setSingleStep(1000)
        exp_layout.addWidget(self.exposure_spin)
        exp_layout.addStretch()
        camera_layout.addLayout(exp_layout)

        # Gain
        gain_layout = QHBoxLayout()
        gain_layout.addWidget(QLabel("Gain:"))
        self.gain_spin = QDoubleSpinBox()
        self.gain_spin.setRange(0.0, 24.0)
        self.gain_spin.setValue(0.0)
        self.gain_spin.setSingleStep(0.1)
        gain_layout.addWidget(self.gain_spin)
        gain_layout.addStretch()
        camera_layout.addLayout(gain_layout)

        # Images per trigger
        img_layout = QHBoxLayout()
        img_layout.addWidget(QLabel("Images/Trigger:"))
        self.images_spin = QSpinBox()
        self.images_spin.setRange(1, 10)
        self.images_spin.setValue(1)
        img_layout.addWidget(self.images_spin)
        img_layout.addStretch()
        camera_layout.addLayout(img_layout)

        # Delay between images
        delay_layout = QHBoxLayout()
        delay_layout.addWidget(QLabel("Delay Between (ms):"))
        self.delay_spin = QSpinBox()
        self.delay_spin.setRange(0, 5000)
        self.delay_spin.setValue(100)
        self.delay_spin.setSingleStep(10)
        delay_layout.addWidget(self.delay_spin)
        delay_layout.addStretch()
        camera_layout.addLayout(delay_layout)

        # First delay
        first_delay_layout = QHBoxLayout()
        first_delay_layout.addWidget(QLabel("First Delay (ms):"))
        self.first_delay_spin = QSpinBox()
        self.first_delay_spin.setRange(0, 5000)
        self.first_delay_spin.setValue(0)
        self.first_delay_spin.setSingleStep(10)
        first_delay_layout.addWidget(self.first_delay_spin)
        first_delay_layout.addStretch()
        camera_layout.addLayout(first_delay_layout)

        # Save folder
        folder_layout = QHBoxLayout()
        folder_layout.addWidget(QLabel("Save Folder:"))
        self.folder_edit = QLineEdit("images")
        folder_layout.addWidget(self.folder_edit)
        self.browse_btn = QPushButton("...")
        self.browse_btn.setMaximumWidth(40)
        self.browse_btn.clicked.connect(self.browse_folder)
        folder_layout.addWidget(self.browse_btn)
        camera_layout.addLayout(folder_layout)

        camera_group.setLayout(camera_layout)
        layout.addWidget(camera_group)

        # Reject System Group
        reject_group = QGroupBox("Reject Output Control")
        reject_layout = QVBoxLayout()

        # Enable checkbox
        self.reject_enable_check = QCheckBox("Enable Reject System")
        self.reject_enable_check.setChecked(True)
        reject_layout.addWidget(self.reject_enable_check)

        # Output pin
        out_pin_layout = QHBoxLayout()
        out_pin_layout.addWidget(QLabel("Output Pin:"))
        self.reject_pin_combo = QComboBox()
        for pin, desc in self.gpio_controller.available_pins.items():
            self.reject_pin_combo.addItem(f"Pin {pin} - {desc}", pin)
        self.reject_pin_combo.setCurrentIndex(8)  # Default Pin 36
        out_pin_layout.addWidget(self.reject_pin_combo)
        reject_layout.addLayout(out_pin_layout)

        # Delay before reject
        reject_delay_layout = QHBoxLayout()
        reject_delay_layout.addWidget(QLabel("Delay Before (ms):"))
        self.reject_delay_spin = QSpinBox()
        self.reject_delay_spin.setRange(0, 5000)
        self.reject_delay_spin.setValue(0)
        self.reject_delay_spin.setSingleStep(10)
        reject_delay_layout.addWidget(self.reject_delay_spin)
        reject_delay_layout.addStretch()
        reject_layout.addLayout(reject_delay_layout)

        # Pulse width
        pulse_layout = QHBoxLayout()
        pulse_layout.addWidget(QLabel("Pulse Width (ms):"))
        self.pulse_spin = QSpinBox()
        self.pulse_spin.setRange(10, 1000)
        self.pulse_spin.setValue(100)
        self.pulse_spin.setSingleStep(10)
        pulse_layout.addWidget(self.pulse_spin)
        pulse_layout.addStretch()
        reject_layout.addLayout(pulse_layout)

        reject_group.setLayout(reject_layout)
        layout.addWidget(reject_group)

        # Control Buttons
        btn_layout = QVBoxLayout()

        self.connect_camera_btn = QPushButton("Connect Camera")
        self.connect_camera_btn.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold; padding: 8px;")
        self.connect_camera_btn.clicked.connect(self.toggle_camera_connection)
        btn_layout.addWidget(self.connect_camera_btn)

        self.apply_settings_btn = QPushButton("Apply Camera Settings")
        self.apply_settings_btn.setEnabled(False)
        self.apply_settings_btn.clicked.connect(self.apply_camera_settings)
        btn_layout.addWidget(self.apply_settings_btn)

        self.test_capture_btn = QPushButton("Test Manual Capture")
        self.test_capture_btn.setEnabled(False)
        self.test_capture_btn.clicked.connect(self.test_manual_capture)
        btn_layout.addWidget(self.test_capture_btn)

        self.start_monitor_btn = QPushButton("Start Monitoring")
        self.start_monitor_btn.setStyleSheet("background-color: #FF9800; color: white; font-weight: bold; padding: 10px;")
        self.start_monitor_btn.setEnabled(False)
        self.start_monitor_btn.clicked.connect(self.toggle_monitoring)
        btn_layout.addWidget(self.start_monitor_btn)

        self.clear_log_btn = QPushButton("Clear Log")
        self.clear_log_btn.clicked.connect(lambda: self.log_text.clear())
        btn_layout.addWidget(self.clear_log_btn)

        layout.addLayout(btn_layout)
        layout.addStretch()

        return panel

    def create_right_panel(self):
        """Create right display and log panel"""
        panel = QWidget()
        layout = QVBoxLayout()
        panel.setLayout(layout)

        # Image Display
        image_group = QGroupBox("Last Captured Image")
        image_layout = QVBoxLayout()

        self.image_label = QLabel("No image captured")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumHeight(500)
        self.image_label.setStyleSheet("background-color: #1e1e1e; color: #888888; border: 2px solid #444444;")
        image_layout.addWidget(self.image_label)

        image_group.setLayout(image_layout)
        layout.addWidget(image_group, stretch=2)

        # Event Log
        log_group = QGroupBox("Event Log")
        log_layout = QVBoxLayout()

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Courier", 9))
        self.log_text.setStyleSheet("background-color: #1e1e1e; color: #00ff00;")
        self.log_text.setMinimumHeight(200)
        log_layout.addWidget(self.log_text)

        log_group.setLayout(log_layout)
        layout.addWidget(log_group, stretch=1)

        return panel

    def create_status_bar(self):
        """Create status bar"""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        self.camera_status_label = QLabel("Camera: ⚫ Disconnected")
        self.gpio_status_label = QLabel("GPIO: ⚫ Ready")
        self.monitor_status_label = QLabel("Monitoring: ⚫ Stopped")
        self.capture_counter_label = QLabel("Total Captures: 0")

        self.status_bar.addWidget(self.camera_status_label)
        self.status_bar.addWidget(QLabel(" | "))
        self.status_bar.addWidget(self.gpio_status_label)
        self.status_bar.addWidget(QLabel(" | "))
        self.status_bar.addWidget(self.monitor_status_label)
        self.status_bar.addWidget(QLabel(" | "))
        self.status_bar.addWidget(self.capture_counter_label)

    def connect_signals(self):
        """Connect all signals to slots"""
        self.signals.log_signal.connect(self.add_log)
        self.signals.trigger_detected_signal.connect(self.on_trigger_detected)
        self.signals.image_captured_signal.connect(self.on_image_captured)
        self.signals.capture_complete_signal.connect(self.on_capture_complete)
        self.signals.capture_error_signal.connect(self.on_capture_error)
        self.signals.status_update_signal.connect(self.on_status_update)
        self.signals.capture_count_signal.connect(self.on_capture_count)

    def set_dark_theme(self):
        """Apply dark theme to window"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2b2b2b;
            }
            QWidget {
                background-color: #2b2b2b;
                color: white;
            }
            QGroupBox {
                border: 2px solid #444444;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QPushButton {
                padding: 5px;
                border-radius: 3px;
            }
            QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
                padding: 3px;
                background-color: #1e1e1e;
                border: 1px solid #444444;
            }
        """)

    # ========================================================================
    # SLOTS
    # ========================================================================

    def add_log(self, message, level="INFO"):
        """Add message to event log with color coding"""
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]

        color_map = {
            "INFO": "#00ff00",
            "SUCCESS": "#00ffff",
            "EVENT": "#ffff00",
            "ERROR": "#ff0000",
            "WARN": "#ff9900"
        }

        color = color_map.get(level, "#ffffff")
        formatted = f'<span style="color: {color};">[{timestamp}] [{level:7s}] {message}</span>'

        self.log_text.append(formatted)

        # Auto-scroll
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def on_trigger_detected(self, edge_type):
        """Handle GPIO trigger detection (main thread)"""
        try:
            # Check if camera is connected
            if not self.camera_controller.connected:
                self.add_log("TRIGGER IGNORED: Camera not connected", "WARN")
                return

            # Check if previous capture still running
            if self.capture_worker is not None and self.capture_worker.isRunning():
                self.add_log("TRIGGER IGNORED: Previous capture still running", "WARN")
                return

            # Increment counter
            self.capture_counter += 1
            self.signals.capture_count_signal.emit(self.capture_counter)

            # Log trigger
            self.add_log(f"TRIGGER #{self.capture_counter} DETECTED ({edge_type}) - Starting capture...", "EVENT")

            # Start capture sequence
            self.start_capture_sequence()

        except Exception as e:
            self.add_log(f"Error processing trigger: {e}", "ERROR")

    def on_image_captured(self, img_array, current_idx, total_count):
        """Handle image captured signal"""
        # Display the image
        self.display_image(img_array)

    def on_capture_complete(self, total_images, file_paths):
        """Handle capture complete signal"""
        # Trigger reject if enabled AND output pin is configured
        if self.reject_enable_check.isChecked() and self.gpio_controller.output_pin is not None:
            delay_ms = self.reject_delay_spin.value()
            pulse_ms = self.pulse_spin.value()
            self.gpio_controller.kick_reject_pulse(pulse_ms, delay_ms)

    def on_capture_error(self, error_msg):
        """Handle capture error signal"""
        self.add_log(f"Capture error: {error_msg}", "ERROR")

    def on_status_update(self, component, status):
        """Handle status update signal"""
        if component == "camera":
            if status == "connected":
                self.camera_status_label.setText("Camera: 🟢 Connected")
            else:
                self.camera_status_label.setText("Camera: ⚫ Disconnected")

    def on_capture_count(self, count):
        """Handle capture count update"""
        self.capture_counter_label.setText(f"Total Captures: {count}")

    # ========================================================================
    # ACTIONS
    # ========================================================================

    def browse_folder(self):
        """Browse for save folder"""
        folder = QFileDialog.getExistingDirectory(self, "Select Save Folder")
        if folder:
            self.folder_edit.setText(folder)

    def toggle_camera_connection(self):
        """Toggle camera connection"""
        if not self.camera_controller.connected:
            # Connect
            exposure = self.exposure_spin.value()
            gain = self.gain_spin.value()

            if self.camera_controller.connect(exposure, gain):
                self.connect_camera_btn.setText("Disconnect Camera")
                self.connect_camera_btn.setStyleSheet("background-color: #f44336; color: white; font-weight: bold; padding: 8px;")
                self.apply_settings_btn.setEnabled(True)
                self.test_capture_btn.setEnabled(True)
                self.start_monitor_btn.setEnabled(True)
        else:
            # Disconnect
            if self.monitoring_active:
                self.toggle_monitoring()

            self.camera_controller.disconnect()
            self.connect_camera_btn.setText("Connect Camera")
            self.connect_camera_btn.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold; padding: 8px;")
            self.apply_settings_btn.setEnabled(False)
            self.test_capture_btn.setEnabled(False)
            self.start_monitor_btn.setEnabled(False)

    def apply_camera_settings(self):
        """Apply camera settings"""
        exposure = self.exposure_spin.value()
        gain = self.gain_spin.value()
        self.camera_controller.apply_settings(exposure, gain)
        self.add_log("Camera settings applied", "SUCCESS")

    def test_manual_capture(self):
        """Test manual capture"""
        self.add_log("Testing manual capture...", "INFO")
        self.capture_counter += 1
        self.signals.capture_count_signal.emit(self.capture_counter)
        self.start_capture_sequence()

    def toggle_monitoring(self):
        """Toggle GPIO monitoring"""
        if not self.monitoring_active:
            # Start monitoring
            trigger_pin = self.trigger_pin_combo.currentData()

            # Get edge type
            if self.rising_radio.isChecked():
                edge_type = GPIO.RISING
                edge_name = "RISING (0→1)"
            elif self.falling_radio.isChecked():
                edge_type = GPIO.FALLING
                edge_name = "FALLING (1→0)"
            else:
                edge_type = GPIO.BOTH
                edge_name = "BOTH"

            debounce = self.debounce_spin.value()

            # Setup GPIO trigger
            if self.gpio_controller.setup_trigger_input(trigger_pin, edge_type, debounce):
                # Setup reject output
                if self.reject_enable_check.isChecked():
                    reject_pin = self.reject_pin_combo.currentData()
                    self.gpio_controller.setup_reject_output(reject_pin)

                self.monitoring_active = True
                self.start_monitor_btn.setText("Stop Monitoring")
                self.start_monitor_btn.setStyleSheet("background-color: #f44336; color: white; font-weight: bold; padding: 10px;")
                self.monitor_status_label.setText("Monitoring: 🟢 Active")
                self.add_log(f"Monitoring started - Waiting for {edge_name} on Pin {trigger_pin}", "EVENT")
        else:
            # Stop monitoring
            self.gpio_controller.remove_event_detection()
            self.monitoring_active = False
            self.start_monitor_btn.setText("Start Monitoring")
            self.start_monitor_btn.setStyleSheet("background-color: #FF9800; color: white; font-weight: bold; padding: 10px;")
            self.monitor_status_label.setText("Monitoring: ⚫ Stopped")
            self.add_log("Monitoring stopped", "INFO")

    def start_capture_sequence(self):
        """Start capture sequence in background thread"""
        # Build config
        config = {
            'num_images': self.images_spin.value(),
            'delay_between_ms': self.delay_spin.value(),
            'first_delay_ms': self.first_delay_spin.value(),
            'save_folder': self.folder_edit.text()
        }

        # Create and start worker
        self.capture_worker = CaptureWorker(
            self.camera_controller,
            config,
            self.signals,
            self.capture_counter
        )
        self.capture_worker.start()

    def display_image(self, img_array):
        """Display captured image in UI"""
        try:
            # Convert BGR to RGB
            rgb_img = img_array[:, :, ::-1]  # BGR to RGB using numpy slicing

            # Convert to PIL Image
            pil_img = Image.fromarray(rgb_img)

            # Get label dimensions
            label_w = self.image_label.width()
            label_h = self.image_label.height()

            if label_w <= 1:
                label_w = 800
            if label_h <= 1:
                label_h = 600

            # Calculate scale
            img_w, img_h = pil_img.size
            scale = min(label_w / img_w, label_h / img_h, 1.0)

            new_w = int(img_w * scale)
            new_h = int(img_h * scale)

            # Resize using PIL (use LANCZOS for older Pillow versions)
            try:
                resized = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            except AttributeError:
                # Fallback for older Pillow versions
                resized = pil_img.resize((new_w, new_h), Image.LANCZOS)

            # Convert to QPixmap
            resized_array = np.array(resized)
            height, width, channel = resized_array.shape
            bytes_per_line = 3 * width
            q_image = QImage(resized_array.data, width, height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)

            # Display
            self.image_label.setPixmap(pixmap)

        except Exception as e:
            self.add_log(f"Display error: {e}", "ERROR")

    def closeEvent(self, event):
        """Handle window close"""
        reply = QMessageBox.question(
            self,
            'Quit',
            'Are you sure you want to quit?',
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            # Stop monitoring
            if self.monitoring_active:
                self.gpio_controller.remove_event_detection()

            # Wait for worker threads
            if self.capture_worker is not None and self.capture_worker.isRunning():
                self.capture_worker.wait(2000)

            # Disconnect camera
            if self.camera_controller.connected:
                self.camera_controller.disconnect()

            # Cleanup GPIO
            self.gpio_controller.cleanup()

            event.accept()
        else:
            event.ignore()


# ============================================================================
# MAIN
# ============================================================================

def main():
    app = QApplication([])
    app.setStyle('Fusion')

    window = MainWindow()
    window.show()

    app.exec_()


if __name__ == "__main__":
    main()
