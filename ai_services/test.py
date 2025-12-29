from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QPushButton, QRadioButton,
                             QGroupBox, QTextEdit, QScrollArea, QFrame,
                             QButtonGroup, QMessageBox, QTabWidget, QComboBox,
                             QSpinBox, QCheckBox, QSlider)
from PyQt5.QtCore import QTimer, Qt, pyqtSignal, QObject
from PyQt5.QtGui import QFont, QPalette, QColor
import sys
import Jetson.GPIO as GPIO
from datetime import datetime
import threading

class GPIOSignals(QObject):
    """Signals for thread-safe GUI updates"""
    log_signal = pyqtSignal(str, str)
    pin_update_signal = pyqtSignal(int, int)
    event_detected_signal = pyqtSignal(int, str)  # pin, edge_type

class GPIOController:
    """GPIO controller class"""
    def __init__(self, signals):
        self.signals = signals
        GPIO.setwarnings(False)
        GPIO.setmode(GPIO.BOARD)

        self.pins = {
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

        self.pin_directions = {}
        self.pin_states = {}
        self.event_detections = {}  # Track which pins have event detection

        # Initialize all pins as INPUT first
        for pin in self.pins.keys():
            try:
                # Try to remove any existing event detection silently
                try:
                    GPIO.remove_event_detect(pin)
                except:
                    pass

                # Setup as input
                GPIO.setup(pin, GPIO.IN)
                self.pin_directions[pin] = "IN"
                self.pin_states[pin] = GPIO.input(pin)
            except Exception as e:
                # If setup fails, log but continue
                print(f"Warning: Could not setup pin {pin}: {e}")
    
    def setup(self, pin, direction):
        """Setup pin direction"""
        if direction == "IN":
            GPIO.setup(pin, GPIO.IN)
        else:
            GPIO.setup(pin, GPIO.OUT)
        self.pin_directions[pin] = direction
    
    def input(self, pin):
        """Read pin value"""
        value = GPIO.input(pin)
        self.pin_states[pin] = value
        return value
    
    def output(self, pin, value):
        """Set output value"""
        if self.pin_directions[pin] == "OUT":
            GPIO.output(pin, value)
            self.pin_states[pin] = value
            return True
        return False
    
    def add_event_detect(self, pin, edge, callback=None, bouncetime=200):
        """Add event detection to a pin"""
        try:
            # Remove existing detection if any
            if pin in self.event_detections:
                GPIO.remove_event_detect(pin)
            
            # Add new detection
            if callback:
                GPIO.add_event_detect(pin, edge, callback=callback, bouncetime=bouncetime)
            else:
                GPIO.add_event_detect(pin, edge, bouncetime=bouncetime)
            
            self.event_detections[pin] = {
                'edge': edge,
                'bouncetime': bouncetime,
                'has_callback': callback is not None
            }
            return True
        except Exception as e:
            print(f"Error adding event detect: {e}")
            return False
    
    def remove_event_detect(self, pin):
        """Remove event detection from a pin"""
        try:
            # Only try to remove if we have it tracked
            if pin in self.event_detections:
                GPIO.remove_event_detect(pin)
                del self.event_detections[pin]
                return True
            return False
        except Exception as e:
            # Silently handle errors - event might already be removed
            if pin in self.event_detections:
                del self.event_detections[pin]
            return False
    
    def add_event_callback(self, pin, callback):
        """Add additional callback to existing event detection"""
        try:
            GPIO.add_event_callback(pin, callback)
            return True
        except Exception as e:
            print(f"Error adding callback: {e}")
            return False
    
    def event_detected(self, pin):
        """Check if event was detected (for polling method)"""
        try:
            return GPIO.event_detected(pin)
        except:
            return False
    
    def wait_for_edge(self, pin, edge, timeout=5000):
        """Wait for edge (blocking)"""
        try:
            # IMPORTANT: Ensure pin is setup as INPUT first
            if self.pin_directions.get(pin) != "IN":
                GPIO.setup(pin, GPIO.IN)
                self.pin_directions[pin] = "IN"
            
            return GPIO.wait_for_edge(pin, edge, timeout=timeout)
        except Exception as e:
            print(f"Error waiting for edge: {e}")
            return None
    
    def cleanup(self):
        """Cleanup GPIO"""
        GPIO.cleanup()

class PinControlWidget(QFrame):
    """Widget for controlling a single pin"""
    def __init__(self, pin, description, controller, signals):
        super().__init__()
        self.pin = pin
        self.description = description
        self.controller = controller
        self.signals = signals
        
        self.setup_ui()
    
    def setup_ui(self):
        """Setup UI for pin control"""
        self.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        self.setLineWidth(2)
        
        layout = QHBoxLayout()
        
        # Pin label
        pin_label = QLabel(f"Pin {self.pin}")
        pin_label.setFont(QFont("Arial", 12, QFont.Bold))
        pin_label.setMinimumWidth(60)
        pin_label.setStyleSheet("color: #ffcc00;")
        layout.addWidget(pin_label)
        
        # Description
        desc_label = QLabel(self.description)
        desc_label.setMinimumWidth(200)
        layout.addWidget(desc_label)
        
        # Direction controls
        dir_group = QGroupBox("Direction")
        dir_layout = QHBoxLayout()
        
        self.dir_button_group = QButtonGroup()
        
        self.in_radio = QRadioButton("INPUT")
        self.in_radio.setChecked(True)
        self.in_radio.toggled.connect(lambda: self.on_direction_changed("IN"))
        self.dir_button_group.addButton(self.in_radio)
        dir_layout.addWidget(self.in_radio)
        
        self.out_radio = QRadioButton("OUTPUT")
        self.out_radio.toggled.connect(lambda: self.on_direction_changed("OUT"))
        self.dir_button_group.addButton(self.out_radio)
        dir_layout.addWidget(self.out_radio)
        
        dir_group.setLayout(dir_layout)
        layout.addWidget(dir_group)
        
        # Output controls
        self.high_btn = QPushButton("HIGH")
        self.high_btn.setStyleSheet("background-color: #00aa00; color: white; font-weight: bold;")
        self.high_btn.clicked.connect(lambda: self.set_output(1))
        self.high_btn.setEnabled(False)
        self.high_btn.setMinimumWidth(70)
        layout.addWidget(self.high_btn)
        
        self.low_btn = QPushButton("LOW")
        self.low_btn.setStyleSheet("background-color: #cc0000; color: white; font-weight: bold;")
        self.low_btn.clicked.connect(lambda: self.set_output(0))
        self.low_btn.setEnabled(False)
        self.low_btn.setMinimumWidth(70)
        layout.addWidget(self.low_btn)
        
        # Value display
        self.value_label = QLabel("●")
        self.value_label.setFont(QFont("Arial", 16, QFont.Bold))
        self.value_label.setAlignment(Qt.AlignCenter)
        self.value_label.setMinimumWidth(30)
        layout.addWidget(self.value_label)
        
        self.value_text = QLabel("LOW (0)")
        self.value_text.setFont(QFont("Courier", 10, QFont.Bold))
        self.value_text.setMinimumWidth(80)
        layout.addWidget(self.value_text)
        
        # NOW update the display
        self.update_value_display(self.controller.pin_states[self.pin])
        
        self.setLayout(layout)
    
    def on_direction_changed(self, direction):
        """Handle direction change"""
        try:
            self.controller.setup(self.pin, direction)
            
            # Enable/disable output buttons
            is_output = (direction == "OUT")
            self.high_btn.setEnabled(is_output)
            self.low_btn.setEnabled(is_output)
            
            self.signals.log_signal.emit(f"Pin {self.pin} set to {direction}", "INFO")
        except Exception as e:
            self.signals.log_signal.emit(f"Error setting pin {self.pin}: {e}", "ERROR")
    
    def set_output(self, value):
        """Set output value"""
        try:
            if self.controller.output(self.pin, value):
                state = "HIGH" if value else "LOW"
                self.signals.log_signal.emit(f"Pin {self.pin} set to {state}", "INFO")
                self.update_value_display(value)
            else:
                QMessageBox.warning(self, "Error", f"Pin {self.pin} is not OUTPUT")
        except Exception as e:
            self.signals.log_signal.emit(f"Error: {e}", "ERROR")
    
    def update_value_display(self, value):
        """Update value display"""
        if value:
            self.value_label.setStyleSheet("color: #00ff00;")
            self.value_text.setText("HIGH (1)")
        else:
            self.value_label.setStyleSheet("color: #ff0000;")
            self.value_text.setText("LOW (0)")

class EventsTab(QWidget):
    # Signal for thread-safe event logging
    event_log_signal = pyqtSignal(str)

    def __init__(self, controller, signals):
        super().__init__()

        self.controller = controller
        self.signals = signals

        # Setup UI first to create widgets
        self.setup_ui()

        # IMPORTANT: Connect signals AFTER setup_ui so widgets exist
        self.event_log_signal.connect(self.log_event)
        self.signals.event_detected_signal.connect(self.on_event_detected)
    
    def setup_ui(self):
        """Setup events tab UI"""
        layout = QVBoxLayout()
        
        # Event Detection Setup
        setup_group = QGroupBox("Event Detection Setup")
        setup_layout = QVBoxLayout()
        
        # Pin selection
        pin_layout = QHBoxLayout()
        pin_layout.addWidget(QLabel("Select Pin:"))
        self.pin_combo = QComboBox()
        for pin, desc in self.controller.pins.items():
            self.pin_combo.addItem(f"Pin {pin} - {desc}", pin)
        pin_layout.addWidget(self.pin_combo)
        pin_layout.addStretch()
        setup_layout.addLayout(pin_layout)
        
        # Edge selection
        edge_layout = QHBoxLayout()
        edge_layout.addWidget(QLabel("Edge Type:"))
        self.edge_combo = QComboBox()
        self.edge_combo.addItem("RISING", GPIO.RISING)
        self.edge_combo.addItem("FALLING", GPIO.FALLING)
        self.edge_combo.addItem("BOTH", GPIO.BOTH)
        edge_layout.addWidget(self.edge_combo)
        edge_layout.addStretch()
        setup_layout.addLayout(edge_layout)
        
        # Debounce time
        debounce_layout = QHBoxLayout()
        debounce_layout.addWidget(QLabel("Debounce Time (ms):"))
        self.debounce_spin = QSpinBox()
        self.debounce_spin.setRange(0, 1000)
        self.debounce_spin.setValue(200)
        self.debounce_spin.setSingleStep(50)
        debounce_layout.addWidget(self.debounce_spin)
        debounce_layout.addStretch()
        setup_layout.addLayout(debounce_layout)
        
        # Options
        self.callback_check = QCheckBox("Enable Callback (shows popup)")
        self.callback_check.setChecked(True)
        setup_layout.addWidget(self.callback_check)
        
        self.popup_check = QCheckBox("Show Popup on Event")
        self.popup_check.setChecked(True)
        setup_layout.addWidget(self.popup_check)
        
        # Buttons
        btn_layout = QHBoxLayout()
        
        add_btn = QPushButton("+ Add Event Detection")
        add_btn.setStyleSheet("background-color: #00aa00; color: white; font-weight: bold; padding: 8px;")
        add_btn.clicked.connect(self.add_event_detection)
        btn_layout.addWidget(add_btn)
        
        remove_btn = QPushButton("- Remove Event Detection")
        remove_btn.setStyleSheet("background-color: #cc0000; color: white; font-weight: bold; padding: 8px;")
        remove_btn.clicked.connect(self.remove_event_detection)
        btn_layout.addWidget(remove_btn)
        
        setup_layout.addLayout(btn_layout)
        setup_group.setLayout(setup_layout)
        layout.addWidget(setup_group)
        
        # Wait for Edge
        wait_group = QGroupBox("Wait for Edge (Blocking)")
        wait_layout = QVBoxLayout()
        
        wait_info = QLabel("This will block until an edge is detected or timeout occurs")
        wait_info.setStyleSheet("color: #ffaa00; font-style: italic;")
        wait_layout.addWidget(wait_info)
        
        wait_btn_layout = QHBoxLayout()
        
        wait_btn = QPushButton("Wait for Edge (5s timeout)")
        wait_btn.setStyleSheet("background-color: #0066cc; color: white; font-weight: bold; padding: 8px;")
        wait_btn.clicked.connect(self.wait_for_edge)
        wait_btn_layout.addWidget(wait_btn)
        
        wait_layout.addLayout(wait_btn_layout)
        wait_group.setLayout(wait_layout)
        layout.addWidget(wait_group)
        
        # Active Event Detections
        active_group = QGroupBox("Active Event Detections")
        active_layout = QVBoxLayout()
        
        self.active_text = QTextEdit()
        self.active_text.setReadOnly(True)
        self.active_text.setMaximumHeight(150)
        self.active_text.setStyleSheet("background-color: #1e1e1e; color: #00ffff;")
        active_layout.addWidget(self.active_text)
        
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self.update_active_detections)
        active_layout.addWidget(refresh_btn)
        
        active_group.setLayout(active_layout)
        layout.addWidget(active_group)
        
        # Event Log
        log_group = QGroupBox("Event Log")
        log_layout = QVBoxLayout()
        
        self.event_log = QTextEdit()
        self.event_log.setReadOnly(True)
        self.event_log.setStyleSheet("background-color: #1e1e1e; color: #ffff00;")
        log_layout.addWidget(self.event_log)
        
        clear_btn = QPushButton("Clear Log")
        clear_btn.clicked.connect(lambda: self.event_log.clear())
        log_layout.addWidget(clear_btn)
        
        log_group.setLayout(log_layout)
        layout.addWidget(log_group)

        self.setLayout(layout)

        # Update active detections
        self.update_active_detections()
    
    def remove_all_events(self):
        """Remove all event detections"""
        count = 0
        for pin in list(self.controller.event_detections.keys()):
            if self.controller.remove_event_detect(pin):
                count += 1

        if count > 0:
            msg = f"Removed {count} event detection(s)"
            self.signals.log_signal.emit(msg, "INFO")
            self.event_log_signal.emit(msg)
            self.update_active_detections()

        return count

    def add_event_detection(self):
        """Add event detection"""
        pin = self.pin_combo.currentData()
        edge = self.edge_combo.currentData()
        edge_name = self.edge_combo.currentText()
        debounce = self.debounce_spin.value()
        use_callback = self.callback_check.isChecked()

        # Auto-remove all existing event detections first
        removed_count = self.remove_all_events()
        if removed_count > 0:
            self.event_log_signal.emit(f"🧹 Auto-cleaned {removed_count} existing event detection(s)")

        # Make sure pin is INPUT
        if self.controller.pin_directions[pin] != "IN":
            self.controller.setup(pin, "IN")
            self.signals.log_signal.emit(f"Pin {pin} set to INPUT for event detection", "INFO")
        
        # Create callback if needed
        callback = None
        if use_callback:
            def event_callback(channel):
                # Emit signal for GUI update
                self.signals.event_detected_signal.emit(channel, edge_name)
            
            callback = event_callback
        
        # Add event detection
        success = self.controller.add_event_detect(pin, edge, callback, debounce)
        
        if success:
            msg = f"Event detection added: Pin {pin}, Edge: {edge_name}, Debounce: {debounce}ms"
            if use_callback:
                msg += ", Callback: YES"
            self.signals.log_signal.emit(msg, "SUCCESS")
            self.log_event(msg)
            self.update_active_detections()
            
            QMessageBox.information(self, "Success", 
                f"Event detection added successfully!\n\n"
                f"Pin: {pin}\n"
                f"Edge: {edge_name}\n"
                f"Debounce: {debounce}ms\n"
                f"Callback: {'Enabled' if use_callback else 'Disabled'}")
        else:
            QMessageBox.warning(self, "Error", "Failed to add event detection")
    
    def remove_event_detection(self):
        """Remove event detection"""
        pin = self.pin_combo.currentData()
        
        success = self.controller.remove_event_detect(pin)
        
        if success:
            msg = f"Event detection removed from Pin {pin}"
            self.signals.log_signal.emit(msg, "INFO")
            self.log_event(msg)
            self.update_active_detections()
            QMessageBox.information(self, "Success", f"Event detection removed from Pin {pin}")
        else:
            QMessageBox.warning(self, "Error", "Failed to remove event detection")
    
    def wait_for_edge(self):
        """Wait for edge (blocking - in thread)"""
        pin = self.pin_combo.currentData()
        edge = self.edge_combo.currentData()
        edge_name = self.edge_combo.currentText()

        # Auto-remove all existing event detections first to avoid conflicts
        removed_count = self.remove_all_events()
        if removed_count > 0:
            self.event_log_signal.emit(f"🧹 Auto-cleaned {removed_count} event detection(s) before waiting")

        self.event_log_signal.emit(f"⏳ Waiting for {edge_name} edge on Pin {pin}...")
        self.signals.log_signal.emit(f"Waiting for {edge_name} edge on Pin {pin}...", "INFO")

        def wait_thread():
            result = self.controller.wait_for_edge(pin, edge, timeout=5000)

            if result:
                msg = f"✓ {edge_name} edge detected on Pin {result}!"
                self.event_log_signal.emit(msg)
                self.signals.log_signal.emit(msg, "SUCCESS")
                self.signals.event_detected_signal.emit(result, edge_name)
            else:
                msg = f"⏱ Timeout: No {edge_name} edge detected on Pin {pin}"
                self.event_log_signal.emit(msg)
                self.signals.log_signal.emit(msg, "INFO")

        thread = threading.Thread(target=wait_thread, daemon=True)
        thread.start()
    
    def on_event_detected(self, pin, edge_type):
        """Handle event detected signal"""
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        msg = f"🔔 EVENT! Pin {pin} - {edge_type} edge detected"
        
        self.log_event(msg)
        self.signals.log_signal.emit(msg, "EVENT")
        
        # Show popup if enabled
        if self.popup_check.isChecked():
            msg_box = QMessageBox(self)
            msg_box.setIcon(QMessageBox.Information)
            msg_box.setWindowTitle("GPIO Event Detected!")
            msg_box.setText(f"Event detected on Pin {pin}")
            msg_box.setInformativeText(
                f"Edge Type: {edge_type}\n"
                f"Time: {timestamp}\n"
                f"Pin: {pin} ({self.controller.pins[pin]})"
            )
            msg_box.setStandardButtons(QMessageBox.Ok)
            msg_box.setStyleSheet("""
                QMessageBox {
                    background-color: #2b2b2b;
                }
                QLabel {
                    color: white;
                }
            """)
            msg_box.show()
    
    def update_active_detections(self):
        """Update active event detections display"""
        self.active_text.clear()
        
        if not self.controller.event_detections:
            self.active_text.append("No active event detections")
            return
        
        self.active_text.append("Active Event Detections:")
        self.active_text.append("="*50)
        
        for pin, info in self.controller.event_detections.items():
            edge_name = {
                GPIO.RISING: "RISING",
                GPIO.FALLING: "FALLING",
                GPIO.BOTH: "BOTH"
            }.get(info['edge'], "UNKNOWN")
            
            text = (f"Pin {pin} ({self.controller.pins[pin]})\n"
                   f"  Edge: {edge_name}\n"
                   f"  Debounce: {info['bouncetime']}ms\n"
                   f"  Callback: {'YES' if info['has_callback'] else 'NO'}\n")
            
            self.active_text.append(text)
    
    def log_event(self, message):
        """Add event to log"""
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]

        # Use append for simplicity
        self.event_log.append(f"[{timestamp}] {message}")

        # Auto scroll
        scrollbar = self.event_log.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

class GPIOMainWindow(QMainWindow):
    """Main application window"""
    def __init__(self):
        super().__init__()
        
        # Fix font rendering issues
        QApplication.setFont(QFont("Sans", 9))
        
        self.setWindowTitle("Jetson GPIO Control - Full Features")
        self.setGeometry(100, 100, 1400, 900)
        
        # Apply dark theme
        self.set_dark_theme()
        
        # Initialize signals
        self.signals = GPIOSignals()
        
        # Initialize GPIO
        self.controller = GPIOController(self.signals)
        self.signals.log_signal.connect(self.add_log)
        self.signals.pin_update_signal.connect(self.update_pin)
        
        # Setup UI
        self.setup_ui()
        
        # Setup monitoring timer
        self.monitor_timer = QTimer()
        self.monitor_timer.timeout.connect(self.monitor_pins)
        self.monitor_timer.start(300)
        
        self.add_log("GPIO initialized successfully!", "SUCCESS")
        self.add_log("You can now add event detections in the Events tab!", "INFO")
    
    def set_dark_theme(self):
        """Apply dark theme"""
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(43, 43, 43))
        palette.setColor(QPalette.WindowText, Qt.white)
        palette.setColor(QPalette.Base, QColor(30, 30, 30))
        palette.setColor(QPalette.AlternateBase, QColor(43, 43, 43))
        palette.setColor(QPalette.ToolTipBase, Qt.white)
        palette.setColor(QPalette.ToolTipText, Qt.white)
        palette.setColor(QPalette.Text, Qt.white)
        palette.setColor(QPalette.Button, QColor(60, 60, 60))
        palette.setColor(QPalette.ButtonText, Qt.white)
        palette.setColor(QPalette.BrightText, Qt.red)
        palette.setColor(QPalette.Highlight, QColor(0, 102, 204))
        palette.setColor(QPalette.HighlightedText, Qt.black)
        
        self.setPalette(palette)
    
    def setup_ui(self):
        """Setup main UI"""
        central = QWidget()
        main_layout = QHBoxLayout()
        
        # Left panel - Pin controls
        left_panel = self.create_left_panel()
        main_layout.addWidget(left_panel, stretch=2)
        
        # Right panel - Tabs
        right_panel = self.create_right_panel()
        main_layout.addWidget(right_panel, stretch=3)
        
        central.setLayout(main_layout)
        self.setCentralWidget(central)
    
    def create_left_panel(self):
        """Create left control panel"""
        panel = QGroupBox("GPIO Pin Controls")
        panel.setFont(QFont("Arial", 14, QFont.Bold))
        layout = QVBoxLayout()
        
        # Info section
        try:
            version = GPIO.VERSION
        except:
            version = "Unknown"
        
        info_label = QLabel(
            f"Model: JETSON_ORIN_NX\n"
            f"GPIO Version: {version}\n"
            f"Available Pins: {len(self.controller.pins)}\n"
            f"Mode: BOARD"
        )
        info_label.setFont(QFont("Courier", 10))
        info_label.setStyleSheet("color: #00ff00; padding: 10px; background-color: #1e1e1e;")
        layout.addWidget(info_label)
        
        # Scrollable pin controls
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumHeight(600)
        
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout()
        
        self.pin_widgets = {}
        for pin, desc in self.controller.pins.items():
            widget = PinControlWidget(pin, desc, self.controller, self.signals)
            self.pin_widgets[pin] = widget
            scroll_layout.addWidget(widget)
        
        scroll_layout.addStretch()
        scroll_widget.setLayout(scroll_layout)
        scroll.setWidget(scroll_widget)
        
        layout.addWidget(scroll)
        panel.setLayout(layout)
        
        return panel
    
    def create_right_panel(self):
        """Create right panel with tabs"""
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 2px solid #3c3c3c;
                background: #2b2b2b;
            }
            QTabBar::tab {
                background: #3c3c3c;
                color: white;
                padding: 8px 20px;
                margin: 2px;
            }
            QTabBar::tab:selected {
                background: #0066cc;
            }
        """)

        # Monitor Tab
        monitor_tab = self.create_monitor_tab()
        self.tabs.addTab(monitor_tab, "📊 Monitor")

        # Events Tab
        self.events_tab = EventsTab(self.controller, self.signals)
        self.tabs.addTab(self.events_tab, "🔔 Events/Interrupts")

        # Connect tab change to auto-cleanup events
        self.tabs.currentChanged.connect(self.on_tab_changed)

        return self.tabs
    
    def create_monitor_tab(self):
        """Create monitor tab"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # Title
        title = QLabel("Activity Monitor")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # Control buttons
        btn_layout = QHBoxLayout()
        
        self.monitor_btn = QPushButton("⏸ Pause")
        self.monitor_btn.setStyleSheet("background-color: #0066cc; color: white; font-weight: bold; padding: 5px;")
        self.monitor_btn.clicked.connect(self.toggle_monitoring)
        btn_layout.addWidget(self.monitor_btn)
        
        clear_btn = QPushButton("🗑 Clear")
        clear_btn.setStyleSheet("background-color: #cc6600; color: white; font-weight: bold; padding: 5px;")
        clear_btn.clicked.connect(self.clear_log)
        btn_layout.addWidget(clear_btn)
        
        read_btn = QPushButton("📊 Read All")
        read_btn.setStyleSheet("background-color: #6600cc; color: white; font-weight: bold; padding: 5px;")
        read_btn.clicked.connect(self.read_all)
        btn_layout.addWidget(read_btn)
        
        layout.addLayout(btn_layout)
        
        # Log display
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Courier", 9))
        self.log_text.setStyleSheet("background-color: #1e1e1e; color: #00ff00;")
        layout.addWidget(self.log_text)
        
        # Quick actions
        actions_group = QGroupBox("Quick Actions")
        actions_layout = QVBoxLayout()
        
        btn_row1 = QHBoxLayout()
        
        all_in_btn = QPushButton("All INPUT")
        all_in_btn.setStyleSheet("background-color: #0066cc; color: white; padding: 8px;")
        all_in_btn.clicked.connect(lambda: self.batch_direction("IN"))
        btn_row1.addWidget(all_in_btn)
        
        all_out_btn = QPushButton("All OUTPUT")
        all_out_btn.setStyleSheet("background-color: #0066cc; color: white; padding: 8px;")
        all_out_btn.clicked.connect(lambda: self.batch_direction("OUT"))
        btn_row1.addWidget(all_out_btn)
        
        actions_layout.addLayout(btn_row1)
        
        btn_row2 = QHBoxLayout()
        
        all_high_btn = QPushButton("All HIGH")
        all_high_btn.setStyleSheet("background-color: #00aa00; color: white; padding: 8px;")
        all_high_btn.clicked.connect(lambda: self.batch_output(1))
        btn_row2.addWidget(all_high_btn)
        
        all_low_btn = QPushButton("All LOW")
        all_low_btn.setStyleSheet("background-color: #cc0000; color: white; padding: 8px;")
        all_low_btn.clicked.connect(lambda: self.batch_output(0))
        btn_row2.addWidget(all_low_btn)
        
        actions_layout.addLayout(btn_row2)
        
        # Blink test
        blink_btn = QPushButton("Blink Test (Pin 32)")
        blink_btn.setStyleSheet("background-color: #cc6600; color: white; padding: 8px;")
        blink_btn.clicked.connect(self.blink_test)
        actions_layout.addWidget(blink_btn)
        
        actions_group.setLayout(actions_layout)
        layout.addWidget(actions_group)
        
        widget.setLayout(layout)
        return widget
    
    def monitor_pins(self):
        """Monitor pin changes"""
        if not self.monitor_timer.isActive():
            return
        
        try:
            for pin in self.controller.pins.keys():
                if self.controller.pin_directions[pin] == "IN":
                    old_value = self.controller.pin_states[pin]
                    new_value = self.controller.input(pin)
                    
                    if new_value != old_value:
                        change = "LOW→HIGH" if new_value else "HIGH→LOW"
                        self.add_log(f"Pin {pin}: {change}", "CHANGE")
                        self.pin_widgets[pin].update_value_display(new_value)
        except Exception as e:
            self.add_log(f"Monitor error: {e}", "ERROR")
    
    def toggle_monitoring(self):
        """Toggle monitoring"""
        if self.monitor_timer.isActive():
            self.monitor_timer.stop()
            self.monitor_btn.setText("▶ Resume")
            self.add_log("Monitoring paused", "INFO")
        else:
            self.monitor_timer.start(300)
            self.monitor_btn.setText("⏸ Pause")
            self.add_log("Monitoring resumed", "INFO")
    
    def add_log(self, message, level="INFO"):
        """Add log message"""
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        
        color_map = {
            "INFO": "#00ff00",
            "CHANGE": "#ffff00",
            "ERROR": "#ff0000",
            "SUCCESS": "#00ffff",
            "EVENT": "#ff00ff"
        }
        
        color = color_map.get(level, "#ffffff")
        formatted = f'<span style="color: {color};">[{timestamp}] [{level:7s}] {message}</span>'
        
        self.log_text.append(formatted)
        
        # Auto-scroll
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def clear_log(self):
        """Clear log"""
        self.log_text.clear()
        self.add_log("Log cleared", "INFO")
    
    def read_all(self):
        """Read all pins"""
        self.add_log("="*50, "INFO")
        
        for pin in sorted(self.controller.pins.keys()):
            value = self.controller.input(pin)
            direction = self.controller.pin_directions[pin]
            state = "HIGH (1)" if value else "LOW  (0)"
            desc = self.controller.pins[pin]
            
            self.add_log(f"Pin {pin:2d} | {direction:3s} | {state} | {desc}", "INFO")
        
        self.add_log("="*50, "INFO")
    
    def batch_direction(self, direction):
        """Set all pins direction"""
        try:
            for pin, widget in self.pin_widgets.items():
                self.controller.setup(pin, direction)
                
                if direction == "IN":
                    widget.in_radio.setChecked(True)
                else:
                    widget.out_radio.setChecked(True)
            
            self.add_log(f"All pins set to {direction}", "SUCCESS")
        except Exception as e:
            self.add_log(f"Batch direction error: {e}", "ERROR")
    
    def batch_output(self, value):
        """Set all output pins"""
        count = 0
        try:
            for pin in self.controller.pins.keys():
                if self.controller.pin_directions[pin] == "OUT":
                    self.controller.output(pin, value)
                    self.pin_widgets[pin].update_value_display(value)
                    count += 1
            
            state = "HIGH" if value else "LOW"
            self.add_log(f"{count} output pins set to {state}", "SUCCESS")
        except Exception as e:
            self.add_log(f"Batch output error: {e}", "ERROR")
    
    def blink_test(self):
        """Blink test on pin 32"""
        pin = 32
        
        # Set as output
        self.controller.setup(pin, "OUT")
        self.pin_widgets[pin].out_radio.setChecked(True)
        
        self.add_log(f"Starting blink test on pin {pin}...", "INFO")
        
        # Create timer for blinking
        self.blink_count = 0
        self.blink_state = False
        
        def blink():
            if self.blink_count >= 10:
                self.blink_timer.stop()
                self.add_log(f"Blink test completed", "SUCCESS")
                return
            
            self.blink_state = not self.blink_state
            self.controller.output(pin, 1 if self.blink_state else 0)
            self.pin_widgets[pin].update_value_display(1 if self.blink_state else 0)
            self.blink_count += 1
        
        self.blink_timer = QTimer()
        self.blink_timer.timeout.connect(blink)
        self.blink_timer.start(500)
    
    def update_pin(self, pin, value):
        """Update pin display"""
        if pin in self.pin_widgets:
            self.pin_widgets[pin].update_value_display(value)

    def on_tab_changed(self, index):
        """Handle tab change - auto cleanup events"""
        # Tab 0 is Monitor, Tab 1 is Events
        # Only cleanup if there are active event detections
        if hasattr(self, 'events_tab') and self.controller.event_detections:
            removed_count = self.events_tab.remove_all_events()
            if removed_count > 0:
                if index == 0:
                    self.add_log(f"🧹 Auto-cleaned {removed_count} event detection(s) when switching to Monitor tab", "INFO")
                else:
                    self.add_log(f"🧹 Auto-cleaned {removed_count} old event detection(s)", "INFO")

    def closeEvent(self, event):
        """Handle window close"""
        reply = QMessageBox.question(self, 'Quit', 'Are you sure you want to quit?',
                                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            self.monitor_timer.stop()
            try:
                self.controller.cleanup()
            except:
                pass
            event.accept()
        else:
            event.ignore()

def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    window = GPIOMainWindow()
    window.show()
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()