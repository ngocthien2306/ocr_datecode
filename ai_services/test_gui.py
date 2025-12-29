#!/usr/bin/env python3
"""
GPIO Testing Application using libgpiod
Safer alternative for non-standard carrier boards
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import threading
import time
from datetime import datetime
import subprocess
import re

class GPIOController:
    """GPIO Controller using gpiod command line tools"""
    
    def __init__(self):
        self.gpio_available = self.check_gpio_available()
        self.pin_to_gpio = self.detect_pin_mapping()
        
    def check_gpio_available(self):
        """Kiểm tra gpiod có sẵn không"""
        try:
            result = subprocess.run(['gpiodetect'], 
                                  capture_output=True, 
                                  timeout=2)
            return result.returncode == 0
        except:
            return False
    
    def detect_pin_mapping(self):
        """Phát hiện GPIO mapping"""
        # Mapping mặc định cho Jetson Orin
        # Format: pin_number -> (chip, line, name)
        return {
            7: (0, 422, "MCLK05"),
            11: (0, 424, "UART1_RTS"),
            12: (0, 417, "I2S2_CLK"),
            19: (0, 493, "SPI1_MOSI"),
            21: (0, 492, "SPI1_MISO"),
            23: (0, 491, "SPI1_SCK"),
            26: (0, 495, "SPI1_CS1_N"),
            32: (0, 395, "GPIO8"),
            36: (0, 425, "UART1_CTS"),
        }
    
    def setup(self, pin, direction):
        """Setup pin direction"""
        if pin not in self.pin_to_gpio:
            return False
        
        chip, line, _ = self.pin_to_gpio[pin]
        
        # libgpiod doesn't require explicit setup
        # Direction is implied by get/set operations
        return True
    
    def input(self, pin):
        """Read pin value"""
        if not self.gpio_available or pin not in self.pin_to_gpio:
            return 0
        
        try:
            chip, line, _ = self.pin_to_gpio[pin]
            result = subprocess.run(
                ['gpioget', f'gpiochip{chip}', str(line)],
                capture_output=True,
                text=True,
                timeout=1
            )
            
            if result.returncode == 0:
                return int(result.stdout.strip())
            return 0
            
        except Exception as e:
            print(f"Error reading pin {pin}: {e}")
            return 0
    
    def output(self, pin, value):
        """Set pin output value"""
        if not self.gpio_available or pin not in self.pin_to_gpio:
            return False
        
        try:
            chip, line, _ = self.pin_to_gpio[pin]
            
            # Use gpioset with --mode=time to set and hold
            # This keeps the value set
            subprocess.Popen(
                ['gpioset', '--mode=signal', f'gpiochip{chip}', 
                 f'{line}={value}'],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            
            return True
            
        except Exception as e:
            print(f"Error setting pin {pin}: {e}")
            return False
    
    def cleanup(self):
        """Cleanup (not needed for gpiod)"""
        pass


class GPIOTestApp:
    def __init__(self, root):
        self.root = root
        self.root.title("GPIO Test Application (libgpiod)")
        self.root.geometry("1400x900")
        self.root.configure(bg='#2b2b2b')
        
        # Initialize GPIO controller
        self.gpio = GPIOController()
        
        # Pin descriptions
        self.pin_descriptions = {
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
        
        # State tracking
        self.pin_states = {pin: 0 for pin in self.pin_descriptions.keys()}
        self.pin_directions = {pin: "IN" for pin in self.pin_descriptions.keys()}
        self.monitoring_active = False
        self.monitor_thread = None
        
        # Create UI
        self.create_ui()
        
        # Start monitoring
        if self.gpio.gpio_available:
            self.start_monitoring()
            self.log_message("GPIO Controller initialized successfully")
        else:
            self.log_message("WARNING: GPIO not available", "ERROR")
            messagebox.showwarning("Warning", 
                "GPIO tools not available!\n\n"
                "Please install: sudo apt-get install gpiod")
    
    def create_ui(self):
        """Tạo giao diện"""
        
        # Top status bar
        status_bar = tk.Frame(self.root, bg='#1e1e1e', height=35)
        status_bar.pack(fill=tk.X, side=tk.TOP)
        
        status_text = "GPIO: Ready (libgpiod)" if self.gpio.gpio_available else "GPIO: Not Available"
        status_color = '#00ff00' if self.gpio.gpio_available else '#ff0000'
        
        tk.Label(status_bar, text="●", fg=status_color, bg='#1e1e1e',
                font=("Arial", 20)).pack(side=tk.LEFT, padx=5)
        
        tk.Label(status_bar, text=status_text, 
                bg='#1e1e1e', fg='white',
                font=("Arial", 11, "bold")).pack(side=tk.LEFT, padx=5)
        
        tk.Label(status_bar, text=f"Pins: {len(self.pin_descriptions)}", 
                bg='#1e1e1e', fg='#aaaaaa',
                font=("Arial", 10)).pack(side=tk.LEFT, padx=20)
        
        # Main container
        main_frame = tk.Frame(self.root, bg='#2b2b2b')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel - Pin controls
        left_panel = self.create_left_panel(main_frame)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        # Right panel - Monitor
        right_panel = self.create_right_panel(main_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
    
    def create_left_panel(self, parent):
        """Create left control panel"""
        panel = tk.Frame(parent, bg='#3c3c3c', relief=tk.RAISED, bd=2)
        
        # Title
        title = tk.Label(panel, text="GPIO Pin Control", 
                        font=("Arial", 18, "bold"), bg='#3c3c3c', fg='white')
        title.pack(pady=15)
        
        # Info card
        info_frame = tk.LabelFrame(panel, text="System Info", 
                                  bg='#3c3c3c', fg='white',
                                  font=("Arial", 11, "bold"))
        info_frame.pack(fill=tk.X, padx=10, pady=10)
        
        info_text = f"Method: libgpiod (gpiod)\n"
        info_text += f"Available Pins: {len(self.pin_descriptions)}\n"
        info_text += f"Status: {'Active' if self.gpio.gpio_available else 'Inactive'}"
        
        tk.Label(info_frame, text=info_text, bg='#3c3c3c', 
                fg='#00ff00', font=("Courier", 10), 
                justify=tk.LEFT).pack(padx=10, pady=8)
        
        # Pin controls with scrollbar
        pins_container = tk.Frame(panel, bg='#3c3c3c')
        pins_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        canvas = tk.Canvas(pins_container, bg='#3c3c3c', highlightthickness=0)
        scrollbar = ttk.Scrollbar(pins_container, orient="vertical", 
                                 command=canvas.yview)
        scrollable = tk.Frame(canvas, bg='#3c3c3c')
        
        scrollable.bind("<Configure>", 
                       lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        
        canvas.create_window((0, 0), window=scrollable, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Create pin controls
        self.pin_widgets = {}
        for pin in sorted(self.pin_descriptions.keys()):
            self.create_pin_control(scrollable, pin)
        
        return panel
    
    def create_pin_control(self, parent, pin):
        """Create control widget for a pin"""
        desc = self.pin_descriptions[pin]
        
        # Main frame
        frame = tk.Frame(parent, bg='#4c4c4c', relief=tk.GROOVE, bd=3)
        frame.pack(fill=tk.X, padx=5, pady=4)
        
        # Header
        header = tk.Frame(frame, bg='#4c4c4c')
        header.pack(fill=tk.X, padx=8, pady=5)
        
        # Pin number
        tk.Label(header, text=f"Pin {pin}", 
                font=("Arial", 12, "bold"), 
                bg='#4c4c4c', fg='#ffcc00', width=6).pack(side=tk.LEFT)
        
        # Description
        tk.Label(header, text=desc, 
                font=("Arial", 10), 
                bg='#4c4c4c', fg='#cccccc').pack(side=tk.LEFT, padx=10)
        
        # Status LED
        status_led = tk.Label(header, text="●", font=("Arial", 18),
                             bg='#4c4c4c', fg='gray')
        status_led.pack(side=tk.RIGHT, padx=5)
        
        # Controls
        controls = tk.Frame(frame, bg='#4c4c4c')
        controls.pack(fill=tk.X, padx=8, pady=5)
        
        # Direction
        tk.Label(controls, text="Mode:", bg='#4c4c4c', 
                fg='white', font=("Arial", 9)).pack(side=tk.LEFT, padx=5)
        
        dir_var = tk.StringVar(value="IN")
        
        tk.Radiobutton(controls, text="INPUT", variable=dir_var, value="IN",
                      bg='#4c4c4c', fg='white', selectcolor='#4c4c4c',
                      font=("Arial", 9),
                      command=lambda: self.set_direction(pin, "IN")).pack(side=tk.LEFT)
        
        tk.Radiobutton(controls, text="OUTPUT", variable=dir_var, value="OUT",
                      bg='#4c4c4c', fg='white', selectcolor='#4c4c4c',
                      font=("Arial", 9),
                      command=lambda: self.set_direction(pin, "OUT")).pack(side=tk.LEFT)
        
        # Output buttons
        btn_frame = tk.Frame(controls, bg='#4c4c4c')
        btn_frame.pack(side=tk.LEFT, padx=15)
        
        high_btn = tk.Button(btn_frame, text="HIGH", width=7,
                            bg='#00aa00', fg='white', font=("Arial", 9, "bold"),
                            command=lambda: self.set_output(pin, 1),
                            state=tk.DISABLED)
        high_btn.pack(side=tk.LEFT, padx=2)
        
        low_btn = tk.Button(btn_frame, text="LOW", width=7,
                           bg='#cc0000', fg='white', font=("Arial", 9, "bold"),
                           command=lambda: self.set_output(pin, 0),
                           state=tk.DISABLED)
        low_btn.pack(side=tk.LEFT, padx=2)
        
        # Value display
        value_label = tk.Label(controls, text="Value: -", 
                              bg='#4c4c4c', fg='white', 
                              font=("Courier", 10, "bold"))
        value_label.pack(side=tk.RIGHT, padx=8)
        
        # Store widgets
        self.pin_widgets[pin] = {
            'frame': frame,
            'status': status_led,
            'dir_var': dir_var,
            'value_label': value_label,
            'high_btn': high_btn,
            'low_btn': low_btn
        }
    
    def create_right_panel(self, parent):
        """Create right monitor panel"""
        panel = tk.Frame(parent, bg='#3c3c3c', relief=tk.RAISED, bd=2)
        
        # Title
        title = tk.Label(panel, text="Activity Monitor", 
                        font=("Arial", 18, "bold"), bg='#3c3c3c', fg='white')
        title.pack(pady=15)
        
        # Control buttons
        btn_frame = tk.Frame(panel, bg='#3c3c3c')
        btn_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.monitor_btn = tk.Button(btn_frame, text="⏸ Pause",
                                     bg='#0066cc', fg='white', width=12,
                                     font=("Arial", 10, "bold"),
                                     command=self.toggle_monitoring)
        self.monitor_btn.pack(side=tk.LEFT, padx=3)
        
        tk.Button(btn_frame, text="🗑 Clear",
                 bg='#cc6600', fg='white', width=10,
                 font=("Arial", 10, "bold"),
                 command=self.clear_log).pack(side=tk.LEFT, padx=3)
        
        tk.Button(btn_frame, text="📊 Read All",
                 bg='#6600cc', fg='white', width=12,
                 font=("Arial", 10, "bold"),
                 command=self.read_all).pack(side=tk.LEFT, padx=3)
        
        # Monitor text
        self.monitor_text = scrolledtext.ScrolledText(
            panel, height=25, bg='#1e1e1e', fg='#00ff00',
            font=("Courier New", 9), wrap=tk.WORD, 
            insertbackground='white')
        self.monitor_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Quick actions
        actions_frame = tk.LabelFrame(panel, text="Quick Actions",
                                     bg='#3c3c3c', fg='white',
                                     font=("Arial", 10, "bold"))
        actions_frame.pack(fill=tk.X, padx=10, pady=10)
        
        btn_row1 = tk.Frame(actions_frame, bg='#3c3c3c')
        btn_row1.pack(pady=5)
        
        tk.Button(btn_row1, text="All INPUT", bg='#0066cc', fg='white', width=12,
                 command=lambda: self.batch_direction("IN")).pack(side=tk.LEFT, padx=3)
        
        tk.Button(btn_row1, text="All OUTPUT", bg='#0066cc', fg='white', width=12,
                 command=lambda: self.batch_direction("OUT")).pack(side=tk.LEFT, padx=3)
        
        btn_row2 = tk.Frame(actions_frame, bg='#3c3c3c')
        btn_row2.pack(pady=5)
        
        tk.Button(btn_row2, text="All HIGH", bg='#00aa00', fg='white', width=12,
                 command=lambda: self.batch_output(1)).pack(side=tk.LEFT, padx=3)
        
        tk.Button(btn_row2, text="All LOW", bg='#cc0000', fg='white', width=12,
                 command=lambda: self.batch_output(0)).pack(side=tk.LEFT, padx=3)
        
        return panel
    
    def set_direction(self, pin, direction):
        """Set pin direction"""
        self.pin_directions[pin] = direction
        self.gpio.setup(pin, direction)
        
        widgets = self.pin_widgets[pin]
        if direction == "OUT":
            widgets['high_btn'].config(state=tk.NORMAL)
            widgets['low_btn'].config(state=tk.NORMAL)
        else:
            widgets['high_btn'].config(state=tk.DISABLED)
            widgets['low_btn'].config(state=tk.DISABLED)
        
        self.log_message(f"Pin {pin} set to {direction}")
    
    def set_output(self, pin, value):
        """Set pin output"""
        if self.pin_directions[pin] != "OUT":
            messagebox.showwarning("Warning", 
                f"Pin {pin} is not configured as OUTPUT")
            return
        
        success = self.gpio.output(pin, value)
        if success:
            self.pin_states[pin] = value
            self.update_pin_display(pin)
            state = "HIGH" if value else "LOW"
            self.log_message(f"Pin {pin} set to {state}")
        else:
            self.log_message(f"Failed to set pin {pin}", "ERROR")
    
    def start_monitoring(self):
        """Start monitoring thread"""
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self.monitor_loop, 
                                              daemon=True)
        self.monitor_thread.start()
    
    def monitor_loop(self):
        """Monitoring loop"""
        while self.monitoring_active:
            try:
                for pin in self.pin_descriptions.keys():
                    if self.pin_directions[pin] == "IN":
                        value = self.gpio.input(pin)
                        old_value = self.pin_states.get(pin)
                        
                        if value != old_value:
                            change = "LOW→HIGH" if value else "HIGH→LOW"
                            self.log_message(f"Pin {pin}: {change}", "CHANGE")
                        
                        self.pin_states[pin] = value
                        self.update_pin_display(pin)
                
                time.sleep(0.3)  # 300ms polling
                
            except Exception as e:
                self.log_message(f"Monitor error: {e}", "ERROR")
                time.sleep(1)
    
    def toggle_monitoring(self):
        """Toggle monitoring"""
        if self.monitoring_active:
            self.monitoring_active = False
            self.monitor_btn.config(text="▶ Resume")
            self.log_message("Monitoring paused")
        else:
            self.monitoring_active = True
            self.monitor_btn.config(text="⏸ Pause")
            self.log_message("Monitoring resumed")
            if not self.monitor_thread or not self.monitor_thread.is_alive():
                self.start_monitoring()
    
    def update_pin_display(self, pin):
        """Update pin display"""
        if pin not in self.pin_widgets:
            return
        
        widgets = self.pin_widgets[pin]
        value = self.pin_states[pin]
        
        # Update LED
        color = '#00ff00' if value else '#ff0000'
        widgets['status'].config(fg=color)
        
        # Update value
        text = "HIGH (1)" if value else "LOW (0)"
        widgets['value_label'].config(text=f"Value: {text}")
    
    def log_message(self, message, level="INFO"):
        """Add log message"""
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        
        color_map = {
            "INFO": "#00ff00",
            "CHANGE": "#ffff00",
            "ERROR": "#ff0000",
        }
        
        formatted = f"[{timestamp}] [{level:6s}] {message}\n"
        
        self.monitor_text.insert(tk.END, formatted)
        self.monitor_text.see(tk.END)
        
        # Limit lines
        lines = int(self.monitor_text.index('end-1c').split('.')[0])
        if lines > 1000:
            self.monitor_text.delete('1.0', '200.0')
    
    def clear_log(self):
        """Clear log"""
        self.monitor_text.delete('1.0', tk.END)
        self.log_message("Log cleared")
    
    def read_all(self):
        """Read all pins"""
        self.log_message("="*50)
        self.log_message("Reading all pins...")
        
        for pin in sorted(self.pin_descriptions.keys()):
            value = self.gpio.input(pin)
            direction = self.pin_directions[pin]
            state = "HIGH (1)" if value else "LOW (0)"
            desc = self.pin_descriptions[pin]
            
            self.log_message(
                f"Pin {pin:2d} | {direction:3s} | {state} | {desc}")
        
        self.log_message("="*50)
    
    def batch_direction(self, direction):
        """Set all pins direction"""
        for pin in self.pin_descriptions.keys():
            self.pin_directions[pin] = direction
            self.pin_widgets[pin]['dir_var'].set(direction)
            self.gpio.setup(pin, direction)
            
            widgets = self.pin_widgets[pin]
            if direction == "OUT":
                widgets['high_btn'].config(state=tk.NORMAL)
                widgets['low_btn'].config(state=tk.NORMAL)
            else:
                widgets['high_btn'].config(state=tk.DISABLED)
                widgets['low_btn'].config(state=tk.DISABLED)
        
        self.log_message(f"All pins set to {direction}")
    
    def batch_output(self, value):
        """Set all output pins"""
        count = 0
        for pin in self.pin_descriptions.keys():
            if self.pin_directions[pin] == "OUT":
                self.gpio.output(pin, value)
                self.pin_states[pin] = value
                self.update_pin_display(pin)
                count += 1
        
        state = "HIGH" if value else "LOW"
        self.log_message(f"{count} output pins set to {state}")
    
    def on_closing(self):
        """Handle window close"""
        if messagebox.askokcancel("Quit", "Exit application?"):
            self.monitoring_active = False
            time.sleep(0.5)
            self.gpio.cleanup()
            self.root.destroy()


def main():
    root = tk.Tk()
    app = GPIOTestApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()