import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog
import cv2
import time
import threading
import os
from datetime import datetime
from PIL import Image, ImageTk
from pypylon import pylon
from pymodbus.client import ModbusTcpClient


class PLCCameraApp:
    def __init__(self, root):
        self.root = root
        self.root.title("PLC Camera Trigger System")
        self.root.geometry("1400x900")

        self.plc = None
        self.camera = None
        self.converter = None

        self.plc_connected = False
        self.camera_connected = False
        self.monitoring = False
        self.capture_count = 0

        self.plc_thread = None
        self.monitor_thread = None
        self.running = False

        self.setup_ui()

    def setup_ui(self):
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        left_frame = tk.Frame(main_frame, width=500)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 10))

        config_container = tk.Frame(left_frame)
        config_container.pack(fill=tk.X, pady=(0, 10))

        plc_frame = tk.LabelFrame(config_container, text="PLC Configuration", padx=10, pady=10)
        plc_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        tk.Label(plc_frame, text="PLC IP:").grid(row=0, column=0, sticky=tk.W, pady=3)
        self.plc_ip_var = tk.StringVar(value="192.168.1.5")
        tk.Entry(plc_frame, textvariable=self.plc_ip_var, width=15).grid(row=0, column=1, pady=3, sticky=tk.EW)

        tk.Label(plc_frame, text="Port:").grid(row=1, column=0, sticky=tk.W, pady=3)
        self.plc_port_var = tk.IntVar(value=502)
        tk.Spinbox(plc_frame, from_=1, to=65535, textvariable=self.plc_port_var, width=13).grid(row=1, column=1, pady=3, sticky=tk.EW)

        tk.Label(plc_frame, text="Register:").grid(row=2, column=0, sticky=tk.W, pady=3)
        self.register_var = tk.IntVar(value=0)
        tk.Spinbox(plc_frame, from_=0, to=9999, textvariable=self.register_var, width=13).grid(row=2, column=1, pady=3, sticky=tk.EW)

        tk.Label(plc_frame, text="Poll (s):").grid(row=3, column=0, sticky=tk.W, pady=3)
        self.poll_interval_var = tk.DoubleVar(value=0.1)
        tk.Spinbox(plc_frame, from_=0.01, to=10.0, increment=0.01, textvariable=self.poll_interval_var, width=13).grid(row=3, column=1, pady=3, sticky=tk.EW)

        self.auto_reset_var = tk.BooleanVar(value=True)
        tk.Checkbutton(plc_frame, text="Auto Reset D0=1", variable=self.auto_reset_var).grid(row=4, column=0, columnspan=2, sticky=tk.W, pady=3)

        self.plc_connect_btn = tk.Button(plc_frame, text="Connect PLC", command=self.toggle_plc_connection, bg="#4CAF50", fg="white", font=("Arial", 9, "bold"))
        self.plc_connect_btn.grid(row=5, column=0, columnspan=2, sticky=tk.EW, pady=(8, 3))

        self.plc_status_label = tk.Label(plc_frame, text="⚫ Disconnected", fg="red", font=("Arial", 8))
        self.plc_status_label.grid(row=6, column=0, columnspan=2, pady=3)

        camera_frame = tk.LabelFrame(config_container, text="Camera Configuration", padx=10, pady=10)
        camera_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 0))

        tk.Label(camera_frame, text="Folder:").grid(row=0, column=0, sticky=tk.W, pady=3)
        folder_container = tk.Frame(camera_frame)
        folder_container.grid(row=0, column=1, sticky=tk.EW, pady=3)
        self.save_folder_var = tk.StringVar(value="images")
        tk.Entry(folder_container, textvariable=self.save_folder_var, width=10).pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Button(folder_container, text="...", command=self.browse_folder, width=3).pack(side=tk.RIGHT, padx=(2, 0))
        camera_frame.columnconfigure(1, weight=1)

        tk.Label(camera_frame, text="Images:").grid(row=1, column=0, sticky=tk.W, pady=3)
        self.images_per_trigger_var = tk.IntVar(value=1)
        tk.Spinbox(camera_frame, from_=1, to=10, textvariable=self.images_per_trigger_var, width=13).grid(row=1, column=1, sticky=tk.EW, pady=3)

        tk.Label(camera_frame, text="Delay (s):").grid(row=2, column=0, sticky=tk.W, pady=3)
        self.delay_between_var = tk.DoubleVar(value=0.1)
        tk.Spinbox(camera_frame, from_=0.01, to=5.0, increment=0.01, textvariable=self.delay_between_var, width=13).grid(row=2, column=1, sticky=tk.EW, pady=3)

        tk.Label(camera_frame, text="First delay (s):").grid(row=3, column=0, sticky=tk.W, pady=3)
        self.first_delay_var = tk.DoubleVar(value=0.0)
        tk.Spinbox(camera_frame, from_=0.0, to=5.0, increment=0.01, textvariable=self.first_delay_var, width=13).grid(row=3, column=1, sticky=tk.EW, pady=3)

        tk.Label(camera_frame, text="Trigger edge:").grid(row=4, column=0, sticky=tk.W, pady=3)
        edge_frame = tk.Frame(camera_frame)
        edge_frame.grid(row=4, column=1, sticky=tk.W, pady=3)
        self.trigger_edge_var = tk.StringVar(value="any")
        tk.Radiobutton(edge_frame, text="0→1", variable=self.trigger_edge_var, value="rising").pack(side=tk.LEFT)
        tk.Radiobutton(edge_frame, text="1→0", variable=self.trigger_edge_var, value="falling").pack(side=tk.LEFT)
        tk.Radiobutton(edge_frame, text="Any", variable=self.trigger_edge_var, value="any").pack(side=tk.LEFT)

        self.camera_connect_btn = tk.Button(camera_frame, text="Connect Camera", command=self.toggle_camera_connection, bg="#2196F3", fg="white", font=("Arial", 9, "bold"))
        self.camera_connect_btn.grid(row=5, column=0, columnspan=2, sticky=tk.EW, pady=(8, 3))

        self.camera_status_label = tk.Label(camera_frame, text="⚫ Disconnected", fg="red", font=("Arial", 8))
        self.camera_status_label.grid(row=6, column=0, columnspan=2, pady=3)

        self.test_capture_btn = tk.Button(camera_frame, text="Test Capture", command=self.test_capture, state=tk.DISABLED)
        self.test_capture_btn.grid(row=7, column=0, columnspan=2, sticky=tk.EW, pady=3)

        monitor_frame = tk.LabelFrame(left_frame, text="Monitoring Control", padx=10, pady=10)
        monitor_frame.pack(fill=tk.X, pady=(0, 10))

        self.start_monitor_btn = tk.Button(monitor_frame, text="Start Monitoring", command=self.toggle_monitoring, state=tk.DISABLED, bg="#FF9800", fg="white", font=("Arial", 11, "bold"), height=2)
        self.start_monitor_btn.pack(fill=tk.X, pady=5)

        stats_frame = tk.Frame(monitor_frame)
        stats_frame.pack(fill=tk.X)

        self.d0_value_label = tk.Label(stats_frame, text="D0: --", font=("Arial", 11, "bold"))
        self.d0_value_label.pack(side=tk.LEFT, padx=10)

        self.capture_count_label = tk.Label(stats_frame, text="Captures: 0", font=("Arial", 11, "bold"), fg="green")
        self.capture_count_label.pack(side=tk.LEFT, padx=10)

        log_frame = tk.LabelFrame(left_frame, text="Log", padx=10, pady=10)
        log_frame.pack(fill=tk.BOTH, expand=True)

        self.log_text = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD, font=("Courier", 9))
        self.log_text.pack(fill=tk.BOTH, expand=True)

        tk.Button(log_frame, text="Clear Log", command=lambda: self.log_text.delete(1.0, tk.END)).pack(fill=tk.X, pady=(5, 0))

        right_frame = tk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        image_frame = tk.LabelFrame(right_frame, text="Last Captured Image", padx=10, pady=10)
        image_frame.pack(fill=tk.BOTH, expand=True)

        self.image_label = tk.Label(image_frame, text="No image captured", bg="#f0f0f0", relief=tk.SUNKEN)
        self.image_label.pack(fill=tk.BOTH, expand=True)

    def add_log(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)

    def browse_folder(self):
        folder = filedialog.askdirectory(title="Select Save Folder")
        if folder:
            self.save_folder_var.set(folder)

    def toggle_plc_connection(self):
        if not self.plc_connected:
            try:
                plc_ip = self.plc_ip_var.get()
                plc_port = self.plc_port_var.get()

                self.plc = ModbusTcpClient(plc_ip, port=plc_port, timeout=2)
                if self.plc.connect():
                    self.plc_connected = True
                    self.plc_status_label.config(text="🟢 Connected", fg="green")
                    self.plc_connect_btn.config(text="Disconnect PLC", bg="#f44336")
                    self.add_log(f"✓ Connected to PLC {plc_ip}:{plc_port}")
                    self.update_monitor_button()
                else:
                    self.add_log("✗ Cannot connect to PLC")
            except Exception as e:
                self.add_log(f"✗ PLC error: {e}")
        else:
            if self.monitoring:
                self.toggle_monitoring()
            if self.plc:
                self.plc.close()
                self.plc = None
            self.plc_connected = False
            self.plc_status_label.config(text="⚫ Disconnected", fg="red")
            self.plc_connect_btn.config(text="Connect PLC", bg="#4CAF50")
            self.add_log("✗ Disconnected from PLC")
            self.update_monitor_button()

    def toggle_camera_connection(self):
        if not self.camera_connected:
            try:
                self.add_log("Connecting to camera...")

                tlFactory = pylon.TlFactory.GetInstance()
                devices = tlFactory.EnumerateDevices()

                if len(devices) == 0:
                    self.add_log("✗ No camera found")
                    return

                self.camera = pylon.InstantCamera(tlFactory.CreateFirstDevice())
                self.camera.Open()

                self.camera.TriggerSelector.SetValue('FrameStart')
                self.camera.TriggerMode.SetValue('On')
                self.camera.TriggerSource.SetValue('Software')

                try:
                    self.camera.ExposureTime.SetValue(10000)
                except:
                    try:
                        self.camera.ExposureTimeAbs.SetValue(10000)
                    except:
                        pass

                self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

                self.converter = pylon.ImageFormatConverter()
                self.converter.OutputPixelFormat = pylon.PixelType_BGR8packed

                self.camera_connected = True
                camera_name = self.camera.GetDeviceInfo().GetModelName()
                self.camera_status_label.config(text="🟢 Connected", fg="green")
                self.camera_connect_btn.config(text="Disconnect Camera", bg="#f44336")
                self.test_capture_btn.config(state=tk.NORMAL)
                self.add_log(f"✓ Connected to {camera_name}")
                self.update_monitor_button()

            except Exception as e:
                self.add_log(f"✗ Camera error: {e}")
                if self.camera:
                    try:
                        self.camera.Close()
                    except:
                        pass
                    self.camera = None
        else:
            if self.monitoring:
                self.toggle_monitoring()
            try:
                if self.camera:
                    if self.camera.IsGrabbing():
                        self.camera.StopGrabbing()
                    self.camera.Close()
                    self.camera = None
            except:
                pass
            self.camera_connected = False
            self.camera_status_label.config(text="⚫ Disconnected", fg="red")
            self.camera_connect_btn.config(text="Connect Camera", bg="#2196F3")
            self.test_capture_btn.config(state=tk.DISABLED)
            self.add_log("✗ Camera disconnected")
            self.update_monitor_button()

    def update_monitor_button(self):
        if self.plc_connected and self.camera_connected:
            self.start_monitor_btn.config(state=tk.NORMAL)
        else:
            self.start_monitor_btn.config(state=tk.DISABLED)

    def test_capture(self):
        self.add_log("Testing manual capture...")
        try:
            num_images = self.images_per_trigger_var.get()
            delay = self.delay_between_var.get()
            first_delay = self.first_delay_var.get()
            save_folder = self.save_folder_var.get()

            if first_delay > 0:
                self.add_log(f"⏱ Waiting {first_delay}s before first capture...")
                time.sleep(first_delay)

            os.makedirs(save_folder, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            if num_images > 1:
                subfolder = os.path.join(save_folder, f"test_{timestamp}")
                os.makedirs(subfolder, exist_ok=True)
                self.add_log(f"Created subfolder: {subfolder}")
            else:
                subfolder = save_folder

            captured_images = []

            for i in range(num_images):
                self.camera.TriggerSoftware.Execute()
                grabResult = self.camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

                if grabResult.GrabSucceeded():
                    image = self.converter.Convert(grabResult)
                    img = image.GetArray()
                    grabResult.Release()

                    if num_images > 1:
                        filename = os.path.join(subfolder, f"image_{i+1:02d}.png")
                    else:
                        filename = os.path.join(subfolder, f"test_capture_{timestamp}.png")

                    cv2.imwrite(filename, img)
                    captured_images.append(img)

                    self.add_log(f"✓ Captured {i+1}/{num_images}: {os.path.basename(filename)} ({img.shape[1]}x{img.shape[0]} px)")

                    if i < num_images - 1:
                        time.sleep(delay)
                else:
                    self.add_log(f"✗ Capture {i+1}/{num_images} failed")

            if captured_images:
                self.display_image(captured_images[-1])
                self.add_log(f"✓ Test complete: {len(captured_images)}/{num_images} images saved")

        except Exception as e:
            self.add_log(f"✗ Capture error: {e}")

    def toggle_monitoring(self):
        if not self.monitoring:
            self.monitoring = True
            self.capture_count = 0
            self.capture_count_label.config(text="Captures: 0")

            edge_type = self.trigger_edge_var.get()
            edge_desc = {"rising": "0→1", "falling": "1→0", "any": "any"}[edge_type]

            self.start_monitor_btn.config(text="Stop Monitoring", bg="#f44336")
            self.add_log(f"▶ Monitoring started - waiting for D0 edge ({edge_desc})...")

            self.running = True
            self.monitor_thread = threading.Thread(target=self.monitor_loop, daemon=True)
            self.monitor_thread.start()
        else:
            self.monitoring = False
            self.running = False

            self.start_monitor_btn.config(text="Start Monitoring", bg="#FF9800")
            self.add_log("■ Monitoring stopped")

    def monitor_loop(self):
        last_value = None
        register_address = self.register_var.get()
        poll_interval = self.poll_interval_var.get()
        trigger_edge = self.trigger_edge_var.get()

        while self.running and self.monitoring:
            try:
                result = self.plc.read_holding_registers(address=register_address, count=1)
                if not result.isError():
                    value = result.registers[0]

                    self.root.after(0, self.d0_value_label.config, {"text": f"D0: {value}"})

                    if last_value is not None and last_value != value:
                        trigger_detected = False

                        if trigger_edge == "rising" and last_value == 0 and value == 1:
                            trigger_detected = True
                        elif trigger_edge == "falling" and last_value == 1 and value == 0:
                            trigger_detected = True
                        elif trigger_edge == "any":
                            trigger_detected = True

                        if trigger_detected:
                            self.capture_count += 1
                            self.root.after(0, self.capture_count_label.config, {"text": f"Captures: {self.capture_count}"})
                            self.root.after(0, self.add_log, f"🎯 EDGE DETECTED ({last_value}→{value}) - Capturing #{self.capture_count}")

                            self.capture_image()

                    last_value = value

            except Exception as e:
                self.root.after(0, self.add_log, f"⚠ Monitor error: {e}")

            time.sleep(poll_interval)

    def capture_image(self):
        try:
            num_images = self.images_per_trigger_var.get()
            delay = self.delay_between_var.get()
            first_delay = self.first_delay_var.get()
            save_folder = self.save_folder_var.get()

            if first_delay > 0:
                self.root.after(0, self.add_log, f"⏱ Waiting {first_delay}s before first capture...")
                time.sleep(first_delay)

            os.makedirs(save_folder, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            if num_images > 1:
                subfolder = os.path.join(save_folder, f"trigger_{self.capture_count:03d}_{timestamp}")
                os.makedirs(subfolder, exist_ok=True)
                self.root.after(0, self.add_log, f"Created subfolder: {os.path.basename(subfolder)}")
            else:
                subfolder = save_folder

            captured_images = []

            for i in range(num_images):
                self.camera.TriggerSoftware.Execute()
                grabResult = self.camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

                if grabResult.GrabSucceeded():
                    image = self.converter.Convert(grabResult)
                    img = image.GetArray()
                    grabResult.Release()

                    if num_images > 1:
                        filename = os.path.join(subfolder, f"image_{i+1:02d}.png")
                    else:
                        filename = os.path.join(subfolder, f"trigger_{self.capture_count:03d}_{timestamp}.png")

                    cv2.imwrite(filename, img)
                    captured_images.append(img)

                    self.root.after(0, self.add_log, f"✓ Captured {i+1}/{num_images}: {os.path.basename(filename)} ({img.shape[1]}x{img.shape[0]} px)")

                    if i < num_images - 1:
                        time.sleep(delay)
                else:
                    self.root.after(0, self.add_log, f"✗ Capture {i+1}/{num_images} failed")

            if captured_images:
                self.root.after(0, self.display_image, captured_images[-1])
                self.root.after(0, self.add_log, f"✓ Complete: {len(captured_images)}/{num_images} images saved")

            if self.auto_reset_var.get():
                try:
                    result = self.plc.write_register(address=0, value=1)
                    if not result.isError():
                        self.root.after(0, self.add_log, "✓ Reset D0 = 1")
                    else:
                        self.root.after(0, self.add_log, "⚠ Failed to reset D0")
                except:
                    pass

        except Exception as e:
            self.root.after(0, self.add_log, f"✗ Capture error: {e}")

    def display_image(self, img):
        try:
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            label_width = self.image_label.winfo_width()
            label_height = self.image_label.winfo_height()

            if label_width <= 1:
                label_width = 800
            if label_height <= 1:
                label_height = 600

            img_height, img_width = rgb_img.shape[:2]

            scale_w = label_width / img_width
            scale_h = label_height / img_height
            scale = min(scale_w, scale_h, 1.0)

            new_width = int(img_width * scale)
            new_height = int(img_height * scale)

            resized = cv2.resize(rgb_img, (new_width, new_height), interpolation=cv2.INTER_AREA)

            pil_img = Image.fromarray(resized)
            photo = ImageTk.PhotoImage(pil_img)

            self.image_label.config(image=photo, text="")
            self.image_label.image = photo

        except Exception as e:
            self.add_log(f"✗ Display error: {e}")

    def on_closing(self):
        if self.monitoring:
            self.toggle_monitoring()
            time.sleep(0.5)

        if self.plc_connected:
            try:
                self.plc.close()
            except:
                pass

        if self.camera_connected:
            try:
                if self.camera and self.camera.IsGrabbing():
                    self.camera.StopGrabbing()
                if self.camera:
                    self.camera.Close()
            except:
                pass

        self.root.destroy()


def main():
    root = tk.Tk()
    app = PLCCameraApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()
