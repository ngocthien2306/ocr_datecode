"""
PLC Modbus TCP Trigger cho Basler Camera
Đọc thanh ghi D0 từ PLC qua Modbus TCP, khi D0=1 thì software trigger camera
"""

import cv2
import time
from pypylon import pylon
from datetime import datetime
from pymodbus.client import ModbusTcpClient

# PLC Configuration
PLC_IP = "192.168.1.5"
PLC_PORT = 502  # Modbus TCP standard port
D0_ADDRESS = 0  # D0 register address (0-based)


def setup_plc_connection(ip=PLC_IP, port=PLC_PORT, timeout=2):
    """Kết nối đến PLC qua Modbus TCP"""
    print(f"\n--- Kết nối PLC (Modbus TCP) ---")
    print(f"IP: {ip}:{port}")

    try:
        client = ModbusTcpClient(ip, port=port, timeout=timeout)
        if client.connect():
            print("✓ Kết nối PLC thành công!")
            return client
        else:
            print("✗ Không thể kết nối PLC")
            return None

    except Exception as e:
        print(f"✗ Lỗi kết nối PLC: {e}")
        return None


def read_plc_register(client, address=D0_ADDRESS):
    """Đọc giá trị thanh ghi từ PLC (Holding Register)"""
    try:
        result = client.read_holding_registers(address=address, count=1)
        if not result.isError():
            return result.registers[0]
        else:
            print(f"✗ Lỗi đọc thanh ghi: {result}")
            return None
    except Exception as e:
        print(f"✗ Exception đọc thanh ghi: {e}")
        return None


def write_plc_register(client, address=D0_ADDRESS, value=0):
    """Ghi giá trị vào thanh ghi PLC"""
    try:
        result = client.write_register(address=address, value=value)
        if not result.isError():
            return True
        else:
            print(f"✗ Lỗi ghi thanh ghi: {result}")
            return False
    except Exception as e:
        print(f"✗ Exception ghi thanh ghi: {e}")
        return False


def setup_software_trigger(camera):
    """Cấu hình software trigger cho camera"""
    print(f"\n--- Cấu hình Software Trigger ---")

    try:
        # Set trigger mode to Software
        camera.TriggerSelector.SetValue('FrameStart')
        camera.TriggerMode.SetValue('On')
        camera.TriggerSource.SetValue('Software')

        print("✓ Software trigger configured!")
        return True

    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def plc_triggered_capture(plc_ip=PLC_IP,
                          plc_port=PLC_PORT,
                          register_address=D0_ADDRESS,
                          polling_interval=0.1,
                          max_captures=0,
                          timeout=300,
                          save_all=True,
                          reset_register=False,
                          trigger_value=0,
                          reset_address=0,
                          reset_value=1):
    """
    Đọc thanh ghi PLC qua Modbus TCP và trigger camera khi phát hiện edge transition

    Cơ chế: ANY Edge detection - trigger khi phát hiện BẤT KỲ thay đổi nào
    - D0: 0 → 1 (rising edge) → Chụp ảnh
    - D0: 1 → 0 (falling edge) → Chụp ảnh

    Args:
        plc_ip: IP của PLC
        plc_port: Port của PLC (default: 502)
        register_address: Địa chỉ thanh ghi đọc (D0 = address 0)
        polling_interval: Thời gian poll (giây)
        max_captures: Số ảnh tối đa (0 = unlimited)
        timeout: Timeout tổng (giây)
        save_all: True = lưu tất cả ảnh
        reset_register: True = software reset sau khi chụp (option)
        trigger_value: Không sử dụng trong any edge mode
        reset_address: Địa chỉ thanh ghi để reset nếu reset_register=True
        reset_value: Giá trị để ghi vào reset_address
    """
    print("\n" + "="*60)
    print("PLC MODBUS TCP TRIGGERED CAMERA CAPTURE")
    print("="*60)

    # Connect to PLC
    plc = setup_plc_connection(plc_ip, plc_port)
    if not plc:
        return

    # Connect to Camera
    try:
        camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
        camera.Open()

        print(f"\nCamera: {camera.GetDeviceInfo().GetModelName()}")
        print(f"Serial: {camera.GetDeviceInfo().GetSerialNumber()}")

    except Exception as e:
        print(f"✗ Lỗi kết nối camera: {e}")
        plc.close()
        return

    # Setup software trigger
    if not setup_software_trigger(camera):
        camera.Close()
        plc.close()
        return

    # Configure camera exposure
    try:
        camera.ExposureTime.SetValue(10000)  # 10ms
        print(f"Exposure: {camera.ExposureTime.GetValue()} µs")
    except:
        try:
            camera.ExposureTimeAbs.SetValue(10000)
            print(f"Exposure: {camera.ExposureTimeAbs.GetValue()} µs")
        except:
            print("Could not set exposure")

    # Start grabbing
    camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
    converter = pylon.ImageFormatConverter()
    converter.OutputPixelFormat = pylon.PixelType_BGR8packed

    print(f"\n{'='*60}")
    print(f"POLLING PLC REGISTER: D{register_address} (Address: {register_address})")
    print(f"PLC IP: {plc_ip}:{plc_port}")
    print(f"Trigger condition: D{register_address} ANY transition (0→1 OR 1→0)")
    if reset_register:
        print(f"Reset after capture: D{reset_address} = {reset_value}")
    print(f"Polling interval: {polling_interval}s")
    if max_captures > 0:
        print(f"Max captures: {max_captures}")
    print(f"Timeout: {timeout}s")
    print(f"Press Ctrl+C to stop")
    print("="*60)
    print(f"\nWaiting for D{register_address} edge (any change)...\n")

    capture_count = 0
    start_time = time.time()
    last_value = None
    poll_count = 0
    error_count = 0

    try:
        while True:
            # Check timeout
            elapsed = time.time() - start_time
            if elapsed > timeout:
                print(f"\n⏱ Timeout reached ({timeout}s)")
                break

            # Check max captures
            if max_captures > 0 and capture_count >= max_captures:
                print(f"\n✓ Reached max captures ({max_captures})")
                break

            # Read PLC register
            value = read_plc_register(plc, register_address)
            poll_count += 1

            if value is None:
                error_count += 1
                if error_count > 5:
                    print(f"\n⚠ Quá nhiều lỗi đọc ({error_count}), dừng chương trình")
                    break
                time.sleep(polling_interval)
                continue
            else:
                error_count = 0  # Reset error count on success

            # Print status every 20 polls or when value changes
            if poll_count % 20 == 0 or value != last_value:
                status = "TRIGGERED!" if (last_value is not None and last_value != value) else "waiting..."
                print(f"[{elapsed:6.1f}s] D{register_address} = {value:5d}  [{status}]", end='\r')

            # Trigger camera on ANY edge (0→1 or 1→0)
            if last_value is not None and last_value != value:
                capture_count += 1

                print(f"\n[{elapsed:6.1f}s] 🎯 EDGE DETECTED ({last_value}→{value})! Capturing image #{capture_count}...")

                # Software trigger
                camera.TriggerSoftware.Execute()

                # Wait for image
                grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

                if grabResult.GrabSucceeded():
                    # Convert to image
                    image = converter.Convert(grabResult)
                    img = image.GetArray()

                    # Get timestamp
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]

                    # Save image
                    if save_all or capture_count == 1:
                        filename = f"plc_trigger_{timestamp}_frame{capture_count:03d}.png"
                        cv2.imwrite(filename, img)
                        print(f"           ✓ Saved: {filename} ({img.shape[1]}x{img.shape[0]} px)")
                    else:
                        print(f"           ✓ Captured: {img.shape[1]}x{img.shape[0]} px (not saved)")

                    grabResult.Release()

                    # Reset register if requested (optional - PLC usually handles this)
                    if reset_register:
                        if write_plc_register(plc, reset_address, reset_value):
                            print(f"           ✓ Set D{reset_address} = {reset_value}")
                        else:
                            print(f"           ⚠ Cannot write to D{reset_address}")

                else:
                    print(f"           ✗ Failed to capture image")

            last_value = value
            time.sleep(polling_interval)

    except KeyboardInterrupt:
        print("\n\n⚠ Stopped by user (Ctrl+C)")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        camera.StopGrabbing()
        camera.Close()
        plc.close()
        print("\n✓ Đã đóng kết nối camera và PLC")

    print("\n" + "="*60)
    if capture_count > 0:
        print(f"✓ COMPLETED! Captured {capture_count} image(s)")
        avg_time = elapsed / capture_count if capture_count > 0 else 0
        print(f"  Average time per capture: {avg_time:.2f}s")
    else:
        print(f"✗ No images captured in {timeout}s")
    print(f"  Total polls: {poll_count}")
    print(f"  Total errors: {error_count}")
    print("="*60 + "\n")


def test_plc_connection(plc_ip=PLC_IP, plc_port=PLC_PORT, register_address=D0_ADDRESS, duration=10):
    """Test kết nối PLC và đọc thanh ghi"""
    print("\n" + "="*60)
    print("TEST PLC MODBUS TCP CONNECTION")
    print("="*60)

    plc = setup_plc_connection(plc_ip, plc_port)
    if not plc:
        return

    print(f"\nMonitoring D{register_address} (Address: {register_address}) for {duration}s...")
    print("="*60 + "\n")

    start_time = time.time()
    last_value = None
    read_count = 0
    error_count = 0

    try:
        while time.time() - start_time < duration:
            value = read_plc_register(plc, register_address)
            elapsed = time.time() - start_time
            read_count += 1

            if value is not None:
                if value != last_value:
                    print(f"[{elapsed:6.2f}s] D{register_address} = {value:5d}  (changed)")
                    last_value = value
                else:
                    print(f"[{elapsed:6.2f}s] D{register_address} = {value:5d}", end='\r')
            else:
                error_count += 1
                print(f"[{elapsed:6.2f}s] Error reading register (total errors: {error_count})")

            time.sleep(0.1)

        print("\n\n" + "="*60)
        print("✓ Test completed!")
        print(f"  Total reads: {read_count}")
        print(f"  Total errors: {error_count}")
        print(f"  Success rate: {(read_count-error_count)/read_count*100:.1f}%")
        print("="*60 + "\n")

    except KeyboardInterrupt:
        print("\n\n⚠ Stopped by user")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        plc.close()


if __name__ == "__main__":
    # Configuration
    PLC_IP = "192.168.1.5"
    PLC_PORT = 502  # Modbus TCP standard port
    REGISTER_ADDRESS = 0  # D0 = address 0
    POLLING_INTERVAL = 0.1  # Poll mỗi 100ms
    MAX_CAPTURES = 0  # 0 = unlimited
    TIMEOUT = 300  # 5 minutes
    SAVE_ALL = True
    RESET_REGISTER = True  # True = tự động reset D0=1 sau khi chụp

    print("\n📋 Configuration:")
    print(f"  PLC IP: {PLC_IP}:{PLC_PORT} (Modbus TCP)")
    print(f"  Monitor Register: D0 (Address: {REGISTER_ADDRESS})")
    print(f"  Trigger Condition: ANY edge (0→1 OR 1→0)")
    print(f"  Action: Capture image on any value change")
    if RESET_REGISTER:
        print(f"  After Capture: Software reset D0 = 1")
    else:
        print(f"  After Capture: PLC handles reset (software does not reset)")
    print(f"  Polling Interval: {POLLING_INTERVAL}s")
    print(f"  Max Captures: {MAX_CAPTURES if MAX_CAPTURES > 0 else 'Unlimited'}")
    print(f"  Timeout: {TIMEOUT}s")
    print(f"  Save All Images: {SAVE_ALL}")

    # Chọn test muốn chạy:

    # Test 1: Test kết nối PLC (uncomment để test)
    # test_plc_connection(
    #     plc_ip=PLC_IP,
    #     plc_port=PLC_PORT,
    #     register_address=REGISTER_ADDRESS,
    #     duration=10
    # )

    # Test 2: Chạy PLC-triggered capture (uncomment để chạy)
    plc_triggered_capture(
        plc_ip=PLC_IP,
        plc_port=PLC_PORT,
        register_address=REGISTER_ADDRESS,  # Đọc từ D0
        polling_interval=POLLING_INTERVAL,
        max_captures=MAX_CAPTURES,
        timeout=TIMEOUT,
        save_all=SAVE_ALL,
        reset_register=RESET_REGISTER,
        trigger_value=0,  # Trigger khi D0 = 0
        reset_address=0,  # Ghi lại vào D0
        reset_value=1     # Reset D0 = 1
    )
