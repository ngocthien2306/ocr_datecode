"""
Test Hardware Trigger cho Basler Camera
Chờ tín hiệu trigger từ Line1/Line2/Line3 và chụp ảnh
"""

import cv2
import time
from pypylon import pylon
from datetime import datetime


def setup_hardware_trigger(camera, trigger_source='Line1', trigger_activation='RisingEdge'):
    """Cấu hình hardware trigger"""
    print(f"\n--- Cấu hình Hardware Trigger ---")
    print(f"Trigger Source: {trigger_source}")
    print(f"Trigger Activation: {trigger_activation}")

    try:
        # Set trigger mode
        camera.TriggerSelector.SetValue('FrameStart')
        camera.TriggerMode.SetValue('On')
        camera.TriggerSource.SetValue(trigger_source)
        camera.TriggerActivation.SetValue(trigger_activation)

        # Configure line as input
        camera.LineSelector.SetValue(trigger_source)
        camera.LineMode.SetValue('Input')

        print("✓ Hardware trigger configured!")
        return True

    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_hardware_trigger_continuous(trigger_source='Line1',
                                     trigger_activation='RisingEdge',
                                     max_frames=0,
                                     timeout=60,
                                     save_all=False):
    """
    Test hardware trigger - chụp ảnh liên tục khi có tín hiệu

    Args:
        trigger_source: 'Line1', 'Line2', 'Line3'
        trigger_activation: 'RisingEdge', 'FallingEdge', 'AnyEdge', 'LevelHigh', 'LevelLow'
        max_frames: Số frame tối đa (0 = unlimited)
        timeout: Timeout in seconds
        save_all: True = lưu tất cả ảnh, False = chỉ lưu ảnh đầu tiên
    """
    print("\n" + "="*60)
    print("TEST HARDWARE TRIGGER - CONTINUOUS MODE")
    print("="*60)

    camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
    camera.Open()

    print(f"\nCamera: {camera.GetDeviceInfo().GetModelName()}")
    print(f"Serial: {camera.GetDeviceInfo().GetSerialNumber()}")

    # Setup hardware trigger
    if not setup_hardware_trigger(camera, trigger_source, trigger_activation):
        camera.Close()
        return

    # Configure camera
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
    camera.StartGrabbing(pylon.GrabStrategy_OneByOne)
    converter = pylon.ImageFormatConverter()
    converter.OutputPixelFormat = pylon.PixelType_BGR8packed

    print(f"\n{'='*60}")
    print("WAITING FOR TRIGGER SIGNAL...")
    if max_frames > 0:
        print(f"Will capture max {max_frames} frames")
    print(f"Timeout: {timeout}s")
    print(f"Press Ctrl+C to stop")
    print("="*60)
    print("\nSend trigger signal to camera now!\n")

    frame_count = 0
    start_time = time.time()
    last_print_time = start_time

    try:
        while True:
            # Check timeout
            elapsed = time.time() - start_time
            if elapsed > timeout:
                print(f"\n⏱ Timeout reached ({timeout}s)")
                break

            # Check max frames
            if max_frames > 0 and frame_count >= max_frames:
                print(f"\n✓ Reached max frames ({max_frames})")
                break

            # Try to get frame (1 second timeout per grab)
            grabResult = camera.RetrieveResult(1000, pylon.TimeoutHandling_Return)

            if grabResult and grabResult.GrabSucceeded():
                frame_count += 1

                # Convert to image
                image = converter.Convert(grabResult)
                img = image.GetArray()

                # Get timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]

                # Print info
                print(f"[{elapsed:.2f}s] Frame {frame_count}: {img.shape[1]}x{img.shape[0]} px")

                # Save image
                if frame_count == 1 or save_all:
                    filename = f"hw_trigger_{timestamp}_frame{frame_count:03d}.png"
                    cv2.imwrite(filename, img)
                    print(f"           Saved: {filename}")

                grabResult.Release()

            else:
                # Show waiting indicator every 2 seconds
                if time.time() - last_print_time > 2:
                    print(f"  Waiting for trigger... ({int(elapsed)}s elapsed)", end='\r')
                    last_print_time = time.time()

    except KeyboardInterrupt:
        print("\n\n⚠ Stopped by user (Ctrl+C)")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        camera.StopGrabbing()
        camera.Close()

    print("\n" + "="*60)
    if frame_count > 0:
        print(f"✓ COMPLETED! Captured {frame_count} frame(s)")
    else:
        print(f"✗ No frames captured in {timeout}s")
    print("="*60 + "\n")


def test_hardware_trigger_single(trigger_source='Line1',
                                 trigger_activation='RisingEdge',
                                 timeout=30):
    """Test hardware trigger - chỉ chụp 1 ảnh"""
    print("\n" + "="*60)
    print("TEST HARDWARE TRIGGER - SINGLE SHOT")
    print("="*60)

    camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
    camera.Open()

    print(f"\nCamera: {camera.GetDeviceInfo().GetModelName()}")

    # Setup hardware trigger
    if not setup_hardware_trigger(camera, trigger_source, trigger_activation):
        camera.Close()
        return

    # Start grabbing
    camera.StartGrabbing(pylon.GrabStrategy_OneByOne)
    converter = pylon.ImageFormatConverter()
    converter.OutputPixelFormat = pylon.PixelType_BGR8packed

    print(f"\n{'='*60}")
    print("WAITING FOR 1 TRIGGER SIGNAL...")
    print(f"Timeout: {timeout}s")
    print("="*60)
    print("\nSend trigger signal now!\n")

    start_time = time.time()

    try:
        while time.time() - start_time < timeout:
            grabResult = camera.RetrieveResult(1000, pylon.TimeoutHandling_Return)

            if grabResult and grabResult.GrabSucceeded():
                image = converter.Convert(grabResult)
                img = image.GetArray()

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"hw_trigger_single_{timestamp}.png"

                cv2.imwrite(filename, img)

                print(f"✓ GOT FRAME!")
                print(f"  Size: {img.shape[1]}x{img.shape[0]} px")
                print(f"  Saved: {filename}")

                grabResult.Release()
                break
            else:
                elapsed = int(time.time() - start_time)
                print(f"  Waiting... ({elapsed}s)", end='\r')

        else:
            print(f"\n✗ No trigger received in {timeout}s")

    except KeyboardInterrupt:
        print("\n\n⚠ Stopped by user")

    except Exception as e:
        print(f"\n✗ Error: {e}")

    finally:
        camera.StopGrabbing()
        camera.Close()

    print()


def monitor_trigger_line(trigger_source='Line1', duration=10):
    """Monitor trạng thái line để debug trigger"""
    print("\n" + "="*60)
    print("MONITOR TRIGGER LINE STATUS")
    print("="*60)

    camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
    camera.Open()

    print(f"\nCamera: {camera.GetDeviceInfo().GetModelName()}")
    print(f"Monitoring: {trigger_source}")
    print(f"Duration: {duration}s")

    try:
        camera.LineSelector.SetValue(trigger_source)
        camera.LineMode.SetValue('Input')

        # Read initial status
        initial_status = camera.LineStatus.GetValue()
        initial_state = "HIGH (1)" if initial_status else "LOW (0)"

        print("\n" + "="*60)
        print(f"CURRENT STATUS: {trigger_source} = {initial_state}")
        print("="*60)

        # Recommendation
        print("\n📌 TRIGGER ACTIVATION RECOMMENDATION:")
        if initial_status:
            print("   Current state: HIGH")
            print("   → Use 'FallingEdge' to trigger when signal goes LOW")
            print("   → Or use 'LevelLow' to trigger while signal is LOW")
        else:
            print("   Current state: LOW")
            print("   → Use 'RisingEdge' to trigger when signal goes HIGH")
            print("   → Or use 'LevelHigh' to trigger while signal is HIGH")

        print("\n   Other options:")
        print("   • RisingEdge:  Trigger on LOW → HIGH transition")
        print("   • FallingEdge: Trigger on HIGH → LOW transition")
        print("   • AnyEdge:     Trigger on any transition (both directions)")
        print("   • LevelHigh:   Trigger continuously while HIGH")
        print("   • LevelLow:    Trigger continuously while LOW")

        print("\n" + "="*60)
        print("Send trigger signals now!")
        print("="*60 + "\n")

        start_time = time.time()
        last_status = initial_status
        change_count = 0

        while time.time() - start_time < duration:
            status = camera.LineStatus.GetValue()

            if status != last_status:
                change_count += 1
                state = "HIGH (1)" if status else "LOW (0)"
                elapsed = time.time() - start_time
                transition = "LOW→HIGH (RisingEdge)" if status else "HIGH→LOW (FallingEdge)"
                print(f"[{elapsed:6.3f}s] {trigger_source}: {state}  [{transition}]  (change #{change_count})")

            last_status = status
            time.sleep(0.001)  # Check every 1ms

        print("\n" + "="*60)
        print(f"✓ Monitoring completed!")
        print(f"  Total state changes: {change_count}")

        if change_count == 0:
            print(f"  ⚠ No changes detected - check your trigger signal connection!")
        elif change_count % 2 == 1:
            print(f"  ⚠ Odd number of changes - signal may still be active")

        print("="*60 + "\n")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()

    camera.Close()


if __name__ == "__main__":
    # Configuration
    TRIGGER_SOURCE = 'Line1'          # Line1, Line2, Line3
    TRIGGER_ACTIVATION = 'RisingEdge' # RisingEdge, FallingEdge, AnyEdge, LevelHigh, LevelLow
    MAX_FRAMES = 10                   # 0 = unlimited
    TIMEOUT = 60                      # seconds
    SAVE_ALL = True                   # True = lưu tất cả, False = chỉ lưu ảnh đầu

    print("\nConfiguration:")
    print(f"  Trigger Source: {TRIGGER_SOURCE}")
    print(f"  Trigger Activation: {TRIGGER_ACTIVATION}")
    print(f"  Max Frames: {MAX_FRAMES if MAX_FRAMES > 0 else 'Unlimited'}")
    print(f"  Timeout: {TIMEOUT}s")
    print(f"  Save All Frames: {SAVE_ALL}")

    # Chọn test nào muốn chạy:

    # Test 1: Chụp liên tục khi có trigger
    # test_hardware_trigger_continuous(
    #     trigger_source=TRIGGER_SOURCE,
    #     trigger_activation=TRIGGER_ACTIVATION,
    #     max_frames=MAX_FRAMES,
    #     timeout=TIMEOUT,
    #     save_all=SAVE_ALL
    # )

    # Test 2: Chỉ chụp 1 ảnh (uncomment để test)
    # test_hardware_trigger_single(
    #     trigger_source=TRIGGER_SOURCE,
    #     trigger_activation=TRIGGER_ACTIVATION,
    #     timeout=30
    # )

    # Test 3: Monitor line status để debug (uncomment để test)
    monitor_trigger_line(
        trigger_source=TRIGGER_SOURCE,
        duration=10
    )
