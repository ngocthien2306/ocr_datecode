"""
Test Basler Camera Trigger Modes
- Software Trigger
- Hardware Trigger
- Free Running
"""

import cv2
import time
from pypylon import pylon


def test_free_running(num_frames=10):
    """Test Free Running mode (no trigger)"""
    print("\n=== TEST FREE RUNNING MODE ===")

    camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
    camera.Open()

    print(f"Camera: {camera.GetDeviceInfo().GetModelName()}")

    # Set free running mode
    camera.TriggerMode.SetValue('Off')
    print("Trigger Mode: OFF")

    # Start grabbing
    camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
    converter = pylon.ImageFormatConverter()
    converter.OutputPixelFormat = pylon.PixelType_BGR8packed

    print(f"Grabbing {num_frames} frames...")
    for i in range(num_frames):
        grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

        if grabResult.GrabSucceeded():
            image = converter.Convert(grabResult)
            img = image.GetArray()
            print(f"  Frame {i+1}: {img.shape[1]}x{img.shape[0]}")

            # Save first frame
            if i == 0:
                cv2.imwrite("free_running_frame.png", img)
                print(f"    Saved: free_running_frame.png")

        grabResult.Release()

    camera.StopGrabbing()
    camera.Close()
    print("✓ Free running test completed!\n")


def test_software_trigger(num_triggers=5):
    """Test Software Trigger mode"""
    print("\n=== TEST SOFTWARE TRIGGER MODE ===")

    camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
    camera.Open()

    print(f"Camera: {camera.GetDeviceInfo().GetModelName()}")

    # Set software trigger mode
    try:
        camera.TriggerSelector.SetValue('FrameStart')
        camera.TriggerMode.SetValue('On')
        camera.TriggerSource.SetValue('Software')
        print("Trigger Mode: Software Trigger")
    except Exception as e:
        print(f"Error setting trigger: {e}")
        camera.Close()
        return

    # Start grabbing
    camera.StartGrabbing(pylon.GrabStrategy_OneByOne)
    converter = pylon.ImageFormatConverter()
    converter.OutputPixelFormat = pylon.PixelType_BGR8packed

    print(f"Executing {num_triggers} software triggers...")
    for i in range(num_triggers):
        print(f"  Trigger {i+1}...")

        # Wait for camera ready
        if camera.WaitForFrameTriggerReady(1000, pylon.TimeoutHandling_Return):
            # Execute software trigger
            camera.ExecuteSoftwareTrigger()

            # Retrieve image
            grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

            if grabResult.GrabSucceeded():
                image = converter.Convert(grabResult)
                img = image.GetArray()
                print(f"    Got frame: {img.shape[1]}x{img.shape[0]}")

                # Save first frame
                if i == 0:
                    cv2.imwrite("software_trigger_frame.png", img)
                    print(f"    Saved: software_trigger_frame.png")

            grabResult.Release()
        else:
            print(f"    Camera not ready for trigger!")

        time.sleep(0.5)  # Wait between triggers

    camera.StopGrabbing()
    camera.Close()
    print("✓ Software trigger test completed!\n")


def test_hardware_trigger(trigger_source='Line1', trigger_activation='RisingEdge', timeout=30):
    """Test Hardware Trigger mode - wait for external trigger signal"""
    print("\n=== TEST HARDWARE TRIGGER MODE ===")
    print(f"Trigger Source: {trigger_source}")
    print(f"Trigger Activation: {trigger_activation}")
    print(f"Timeout: {timeout}s")

    camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
    camera.Open()

    print(f"Camera: {camera.GetDeviceInfo().GetModelName()}")

    # Set hardware trigger mode
    try:
        camera.TriggerSelector.SetValue('FrameStart')
        camera.TriggerMode.SetValue('On')
        camera.TriggerSource.SetValue(trigger_source)
        camera.TriggerActivation.SetValue(trigger_activation)

        # Configure line as input
        camera.LineSelector.SetValue(trigger_source)
        camera.LineMode.SetValue('Input')

        print(f"Trigger Mode: Hardware Trigger ({trigger_source}, {trigger_activation})")
    except Exception as e:
        print(f"Error setting trigger: {e}")
        camera.Close()
        return

    # Start grabbing
    camera.StartGrabbing(pylon.GrabStrategy_OneByOne)
    converter = pylon.ImageFormatConverter()
    converter.OutputPixelFormat = pylon.PixelType_BGR8packed

    print(f"\nWaiting for hardware trigger signal (timeout: {timeout}s)...")
    print("Send trigger signal to camera now!")

    frame_count = 0
    start_time = time.time()

    while time.time() - start_time < timeout:
        try:
            # Try to get triggered frame
            grabResult = camera.RetrieveResult(1000, pylon.TimeoutHandling_Return)

            if grabResult and grabResult.GrabSucceeded():
                frame_count += 1
                image = converter.Convert(grabResult)
                img = image.GetArray()
                print(f"  Frame {frame_count}: {img.shape[1]}x{img.shape[0]}")

                # Save first frame
                if frame_count == 1:
                    cv2.imwrite("hardware_trigger_frame.png", img)
                    print(f"    Saved: hardware_trigger_frame.png")

                grabResult.Release()
            else:
                # Show waiting indicator
                elapsed = int(time.time() - start_time)
                print(f"  Waiting... ({elapsed}s)", end='\r')

        except Exception as e:
            print(f"Error: {e}")
            break

    camera.StopGrabbing()
    camera.Close()

    if frame_count > 0:
        print(f"\n✓ Hardware trigger test completed! Got {frame_count} frame(s)\n")
    else:
        print(f"\n✗ No trigger received within {timeout}s\n")


def monitor_line_status(trigger_source='Line1', duration=10):
    """Monitor hardware trigger line status"""
    print("\n=== MONITOR LINE STATUS ===")
    print(f"Monitoring {trigger_source} for {duration}s...")

    camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
    camera.Open()

    try:
        camera.LineSelector.SetValue(trigger_source)
        camera.LineMode.SetValue('Input')

        print("Send trigger signals now!\n")

        start_time = time.time()
        last_status = None

        while time.time() - start_time < duration:
            status = camera.LineStatus.GetValue()

            if status != last_status:
                state = "HIGH" if status else "LOW"
                timestamp = time.time() - start_time
                print(f"[{timestamp:.2f}s] Line {trigger_source}: {state}")
                last_status = status

            time.sleep(0.01)  # Check every 10ms

        print("\n✓ Monitoring completed!\n")

    except Exception as e:
        print(f"Error: {e}")

    camera.Close()


def test_auto_trigger(interval_ms=1000, num_triggers=5):
    """Test auto software trigger with configurable interval"""
    print("\n=== TEST AUTO SOFTWARE TRIGGER ===")
    print(f"Interval: {interval_ms}ms")
    print(f"Number of triggers: {num_triggers}")

    camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
    camera.Open()

    # Set software trigger mode
    camera.TriggerSelector.SetValue('FrameStart')
    camera.TriggerMode.SetValue('On')
    camera.TriggerSource.SetValue('Software')

    camera.StartGrabbing(pylon.GrabStrategy_OneByOne)
    converter = pylon.ImageFormatConverter()
    converter.OutputPixelFormat = pylon.PixelType_BGR8packed

    print("Starting auto trigger...")
    for i in range(num_triggers):
        if camera.WaitForFrameTriggerReady(1000, pylon.TimeoutHandling_Return):
            camera.ExecuteSoftwareTrigger()

            grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
            if grabResult.GrabSucceeded():
                image = converter.Convert(grabResult)
                img = image.GetArray()
                print(f"  [{i+1}/{num_triggers}] Frame: {img.shape[1]}x{img.shape[0]}")
            grabResult.Release()

        time.sleep(interval_ms / 1000.0)

    camera.StopGrabbing()
    camera.Close()
    print("✓ Auto trigger test completed!\n")


if __name__ == "__main__":
    print("=" * 50)
    print("BASLER CAMERA TRIGGER TEST")
    print("=" * 50)

    # Test 1: Free running
    test_free_running(num_frames=10)

    # Test 2: Software trigger
    test_software_trigger(num_triggers=5)

    # Test 3: Auto software trigger
    test_auto_trigger(interval_ms=500, num_triggers=5)

    # Test 4: Hardware trigger (uncomment to test)
    # test_hardware_trigger(trigger_source='Line1', trigger_activation='RisingEdge', timeout=30)

    # Test 5: Monitor line status (uncomment to test)
    # monitor_line_status(trigger_source='Line1', duration=10)

    print("=" * 50)
    print("ALL TESTS COMPLETED!")
    print("=" * 50)
