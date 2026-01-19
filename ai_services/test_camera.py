#!/usr/bin/env python3
"""Simple camera settings test"""
from pypylon import pylon
import sys

def test_camera(serial_number=None):
    pylon.PylonInitialize()

    try:
        if serial_number:
            tlFactory = pylon.TlFactory.GetInstance()
            devices = tlFactory.EnumerateDevices()
            device = next((d for d in devices if d.GetSerialNumber() == serial_number), None)
            if not device:
                print(f"Camera {serial_number} not found")
                return False
            camera = pylon.InstantCamera(tlFactory.CreateDevice(device))
        else:
            camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())

        camera.Open()
        print(f"Camera: {camera.GetDeviceInfo().GetModelName()} (SN: {camera.GetDeviceInfo().GetSerialNumber()})")

        # Turn off auto exposure
        if camera.ExposureAuto.IsWritable():
            camera.ExposureAuto.SetValue("Off")
            print("ExposureAuto: Off")

        # Set exposure mode to Timed
        if camera.ExposureMode.IsWritable():
            camera.ExposureMode.SetValue("Timed")
            print("ExposureMode: Timed")

        # Set exposure time
        exposure_us = 500
        if camera.ExposureTimeAbs.IsWritable():
            camera.ExposureTimeAbs.SetValue(exposure_us)
            print(f"ExposureTimeAbs: {camera.ExposureTimeAbs.Value} µs")
        elif camera.ExposureTime.IsWritable():
            camera.ExposureTime.SetValue(exposure_us)
            print(f"ExposureTime: {camera.ExposureTime.Value} µs")
        else:
            print("Cannot set exposure time")

        # Set gain
        gain_value = 5
        if camera.GainAuto.IsWritable():
            camera.GainAuto.SetValue("Off")
            print("GainAuto: Off")

        if camera.Gain.IsWritable():
            camera.Gain.SetValue(float(gain_value))
            print(f"Gain: {camera.Gain.Value}")
        elif camera.GainRaw.IsWritable():
            camera.GainRaw.SetValue(int(gain_value))
            print(f"GainRaw: {camera.GainRaw.Value}")
        else:
            print("Cannot set gain")

        # Test grabbing
        camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        print("Grabbing started...")

        grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
        if grabResult.GrabSucceeded():
            print(f"✅ Grab successful! Image size: {grabResult.Width}x{grabResult.Height}")
        grabResult.Release()

        camera.StopGrabbing()
        camera.Close()
        print("✅ Test passed!")
        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        pylon.PylonTerminate()

if __name__ == '__main__':
    serial = sys.argv[1] if len(sys.argv) > 1 else None
    success = test_camera(serial)
    sys.exit(0 if success else 1)


