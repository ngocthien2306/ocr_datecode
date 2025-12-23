import cv2
from pypylon import pylon
from datetime import datetime
import traceback

def list_cameras():
    try:
        tlFactory = pylon.TlFactory.GetInstance()
        devices = tlFactory.EnumerateDevices()
        
        if len(devices) == 0:
            print("No cameras found")
            return None
            
        print(f"Found {len(devices)} camera(s):")
        for i, device in enumerate(devices):
            print(f"  [{i}] {device.GetModelName()} (SN: {device.GetSerialNumber()})")
        
        return devices
    except Exception as e:
        print(f"Error listing cameras: {e}")
        return None

def capture_image():
    try:
        devices = list_cameras()
        if not devices:
            return
        
        camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateDevice(devices[0]))
        camera.Open()
        
        print(f"\nOpened: {camera.GetDeviceInfo().GetModelName()}")
        
        try:
            camera.ExposureTime.SetValue(10000)
            print(f"Exposure: {camera.ExposureTime.GetValue()} µs")
        except:
            pass
        
        camera.StartGrabbingMax(1)
        grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
        
        if grabResult.GrabSucceeded():
            converter = pylon.ImageFormatConverter()
            converter.OutputPixelFormat = pylon.PixelType_BGR8packed
            converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
            
            image = converter.Convert(grabResult)
            img_array = image.GetArray()
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"image_{timestamp}.png"
            cv2.imwrite(filename, img_array)
            
            print(f"Saved: {filename} ({img_array.shape[1]}x{img_array.shape[0]})")
            
            cv2.imshow("Captured", img_array)
            cv2.waitKey(2000)
            cv2.destroyAllWindows()
        
        grabResult.Release()
        camera.Close()
        
    except Exception as e:
        traceback.print_exc()

def capture_multiple(count=5):
    try:
        devices = list_cameras()
        if not devices:
            return
            
        camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateDevice(devices[0]))
        camera.Open()
        
        converter = pylon.ImageFormatConverter()
        converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
        
        camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        
        for i in range(count):
            grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
            
            if grabResult.GrabSucceeded():
                image = converter.Convert(grabResult)
                img_array = image.GetArray()
                
                filename = f"image_{i+1:03d}.png"
                cv2.imwrite(filename, img_array)
                print(f"[{i+1}/{count}] {filename}")
            
            grabResult.Release()
        
        camera.StopGrabbing()
        camera.Close()
        print("Done!")
        
    except Exception as e:
        traceback.print_exc()

if __name__ == "__main__":
    capture_image()
    # capture_multiple(count=5)