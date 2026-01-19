from pypylon import pylon
import cv2

def open_both_cameras():
    devices = pylon.TlFactory.GetInstance().EnumerateDevices()
    
    cameras = []
    for i, device in enumerate(devices):
        camera = pylon.InstantCamera(
            pylon.TlFactory.GetInstance().CreateDevice(device)
        )
        camera.Open()
        camera.TriggerMode.SetValue("Off")
        print(f"Camera {i}: {camera.GetDeviceInfo().GetModelName()}")
        cameras.append(camera)
    
    return cameras

# Mở cả 2 cameras
cameras = open_both_cameras()

# Grab từ camera thứ 2 (GigE - acA2500-14gm)
camera_gige = cameras[0]
camera_gige.StartGrabbingMax(10)

frame_count = 0
while camera_gige.IsGrabbing():
    grabResult = camera_gige.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
    
    if grabResult.GrabSucceeded():
        img = grabResult.Array
        cv2.imwrite(f"gige_frame_{frame_count}.png", img)
        print(f"Saved frame {frame_count}: {img.shape}")
        frame_count += 1
    
    grabResult.Release()

# Đóng tất cả cameras
for camera in cameras:
    camera.Close()