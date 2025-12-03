"""
Simple Basler Camera Capture
Kết nối camera, lấy ảnh và lưu file
"""

import cv2
import numpy as np
from pypylon import pylon
from datetime import datetime


def capture_image():
    """Kết nối camera và chụp ảnh"""
    try:
        # Khởi tạo camera
        camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
        camera.Open()

        print(f"Camera: {camera.GetDeviceInfo().GetModelName()}")
        print(f"Serial: {camera.GetDeviceInfo().GetSerialNumber()}")

        # Cấu hình camera (tùy chọn)
        camera.ExposureTime.SetValue(10000)  # 10ms
        # camera.Gain.SetValue(0)

        # Chụp ảnh
        camera.StartGrabbingMax(1)
        grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

        if grabResult.GrabSucceeded():
            # Convert sang numpy array
            converter = pylon.ImageFormatConverter()
            converter.OutputPixelFormat = pylon.PixelType_BGR8packed
            converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

            image = converter.Convert(grabResult)
            img_array = image.GetArray()

            # Lưu ảnh
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"image_{timestamp}.png"
            cv2.imwrite(filename, img_array)

            print(f"✓ Đã lưu ảnh: {filename}")
            print(f"  Kích thước: {img_array.shape[1]}x{img_array.shape[0]}")

            # Hiển thị ảnh (tùy chọn)
            cv2.imshow("Captured Image", img_array)
            cv2.waitKey(2000)  # Hiển thị 2 giây
            cv2.destroyAllWindows()

        grabResult.Release()
        camera.Close()

    except Exception as e:
        print(f"Lỗi: {str(e)}")


def capture_multiple(count=5, interval=1):
    """Chụp nhiều ảnh liên tiếp"""
    try:
        camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
        camera.Open()

        print(f"Camera: {camera.GetDeviceInfo().GetModelName()}")
        print(f"Chụp {count} ảnh...")

        # Converter
        converter = pylon.ImageFormatConverter()
        converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

        # Bắt đầu chụp liên tục
        camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

        for i in range(count):
            grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

            if grabResult.GrabSucceeded():
                image = converter.Convert(grabResult)
                img_array = image.GetArray()

                filename = f"image_{i+1:03d}.png"
                cv2.imwrite(filename, img_array)
                print(f"  [{i+1}/{count}] {filename}")

            grabResult.Release()

        camera.StopGrabbing()
        camera.Close()
        print("✓ Hoàn thành!")

    except Exception as e:
        print(f"Lỗi: {str(e)}")


if __name__ == "__main__":
    # Chụp 1 ảnh
    capture_image()

    # Hoặc chụp nhiều ảnh:
    # capture_multiple(count=5)
