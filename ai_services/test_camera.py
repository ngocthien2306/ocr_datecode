#!/usr/bin/env python3
"""Test camera detection and streaming"""

import sys
import os
import time
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'core'))

from core.camera_manager import CameraManager

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def main():
    print("\n" + "="*60)
    print("CAMERA AUTO DETECTION TEST")
    print("="*60 + "\n")

    manager = CameraManager()

    # Auto detect and start all cameras
    num_cameras = manager.enumerate_and_start_cameras()

    if num_cameras == 0:
        print("\n❌ No cameras found!")
        return

    running = manager.get_running_cameras()
    print(f"\n✅ Running cameras: {running}\n")

    duration = 10
    print(f"📹 Monitoring frames for {duration} seconds...\n")
    print("Press Ctrl+C to stop\n")

    start_time = time.time()
    frame_count = {}
    save_dir = "camera_test_frames"
    os.makedirs(save_dir, exist_ok=True)

    try:
        while time.time() - start_time < duration:
            # Get frame
            frame_data = manager.get_frame(timeout=0.1)
            if frame_data:
                camera_id = frame_data['camera_id']
                metadata = frame_data['metadata']
                img = frame_data['image']

                frame_count[camera_id] = frame_count.get(camera_id, 0) + 1

                # Print every 30 frames
                if frame_count[camera_id] % 30 == 0:
                    print(f"[{camera_id}] Frame #{metadata['frame_number']} ({metadata['width']}x{metadata['height']})")

                # Save first frame
                if frame_count[camera_id] == 1:
                    filename = os.path.join(save_dir, f"{camera_id}_first.png")
                    cv2.imwrite(filename, img)
                    print(f"   ✅ Saved: {filename}")

            # Get results
            result = manager.get_result(timeout=0.01)
            if result:
                print(f"📨 {result}")

    except KeyboardInterrupt:
        print("\n⚠️  Interrupted")

    print("\n" + "="*60)
    print("Stopping cameras...")
    manager.cleanup()

    print("\n✅ Test completed!")
    print("="*60)
    print(f"\nResults:")
    for cam_id, count in frame_count.items():
        fps = count / duration
        print(f"  {cam_id}: {count} frames ({fps:.1f} fps)")
    print("="*60 + "\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
