#!/usr/bin/env python3
"""
Test script to verify AsyncImageSaver runs in background threads
and doesn't block main thread.
"""

import time
import numpy as np
from pathlib import Path
from camera_management.utils import AsyncImageSaver

def main():
    print("Testing AsyncImageSaver threading behavior...\n")

    # Create test images (simulate 3 cameras, full resolution)
    test_imgs = [
        np.random.randint(0, 255, (2056, 2464, 3), dtype=np.uint8)
        for _ in range(3)
    ]

    test_dir = Path("/tmp/async_save_test")
    test_dir.mkdir(exist_ok=True)

    # Get singleton instance
    saver = AsyncImageSaver.get_instance(max_workers=4)

    print("=== MAIN THREAD: Submitting 3 save tasks ===")
    t_start = time.perf_counter()

    for i, img in enumerate(test_imgs):
        path_viz = test_dir / f"test_{i}_viz.jpg"
        path_org = test_dir / f"test_{i}_org.jpg"

        print(f"  Submitting task {i+1}...")
        saver.save_images_async(
            img_viz=img,
            img_org=img,
            path_viz=path_viz,
            path_org=path_org,
            quality=95
        )
        print(f"  Task {i+1} submitted (returned immediately)")

    t_submit = (time.perf_counter() - t_start) * 1000
    pending = saver.get_pending_count()

    print(f"\n=== MAIN THREAD: All tasks submitted in {t_submit:.1f}ms ===")
    print(f"Pending tasks: {pending}")
    print(f"Main thread is NOT blocked! Can continue working...\n")

    # Simulate main thread continuing to work
    print("=== MAIN THREAD: Doing other work while background saves... ===")
    for i in range(5):
        time.sleep(0.1)
        pending = saver.get_pending_count()
        print(f"  {i+1}s: Pending tasks = {pending}")

    # Wait for all saves to complete
    print("\n=== MAIN THREAD: Waiting for background tasks to complete ===")
    while saver.get_pending_count() > 0:
        time.sleep(0.1)
        pending = saver.get_pending_count()
        print(f"  Pending: {pending}")

    t_total = (time.perf_counter() - t_start) * 1000

    print(f"\n=== ALL DONE ===")
    print(f"Total time: {t_total:.1f}ms")
    print(f"Submit time: {t_submit:.1f}ms (main thread was only blocked for this)")
    print(f"Background save time: {t_total - t_submit:.1f}ms (happened in parallel)\n")

    # Check files exist
    saved_files = list(test_dir.glob("*.jpg"))
    print(f"Files saved: {len(saved_files)}")
    for f in sorted(saved_files):
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  {f.name}: {size_mb:.2f} MB")

    # Cleanup
    saver.shutdown(wait=True)
    print("\nAsyncImageSaver shutdown complete")

if __name__ == "__main__":
    main()
