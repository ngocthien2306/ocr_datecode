"""
Script to generate test inference results data
Tạo 100 records mẫu cho database inference_results
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime, timedelta
import random

# Add backend to path
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

from motor.motor_asyncio import AsyncIOMotorClient
from app.core.config import settings


# Sample data configuration
RECIPE_IDS = ["test_recipe_001", "test_recipe_002", "test_recipe_003"]
RECIPE_NAMES = ["Test Recipe Alpha", "Test Recipe Beta", "Test Recipe Gamma"]
CAMERA_SERIALS = ["CAM_001", "CAM_002", "CAM_003"]
TEMPLATE_NAMES = ["Template_A", "Template_B", "Template_C", "Template_D"]


def generate_frame_result(frame_idx: int, pass_fail: str) -> dict:
    """Generate a single frame result"""
    confidence = random.uniform(0.7, 0.99) if pass_fail == "PASS" else random.uniform(0.2, 0.6)

    frame = {
        "template_name": random.choice(TEMPLATE_NAMES),
        "frame_idx": frame_idx,
        "pass_fail": pass_fail,
        "confidence": round(confidence, 4),
        "detected_regions": [
            {
                "type": "text",
                "text": f"TEST{random.randint(100, 999)}",
                "points": [[100, 100], [200, 100], [200, 150], [100, 150]]
            },
            {
                "type": "barcode",
                "text": f"BC{random.randint(10000, 99999)}",
                "points": [[100, 200], [300, 200], [300, 250], [100, 250]]
            }
        ],
        "timings": {
            "feature_extraction": round(random.uniform(10, 30), 2),
            "matching": round(random.uniform(5, 15), 2),
            "ransac": round(random.uniform(2, 8), 2),
            "total": round(random.uniform(20, 50), 2)
        }
    }

    # Add image_path only for FAIL cases
    if pass_fail == "FAIL":
        timestamp_str = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        frame["image_path"] = f"inference_results/{timestamp_str}_CAM_00{frame_idx}_FAIL.jpg"
    else:
        frame["image_path"] = None

    # Add text_verification for some frames (Check_Type_Product function)
    if random.random() > 0.5:
        all_match = pass_fail == "PASS"
        frame["text_verification"] = {
            "all_match": all_match,
            "results": [
                {
                    "region_idx": 0,
                    "expected": "TEST123",
                    "recognized": "TEST123" if all_match else "TEST124",
                    "match": all_match,
                    "confidence": round(random.uniform(0.85, 0.99), 4)
                }
            ]
        }

    return frame


def generate_camera_result(camera_serial: str, num_frames: int = 2) -> dict:
    """Generate result for a single camera"""
    # Random pass/fail for frames
    frames = []
    for frame_idx in range(num_frames):
        # 70% PASS, 30% FAIL
        pass_fail = "PASS" if random.random() < 0.7 else "FAIL"
        frames.append(generate_frame_result(frame_idx, pass_fail))

    return {
        "camera_id": camera_serial,
        "serial_number": camera_serial,
        "delay_trigger": random.randint(0, 500),
        "frames": frames
    }


def generate_inference_result(timestamp: datetime) -> dict:
    """Generate a complete inference result"""
    recipe_idx = random.randint(0, len(RECIPE_IDS) - 1)
    recipe_id = RECIPE_IDS[recipe_idx]
    recipe_name = RECIPE_NAMES[recipe_idx]

    # Generate results for 3 cameras
    camera_results = []
    overall_pass_fail = "PASS"

    for camera_serial in CAMERA_SERIALS:
        camera_result = generate_camera_result(camera_serial, num_frames=random.randint(1, 3))
        camera_results.append(camera_result)

        # Check if any frame failed
        for frame in camera_result["frames"]:
            if frame["pass_fail"] == "FAIL":
                overall_pass_fail = "FAIL"

    # Calculate metadata stats
    total_frames = sum(len(cr["frames"]) for cr in camera_results)
    all_confidences = []
    total_inliers = 0
    total_matches = 0

    for cr in camera_results:
        for frame in cr["frames"]:
            all_confidences.append(frame["confidence"])
            # Random inliers/matches for stats
            total_inliers += random.randint(50, 200)
            total_matches += random.randint(100, 300)

    avg_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0.0

    per_camera_stats = []
    for camera_serial in CAMERA_SERIALS:
        camera_result = next((cr for cr in camera_results if cr["serial_number"] == camera_serial), None)
        if camera_result:
            frames = camera_result["frames"]
            pass_count = sum(1 for f in frames if f["pass_fail"] == "PASS")
            fail_count = sum(1 for f in frames if f["pass_fail"] == "FAIL")

            per_camera_stats.append({
                "serial_number": camera_serial,
                "confidence": round(random.uniform(0.7, 0.95), 4),
                "inliers": random.randint(50, 200),
                "total_matches": random.randint(100, 300),
                "timings": {
                    "feature_extraction": round(random.uniform(10, 30), 2),
                    "matching": round(random.uniform(5, 15), 2),
                    "total": round(random.uniform(20, 50), 2)
                },
                "frame_stats": {
                    "total_frames": len(frames),
                    "pass_count": pass_count,
                    "fail_count": fail_count,
                    "error_count": 0,
                    "avg_confidence": round(sum(f["confidence"] for f in frames) / len(frames), 4)
                }
            })

    return {
        "recipe_id": recipe_id,
        "recipe_name": recipe_name,
        "product_pass_fail": overall_pass_fail,
        "camera_results": camera_results,
        "metadata": {
            "total_cameras": len(CAMERA_SERIALS),
            "total_frames": total_frames,
            "inference_stats": {
                "avg_confidence": round(avg_confidence, 4),
                "total_inliers": total_inliers,
                "total_matches": total_matches,
                "per_camera_stats": per_camera_stats,
                "overall_timings": {
                    "batch_total": round(random.uniform(50, 150), 2),
                    "trt_inference": round(random.uniform(30, 100), 2)
                }
            }
        },
        "timestamp": timestamp,
        "created_at": timestamp
    }


async def generate_and_insert_data(num_records: int = 100):
    """Generate and insert test data into MongoDB"""
    print(f"🚀 Starting data generation: {num_records} records")
    print(f"📦 Connecting to MongoDB: {settings.MONGODB_URL}")

    # Connect to MongoDB
    client = AsyncIOMotorClient(settings.MONGODB_URL)
    db = client[settings.DATABASE_NAME]
    collection = db["inference_results"]

    print(f"✅ Connected to database: {settings.DATABASE_NAME}")

    # Generate timestamps for last 7 days
    now = datetime.utcnow()
    timestamps = []

    for i in range(num_records):
        # Random time within last 7 days
        days_ago = random.uniform(0, 7)
        timestamp = now - timedelta(days=days_ago)
        timestamps.append(timestamp)

    # Sort timestamps (oldest first)
    timestamps.sort()

    # Generate and insert records
    records = []
    for i, timestamp in enumerate(timestamps):
        record = generate_inference_result(timestamp)
        records.append(record)

        if (i + 1) % 10 == 0:
            print(f"📝 Generated {i + 1}/{num_records} records...")

    print(f"💾 Inserting {len(records)} records into database...")
    result = await collection.insert_many(records)

    print(f"✅ Successfully inserted {len(result.inserted_ids)} records!")

    # Print statistics
    pass_count = sum(1 for r in records if r["product_pass_fail"] == "PASS")
    fail_count = sum(1 for r in records if r["product_pass_fail"] == "FAIL")

    print("\n📊 Statistics:")
    print(f"   Total records: {len(records)}")
    print(f"   PASS: {pass_count} ({pass_count/len(records)*100:.1f}%)")
    print(f"   FAIL: {fail_count} ({fail_count/len(records)*100:.1f}%)")
    print(f"   Recipes used: {len(set(r['recipe_id'] for r in records))}")
    print(f"   Date range: {timestamps[0].strftime('%Y-%m-%d')} to {timestamps[-1].strftime('%Y-%m-%d')}")

    # Close connection
    client.close()
    print("\n✨ Done!")


async def clear_test_data():
    """Clear all test data (optional)"""
    print("🗑️  Clearing test data...")

    client = AsyncIOMotorClient(settings.MONGODB_URL)
    db = client[settings.DATABASE_NAME]
    collection = db["inference_results"]

    # Delete all records with test recipe IDs
    result = await collection.delete_many({"recipe_id": {"$in": RECIPE_IDS}})

    print(f"✅ Deleted {result.deleted_count} test records")

    client.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate test inference results data")
    parser.add_argument("--num", type=int, default=100, help="Number of records to generate (default: 100)")
    parser.add_argument("--clear", action="store_true", help="Clear existing test data first")

    args = parser.parse_args()

    if args.clear:
        asyncio.run(clear_test_data())

    # asyncio.run(generate_and_insert_data(args.num))
