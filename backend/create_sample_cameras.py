"""
Create Sample Cameras
Script to create sample camera entries in the database
"""
import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
from datetime import datetime
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
DATABASE_NAME = os.getenv("DATABASE_NAME", "ocr_datecode")


async def create_sample_cameras():
    """Create sample cameras in the database"""
    
    # Connect to MongoDB
    client = AsyncIOMotorClient(MONGODB_URL)
    db = client[DATABASE_NAME]
    cameras_collection = db.cameras
    
    print(f"Connected to MongoDB: {MONGODB_URL}")
    print(f"Database: {DATABASE_NAME}")
    print("-" * 60)
    
    # Sample camera data
    sample_cameras = [
        {
            "camera_id": "CAM001",
            "model_name": "Basler acA1920-40gm",
            "serial_number": "40123456",
            "resolution_width": 1920,
            "resolution_height": 1200,
            "is_connected": True,
            "is_active": True,
            "ip_address": "192.168.1.100",
            "location": "Production Line 1 - Position A",
            "description": "Main inspection camera for date code reading",
            "max_frame_rate": 40.0,
            "pixel_format": "Mono8",
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
            "created_by": "admin"
        },
        {
            "camera_id": "CAM002",
            "model_name": "Basler acA1920-40gm",
            "serial_number": "40123457",
            "resolution_width": 1920,
            "resolution_height": 1200,
            "is_connected": True,
            "is_active": True,
            "ip_address": "192.168.1.101",
            "location": "Production Line 1 - Position B",
            "description": "Secondary inspection camera for date code verification",
            "max_frame_rate": 40.0,
            "pixel_format": "Mono8",
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
            "created_by": "admin"
        },
        {
            "camera_id": "CAM003",
            "model_name": "Basler acA2440-75um",
            "serial_number": "40234567",
            "resolution_width": 2448,
            "resolution_height": 2048,
            "is_connected": False,
            "is_active": True,
            "ip_address": "192.168.1.102",
            "location": "Production Line 2 - Position A",
            "description": "High resolution camera for detailed inspection",
            "max_frame_rate": 75.0,
            "pixel_format": "Mono8",
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
            "created_by": "admin"
        }
    ]
    
    # Clear existing cameras (optional)
    print("Clearing existing cameras...")
    result = await cameras_collection.delete_many({})
    print(f"Deleted {result.deleted_count} existing cameras")
    print("-" * 60)
    
    # Insert sample cameras
    print("Creating sample cameras...")
    for camera in sample_cameras:
        try:
            # Check if camera already exists
            existing = await cameras_collection.find_one({"camera_id": camera["camera_id"]})
            
            if existing:
                print(f"⚠ Camera {camera['camera_id']} already exists, skipping...")
                continue
            
            result = await cameras_collection.insert_one(camera)
            print(f"✅ Created camera: {camera['camera_id']} - {camera['model_name']}")
            print(f"   Serial: {camera['serial_number']}")
            print(f"   Resolution: {camera['resolution_width']}x{camera['resolution_height']}")
            print(f"   Location: {camera['location']}")
            print(f"   Status: {'Connected' if camera['is_connected'] else 'Disconnected'}")
            print(f"   MongoDB ID: {result.inserted_id}")
            print()
            
        except Exception as e:
            print(f"❌ Error creating camera {camera['camera_id']}: {e}")
            print()
    
    # Display summary
    print("-" * 60)
    total_cameras = await cameras_collection.count_documents({})
    connected_cameras = await cameras_collection.count_documents({"is_connected": True})
    active_cameras = await cameras_collection.count_documents({"is_active": True})
    
    print("📊 Summary:")
    print(f"   Total cameras: {total_cameras}")
    print(f"   Connected cameras: {connected_cameras}")
    print(f"   Active cameras: {active_cameras}")
    print("-" * 60)
    
    # Close connection
    client.close()
    print("✅ Done! Database connection closed.")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("CREATE SAMPLE CAMERAS")
    print("=" * 60 + "\n")
    
    asyncio.run(create_sample_cameras())
    
    print("\n" + "=" * 60)
    print("To test the API, run:")
    print("  curl http://localhost:8000/api/cameras")
    print("  curl http://localhost:8000/api/cameras/CAM001")
    print("=" * 60 + "\n")
