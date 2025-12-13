"""
Camera Repository
Database operations for cameras
"""
from datetime import datetime
from typing import List, Optional
from motor.motor_asyncio import AsyncIOMotorDatabase
from bson import ObjectId

from app.models.camera import CameraModel
from app.schemas.camera import CameraCreate, CameraUpdate


class CameraRepository:
    """Repository for camera database operations"""
    
    def __init__(self, database: AsyncIOMotorDatabase):
        self.database = database
        self.collection = database.cameras
    
    async def create(self, camera: CameraCreate, created_by: str = "system") -> dict:
        """Create a new camera"""
        camera_dict = camera.model_dump()
        camera_dict["created_at"] = datetime.utcnow()
        camera_dict["updated_at"] = datetime.utcnow()
        camera_dict["created_by"] = created_by
        camera_dict["is_connected"] = False
        
        result = await self.collection.insert_one(camera_dict)
        camera_dict["_id"] = str(result.inserted_id)
        
        return camera_dict
    
    async def get_by_id(self, camera_id: str) -> Optional[dict]:
        """Get camera by camera_id"""
        camera = await self.collection.find_one({"camera_id": camera_id})
        if camera:
            camera["_id"] = str(camera["_id"])
        return camera
    
    async def get_by_serial(self, serial_number: str) -> Optional[dict]:
        """Get camera by serial number"""
        camera = await self.collection.find_one({"serial_number": serial_number})
        if camera:
            camera["_id"] = str(camera["_id"])
        return camera
    
    async def get_all(
        self,
        skip: int = 0,
        limit: int = 100,
        is_active: Optional[bool] = None
    ) -> List[dict]:
        """Get all cameras with pagination"""
        query = {}
        if is_active is not None:
            query["is_active"] = is_active
        
        cursor = self.collection.find(query).skip(skip).limit(limit)
        cameras = await cursor.to_list(length=limit)
        
        for camera in cameras:
            camera["_id"] = str(camera["_id"])
        
        return cameras
    
    async def get_connected_cameras(self) -> List[dict]:
        """Get all connected cameras"""
        cursor = self.collection.find({"is_connected": True, "is_active": True})
        cameras = await cursor.to_list(length=None)
        
        for camera in cameras:
            camera["_id"] = str(camera["_id"])
        
        return cameras
    
    async def update(self, camera_id: str, camera_update: CameraUpdate) -> Optional[dict]:
        """Update camera"""
        update_data = camera_update.model_dump(exclude_unset=True)
        if not update_data:
            return await self.get_by_id(camera_id)
        
        update_data["updated_at"] = datetime.utcnow()
        
        result = await self.collection.update_one(
            {"camera_id": camera_id},
            {"$set": update_data}
        )
        
        if result.modified_count:
            return await self.get_by_id(camera_id)
        return None
    
    async def update_connection_status(
        self,
        camera_id: str,
        is_connected: bool
    ) -> Optional[dict]:
        """Update camera connection status"""
        result = await self.collection.update_one(
            {"camera_id": camera_id},
            {
                "$set": {
                    "is_connected": is_connected,
                    "updated_at": datetime.utcnow()
                }
            }
        )
        
        if result.modified_count:
            return await self.get_by_id(camera_id)
        return None
    
    async def delete(self, camera_id: str) -> bool:
        """Delete camera"""
        result = await self.collection.delete_one({"camera_id": camera_id})
        return result.deleted_count > 0
    
    async def count(self, is_active: Optional[bool] = None) -> int:
        """Count cameras"""
        query = {}
        if is_active is not None:
            query["is_active"] = is_active
        
        return await self.collection.count_documents(query)
