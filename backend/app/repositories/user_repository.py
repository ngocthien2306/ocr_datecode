from typing import Optional, List
from datetime import datetime
from bson import ObjectId
from app.models.user import UserCreate, UserUpdate, UserInDB, UserRole
from app.core.security import get_password_hash


class UserRepository:
    def __init__(self, database):
        self.collection = database.users

    async def create_user(self, user: UserCreate) -> UserInDB:
        """Create a new user"""
        user_dict = user.model_dump()
        user_dict["hashed_password"] = get_password_hash(user_dict.pop("password"))
        user_dict["created_at"] = datetime.utcnow()
        user_dict["updated_at"] = datetime.utcnow()
        user_dict["last_login"] = None

        result = await self.collection.insert_one(user_dict)
        user_dict["_id"] = str(result.inserted_id)

        return UserInDB(**user_dict)

    async def get_user_by_username(self, username: str) -> Optional[UserInDB]:
        """Get user by username"""
        user = await self.collection.find_one({"username": username})
        if user:
            user["_id"] = str(user["_id"])
            return UserInDB(**user)
        return None

    async def get_user_by_id(self, user_id: str) -> Optional[UserInDB]:
        """Get user by ID"""
        if not ObjectId.is_valid(user_id):
            return None

        user = await self.collection.find_one({"_id": ObjectId(user_id)})
        if user:
            user["_id"] = str(user["_id"])
            return UserInDB(**user)
        return None

    async def get_all_users(self, skip: int = 0, limit: int = 100) -> List[UserInDB]:
        """Get all users with pagination"""
        cursor = self.collection.find().skip(skip).limit(limit)
        users = []
        async for user in cursor:
            user["_id"] = str(user["_id"])
            users.append(UserInDB(**user))
        return users

    async def update_user(self, user_id: str, user_update: UserUpdate) -> Optional[UserInDB]:
        """Update user"""
        if not ObjectId.is_valid(user_id):
            return None

        update_data = {k: v for k, v in user_update.model_dump(exclude_unset=True).items() if v is not None}
        if not update_data:
            return await self.get_user_by_id(user_id)

        update_data["updated_at"] = datetime.utcnow()

        result = await self.collection.update_one(
            {"_id": ObjectId(user_id)},
            {"$set": update_data}
        )

        if result.modified_count:
            return await self.get_user_by_id(user_id)
        return None

    async def update_password(self, user_id: str, new_password: str) -> bool:
        """Update user password"""
        if not ObjectId.is_valid(user_id):
            return False

        hashed_password = get_password_hash(new_password)
        result = await self.collection.update_one(
            {"_id": ObjectId(user_id)},
            {"$set": {"hashed_password": hashed_password, "updated_at": datetime.utcnow()}}
        )

        return result.modified_count > 0

    async def update_last_login(self, user_id: str) -> bool:
        """Update user's last login timestamp"""
        if not ObjectId.is_valid(user_id):
            return False

        result = await self.collection.update_one(
            {"_id": ObjectId(user_id)},
            {"$set": {"last_login": datetime.utcnow()}}
        )

        return result.modified_count > 0

    async def delete_user(self, user_id: str) -> bool:
        """Delete user"""
        if not ObjectId.is_valid(user_id):
            return False

        result = await self.collection.delete_one({"_id": ObjectId(user_id)})
        return result.deleted_count > 0

    async def create_indexes(self):
        """Create database indexes"""
        await self.collection.create_index("username", unique=True)
        await self.collection.create_index("email", unique=True)
