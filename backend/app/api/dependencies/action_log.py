from app.repositories.action_log_repository import ActionLogRepository
from app.db.mongodb import get_database
from fastapi import Depends


async def get_action_log_repository(db=Depends(get_database)) -> ActionLogRepository:
    """Get ActionLogRepository instance"""
    return ActionLogRepository(db)