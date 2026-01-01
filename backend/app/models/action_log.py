from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, Dict, Any
from datetime import datetime
from enum import Enum


class ActionType(str, Enum):
    # Authentication
    LOGIN = "login"
    LOGOUT = "logout"

    # User Management
    CREATE_USER = "create_user"
    UPDATE_USER = "update_user"
    DELETE_USER = "delete_user"
    RESET_USER_PASSWORD = "reset_user_password"

    # Recipe Management
    CREATE_RECIPE = "create_recipe"
    UPDATE_RECIPE = "update_recipe"
    DELETE_RECIPE = "delete_recipe"
    LOAD_RECIPE = "load_recipe"

    # Camera Management
    CREATE_CAMERA = "create_camera"
    UPDATE_CAMERA = "update_camera"
    DELETE_CAMERA = "delete_camera"


class ActionLogBase(BaseModel):
    user_id: str
    username: str
    action_type: ActionType
    resource_type: str  # "user", "recipe", "camera", "auth"
    resource_id: Optional[str] = None
    description: str
    old_value: Optional[Dict[str, Any]] = None
    new_value: Optional[Dict[str, Any]] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None


class ActionLogCreate(ActionLogBase):
    pass


class ActionLogResponse(ActionLogBase):
    model_config = ConfigDict(populate_by_name=True)

    id: str = Field(alias="_id")
    timestamp: datetime


class ActionLogInDB(ActionLogBase):
    model_config = ConfigDict(populate_by_name=True)

    id: str = Field(alias="_id")
    timestamp: datetime