from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File
from fastapi.responses import FileResponse
import uuid
from pathlib import Path
from PIL import Image
import io

from app.models.user import UserInDB
from app.api.dependencies.auth import get_current_user

# Handle Pillow version compatibility
try:
    # Pillow >= 9.1.0
    LANCZOS = Image.Resampling.LANCZOS
except AttributeError:
    # Pillow < 9.1.0
    LANCZOS = Image.LANCZOS

router = APIRouter()

# Avatar upload directory
AVATAR_UPLOAD_DIR = Path("uploads/avatar")
AVATAR_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@router.post("/avatar")
async def upload_avatar(
    file: UploadFile = File(...),
    current_user: UserInDB = Depends(get_current_user)
):
    """
    Upload a user avatar image and return its URL.

    **Permission**: All authenticated users

    Args:
        file: Image file to upload (PNG, JPG, etc.)

    Returns:
        {
            "url": "/api/upload/avatars/{filename}"
        }
    """
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must be an image"
        )

    # Read image data
    image_data = await file.read()

    # Validate file size (max 5MB)
    if len(image_data) > 5 * 1024 * 1024:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Image size must be less than 5MB"
        )

    # Validate and get image format using PIL
    try:
        image = Image.open(io.BytesIO(image_data))
        image_format = image.format.lower() if image.format else 'jpg'

        # Optional: Resize image if too large (e.g., max 500x500 for avatars)
        max_size = (500, 500)
        if image.size[0] > max_size[0] or image.size[1] > max_size[1]:
            image.thumbnail(max_size, LANCZOS)

            # Convert resized image back to bytes
            output = io.BytesIO()
            image.save(output, format=image_format.upper())
            image_data = output.getvalue()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid image file: {str(e)}"
        )

    # Generate unique filename
    file_extension = file.filename.split('.')[-1] if '.' in file.filename else image_format
    unique_filename = f"{uuid.uuid4()}.{file_extension}"
    file_path = AVATAR_UPLOAD_DIR / unique_filename

    # Save file
    with open(file_path, 'wb') as f:
        f.write(image_data)

    # Return URL
    return {
        "url": f"/api/upload/avatars/{unique_filename}"
    }


@router.get("/avatars/{filename}")
async def get_avatar_image(filename: str):
    """
    Serve an avatar image file.

    **Permission**: Public (no auth required for image serving)
    """
    file_path = AVATAR_UPLOAD_DIR / filename

    if not file_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Avatar image not found"
        )

    return FileResponse(file_path)


@router.delete("/avatar")
async def delete_avatar(
    url: str,
    current_user: UserInDB = Depends(get_current_user)
):
    """
    Delete an avatar image.

    **Permission**: All authenticated users

    Args:
        url: The URL of the avatar to delete

    Returns:
        {"message": "Avatar deleted successfully"}
    """
    # Extract filename from URL
    try:
        filename = url.split('/')[-1]
        file_path = AVATAR_UPLOAD_DIR / filename

        if file_path.exists():
            file_path.unlink()
            return {"message": "Avatar deleted successfully"}
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Avatar not found"
            )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to delete avatar: {str(e)}"
        )
