import os
import uuid
from fastapi import HTTPException, UploadFile
import shutil
from typing import Tuple

# Define the upload directory
UPLOAD_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

def save_upload_file(upload_file: UploadFile, directory: str = UPLOAD_DIR) -> Tuple[str, str]:
    """
    Save an uploaded file to the specified directory
    Returns a tuple of (original_filename, stored_filename)
    """
    # Validate file extension (only PDF allowed)
    if not upload_file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    # Create a unique filename for storage
    file_extension = os.path.splitext(upload_file.filename)[1]
    stored_filename = f"{uuid.uuid4()}{file_extension}"
    file_path = os.path.join(directory, stored_filename)
    
    try:
        # Create a file in the uploads directory
        with open(file_path, "wb") as buffer:
            # Copy the file content
            shutil.copyfileobj(upload_file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")
    
    return upload_file.filename, stored_filename

def delete_file(stored_filename: str, directory: str = UPLOAD_DIR) -> bool:
    """
    Delete a file from the uploads directory
    Returns True if successful, False otherwise
    """
    file_path = os.path.join(directory, stored_filename)
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
            return True
        except Exception:
            return False
    return False

def get_file_path(stored_filename: str, directory: str = UPLOAD_DIR) -> str:
    """
    Get the full path to a stored file
    """
    return os.path.join(directory, stored_filename)