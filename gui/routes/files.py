"""
File management routes for GCS operations.
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from typing import List, Optional
import os
import shutil
from pathlib import Path

try:
    from gui.utils.gcs_operations import list_directories, create_zip_from_gcs
except ImportError:
    # Fallback implementations
    def list_directories(bucket_name: str, prefix: str = ""):
        return []
    
    def create_zip_from_gcs(bucket_name: str, prefix: str, output_path: str):
        return False

router = APIRouter(tags=["files"])

@router.get("/browse")
async def browse_gcs_files(prefix: Optional[str] = ""):
    """Browse files in GCS bucket."""
    try:
        # Mock bucket name - in production, get from config
        bucket_name = "your-bucket-name"
        
        # List directories and files
        files = list_directories(bucket_name, prefix)
        
        # Group by type
        directories = []
        images = []
        other_files = []
        
        for file_info in files:
            name = file_info.get('name', '')
            if name.endswith('/'):
                directories.append(file_info)
            elif name.lower().endswith(('.jpg', '.jpeg', '.png', '.tiff', '.tif')):
                images.append(file_info)
            else:
                other_files.append(file_info)
        
        return {
            "prefix": prefix,
            "directories": directories,
            "images": images,
            "other_files": other_files,
            "total_count": len(files)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to browse files: {str(e)}")

@router.post("/upload")
async def upload_files(
    files: List[UploadFile] = File(...),
    upload_path: str = Form("/dataset/images/")
):
    """Upload files to local storage and optionally to GCS."""
    try:
        uploaded_files = []
        
        # Create upload directory if it doesn't exist
        upload_dir = Path("uploads") / upload_path.strip("/")
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        for file in files:
            # Save file locally
            file_path = upload_dir / file.filename
            
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            uploaded_files.append({
                "filename": file.filename,
                "size": file_path.stat().st_size,
                "path": str(file_path)
            })
        
        return {
            "message": f"Successfully uploaded {len(uploaded_files)} files",
            "files": uploaded_files
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload files: {str(e)}")

@router.get("/download/{file_path:path}")
async def download_file(file_path: str):
    """Download a file from local storage or GCS."""
    try:
        # For now, just return info about the file
        # In production, this would serve the actual file
        return {
            "message": f"Download functionality for {file_path} will be implemented",
            "file_path": file_path
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to download file: {str(e)}")

@router.post("/create-zip")
async def create_results_zip(prefix: str = "results/"):
    """Create a ZIP file from GCS files."""
    try:
        bucket_name = "your-bucket-name"
        output_path = f"downloads/results_{prefix.replace('/', '_')}.zip"
        
        success = create_zip_from_gcs(bucket_name, prefix, output_path)
        
        if success:
            return {"message": "ZIP file created successfully", "download_path": output_path}
        else:
            return {"message": "No files found to zip", "download_path": None}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create ZIP: {str(e)}")

@router.get("/list-local")
async def list_local_files(directory: str = "uploads"):
    """List files in local storage."""
    try:
        base_path = Path(directory)
        if not base_path.exists():
            return {"files": [], "message": "Directory does not exist"}
        
        files = []
        for file_path in base_path.rglob("*"):
            if file_path.is_file():
                files.append({
                    "name": file_path.name,
                    "path": str(file_path.relative_to(base_path)),
                    "size": file_path.stat().st_size,
                    "modified": file_path.stat().st_mtime
                })
        
        return {"files": files, "count": len(files)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list files: {str(e)}")
