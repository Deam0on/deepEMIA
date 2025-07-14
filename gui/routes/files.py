"""
File management routes for GCS operations.
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Header
from typing import List, Optional
import os
import shutil
from pathlib import Path

try:
    from gui.utils.gcs_operations import list_directories, create_zip_from_gcs, upload_file_to_gcs
except ImportError:
    # Fallback implementations
    def list_directories(bucket_name: str, prefix: str = ""):
        return []
    
    def create_zip_from_gcs(bucket_name: str, prefix: str, output_path: str):
        return False
    
    def upload_file_to_gcs(bucket_name: str, file_obj, destination: str):
        return None

router = APIRouter(tags=["files"])

@router.get("/browse")
async def browse_gcs_files(prefix: Optional[str] = ""):
    """Browse files in GCS bucket."""
    try:
        # Get bucket name from config
        try:
            import sys
            import os
            sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
            from src.utils.config import get_config
            config = get_config()
            bucket_name = config.get("bucket", "deepemia-bucket")
        except:
            bucket_name = "deepemia-bucket"  # fallback
        
        # List directories and files
        files = list_directories(bucket_name, prefix)
        
        # Convert string list to proper objects
        directories = []
        images = []
        other_files = []
        
        for file_path in files:
            # Create file info object
            file_info = {
                "name": file_path,
                "path": file_path,
                "size": 0,  # Size not available from directory listing
                "type": "directory" if file_path.endswith('/') else "file"
            }
            
            if file_path.endswith('/'):
                directories.append(file_info)
            elif file_path.lower().endswith(('.jpg', '.jpeg', '.png', '.tiff', '.tif')):
                images.append(file_info)
            else:
                other_files.append(file_info)
        
        return {
            "prefix": prefix,
            "directories": directories,
            "images": images,
            "other_files": other_files,
            "total_count": len(files),
            "bucket": bucket_name
        }
    except Exception as e:
        # Return graceful fallback
        return {
            "prefix": prefix,
            "directories": [],
            "images": [],
            "other_files": [],
            "total_count": 0,
            "error": str(e),
            "message": "GCS access not configured or permission denied"
        }

@router.post("/upload")
async def upload_files(
    files: List[UploadFile] = File(...),
    upload_path: str = Form("/dataset/images/"),
    x_admin_password: str = Header(None)
):
    """Upload files to local storage and optionally to GCS."""
    try:
        # Verify admin authentication if header is provided
        if x_admin_password:
            from gui.utils.gcs_operations import verify_admin_password
            if not verify_admin_password(x_admin_password):
                raise HTTPException(status_code=401, detail="Invalid admin password")
        
        uploaded_files = []
        
        # Create upload directory if it doesn't exist
        upload_dir = Path("uploads") / upload_path.strip("/")
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        for file in files:
            # Save file locally first
            file_path = upload_dir / file.filename
            
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            # Try to upload to GCS as well
            gcs_path = None
            try:
                # Reset file pointer for GCS upload
                file.file.seek(0)
                
                # Get bucket name from config
                try:
                    import sys
                    import os
                    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
                    from src.utils.config import get_config
                    config = get_config()
                    bucket_name = config.get("bucket", "deepemia-bucket")
                except:
                    bucket_name = "deepemia-bucket"  # fallback
                
                # Upload to GCS
                gcs_destination = f"{upload_path.strip('/')}/{file.filename}"
                gcs_path = upload_file_to_gcs(bucket_name, file.file, gcs_destination)
                
            except Exception as gcs_error:
                print(f"GCS upload failed for {file.filename}: {gcs_error}")
                # Continue with local upload even if GCS fails
            
            uploaded_files.append({
                "filename": file.filename,
                "size": file_path.stat().st_size,
                "local_path": str(file_path),
                "gcs_path": gcs_path,
                "uploaded_to_gcs": gcs_path is not None
            })
        
        return {
            "message": f"Successfully uploaded {len(uploaded_files)} files",
            "files": uploaded_files,
            "gcs_uploads": sum(1 for f in uploaded_files if f["uploaded_to_gcs"])
        }
    except HTTPException:
        raise
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
        # Get bucket name from config
        try:
            import sys
            import os
            sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
            from src.utils.config import get_config
            config = get_config()
            bucket_name = config.get("bucket", "deepemia-bucket")
        except:
            bucket_name = "deepemia-bucket"  # fallback
        
        output_path = f"downloads/results_{prefix.replace('/', '_')}.zip"
        
        # Create downloads directory if it doesn't exist
        os.makedirs("downloads", exist_ok=True)
        
        success = create_zip_from_gcs(bucket_name, prefix, output_path)
        
        if success:
            return {"message": "ZIP file created successfully", "download_path": output_path}
        else:
            return {"message": "No files found to zip or GCS access not configured", "download_path": None}
    except Exception as e:
        return {"message": f"Failed to create ZIP: {str(e)}", "download_path": None}

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

@router.get("/browse-archive")
async def browse_archive_files(prefix: Optional[str] = "Archive/"):
    """Browse archive files in GCS bucket."""
    try:
        # Get bucket name from config
        try:
            import sys
            import os
            sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
            from src.utils.config import get_config
            config = get_config()
            bucket_name = config.get("bucket", "deepemia-bucket")
        except:
            bucket_name = "deepemia-bucket"  # fallback
        
        # List archive files
        files = list_directories(bucket_name, prefix)
        
        # Process archive structure
        archive_folders = []
        for file_path in files:
            if file_path.endswith('/') and 'Archive/' in file_path:
                # Extract timestamp from folder name
                folder_name = file_path.replace('Archive/', '').replace('/', '')
                if folder_name:  # Not empty
                    archive_folders.append({
                        "name": folder_name,
                        "path": file_path,
                        "timestamp": folder_name,
                        "type": "archive_folder"
                    })
        
        # Sort by timestamp (newest first)
        archive_folders.sort(key=lambda x: x['timestamp'], reverse=True)
        
        return {
            "prefix": prefix,
            "archive_folders": archive_folders,
            "total_count": len(archive_folders),
            "bucket": bucket_name
        }
    except Exception as e:
        return {
            "prefix": prefix,
            "archive_folders": [],
            "total_count": 0,
            "error": str(e),
            "message": "Failed to browse archive files"
        }

@router.get("/list-archive-contents/{folder_name}")
async def list_archive_contents(folder_name: str):
    """List contents of a specific archive folder."""
    try:
        # Get bucket name from config
        try:
            import sys
            import os
            sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
            from src.utils.config import get_config
            config = get_config()
            bucket_name = config.get("bucket", "deepemia-bucket")
        except:
            bucket_name = "deepemia-bucket"  # fallback
        
        prefix = f"Archive/{folder_name}/"
        files = list_directories(bucket_name, prefix)
        
        # Categorize files
        png_files = []
        csv_files = []
        other_files = []
        
        for file_path in files:
            if not file_path.endswith('/'):  # Not a directory
                file_info = {
                    "name": os.path.basename(file_path),
                    "path": file_path,
                    "download_url": f"/api/files/download-from-gcs?file_path={file_path}"
                }
                
                if file_path.lower().endswith('.png'):
                    png_files.append(file_info)
                elif file_path.lower().endswith('.csv'):
                    csv_files.append(file_info)
                else:
                    other_files.append(file_info)
        
        return {
            "folder_name": folder_name,
            "png_files": png_files,
            "csv_files": csv_files,
            "other_files": other_files,
            "total_files": len(png_files) + len(csv_files) + len(other_files)
        }
    except Exception as e:
        return {
            "folder_name": folder_name,
            "png_files": [],
            "csv_files": [],
            "other_files": [],
            "total_files": 0,
            "error": str(e)
        }

@router.get("/download-from-gcs")
async def download_from_gcs(file_path: str):
    """Download a file from GCS."""
    try:
        # This would implement actual GCS file download
        # For now, return file info
        return {
            "message": f"Download link for {file_path}",
            "file_path": file_path,
            "download_url": f"gs://bucket/{file_path}",
            "note": "Direct download implementation would go here"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to download file: {str(e)}")
