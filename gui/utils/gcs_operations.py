"""
Google Cloud Storage operations and utilities.
"""

import json
import os
import hashlib
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, List, Any
import sys

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

try:
    from google.cloud import storage
    from google.api_core import page_iterator
    from src.utils.config import get_config
    from src.utils.logger_utils import system_logger
    GCS_AVAILABLE = True
except ImportError as e:
    print(f"GCS dependencies not available: {e}")
    GCS_AVAILABLE = False
    
    # Fallback implementations
    class FallbackLogger:
        def info(self, msg): print(f"INFO: {msg}")
        def error(self, msg): print(f"ERROR: {msg}")
    
    def get_config():
        return {"bucket": "not-configured"}
    
    system_logger = FallbackLogger()

if GCS_AVAILABLE:
    config = get_config()
    GCS_BUCKET_NAME = config["bucket"]
else:
    GCS_BUCKET_NAME = "not-configured"

GCS_DATASET_INFO_PATH = "dataset_info.json"

def verify_admin_password(input_password: str) -> bool:
    """Verify admin password using environment variable hash."""
    stored_hash = os.environ.get('ADMIN_PASSWORD_HASH')
    if not stored_hash:
        # Fallback for development - hash of "admin"
        stored_hash = "8c6976e5b5410415bde908bd4dee15dfb167a9c873fc4bb8a81f6f2ab448a918"
    
    input_hash = hashlib.sha256(input_password.encode()).hexdigest()
    return stored_hash == input_hash

def load_dataset_names_from_gcs() -> Dict[str, List[Any]]:
    """Load dataset names from a JSON file in GCS."""
    if not GCS_AVAILABLE:
        return {
            "default": ["/home/DATASET/default/", "/home/DATASET/default/", ["class1", "class2"]],
            "polyhipes": ["/home/DATASET/polyhipes/", "/home/DATASET/polyhipes/", ["pores", "throats"]]
        }
    
    try:
        client = storage.Client()
        bucket = client.bucket(GCS_BUCKET_NAME)
        blob = bucket.blob(GCS_DATASET_INFO_PATH)
        
        if blob.exists():
            data = json.loads(blob.download_as_text())
            system_logger.info("Dataset info loaded from GCS")
            return data
        else:
            # Return default datasets if file doesn't exist
            default_data = {
                "default": ["/home/DATASET/default/", "/home/DATASET/default/", ["class1", "class2"]],
                "polyhipes": ["/home/DATASET/polyhipes/", "/home/DATASET/polyhipes/", ["pores", "throats"]]
            }
            save_dataset_names_to_gcs(default_data)
            return default_data
    except Exception as e:
        system_logger.error(f"Failed to load dataset info from GCS: {e}")
        return {
            "default": ["/home/DATASET/default/", "/home/DATASET/default/", ["class1", "class2"]]
        }

def save_dataset_names_to_gcs(data: Dict[str, List[Any]]) -> bool:
    """Save dataset names to a JSON file in GCS."""
    if not GCS_AVAILABLE:
        return False
    
    try:
        client = storage.Client()
        bucket = client.bucket(GCS_BUCKET_NAME)
        blob = bucket.blob(GCS_DATASET_INFO_PATH)
        
        blob.upload_from_string(json.dumps(data, indent=2))
        system_logger.info("Dataset info saved to GCS")
        return True
    except Exception as e:
        system_logger.error(f"Failed to save dataset info to GCS: {e}")
        return False

def list_directories(bucket_name: str, prefix: str) -> List[str]:
    """List directories in a GCS bucket with the given prefix."""
    if not GCS_AVAILABLE:
        return []
    
    if prefix and not prefix.endswith("/"):
        prefix = prefix + "/"

    extra_params = {"projection": "noAcl", "prefix": prefix, "delimiter": "/"}

    try:
        gcs = storage.Client()
        path = "/b/" + bucket_name + "/o"

        iterator = page_iterator.HTTPIterator(
            client=gcs,
            api_request=gcs._connection.api_request,
            path=path,
            items_key="prefixes",
            item_to_value=lambda iterator, item: item,
            extra_params=extra_params,
        )

        return [x for x in iterator]
    except Exception as e:
        system_logger.error(f"Failed to list directories: {e}")
        return []

def create_zip_from_gcs(bucket_name: str, folder: str, zip_name: str = "archive.zip") -> bytes:
    """Create a ZIP archive from files in a GCS bucket folder."""
    if not GCS_AVAILABLE:
        return b""
    
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blobs = bucket.list_blobs(prefix=folder)

        with TemporaryDirectory() as tempdir:
            zip_path = os.path.join(tempdir, zip_name)
            with zipfile.ZipFile(zip_path, "w") as zipf:
                for blob in blobs:
                    if (
                        blob.name.endswith(".png")
                        or blob.name.endswith("results_x_pred_1.csv")
                        or blob.name.endswith("results_x_pred_0.csv")
                    ):
                        file_path = os.path.join(tempdir, os.path.basename(blob.name))
                        blob.download_to_filename(file_path)
                        zipf.write(file_path, os.path.basename(blob.name))

            with open(zip_path, "rb") as f:
                return f.read()
    except Exception as e:
        system_logger.error(f"Failed to create ZIP from GCS: {e}")
        return b""

def format_and_sort_folders(folders: List[str]) -> List[tuple]:
    """Format and sort folder names for display."""
    formatted_folders = []
    for folder in folders:
        if folder:
            # Extract timestamp and format for display
            folder_parts = folder.strip("/").split("/")
            if len(folder_parts) >= 2:
                timestamp = folder_parts[-1]
                display_name = f"Archive/{timestamp}"
                formatted_folders.append((folder, display_name))
    
    # Sort by timestamp (newest first)
    formatted_folders.sort(key=lambda x: x[1], reverse=True)
    return formatted_folders
