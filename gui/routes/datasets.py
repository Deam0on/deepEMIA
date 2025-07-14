"""
API routes for dataset management operations.
"""

from fastapi import APIRouter, HTTPException, Depends, Header
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import json
import hashlib
import os

from gui.utils.gcs_operations import (
    load_dataset_names_from_gcs,
    save_dataset_names_to_gcs,
    verify_admin_password
)

router = APIRouter(tags=["datasets"])

class DatasetCreate(BaseModel):
    name: str
    description: Optional[str] = ""
    classes: List[str] = []

class DatasetResponse(BaseModel):
    name: str
    path1: str
    path2: str
    classes: List[str]

def verify_admin_header(x_admin_password: str = Header(...)):
    """Dependency to verify admin authentication via header."""
    if not verify_admin_password(x_admin_password):
        raise HTTPException(status_code=401, detail="Invalid admin password")
    return True

@router.get("/", response_model=Dict[str, List[Any]])
async def get_datasets():
    """Get all available datasets."""
    try:
        datasets = load_dataset_names_from_gcs()
        return datasets
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load datasets: {str(e)}")

@router.post("/", dependencies=[Depends(verify_admin_header)])
async def create_dataset(dataset: DatasetCreate):
    """Create a new dataset (admin only)."""
    try:
        datasets = load_dataset_names_from_gcs()
        
        if dataset.name in datasets:
            raise HTTPException(status_code=400, detail="Dataset already exists")
        
        # Create timestamp for creation date
        from datetime import datetime
        created = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        datasets[dataset.name] = [created, dataset.description, dataset.classes]
        
        save_dataset_names_to_gcs(datasets)
        return {"message": f"Dataset '{dataset.name}' created successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create dataset: {str(e)}")

@router.delete("/{dataset_name}", dependencies=[Depends(verify_admin_header)])
async def delete_dataset(dataset_name: str):
    """Delete a dataset (admin only)."""
    try:
        datasets = load_dataset_names_from_gcs()
        
        if dataset_name not in datasets:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        del datasets[dataset_name]
        save_dataset_names_to_gcs(datasets)
        return {"message": f"Dataset '{dataset_name}' deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete dataset: {str(e)}")

@router.post("/load-from-gcs", dependencies=[Depends(verify_admin_header)])
async def load_datasets_from_gcs():
    """Load datasets from GCS (admin only)."""
    try:
        datasets = load_dataset_names_from_gcs()
        count = len(datasets)
        return {"message": f"Loaded {count} datasets from GCS", "count": count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load from GCS: {str(e)}")

@router.post("/save-to-gcs", dependencies=[Depends(verify_admin_header)])
async def save_datasets_to_gcs():
    """Save datasets to GCS (admin only)."""
    try:
        datasets = load_dataset_names_from_gcs()
        save_dataset_names_to_gcs(datasets)
        return {"message": "Datasets saved to GCS successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save to GCS: {str(e)}")
