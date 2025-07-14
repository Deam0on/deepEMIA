"""
API routes for dataset management operations.
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Dict, Any
import json
import hashlib
import os

from gui.utils.gcs_operations import (
    load_dataset_names_from_gcs,
    save_dataset_names_to_gcs,
    verify_admin_password
)

router = APIRouter(prefix="/api/datasets", tags=["datasets"])

class DatasetCreate(BaseModel):
    name: str
    classes: List[str]

class DatasetResponse(BaseModel):
    name: str
    path1: str
    path2: str
    classes: List[str]

class AdminAuth(BaseModel):
    password: str

def verify_admin(auth: AdminAuth):
    """Dependency to verify admin authentication."""
    if not verify_admin_password(auth.password):
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

@router.post("/", dependencies=[Depends(verify_admin)])
async def create_dataset(dataset: DatasetCreate, auth: AdminAuth):
    """Create a new dataset (admin only)."""
    try:
        datasets = load_dataset_names_from_gcs()
        
        if dataset.name in datasets:
            raise HTTPException(status_code=400, detail="Dataset already exists")
        
        path1 = f"/home/DATASET/{dataset.name}/"
        path2 = path1
        datasets[dataset.name] = [path1, path2, dataset.classes]
        
        save_dataset_names_to_gcs(datasets)
        return {"message": f"Dataset '{dataset.name}' created successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create dataset: {str(e)}")

@router.delete("/{dataset_name}", dependencies=[Depends(verify_admin)])
async def delete_dataset(dataset_name: str, auth: AdminAuth):
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

@router.post("/verify-admin")
async def verify_admin_endpoint(auth: AdminAuth):
    """Verify admin password."""
    if verify_admin_password(auth.password):
        return {"valid": True}
    else:
        return {"valid": False}
