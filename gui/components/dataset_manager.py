"""
Dataset management components for Gradio interface.

This module provides components for dataset creation, deletion, and management.
"""
import gradio as gr
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from gui.streamlit_functions import (
    load_dataset_names_from_gcs,
    save_dataset_names_to_gcs,
    list_directories,
    upload_files_to_gcs
)
from gui.utils.gradio_helpers import (
    verify_admin_password,
    validate_dataset_name,
    parse_class_list,
    create_status_message,
    session_state
)
from src.utils.config import get_config

config = get_config()
GCS_BUCKET_NAME = config["bucket"]
GCS_DATASET_FOLDER = "DATASET"
GCS_INFERENCE_FOLDER = "DATASET/INFERENCE"


def create_dataset_manager():
    """
    Create dataset management interface.
    
    Returns:
        Tuple[gr.Column, gr.Dropdown]: Dataset management component and dropdown
    """
    with gr.Column() as dataset_group:
        gr.Markdown("## 📁 Dataset Management")
        
        # Dataset selection
        with gr.Row():
            with gr.Column(scale=3):
                dataset_dropdown = gr.Dropdown(
                    label="Select Dataset",
                    choices=[],
                    value=None,
                    interactive=True,
                    info="Choose a dataset for operations"
                )
            with gr.Column(scale=1):
                refresh_btn = gr.Button("🔄 Refresh", size="sm")
        
        # Admin controls (initially hidden)
        with gr.Column(visible=False) as admin_controls:
            gr.Markdown("### 🔐 Admin Controls")
            
            # Password input
            with gr.Row():
                password_input = gr.Textbox(
                    label="Admin Password",
                    type="password",
                    placeholder="Enter admin password",
                    scale=3
                )
                auth_btn = gr.Button("Authenticate", scale=1)
            
            auth_status = gr.HTML("")
            
            # Dataset creation (initially hidden)
            with gr.Column(visible=False) as creation_controls:
                gr.Markdown("#### ➕ Create New Dataset")
                with gr.Row():
                    new_dataset_name = gr.Textbox(
                        label="Dataset Name",
                        placeholder="e.g., polyhipes_v2",
                        scale=2
                    )
                    new_dataset_classes = gr.Textbox(
                        label="Classes (comma-separated)",
                        placeholder="e.g., particle, background",
                        scale=2
                    )
                    create_btn = gr.Button("Create", scale=1, variant="primary")
                
                # Dataset deletion
                gr.Markdown("#### 🗑️ Delete Dataset")
                with gr.Row():
                    delete_confirm = gr.Checkbox(
                        label="I understand this will permanently delete the dataset",
                        value=False
                    )
                    delete_btn = gr.Button("Delete Selected Dataset", scale=1, variant="stop")
        
        # Show/Hide admin controls button
        admin_toggle = gr.Button("🔧 Admin Controls", variant="secondary")
        
        # Status display
        status_display = gr.HTML("")
    
    # Event handlers
    def load_datasets():
        """Load available datasets."""
        try:
            datasets = load_dataset_names_from_gcs()
            choices = list(datasets.keys()) if datasets else []
            session_state.set("datasets", datasets)
            return gr.Dropdown(choices=choices, value=choices[0] if choices else None)
        except Exception as e:
            return gr.Dropdown(choices=[], value=None)
    
    def toggle_admin_controls(current_state):
        """Toggle admin controls visibility."""
        return gr.Column(visible=not current_state)
    
    def authenticate_admin(password):
        """Authenticate admin user."""
        if verify_admin_password(password):
            session_state.set("admin_authenticated", True)
            return (
                create_status_message("success", "Authentication successful"),
                gr.Column(visible=True),
                gr.Textbox(value="")  # Clear password
            )
        else:
            return (
                create_status_message("error", "Invalid password"),
                gr.Column(visible=False),
                gr.Textbox(value="")  # Clear password
            )
    
    def create_dataset(name, classes_str):
        """Create a new dataset."""
        if not session_state.get("admin_authenticated", False):
            return create_status_message("error", "Admin authentication required")
        
        # Validate inputs
        is_valid, error_msg = validate_dataset_name(name)
        if not is_valid:
            return create_status_message("error", error_msg)
        
        classes = parse_class_list(classes_str)
        if not classes:
            return create_status_message("error", "At least one class must be specified")
        
        try:
            # Load current datasets
            datasets = session_state.get("datasets", {})
            
            if name in datasets:
                return create_status_message("error", f"Dataset '{name}' already exists")
            
            # Create new dataset entry
            path1 = f"/home/DATASET/{name}/"
            datasets[name] = {
                "path1": path1,
                "path2": path1,
                "classes": classes
            }
            
            # Save to GCS
            save_dataset_names_to_gcs(datasets)
            session_state.set("datasets", datasets)
            
            return create_status_message("success", f"Dataset '{name}' created successfully")
        
        except Exception as e:
            return create_status_message("error", f"Failed to create dataset: {str(e)}")
    
    def delete_dataset(dataset_name, confirm_delete):
        """Delete a dataset."""
        if not session_state.get("admin_authenticated", False):
            return create_status_message("error", "Admin authentication required")
        
        if not confirm_delete:
            return create_status_message("error", "Please confirm deletion by checking the box")
        
        if not dataset_name:
            return create_status_message("error", "No dataset selected")
        
        try:
            datasets = session_state.get("datasets", {})
            
            if dataset_name not in datasets:
                return create_status_message("error", f"Dataset '{dataset_name}' not found")
            
            # Remove dataset
            del datasets[dataset_name]
            
            # Save to GCS
            save_dataset_names_to_gcs(datasets)
            session_state.set("datasets", datasets)
            
            return create_status_message("success", f"Dataset '{dataset_name}' deleted successfully")
        
        except Exception as e:
            return create_status_message("error", f"Failed to delete dataset: {str(e)}")
    
    # Connect event handlers
    refresh_btn.click(load_datasets, outputs=[dataset_dropdown])
    
    admin_toggle.click(
        toggle_admin_controls,
        inputs=[admin_controls],
        outputs=[admin_controls]
    )
    
    auth_btn.click(
        authenticate_admin,
        inputs=[password_input],
        outputs=[auth_status, creation_controls, password_input]
    )
    
    create_btn.click(
        create_dataset,
        inputs=[new_dataset_name, new_dataset_classes],
        outputs=[status_display]
    )
    
    delete_btn.click(
        delete_dataset,
        inputs=[dataset_dropdown, delete_confirm],
        outputs=[status_display]
    )
    
    return dataset_group, dataset_dropdown


def create_file_upload():
    """
    Create file upload interface.
    
    Returns:
        Tuple[gr.Column, gr.File]: Upload component and file component
    """
    with gr.Column() as upload_group:
        gr.Markdown("## 📤 Upload Data")
        
        with gr.Row():
            upload_target = gr.Dropdown(
                label="Upload Target",
                choices=[
                    ("Measurement Data", GCS_INFERENCE_FOLDER),
                    ("Training Data", GCS_DATASET_FOLDER)
                ],
                value=GCS_INFERENCE_FOLDER,
                scale=2
            )
            overwrite_files = gr.Checkbox(
                label="Overwrite existing files",
                value=True,
                scale=1
            )
        
        file_upload = gr.File(
            label="Select Files",
            file_count="multiple",
            file_types=["image", ".json", ".csv"],
            height=200
        )
        
        upload_btn = gr.Button("📤 Upload Files", variant="primary")
        upload_status = gr.HTML("")
    
    def upload_files(files, target_folder, overwrite):
        """Upload files to GCS."""
        if not files:
            return create_status_message("error", "No files selected")
        
        try:
            upload_files_to_gcs(GCS_BUCKET_NAME, target_folder, files, overwrite)
            return create_status_message("success", f"Successfully uploaded {len(files)} file(s)")
        except Exception as e:
            return create_status_message("error", f"Upload failed: {str(e)}")
    
    upload_btn.click(
        upload_files,
        inputs=[file_upload, upload_target, overwrite_files],
        outputs=[upload_status]
    )
    
    return upload_group, file_upload
