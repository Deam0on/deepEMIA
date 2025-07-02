"""
Results visualization components for Gradio interface.

This module provides components for viewing results, images, and downloading outputs.
"""
import gradio as gr
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import sys
from io import BytesIO
from PIL import Image

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from gui.streamlit_functions import (
    list_directories,
    list_png_files_in_gcs_folder,
    list_specific_csv_files_in_gcs_folder,
    create_zip_from_gcs,
    format_and_sort_folders
)
from gui.utils.gradio_helpers import (
    create_status_message,
    format_file_size,
    session_state
)
from src.utils.config import get_config

config = get_config()
GCS_BUCKET_NAME = config["bucket"]
GCS_DATASET_FOLDER = "DATASET"
GCS_INFERENCE_FOLDER = "DATASET/INFERENCE"
GCS_ARCHIVE_FOLDER = "Archive"


def create_results_viewer() -> gr.Group:
    """
    Create results viewing interface.
    
    Returns:
        gr.Group: Results viewer component group
    """
    with gr.Group() as results_group:
        gr.Markdown("## 📊 Results & Cloud Storage")
        
        # Folder browser
        with gr.Row():
            with gr.Column(scale=3):
                folder_dropdown = gr.Dropdown(
                    label="Browse GCS Folders",
                    choices=[],
                    value=None,
                    info="Select a folder to explore"
                )
            with gr.Column(scale=1):
                refresh_folders_btn = gr.Button("🔄 Refresh", size="sm")
        
        # Quick actions
        with gr.Row():
            show_images_btn = gr.Button("🖼️ Show Images", variant="secondary")
            show_results_btn = gr.Button("📈 Show Results", variant="secondary") 
            download_btn = gr.Button("📥 Download Folder", variant="primary")
        
        # Image gallery
        with gr.Accordion("🖼️ Image Gallery", open=False) as image_accordion:
            image_gallery = gr.Gallery(
                label="Inference Results",
                show_label=False,
                elem_id="results_gallery",
                columns=3,
                rows=2,
                object_fit="contain",
                height="auto"
            )
            
            image_info = gr.HTML("")
        
        # Results data
        with gr.Accordion("📊 Analysis Results", open=False) as results_accordion:
            results_data = gr.DataFrame(
                label="Measurement Results",
                interactive=False,
                wrap=True,
                height=400
            )
            
            results_info = gr.HTML("")
        
        # Download section
        with gr.Accordion("📥 Downloads", open=False) as download_accordion:
            download_status = gr.HTML("")
            download_file = gr.File(
                label="Download Ready",
                visible=False
            )
        
        # Status display
        status_display = gr.HTML("")
    
    def load_folders():
        """Load available folders from GCS."""
        try:
            folders = list_directories(GCS_BUCKET_NAME, "")
            formatted_folders = format_and_sort_folders(folders)
            session_state.set("folders", folders)
            
            return gr.Dropdown(
                choices=formatted_folders,
                value=formatted_folders[0] if formatted_folders else None
            )
        except Exception as e:
            return gr.Dropdown(choices=[], value=None)
    
    def show_images(selected_folder):
        """Display images from selected folder."""
        if not selected_folder:
            return (
                gr.Gallery(value=[]),
                create_status_message("error", "Please select a folder"),
                gr.Accordion(open=False)
            )
        
        try:
            # Extract folder path (remove formatting)
            folder_path = selected_folder.split(" - ")[0] if " - " in selected_folder else selected_folder
            
            # Get PNG files
            png_blobs = list_png_files_in_gcs_folder(GCS_BUCKET_NAME, folder_path)
            
            if not png_blobs:
                return (
                    gr.Gallery(value=[]),
                    create_status_message("info", f"No images found in {folder_path}"),
                    gr.Accordion(open=False)
                )
            
            # Load images (limit to first 20 for performance)
            images = []
            max_images = 20
            
            for i, blob in enumerate(png_blobs[:max_images]):
                try:
                    # Download image data
                    image_data = blob.download_as_bytes()
                    
                    # Open with PIL
                    image = Image.open(BytesIO(image_data))
                    
                    # Convert to RGB if needed
                    if image.mode != 'RGB':
                        image = image.convert('RGB')
                    
                    images.append(image)
                except Exception as e:
                    continue
            
            info_text = f"Showing {len(images)} images from {folder_path}"
            if len(png_blobs) > max_images:
                info_text += f" (limited to first {max_images} of {len(png_blobs)} total)"
            
            return (
                gr.Gallery(value=images),
                create_status_message("success", info_text),
                gr.Accordion(open=True)
            )
            
        except Exception as e:
            return (
                gr.Gallery(value=[]),
                create_status_message("error", f"Failed to load images: {str(e)}"),
                gr.Accordion(open=False)
            )
    
    def show_results(selected_folder):
        """Display CSV results from selected folder."""
        if not selected_folder:
            return (
                gr.DataFrame(value=None),
                create_status_message("error", "Please select a folder"),
                gr.Accordion(open=False)
            )
        
        try:
            # Extract folder path
            folder_path = selected_folder.split(" - ")[0] if " - " in selected_folder else selected_folder
            
            # Get CSV files
            csv_blobs = list_specific_csv_files_in_gcs_folder(GCS_BUCKET_NAME, folder_path)
            
            if not csv_blobs:
                return (
                    gr.DataFrame(value=None),
                    create_status_message("info", f"No result files found in {folder_path}"),
                    gr.Accordion(open=False)
                )
            
            # Load first CSV file
            blob = csv_blobs[0]
            csv_data = blob.download_as_text()
            
            # Parse CSV
            import pandas as pd
            from io import StringIO
            
            df = pd.read_csv(StringIO(csv_data))
            
            # Limit rows for display performance
            if len(df) > 1000:
                df_display = df.head(1000)
                info_text = f"Showing first 1000 rows of {len(df)} total from {blob.name}"
            else:
                df_display = df
                info_text = f"Showing {len(df)} rows from {blob.name}"
            
            return (
                gr.DataFrame(value=df_display),
                create_status_message("success", info_text),
                gr.Accordion(open=True)
            )
            
        except Exception as e:
            return (
                gr.DataFrame(value=None),
                create_status_message("error", f"Failed to load results: {str(e)}"),
                gr.Accordion(open=False)
            )
    
    def download_folder(selected_folder):
        """Create and provide download for selected folder."""
        if not selected_folder:
            return (
                create_status_message("error", "Please select a folder"),
                gr.File(visible=False)
            )
        
        try:
            # Extract folder path
            folder_path = selected_folder.split(" - ")[0] if " - " in selected_folder else selected_folder
            
            # Create safe filename
            safe_name = folder_path.replace("/", "_").replace("\\", "_")
            zip_filename = f"{safe_name}.zip"
            
            # Create ZIP archive
            zip_bytes = create_zip_from_gcs(GCS_BUCKET_NAME, folder_path, zip_filename)
            
            if not zip_bytes:
                return (
                    create_status_message("error", "Failed to create archive - folder may be empty"),
                    gr.File(visible=False)
                )
            
            # Save to temporary file
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp_file:
                tmp_file.write(zip_bytes)
                tmp_path = tmp_file.name
            
            file_size = format_file_size(len(zip_bytes))
            
            return (
                create_status_message("success", f"Archive created successfully ({file_size})"),
                gr.File(value=tmp_path, visible=True)
            )
            
        except Exception as e:
            return (
                create_status_message("error", f"Download failed: {str(e)}"),
                gr.File(visible=False)
            )
    
    # Connect event handlers
    refresh_folders_btn.click(load_folders, outputs=[folder_dropdown])
    
    show_images_btn.click(
        show_images,
        inputs=[folder_dropdown],
        outputs=[image_gallery, image_info, image_accordion]
    )
    
    show_results_btn.click(
        show_results,
        inputs=[folder_dropdown],
        outputs=[results_data, results_info, results_accordion]
    )
    
    download_btn.click(
        download_folder,
        inputs=[folder_dropdown],
        outputs=[download_status, download_file]
    )
    
    # Load folders on component creation
    results_group.load(load_folders, outputs=[folder_dropdown])
    
    return results_group
