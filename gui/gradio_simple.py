"""
Simple Gradio web interface for the deepEMIA project (Compatible with Gradio 4.15.0).

This module provides a modern, simplified web interface that works with older Gradio versions.
"""
import gradio as gr
from pathlib import Path
import sys
import os

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).resolve().parent.parent))

from gui.utils.gradio_helpers import session_state, create_status_message
from gui.streamlit_functions import (
    load_dataset_names_from_gcs,
    save_dataset_names_to_gcs,
    list_directories,
    upload_files_to_gcs,
    format_and_sort_folders,
    verify_admin_password
)
from src.utils.config import get_config
from src.utils.logger_utils import system_logger

# Load configuration
config = get_config()
GCS_BUCKET_NAME = config["bucket"]
GCS_DATASET_FOLDER = "DATASET"
GCS_INFERENCE_FOLDER = "DATASET/INFERENCE"


def create_simple_app():
    """Create a simplified Gradio application compatible with version 4.15.0."""
    
    # Custom CSS for better styling
    custom_css = """
    .gradio-container {
        max-width: 1200px !important;
        margin: auto !important;
    }
    """
    
    with gr.Blocks(title="deepEMIA Control Panel", css=custom_css) as app:
        
        # Header
        gr.HTML("""
        <div style="text-align: center; padding: 20px; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 10px; margin-bottom: 20px;">
            <h1 style="margin: 0; font-size: 2.5em; font-weight: bold;">🔬 deepEMIA Control Panel</h1>
            <p style="margin: 10px 0 0 0; font-size: 1.2em; opacity: 0.9;">
                Deep Learning Image Analysis • Modern Web Interface
            </p>
        </div>
        """)
        
        # Main interface
        with gr.Row():
            with gr.Column(scale=3):
                
                # File Upload Section
                gr.Markdown("## 📤 Upload Data")
                with gr.Row():
                    upload_target = gr.Dropdown(
                        label="Upload Target",
                        choices=[
                            ("Measurement Data", GCS_INFERENCE_FOLDER),
                            ("Training Data", GCS_DATASET_FOLDER)
                        ],
                        value=GCS_INFERENCE_FOLDER
                    )
                    overwrite_files = gr.Checkbox(
                        label="Overwrite existing files",
                        value=True
                    )
                
                file_upload = gr.File(
                    label="Select Files",
                    file_count="multiple",
                    file_types=["image", ".json", ".csv"]
                )
                
                upload_btn = gr.Button("📤 Upload Files", variant="primary")
                upload_status = gr.HTML("")
                
                # Dataset Management
                gr.Markdown("## 📁 Dataset Management")
                with gr.Row():
                    dataset_dropdown = gr.Dropdown(
                        label="Select Dataset",
                        choices=[],
                        value=None
                    )
                    refresh_datasets_btn = gr.Button("🔄 Refresh")
                
                # Admin Section
                gr.Markdown("### 🔐 Admin Controls")
                admin_password = gr.Textbox(
                    label="Admin Password",
                    type="password",
                    placeholder="Enter admin password"
                )
                
                with gr.Row():
                    new_dataset_name = gr.Textbox(
                        label="New Dataset Name",
                        placeholder="e.g., polyhipes_v2"
                    )
                    new_dataset_classes = gr.Textbox(
                        label="Classes (comma-separated)",
                        placeholder="e.g., particle, background"
                    )
                
                with gr.Row():
                    create_dataset_btn = gr.Button("➕ Create Dataset", variant="secondary")
                    delete_dataset_btn = gr.Button("🗑️ Delete Dataset", variant="stop")
                
                admin_status = gr.HTML("")
                
                # Task Execution
                gr.Markdown("## 🚀 Task Execution")
                with gr.Row():
                    task_selection = gr.Dropdown(
                        label="Select Task",
                        choices=[
                            ("🔍 Measurement (Inference)", "inference"),
                            ("📊 Model Evaluation", "evaluate"),
                            ("📋 Data Preparation", "prepare"),
                            ("🎯 Model Training", "train")
                        ],
                        value="inference"
                    )
                    
                    rcnn_model = gr.Dropdown(
                        label="RCNN Backbone",
                        choices=[
                            ("R50 (small particles)", "50"),
                            ("R101 (large particles)", "101"),
                            ("Dual model (universal)", "combo")
                        ],
                        value="combo"
                    )
                
                threshold_slider = gr.Slider(
                    label="Detection Threshold",
                    minimum=0.0,
                    maximum=1.0,
                    value=0.65,
                    step=0.01
                )
                
                with gr.Row():
                    enable_visualization = gr.Checkbox(
                        label="Generate Visualizations",
                        value=True
                    )
                    enable_augmentation = gr.Checkbox(
                        label="Enable Data Augmentation",
                        value=False
                    )
                
                run_task_btn = gr.Button("▶️ Run Task", variant="primary")
                task_status = gr.HTML("")
                
                # Cloud Storage Browser
                gr.Markdown("## 📊 Cloud Storage Browser")
                with gr.Row():
                    folder_dropdown = gr.Dropdown(
                        label="Browse GCS Folders",
                        choices=[],
                        value=None
                    )
                    refresh_folders_btn = gr.Button("🔄 Refresh")
                
                with gr.Row():
                    show_images_btn = gr.Button("🖼️ Show Images")
                    download_folder_btn = gr.Button("📥 Download Folder")
                
                folder_status = gr.HTML("")
                
            # Sidebar
            with gr.Column(scale=1):
                gr.Markdown("""
                ### 💡 Quick Tips
                
                **Getting Started:**
                1. Upload your data files
                2. Select or create a dataset
                3. Choose a task to run
                4. View results in cloud storage
                
                **Model Selection:**
                - **R50**: Fast, good for small particles
                - **R101**: Slower, better for large particles  
                - **Dual**: Best overall performance
                
                **Threshold:**
                - Higher (0.8+): Fewer, more confident detections
                - Lower (0.5-0.6): More detections, some false positives
                """)
        
        # Event handlers
        def upload_files(files, target_folder, overwrite):
            """Upload files to GCS."""
            if not files:
                return create_status_message("error", "No files selected")
            
            try:
                upload_files_to_gcs(GCS_BUCKET_NAME, target_folder, files, overwrite)
                return create_status_message("success", f"Successfully uploaded {len(files)} file(s)")
            except Exception as e:
                return create_status_message("error", f"Upload failed: {str(e)}")
        
        def load_datasets():
            """Load available datasets."""
            try:
                datasets = load_dataset_names_from_gcs()
                choices = list(datasets.keys()) if datasets else []
                session_state.set("datasets", datasets)
                return gr.Dropdown.update(choices=choices, value=choices[0] if choices else None)
            except Exception as e:
                return gr.Dropdown.update(choices=[], value=None)
        
        def create_dataset(password, name, classes_str):
            """Create a new dataset."""
            if not verify_admin_password(password):
                return create_status_message("error", "Invalid admin password")
            
            if not name or not classes_str:
                return create_status_message("error", "Please provide dataset name and classes")
            
            try:
                datasets = session_state.get("datasets", {})
                
                if name in datasets:
                    return create_status_message("error", f"Dataset '{name}' already exists")
                
                classes = [cls.strip() for cls in classes_str.split(",")]
                path1 = f"/home/DATASET/{name}/"
                
                datasets[name] = {
                    "path1": path1,
                    "path2": path1,
                    "classes": classes
                }
                
                save_dataset_names_to_gcs(datasets)
                session_state.set("datasets", datasets)
                
                return create_status_message("success", f"Dataset '{name}' created successfully")
            
            except Exception as e:
                return create_status_message("error", f"Failed to create dataset: {str(e)}")
        
        def delete_dataset(password, dataset_name):
            """Delete a dataset."""
            if not verify_admin_password(password):
                return create_status_message("error", "Invalid admin password")
            
            if not dataset_name:
                return create_status_message("error", "No dataset selected")
            
            try:
                datasets = session_state.get("datasets", {})
                
                if dataset_name not in datasets:
                    return create_status_message("error", f"Dataset '{dataset_name}' not found")
                
                del datasets[dataset_name]
                save_dataset_names_to_gcs(datasets)
                session_state.set("datasets", datasets)
                
                return create_status_message("success", f"Dataset '{dataset_name}' deleted successfully")
            
            except Exception as e:
                return create_status_message("error", f"Failed to delete dataset: {str(e)}")
        
        def run_task(task, dataset, threshold, rcnn, visualize, augment):
            """Run a task (simplified version)."""
            if not dataset:
                return create_status_message("error", "Please select a dataset")
            
            # In a real implementation, this would execute the task
            # For now, just show a placeholder message
            return create_status_message("info", f"Task '{task}' would run on dataset '{dataset}' with threshold {threshold}")
        
        def load_folders():
            """Load available folders from GCS."""
            try:
                folders = list_directories(GCS_BUCKET_NAME, "")
                formatted_folders = format_and_sort_folders(folders)
                session_state.set("folders", folders)
                
                return gr.Dropdown.update(choices=formatted_folders, value=formatted_folders[0] if formatted_folders else None)
            except Exception as e:
                return gr.Dropdown.update(choices=[], value=None)
        
        def show_folder_images(selected_folder):
            """Show images from selected folder."""
            if not selected_folder:
                return create_status_message("error", "Please select a folder")
            
            return create_status_message("info", f"Would show images from {selected_folder}")
        
        def download_folder(selected_folder):
            """Download selected folder."""
            if not selected_folder:
                return create_status_message("error", "Please select a folder")
            
            return create_status_message("info", f"Would download {selected_folder}")
        
        # Connect event handlers
        upload_btn.click(
            upload_files,
            inputs=[file_upload, upload_target, overwrite_files],
            outputs=[upload_status]
        )
        
        refresh_datasets_btn.click(
            load_datasets,
            outputs=[dataset_dropdown]
        )
        
        create_dataset_btn.click(
            create_dataset,
            inputs=[admin_password, new_dataset_name, new_dataset_classes],
            outputs=[admin_status]
        )
        
        delete_dataset_btn.click(
            delete_dataset,
            inputs=[admin_password, dataset_dropdown],
            outputs=[admin_status]
        )
        
        run_task_btn.click(
            run_task,
            inputs=[task_selection, dataset_dropdown, threshold_slider, rcnn_model, enable_visualization, enable_augmentation],
            outputs=[task_status]
        )
        
        refresh_folders_btn.click(
            load_folders,
            outputs=[folder_dropdown]
        )
        
        show_images_btn.click(
            show_folder_images,
            inputs=[folder_dropdown],
            outputs=[folder_status]
        )
        
        download_folder_btn.click(
            download_folder,
            inputs=[folder_dropdown],
            outputs=[folder_status]
        )
    
    return app


def launch_simple_app(
    server_name: str = "127.0.0.1",
    server_port: int = 7860,
    share: bool = False,
    debug: bool = False
):
    """
    Launch the simplified Gradio application.
    
    Args:
        server_name: Server hostname/IP
        server_port: Server port
        share: Whether to create a public link
        debug: Enable debug mode
    """
    app = create_simple_app()
    
    try:
        system_logger.info(f"Starting deepEMIA Gradio interface on {server_name}:{server_port}")
        
        app.launch(
            server_name=server_name,
            server_port=server_port,
            share=share,
            debug=debug,
            show_error=True,
            quiet=False
        )
        
    except Exception as e:
        system_logger.error(f"Failed to launch Gradio interface: {str(e)}")
        raise


if __name__ == "__main__":
    launch_simple_app()
