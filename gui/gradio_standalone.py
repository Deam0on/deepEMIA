"""
Standalone Gradio interface for deepEMIA - No external component dependencies.
"""
import gradio as gr
import hashlib
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.utils.config import get_config
from src.utils.logger_utils import system_logger

# Load configuration
try:
    config = get_config()
    GCS_BUCKET_NAME = config["bucket"]
    GCS_DATASET_FOLDER = "DATASET"
    GCS_INFERENCE_FOLDER = "DATASET/INFERENCE"
except Exception as e:
    print(f"Warning: Could not load config: {e}")
    GCS_BUCKET_NAME = "your-bucket"
    GCS_DATASET_FOLDER = "DATASET"
    GCS_INFERENCE_FOLDER = "DATASET/INFERENCE"


def verify_admin_password(input_password: str) -> bool:
    """Verify admin password using environment variable hash."""
    stored_hash = os.environ.get('ADMIN_PASSWORD_HASH')
    if not stored_hash:
        return False
    
    input_hash = hashlib.sha256(input_password.encode()).hexdigest()
    return stored_hash == input_hash


def create_status_message(status: str, message: str) -> str:
    """Create a formatted status message."""
    status_colors = {
        "success": "#28a745",
        "error": "#dc3545", 
        "info": "#17a2b8",
        "warning": "#ffc107"
    }
    
    color = status_colors.get(status, "#6c757d")
    
    html = f'<div style="padding: 10px; margin: 10px 0; border-left: 4px solid {color}; background-color: #f8f9fa;">'
    html += f'<strong style="color: {color};">{status.upper()}:</strong> {message}'
    html += '</div>'
    return html


def create_standalone_app():
    """Create a completely standalone Gradio application."""
    
    with gr.Blocks(title="deepEMIA Control Panel") as app:
        
        # Header
        gr.HTML("""
        <div style="text-align: center; padding: 20px; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 10px; margin-bottom: 20px;">
            <h1 style="margin: 0; font-size: 2.5em; font-weight: bold;">🔬 deepEMIA Control Panel</h1>
            <p style="margin: 10px 0 0 0; font-size: 1.2em; opacity: 0.9;">
                Deep Learning Image Analysis • Standalone Interface
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
                            "Measurement Data",
                            "Training Data"
                        ],
                        value="Measurement Data"
                    )
                    overwrite_files = gr.Checkbox(
                        label="Overwrite existing files",
                        value=True
                    )
                
                file_upload = gr.File(
                    label="Select Files",
                    file_count="multiple"
                )
                
                upload_btn = gr.Button("📤 Upload Files", variant="primary")
                upload_status = gr.HTML("")
                
                # Dataset Management
                gr.Markdown("## 📁 Dataset Management")
                with gr.Row():
                    dataset_dropdown = gr.Dropdown(
                        label="Select Dataset",
                        choices=["polyhipes", "test_dataset"],
                        value="polyhipes"
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
                            "inference",
                            "evaluate", 
                            "prepare",
                            "train"
                        ],
                        value="inference"
                    )
                    
                    rcnn_model = gr.Dropdown(
                        label="RCNN Backbone",
                        choices=[
                            "50",
                            "101", 
                            "combo"
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
                        choices=["DATASET/", "DATASET/INFERENCE/", "Archive/"],
                        value="DATASET/"
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
                - **50**: Fast, good for small particles
                - **101**: Slower, better for large particles  
                - **combo**: Best overall performance
                
                **Threshold:**
                - Higher (0.8+): Fewer, more confident detections
                - Lower (0.5-0.6): More detections, some false positives
                
                ### 🔧 System Status
                - **Status**: 🟢 Online
                - **Backend**: Connected
                - **Version**: Standalone
                """)
        
        # Event handlers
        def upload_files(files, target, overwrite):
            """Handle file upload."""
            if not files:
                return create_status_message("error", "No files selected")
            
            return create_status_message("success", f"Would upload {len(files)} file(s) to {target}")
        
        def refresh_datasets():
            """Refresh dataset list."""
            return gr.Dropdown.update(choices=["polyhipes", "test_dataset", "new_dataset"])
        
        def create_dataset(password, name, classes_str):
            """Create a new dataset."""
            if not password:
                return create_status_message("error", "Admin password required")
            
            if not verify_admin_password(password):
                return create_status_message("error", "Invalid admin password")
            
            if not name or not classes_str:
                return create_status_message("error", "Please provide dataset name and classes")
            
            return create_status_message("success", f"Dataset '{name}' would be created with classes: {classes_str}")
        
        def delete_dataset(password, dataset_name):
            """Delete a dataset."""
            if not password:
                return create_status_message("error", "Admin password required")
            
            if not verify_admin_password(password):
                return create_status_message("error", "Invalid admin password")
            
            if not dataset_name:
                return create_status_message("error", "No dataset selected")
            
            return create_status_message("warning", f"Dataset '{dataset_name}' would be deleted")
        
        def run_task(task, dataset, threshold, rcnn, visualize, augment):
            """Run a task."""
            if not dataset:
                return create_status_message("error", "Please select a dataset")
            
            return create_status_message("info", f"Would run {task} on {dataset} with threshold {threshold} using {rcnn} model")
        
        def refresh_folders():
            """Refresh folder list."""
            return gr.Dropdown.update(choices=["DATASET/", "DATASET/INFERENCE/", "Archive/", "NEW_FOLDER/"])
        
        def show_images(folder):
            """Show images from folder."""
            if not folder:
                return create_status_message("error", "Please select a folder")
            
            return create_status_message("info", f"Would show images from {folder}")
        
        def download_folder(folder):
            """Download folder."""
            if not folder:
                return create_status_message("error", "Please select a folder")
            
            return create_status_message("info", f"Would download {folder}")
        
        # Connect event handlers
        upload_btn.click(
            upload_files,
            inputs=[file_upload, upload_target, overwrite_files],
            outputs=[upload_status]
        )
        
        refresh_datasets_btn.click(
            refresh_datasets,
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
            refresh_folders,
            outputs=[folder_dropdown]
        )
        
        show_images_btn.click(
            show_images,
            inputs=[folder_dropdown],
            outputs=[folder_status]
        )
        
        download_folder_btn.click(
            download_folder,
            inputs=[folder_dropdown],
            outputs=[folder_status]
        )
    
    return app


def launch_standalone_app(
    server_name: str = "0.0.0.0",
    server_port: int = 7860,
    share: bool = False,
    debug: bool = False
):
    """Launch the standalone Gradio application."""
    app = create_standalone_app()
    
    try:
        print(f"Starting deepEMIA standalone interface on {server_name}:{server_port}")
        
        app.launch(
            server_name=server_name,
            server_port=server_port,
            share=share,
            debug=debug,
            show_error=True,
            quiet=False
        )
        
    except Exception as e:
        print(f"Failed to launch interface: {str(e)}")
        raise


if __name__ == "__main__":
    launch_standalone_app()
