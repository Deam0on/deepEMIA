"""
Minimal working Gradio interface for deepEMIA - Maximum compatibility.
"""
import gradio as gr
import hashlib
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).resolve().parent.parent))

def verify_admin_password(input_password: str) -> bool:
    """Verify admin password using environment variable hash."""
    stored_hash = os.environ.get('ADMIN_PASSWORD_HASH')
    if not stored_hash:
        return False
    
    input_hash = hashlib.sha256(input_password.encode()).hexdigest()
    return stored_hash == input_hash


def create_minimal_app():
    """Create a minimal Gradio application with maximum compatibility."""
    
    # Create interface using the Interface class (more compatible)
    def process_request(
        task,
        dataset, 
        threshold,
        rcnn_model,
        admin_password,
        new_dataset_name,
        new_dataset_classes
    ):
        """Process any request and return appropriate response."""
        
        results = []
        
        # Task execution
        if task and dataset:
            results.append(f"✅ Would execute {task} on dataset '{dataset}' with threshold {threshold} using {rcnn_model} model")
        
        # Admin operations
        if admin_password and new_dataset_name and new_dataset_classes:
            if verify_admin_password(admin_password):
                results.append(f"✅ Would create dataset '{new_dataset_name}' with classes: {new_dataset_classes}")
            else:
                results.append("❌ Invalid admin password")
        
        if not results:
            results.append("ℹ️ Configure your settings and click Submit")
        
        return "\n".join(results)
    
    # Create interface
    interface = gr.Interface(
        fn=process_request,
        inputs=[
            gr.Dropdown(
                choices=["inference", "evaluate", "prepare", "train"],
                value="inference",
                label="Select Task"
            ),
            gr.Dropdown(
                choices=["polyhipes", "test_dataset", "custom_dataset"],
                value="polyhipes", 
                label="Select Dataset"
            ),
            gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=0.65,
                step=0.01,
                label="Detection Threshold"
            ),
            gr.Dropdown(
                choices=["50", "101", "combo"],
                value="combo",
                label="RCNN Model"
            ),
            gr.Textbox(
                type="password",
                label="Admin Password (for dataset management)",
                placeholder="Leave empty if not creating datasets"
            ),
            gr.Textbox(
                label="New Dataset Name",
                placeholder="Enter name for new dataset"
            ),
            gr.Textbox(
                label="Dataset Classes (comma-separated)",
                placeholder="e.g., particle, background"
            )
        ],
        outputs=gr.Textbox(label="Results", lines=10),
        title="🔬 deepEMIA Control Panel",
        description="""
        **Deep Learning Image Analysis Interface**
        
        **Quick Start:**
        1. Select your task (inference is most common)
        2. Choose a dataset 
        3. Adjust threshold if needed
        4. Click Submit to execute
        
        **Admin Features:**
        - Enter admin password to create new datasets
        - Provide dataset name and classes
        
        **Model Guide:**
        - **50**: Fast, good for small particles
        - **101**: Slower, better for large particles  
        - **combo**: Best overall performance
        """,
        examples=[
            ["inference", "polyhipes", 0.65, "combo", "", "", ""],
            ["train", "polyhipes", 0.7, "101", "", "", ""],
            ["evaluate", "test_dataset", 0.6, "50", "", "", ""]
        ],
        theme=gr.themes.Soft(),
        allow_flagging="never"
    )
    
    return interface


def launch_minimal_app(
    server_name: str = "0.0.0.0",
    server_port: int = 7860,
    share: bool = True,  # Enable sharing by default
    debug: bool = False
):
    """Launch the minimal Gradio application."""
    interface = create_minimal_app()
    
    try:
        print(f"Starting deepEMIA minimal interface...")
        
        interface.launch(
            server_name=server_name,
            server_port=server_port,
            share=share,
            debug=debug,
            show_error=True,
            quiet=False,
            prevent_thread_lock=False
        )
        
    except Exception as e:
        print(f"Failed to launch interface: {str(e)}")
        raise


if __name__ == "__main__":
    launch_minimal_app()
