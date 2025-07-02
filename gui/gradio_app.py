"""
Modern Gradio web interface for the deepEMIA project.

This module provides a modern, responsive web interface for:
- Dataset management and creation
- Model training, evaluation, and inference
- Real-time progress tracking and ETA estimation
- Results visualization and download
- File upload and cloud storage management

The interface is built with Gradio for optimal ML workflow integration
and provides a much improved user experience over the legacy Streamlit interface.
"""
import gradio as gr
from pathlib import Path
import sys
import os

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).resolve().parent.parent))

from gui.components.dataset_manager import create_dataset_manager, create_file_upload
from gui.components.task_runner import create_task_runner
from gui.components.results_viewer import create_results_viewer
from gui.utils.gradio_helpers import session_state, create_status_message
from src.utils.config import get_config
from src.utils.logger_utils import system_logger

# Load configuration
config = get_config()


def create_header():
    """Create the application header."""
    return gr.HTML("""
    <div style="text-align: center; padding: 20px; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 10px; margin-bottom: 20px;">
        <h1 style="margin: 0; font-size: 2.5em; font-weight: bold;">🔬 deepEMIA Control Panel</h1>
        <p style="margin: 10px 0 0 0; font-size: 1.2em; opacity: 0.9;">
            Deep Learning Image Analysis • Modern Web Interface
        </p>
    </div>
    """)


def create_help_sidebar():
    """Create help and tips sidebar."""
    with gr.Accordion("ℹ️ Help & Tips", open=False):
        gr.Markdown("""
        ### 🚀 Quick Start Guide
        
        1. **Upload Data**: Use the upload section to add images and annotations
        2. **Select Dataset**: Choose or create a dataset for your analysis
        3. **Configure Task**: Select inference, training, evaluation, or data preparation
        4. **Run Analysis**: Execute your task with real-time progress tracking
        5. **View Results**: Browse outputs, visualizations, and download results
        
        ### 💡 Tips
        
        - **Inference**: Best for analyzing new images with existing models
        - **Training**: Use when you have new annotated data to improve models
        - **Evaluation**: Test model performance on validation datasets
        - **Data Preparation**: Split datasets into training/testing sets
        
        ### ⚙️ Model Selection
        
        - **R50**: Optimized for small particles and faster inference
        - **R101**: Better accuracy for large particles, slower inference  
        - **Dual Model**: Universal model, good balance of speed and accuracy
        
        ### 🔧 Threshold Tuning
        
        - **Higher threshold (0.8+)**: Fewer detections, higher precision
        - **Lower threshold (0.5-0.6)**: More detections, may include false positives
        - **Recommended**: Start with 0.65 and adjust based on results
        
        ### 📊 Understanding Results
        
        - **Visualizations**: Check inference images for detection quality
        - **CSV Files**: Contain detailed measurements and statistics
        - **Progress Tracking**: ETA estimates become more accurate over time
        """)


def create_footer():
    """Create the application footer."""
    return gr.HTML("""
    <div style="text-align: center; padding: 15px; margin-top: 30px; border-top: 1px solid #e0e0e0; color: #666;">
        <p style="margin: 0;">
            🔬 deepEMIA Project • Built with ❤️ using Gradio • 
            <a href="https://github.com" style="color: #667eea; text-decoration: none;">GitHub</a>
        </p>
    </div>
    """)


def create_app():
    """Create the main Gradio application."""
    
    # Custom CSS for better styling
    custom_css = """
    .gradio-container {
        max-width: 1200px !important;
        margin: auto !important;
    }
    
    .gr-button {
        transition: all 0.3s ease;
    }
    
    .gr-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    .gr-form {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }
    
    #results_gallery {
        border-radius: 10px;
        overflow: hidden;
    }
    
    .gr-accordion {
        margin: 10px 0;
    }
    
    .gr-group {
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        padding: 15px;
        margin: 10px 0;
        background: white;
    }
    """
    
    # Create the main interface
    with gr.Blocks(
        title="deepEMIA Control Panel",
        theme=gr.themes.Soft(),
        css=custom_css
    ) as app:
        
        # Header
        create_header()
        
        # Main content area
        with gr.Row():
            # Main content (left side)
            with gr.Column(scale=4):
                
                # File upload section
                upload_group, file_upload = create_file_upload()
                
                # Dataset management
                dataset_group, dataset_dropdown = create_dataset_manager()
                
                # Task execution
                task_group = create_task_runner()
                
                # Results and visualization
                results_group = create_results_viewer()
            
            # Sidebar (right side)
            with gr.Column(scale=1):
                create_help_sidebar()
                
                # System status
                with gr.Accordion("🖥️ System Status", open=False):
                    system_info = gr.HTML("")
                    
                    def update_system_info():
                        """Update system information."""
                        try:
                            import psutil
                            cpu_percent = psutil.cpu_percent()
                            memory = psutil.virtual_memory()
                            disk = psutil.disk_usage('/')
                            
                            return f"""
                            <div style="font-size: 0.9em;">
                                <p><strong>CPU:</strong> {cpu_percent}%</p>
                                <p><strong>Memory:</strong> {memory.percent}% used</p>
                                <p><strong>Disk:</strong> {disk.percent}% used</p>
                                <p><strong>Status:</strong> <span style="color: green;">●</span> Online</p>
                            </div>
                            """
                        except:
                            return """
                            <div style="font-size: 0.9em;">
                                <p><strong>Status:</strong> <span style="color: green;">●</span> Online</p>
                                <p><strong>Backend:</strong> Connected</p>
                            </div>
                            """
                    
                    # Update system info periodically
                    app.load(update_system_info, outputs=[system_info], every=30)
        
        # Footer
        create_footer()
        
        # Global error handling
        def handle_global_error(error_msg):
            """Handle global application errors."""
            system_logger.error(f"Gradio interface error: {error_msg}")
            return create_status_message("error", f"Application error: {error_msg}")
        
        # Initialize application state
        def initialize_app():
            """Initialize the application on startup."""
            try:
                # Check environment variables
                if not os.environ.get('ADMIN_PASSWORD_HASH'):
                    system_logger.warning("ADMIN_PASSWORD_HASH not set - admin features will be disabled")
                
                # Initialize session state
                session_state.set("initialized", True)
                
                system_logger.info("Gradio interface initialized successfully")
                return create_status_message("success", "Application initialized successfully")
                
            except Exception as e:
                system_logger.error(f"Failed to initialize application: {str(e)}")
                return create_status_message("error", f"Initialization failed: {str(e)}")
        
        # Run initialization on app load
        app.load(initialize_app)
    
    return app


def launch_app(
    server_name: str = "127.0.0.1",
    server_port: int = 7860,
    share: bool = False,
    debug: bool = False
):
    """
    Launch the Gradio application.
    
    Args:
        server_name: Server hostname/IP
        server_port: Server port
        share: Whether to create a public link
        debug: Enable debug mode
    """
    app = create_app()
    
    try:
        system_logger.info(f"Starting deepEMIA Gradio interface on {server_name}:{server_port}")
        
        app.launch(
            server_name=server_name,
            server_port=server_port,
            share=share,
            debug=debug,
            favicon_path=None,  # Could add a custom favicon
            show_error=True,
            quiet=False
        )
        
    except Exception as e:
        system_logger.error(f"Failed to launch Gradio interface: {str(e)}")
        raise


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Launch deepEMIA Gradio Interface")
    parser.add_argument("--host", default="127.0.0.1", help="Server hostname")
    parser.add_argument("--port", type=int, default=7860, help="Server port")
    parser.add_argument("--share", action="store_true", help="Create public link")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    launch_app(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        debug=args.debug
    )
