"""
Modern Gradio interface for deepEMIA - Real backend integration.
"""
import gradio as gr
import hashlib
import os
import sys
import json
import traceback
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Import backend functions
from src.data.datasets import split_dataset
from src.functions.train_model import train_on_dataset
from src.functions.evaluate_model import evaluate_model
from src.functions.inference import run_inference
from src.utils.config import get_config
from src.utils.logger_utils import system_logger

# Load config
config = get_config()

def verify_admin_password(input_password: str) -> bool:
    """Verify admin password using environment variable hash."""
    stored_hash = os.environ.get('ADMIN_PASSWORD_HASH')
    if not stored_hash:
        return False
    
    input_hash = hashlib.sha256(input_password.encode()).hexdigest()
    return stored_hash == input_hash

def get_available_datasets():
    """Get list of available datasets from dataset_info.json."""
    try:
        dataset_info_path = Path(config["paths"]["category_json"]).expanduser().resolve()
        if dataset_info_path.exists():
            with open(dataset_info_path, 'r') as f:
                dataset_info = json.load(f)
            return list(dataset_info.keys())
        else:
            return ["polyhipes", "test_dataset"]  # Default fallback
    except Exception as e:
        system_logger.warning(f"Could not load dataset list: {e}")
        return ["polyhipes", "test_dataset"]  # Default fallback

def create_modern_app():
    """Create a modern Gradio application with real backend integration."""
    
    # Get available datasets
    available_datasets = get_available_datasets()
    
    def run_backend_task(
        task,
        dataset_name, 
        threshold,
        rcnn_model,
        dataset_format,
        visualize,
        draw_id,
        augment,
        optimize,
        n_trials,
        pass_mode,
        max_iters,
        admin_password,
        new_dataset_name,
        new_dataset_classes,
        test_size
    ):
        """Execute the selected task with real backend functions."""
        
        try:
            results = []
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            results.append(f"🕐 Started at {timestamp}")
            
            # Validate inputs based on task
            if task in ["prepare", "train", "evaluate", "inference"] and not dataset_name:
                return "❌ Error: Dataset name is required for this task."
            
            # Setup output directory
            output_dir = Path.home() / "deepEMIA" / "output"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            if task == "prepare":
                results.append(f"📁 Preparing dataset '{dataset_name}'...")
                
                # Find dataset directory
                dataset_path = Path.home() / dataset_name
                if not dataset_path.exists():
                    dataset_path = Path.home() / "datasets" / dataset_name
                if not dataset_path.exists():
                    return f"❌ Error: Dataset directory not found for '{dataset_name}'"
                
                # Split dataset
                train_files, test_files = split_dataset(
                    str(dataset_path), 
                    dataset_name, 
                    test_size=test_size
                )
                
                results.append(f"✅ Dataset split completed:")
                results.append(f"   • Training files: {len(train_files)}")
                results.append(f"   • Test files: {len(test_files)}")
                results.append(f"   • Test size: {test_size * 100:.1f}%")
                
            elif task == "train":
                results.append(f"🚀 Training model on '{dataset_name}'...")
                results.append(f"   • RCNN backbone: {rcnn_model}")
                results.append(f"   • Augmentation: {'Enabled' if augment else 'Disabled'}")
                results.append(f"   • HPO: {'Enabled' if optimize else 'Disabled'}")
                
                train_on_dataset(
                    dataset_name=dataset_name,
                    output_dir=str(output_dir),
                    dataset_format=dataset_format,
                    rcnn=rcnn_model,
                    augment=augment,
                    optimize=optimize,
                    n_trials=n_trials
                )
                
                results.append("✅ Training completed successfully!")
                
            elif task == "evaluate":
                results.append(f"📊 Evaluating model on '{dataset_name}'...")
                results.append(f"   • RCNN backbone: {rcnn_model}")
                results.append(f"   • Visualization: {'Enabled' if visualize else 'Disabled'}")
                
                # Convert rcnn_model to int if it's not "combo"
                rcnn_int = int(rcnn_model) if rcnn_model != "combo" else 101
                
                evaluate_model(
                    dataset_name=dataset_name,
                    output_dir=str(output_dir),
                    visualize=visualize,
                    dataset_format=dataset_format,
                    rcnn=rcnn_int
                )
                
                results.append("✅ Evaluation completed successfully!")
                
            elif task == "inference":
                results.append(f"🔍 Running inference on '{dataset_name}'...")
                results.append(f"   • RCNN backbone: {rcnn_model}")
                results.append(f"   • Threshold: {threshold}")
                results.append(f"   • Draw IDs: {'Yes' if draw_id else 'No'}")
                results.append(f"   • Pass mode: {pass_mode}")
                
                # Convert rcnn_model for inference
                if rcnn_model == "combo":
                    rcnn_val = "combo"
                else:
                    rcnn_val = int(rcnn_model)
                
                run_inference(
                    dataset_name=dataset_name,
                    output_dir=str(output_dir),
                    visualize=visualize,
                    threshold=threshold,
                    draw_id=draw_id,
                    dataset_format=dataset_format,
                    rcnn=rcnn_val,
                    pass_mode=pass_mode,
                    max_iters=max_iters
                )
                
                results.append("✅ Inference completed successfully!")
                
            # Admin dataset creation
            if admin_password and new_dataset_name and new_dataset_classes:
                if verify_admin_password(admin_password):
                    results.append(f"👤 Admin: Creating dataset '{new_dataset_name}'")
                    results.append(f"   • Classes: {new_dataset_classes}")
                    results.append("⚠️  Note: Dataset creation requires manual setup")
                else:
                    results.append("❌ Invalid admin password")
            
            # Add completion timestamp
            end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            results.append(f"🏁 Completed at {end_time}")
            
            return "\n".join(results)
            
        except Exception as e:
            error_msg = f"❌ Error: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            system_logger.error(f"GUI task failed: {e}")
            return error_msg

    def update_task_visibility(task):
        """Update the visibility of components based on selected task."""
        # Define which components are visible for each task
        show_dataset = task in ["prepare", "train", "evaluate", "inference"]
        show_threshold = task == "inference"
        show_visualize = task in ["evaluate", "inference"]
        show_draw_id = task == "inference"
        show_augment = task == "train"
        show_optimize = task == "train"
        show_n_trials = task == "train"
        show_pass_mode = task == "inference"
        show_max_iters = task == "inference"
        show_test_size = task == "prepare"
        
        return {
            dataset_dropdown: gr.update(visible=show_dataset),
            threshold_slider: gr.update(visible=show_threshold),
            visualize_checkbox: gr.update(visible=show_visualize),
            draw_id_checkbox: gr.update(visible=show_draw_id),
            augment_checkbox: gr.update(visible=show_augment),
            optimize_checkbox: gr.update(visible=show_optimize),
            n_trials_slider: gr.update(visible=show_n_trials),
            pass_mode_dropdown: gr.update(visible=show_pass_mode),
            max_iters_slider: gr.update(visible=show_max_iters),
            test_size_slider: gr.update(visible=show_test_size)
        }

    # Create the interface using Blocks for better control
    with gr.Blocks(
        theme=gr.themes.Soft(),
        title="🔬 deepEMIA Control Panel",
        css="""
        .main-container { max-width: 1200px; margin: 0 auto; }
        .task-section { border: 2px solid #e0e0e0; border-radius: 10px; padding: 20px; margin: 10px 0; }
        .admin-section { border: 2px solid #ffeb3b; border-radius: 10px; padding: 15px; margin: 10px 0; background-color: #fffef7; }
        """
    ) as interface:
        
        gr.Markdown("""
        # 🔬 deepEMIA Control Panel
        **Deep Learning Electron Microscopy Image Analysis**
        
        Select a task below and configure the parameters. Only relevant options will be shown.
        """)
        
        with gr.Row():
            with gr.Column(scale=2):
                with gr.Group():
                    gr.Markdown("### 🎯 Task Configuration")
                    
                    task_dropdown = gr.Dropdown(
                        choices=["prepare", "train", "evaluate", "inference"],
                        value="inference",
                        label="Select Task",
                        info="Choose what you want to do"
                    )
                    
                    dataset_dropdown = gr.Dropdown(
                        choices=available_datasets,
                        value=available_datasets[0] if available_datasets else None,
                        label="Dataset",
                        info="Select dataset to work with",
                        visible=True
                    )
                    
                    rcnn_dropdown = gr.Dropdown(
                        choices=["50", "101", "combo"],
                        value="101",
                        label="RCNN Model",
                        info="50=Fast, 101=Accurate, combo=Best"
                    )
                    
                    dataset_format_dropdown = gr.Dropdown(
                        choices=["json", "coco"],
                        value="json",
                        label="Dataset Format",
                        info="Annotation format"
                    )
                
                # Task-specific parameters (initially hidden)
                with gr.Group():
                    gr.Markdown("### ⚙️ Task-Specific Parameters")
                    
                    threshold_slider = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.65,
                        step=0.01,
                        label="Detection Threshold",
                        info="Higher = fewer, more confident detections",
                        visible=False
                    )
                    
                    test_size_slider = gr.Slider(
                        minimum=0.1,
                        maximum=0.5,
                        value=0.2,
                        step=0.05,
                        label="Test Set Size",
                        info="Proportion of data for testing",
                        visible=False
                    )
                    
                    visualize_checkbox = gr.Checkbox(
                        label="Generate Visualizations",
                        info="Create overlay images showing predictions",
                        visible=False
                    )
                    
                    draw_id_checkbox = gr.Checkbox(
                        label="Draw Instance IDs",
                        info="Show ID numbers on detected particles",
                        visible=False
                    )
                    
                    augment_checkbox = gr.Checkbox(
                        label="Data Augmentation",
                        info="Enable rotation, flip, brightness changes",
                        visible=False
                    )
                    
                    optimize_checkbox = gr.Checkbox(
                        label="Hyperparameter Optimization",
                        info="Use Optuna to find best parameters",
                        visible=False
                    )
                    
                    n_trials_slider = gr.Slider(
                        minimum=5,
                        maximum=50,
                        value=10,
                        step=1,
                        label="Optimization Trials",
                        info="More trials = better optimization",
                        visible=False
                    )
                    
                    pass_mode_dropdown = gr.Dropdown(
                        choices=["single", "multi"],
                        value="single",
                        label="Inference Pass Mode",
                        info="Single=Fast, Multi=More accurate",
                        visible=False
                    )
                    
                    max_iters_slider = gr.Slider(
                        minimum=1,
                        maximum=20,
                        value=10,
                        step=1,
                        label="Max Iterations (Multi-pass)",
                        info="Maximum iterations for multi-pass mode",
                        visible=False
                    )
                
                # Admin section
                with gr.Group():
                    gr.Markdown("### 👤 Admin Features")
                    with gr.Row():
                        admin_password = gr.Textbox(
                            type="password",
                            label="Admin Password",
                            placeholder="Enter admin password",
                            scale=2
                        )
                    with gr.Row():
                        new_dataset_name = gr.Textbox(
                            label="New Dataset Name",
                            placeholder="dataset_name",
                            scale=1
                        )
                        new_dataset_classes = gr.Textbox(
                            label="Classes (comma-separated)",
                            placeholder="particle,background",
                            scale=2
                        )
            
            with gr.Column(scale=1):
                gr.Markdown("### 📊 Results")
                
                # Submit button
                submit_btn = gr.Button(
                    "🚀 Execute Task",
                    variant="primary",
                    size="lg"
                )
                
                # Results output
                results_output = gr.Textbox(
                    label="Output",
                    lines=20,
                    max_lines=30,
                    value="ℹ️ Select a task and click Execute to begin...",
                    interactive=False
                )
        
        # Add examples section
        gr.Markdown("""
        ### 📋 Quick Examples
        
        **Common Workflows:**
        1. **New Dataset**: prepare → train → evaluate → inference
        2. **Quick Inference**: Just select inference task with existing model
        3. **Model Comparison**: evaluate with different RCNN models
        """)
        
        # Event handlers
        task_dropdown.change(
            fn=update_task_visibility,
            inputs=[task_dropdown],
            outputs=[
                dataset_dropdown, threshold_slider, visualize_checkbox, 
                draw_id_checkbox, augment_checkbox, optimize_checkbox, 
                n_trials_slider, pass_mode_dropdown, max_iters_slider, 
                test_size_slider
            ]
        )
        
        submit_btn.click(
            fn=run_backend_task,
            inputs=[
                task_dropdown, dataset_dropdown, threshold_slider, rcnn_dropdown,
                dataset_format_dropdown, visualize_checkbox, draw_id_checkbox,
                augment_checkbox, optimize_checkbox, n_trials_slider,
                pass_mode_dropdown, max_iters_slider,
                admin_password, new_dataset_name, new_dataset_classes,
                test_size_slider
            ],
            outputs=[results_output]
        )
    
    return interface


def launch_modern_app(
    server_name: str = "0.0.0.0",
    server_port: int = 7860,
    share: bool = False,  # Disable sharing - use only VM IP
    debug: bool = False
):
    """Launch the modern Gradio application."""
    interface = create_modern_app()
    
    try:
        print(f"Starting deepEMIA modern interface...")
        print(f"Access the interface at: http://{server_name}:{server_port}")
        
        interface.launch(
            server_name=server_name,
            server_port=server_port,
            share=share,  # No public sharing
            debug=debug,
            show_error=True,
            quiet=False,
            prevent_thread_lock=False
        )
        
    except Exception as e:
        print(f"Failed to launch interface: {str(e)}")
        raise


# Keep the old function name for backward compatibility
def launch_minimal_app(*args, **kwargs):
    """Backward compatibility wrapper."""
    return launch_modern_app(*args, **kwargs)


if __name__ == "__main__":
    launch_modern_app()
