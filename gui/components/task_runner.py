"""
Task execution components for Gradio interface.

This module provides components for running training, evaluation, and inference tasks.
"""
import gradio as gr
import subprocess
import threading
import time
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from gui.streamlit_functions import estimate_eta, contains_errors
from gui.utils.gradio_helpers import (
    create_status_message,
    format_time_remaining,
    session_state,
    progress_tracker
)
from src.utils.config import get_config

config = get_config()
MAIN_SCRIPT_PATH = Path(config["paths"]["main_script"]).expanduser().resolve()


def create_task_runner() -> gr.Group:
    """
    Create task execution interface.
    
    Returns:
        gr.Group: Task runner component group
    """
    with gr.Group() as task_group:
        gr.Markdown("## 🚀 Task Execution")
        
        # Task configuration
        with gr.Row():
            with gr.Column(scale=2):
                task_selection = gr.Dropdown(
                    label="Select Task",
                    choices=[
                        ("🔍 Measurement (Inference)", "inference"),
                        ("📊 Model Evaluation", "evaluate"),
                        ("📋 Data Preparation", "prepare"),
                        ("🎯 Model Training", "train")
                    ],
                    value="inference",
                    info="Choose the task to execute"
                )
            
            with gr.Column(scale=2):
                dataset_selection = gr.Dropdown(
                    label="Dataset",
                    choices=[],
                    value=None,
                    info="Select dataset for the task"
                )
        
        # Advanced parameters
        with gr.Accordion("⚙️ Advanced Parameters", open=False):
            with gr.Row():
                threshold_slider = gr.Slider(
                    label="Detection Threshold",
                    minimum=0.0,
                    maximum=1.0,
                    value=0.65,
                    step=0.01,
                    info="Confidence threshold for detection"
                )
                
                rcnn_model = gr.Dropdown(
                    label="RCNN Backbone",
                    choices=[
                        ("R50 (small particles)", "50"),
                        ("R101 (large particles)", "101"),
                        ("Dual model (universal)", "combo")
                    ],
                    value="combo",
                    info="Choose model architecture"
                )
            
            with gr.Row():
                enable_augmentation = gr.Checkbox(
                    label="Enable Data Augmentation",
                    value=False,
                    info="Apply data augmentation during training"
                )
                
                enable_visualization = gr.Checkbox(
                    label="Generate Visualizations",
                    value=True,
                    info="Create visual outputs and reports"
                )
                
                use_gpu = gr.Checkbox(
                    label="Use GPU",
                    value=True,
                    info="Enable GPU acceleration if available"
                )
        
        # Execution controls
        with gr.Row():
            run_btn = gr.Button("▶️ Run Task", variant="primary", scale=2)
            stop_btn = gr.Button("⏹️ Stop", variant="stop", scale=1, interactive=False)
            
        # Progress and status
        progress_bar = gr.Progress()
        status_display = gr.HTML("")
        
        # Real-time logs (collapsible)
        with gr.Accordion("📋 Task Logs", open=False):
            log_display = gr.Textbox(
                label="Live Output",
                lines=15,
                max_lines=20,
                interactive=False,
                show_copy_button=True
            )
    
    # Task execution state
    current_process = None
    
    def update_dataset_choices():
        """Update dataset choices from session state."""
        datasets = session_state.get("datasets", {})
        choices = list(datasets.keys()) if datasets else []
        return gr.Dropdown(choices=choices, value=choices[0] if choices else None)
    
    def run_task(task, dataset, threshold, rcnn, augment, visualize, gpu):
        """Execute a task with given parameters."""
        nonlocal current_process
        
        if not dataset:
            return create_status_message("error", "Please select a dataset")
        
        # Build command
        cmd = [
            "python", str(MAIN_SCRIPT_PATH),
            "--task", task,
            "--dataset_name", dataset,
            "--threshold", str(threshold),
            "--rcnn", rcnn
        ]
        
        if augment and task == "train":
            cmd.append("--augment")
        
        if visualize:
            cmd.append("--visualize")
        
        if not gpu:
            cmd.extend(["--device", "cpu"])
        
        try:
            # Start progress tracking
            progress_tracker.start(f"Starting {task} task...")
            
            # Estimate ETA
            num_images = 100  # Default estimate, could be improved
            eta_seconds = estimate_eta(task, num_images)
            
            progress_tracker.update(5, f"Executing {task} on dataset '{dataset}'", eta_seconds)
            
            # Execute command
            current_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # Start monitoring thread
            threading.Thread(
                target=monitor_process,
                args=(current_process, task),
                daemon=True
            ).start()
            
            return (
                create_status_message("info", f"Task '{task}' started successfully", 5, eta_seconds),
                gr.Button(interactive=False),  # Disable run button
                gr.Button(interactive=True)    # Enable stop button
            )
            
        except Exception as e:
            progress_tracker.finish(f"Failed to start task: {str(e)}")
            return (
                create_status_message("error", f"Failed to start task: {str(e)}"),
                gr.Button(interactive=True),   # Re-enable run button
                gr.Button(interactive=False)   # Disable stop button
            )
    
    def stop_task():
        """Stop the currently running task."""
        nonlocal current_process
        
        if current_process and current_process.poll() is None:
            current_process.terminate()
            time.sleep(2)
            
            if current_process.poll() is None:
                current_process.kill()
            
            progress_tracker.finish("Task stopped by user")
            return (
                create_status_message("warning", "Task stopped by user"),
                gr.Button(interactive=True),   # Re-enable run button
                gr.Button(interactive=False)   # Disable stop button
            )
        else:
            return (
                create_status_message("info", "No running task to stop"),
                gr.Button(interactive=True),   # Re-enable run button
                gr.Button(interactive=False)   # Disable stop button
            )
    
    def monitor_process(process, task_name):
        """Monitor process execution and update progress."""
        log_lines = []
        
        try:
            while process.poll() is None:
                # Read stdout
                if process.stdout:
                    line = process.stdout.readline()
                    if line:
                        log_lines.append(line.strip())
                        
                        # Update progress based on log content
                        progress = estimate_progress_from_log(line, task_name)
                        if progress is not None:
                            progress_tracker.update(progress, f"Processing... {line.strip()[:50]}...")
                
                time.sleep(0.1)
            
            # Process finished
            return_code = process.returncode
            
            # Read any remaining output
            if process.stdout:
                remaining_stdout = process.stdout.read()
                if remaining_stdout:
                    log_lines.extend(remaining_stdout.split('\n'))
            
            if process.stderr:
                stderr = process.stderr.read()
                if stderr and contains_errors(stderr):
                    log_lines.append(f"ERRORS:\n{stderr}")
            
            # Update final status
            if return_code == 0:
                progress_tracker.finish(f"Task '{task_name}' completed successfully")
            else:
                progress_tracker.finish(f"Task '{task_name}' failed with code {return_code}")
                
        except Exception as e:
            progress_tracker.finish(f"Task monitoring failed: {str(e)}")
    
    def estimate_progress_from_log(log_line, task_name):
        """Estimate progress percentage from log line."""
        # Simple heuristic - could be improved with more sophisticated parsing
        if "epoch" in log_line.lower():
            try:
                # Look for epoch information like "Epoch 5/10"
                parts = log_line.lower().split("epoch")
                for part in parts:
                    if "/" in part:
                        epoch_info = part.strip().split()[0]
                        if "/" in epoch_info:
                            current, total = epoch_info.split("/")
                            return (int(current) / int(total)) * 100
            except:
                pass
        
        # Default incremental progress
        current_progress = progress_tracker.get_status()["progress"]
        if current_progress < 90:
            return min(current_progress + 1, 90)
        
        return None
    
    def get_live_status():
        """Get current status for live updates."""
        status = progress_tracker.get_status()
        
        if status["is_running"]:
            status_type = "info"
        elif status["progress"] >= 100:
            status_type = "success"
        else:
            status_type = "info"
        
        return create_status_message(
            status_type,
            status["message"],
            status["progress"] if status["is_running"] else None,
            status["eta"]
        )
    
    # Connect event handlers
    run_btn.click(
        run_task,
        inputs=[
            task_selection, dataset_selection, threshold_slider,
            rcnn_model, enable_augmentation, enable_visualization, use_gpu
        ],
        outputs=[status_display, run_btn, stop_btn]
    )
    
    stop_btn.click(
        stop_task,
        outputs=[status_display, run_btn, stop_btn]
    )
    
    return task_group
