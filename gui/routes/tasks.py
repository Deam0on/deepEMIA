"""
Task execution routes for running deepEMIA commands.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, Dict, Any
import subprocess
import asyncio
import uuid
import time
from pathlib import Path
import json

router = APIRouter(tags=["tasks"])

# Store running tasks
running_tasks: Dict[str, Dict[str, Any]] = {}

class TaskRequest(BaseModel):
    task: str  # prepare, train, evaluate, inference
    dataset_name: str
    threshold: Optional[float] = 0.65
    rcnn_model: Optional[str] = "101"  # 50, 101, combo
    visualize: Optional[bool] = True
    download: Optional[bool] = True
    upload: Optional[bool] = True

class TaskResponse(BaseModel):
    task_id: str
    status: str
    message: str

@router.post("/run", response_model=TaskResponse)
async def run_task(task_request: TaskRequest, background_tasks: BackgroundTasks):
    """Execute a deepEMIA task in the background."""
    task_id = str(uuid.uuid4())
    
    # Add task to running tasks
    running_tasks[task_id] = {
        "status": "starting",
        "progress": 0,
        "stdout": "",
        "stderr": "",
        "start_time": time.time(),
        "command": "",
        "task_request": task_request.dict()
    }
    
    # Start task in background
    background_tasks.add_task(execute_task, task_id, task_request)
    
    return TaskResponse(
        task_id=task_id,
        status="started",
        message=f"Task {task_request.task} started with ID {task_id}"
    )

@router.get("/status/{task_id}")
async def get_task_status(task_id: str):
    """Get the status of a running task."""
    if task_id not in running_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task_info = running_tasks[task_id]
    return {
        "task_id": task_id,
        "status": task_info["status"],
        "progress": task_info["progress"],
        "stdout": task_info["stdout"][-2000:],  # Last 2000 chars
        "stderr": task_info["stderr"][-1000:],  # Last 1000 chars
        "elapsed_time": time.time() - task_info["start_time"],
        "command": task_info["command"]
    }

@router.get("/")
async def list_tasks():
    """List all tasks (running and completed)."""
    return {
        task_id: {
            "status": info["status"],
            "progress": info["progress"],
            "start_time": info["start_time"],
            "elapsed_time": time.time() - info["start_time"],
            "task_type": info["task_request"]["task"]
        }
        for task_id, info in running_tasks.items()
    }

@router.delete("/{task_id}")
async def stop_task(task_id: str):
    """Stop a running task."""
    if task_id not in running_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task_info = running_tasks[task_id]
    if task_info["status"] == "running":
        task_info["status"] = "cancelled"
        return {"message": f"Task {task_id} marked for cancellation"}
    else:
        return {"message": f"Task {task_id} is not running (status: {task_info['status']})"}

async def execute_task(task_id: str, task_request: TaskRequest):
    """Execute the actual deepEMIA command."""
    task_info = running_tasks[task_id]
    
    try:
        # Build command - try to find main.py
        main_script = None
        possible_paths = [
            Path("main.py"),  # Current directory
            Path("../main.py"),  # Parent directory
            Path(__file__).parent.parent.parent / "main.py",  # Project root
        ]
        
        for path in possible_paths:
            if path.exists():
                main_script = path.resolve()
                break
        
        if not main_script:
            raise FileNotFoundError("Could not find main.py script")
        
        # Use python instead of python3 for Windows
        python_cmd = "python"
        
        command_parts = [
            python_cmd, str(main_script),
            "--task", task_request.task,
            "--dataset_name", task_request.dataset_name,
            "--threshold", str(task_request.threshold),
            "--rcnn", task_request.rcnn_model
        ]
        
        if task_request.visualize:
            command_parts.append("--visualize")
        if task_request.download:
            command_parts.append("--download")
        if task_request.upload:
            command_parts.append("--upload")
        
        command = " ".join(command_parts)
        task_info["command"] = command
        task_info["status"] = "running"
        
        print(f"Executing command: {command}")  # Debug log
        
        # Small delay to ensure the "running" status is visible
        import time
        time.sleep(0.5)
        
        # Execute command
        process = subprocess.Popen(
            command_parts,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True,
            cwd=str(main_script.parent)  # Set working directory
        )
        
        # Store process for potential cancellation
        task_info["process"] = process
        
        # Read output in real-time
        stdout_buffer = []
        stderr_buffer = []
        
        # Set initial progress
        task_info["progress"] = 15
        
        while True:
            if task_info["status"] == "cancelled":
                process.terminate()
                break
                
            # Check if process is still running
            if process.poll() is not None:
                break
            
            # Read available output
            try:
                line = process.stdout.readline()
                if line:
                    stdout_buffer.append(line)
                    task_info["stdout"] = "".join(stdout_buffer)
                    
                    # Update progress based on output patterns
                    line_lower = line.lower()
                    if "loading" in line_lower or "initializing" in line_lower:
                        task_info["progress"] = max(task_info["progress"], 20)
                    elif "download" in line_lower:
                        task_info["progress"] = max(task_info["progress"], 30)
                    elif "processing" in line_lower or "inference" in line_lower:
                        task_info["progress"] = max(task_info["progress"], 50)
                    elif "training" in line_lower or "prepare" in line_lower:
                        task_info["progress"] = max(task_info["progress"], 60)
                    elif "saving" in line_lower or "upload" in line_lower:
                        task_info["progress"] = max(task_info["progress"], 80)
                    elif "complete" in line_lower or "finished" in line_lower:
                        task_info["progress"] = max(task_info["progress"], 95)
            except:
                pass
            
            await asyncio.sleep(0.1)
        
        # Get final output
        stdout, stderr = process.communicate()
        if stdout:
            stdout_buffer.append(stdout)
        if stderr:
            stderr_buffer.append(stderr)
            
        task_info["stdout"] = "".join(stdout_buffer)
        task_info["stderr"] = "".join(stderr_buffer)
        
        # Set final status
        if process.returncode == 0:
            task_info["status"] = "completed"
            task_info["progress"] = 100
        else:
            task_info["status"] = "failed"
            task_info["progress"] = -1
            
    except Exception as e:
        task_info["status"] = "error"
        task_info["stderr"] = str(e)
        task_info["progress"] = -1
