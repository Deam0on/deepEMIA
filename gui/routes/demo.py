"""
Demo routes for testing HTMX functionality.
These will be removed in the final implementation.
"""

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path
import time

router = APIRouter(prefix="/demo", tags=["demo"])

templates_dir = Path(__file__).parent.parent / "templates"
templates = Jinja2Templates(directory=str(templates_dir))

@router.get("/hello", response_class=HTMLResponse)
async def demo_hello(request: Request):
    """Demo endpoint to test HTMX functionality."""
    # Simulate some processing time
    time.sleep(0.5)
    
    current_time = time.strftime("%H:%M:%S")
    return f"""
    <div class="alert alert-success fade-in">
        <strong>Success!</strong> HTMX is working correctly. 
        The FastAPI backend responded at {current_time}.
        <br><small>This demo shows how interactive components will work in the full interface.</small>
    </div>
    """

@router.get("/health-check", response_class=HTMLResponse)
async def quick_health_check(request: Request):
    """Quick health check action."""
    time.sleep(0.3)  # Simulate check
    return """
    <div class="alert alert-success">
        <i class="bi bi-check-circle"></i> <strong>System Healthy</strong><br>
        All core services are operational.
    </div>
    """

@router.get("/system-status", response_class=HTMLResponse)
async def quick_system_status(request: Request):
    """Quick system status check."""
    time.sleep(0.5)  # Simulate status check
    return """
    <div class="alert alert-info">
        <i class="bi bi-info-circle"></i> <strong>System Status</strong><br>
        • FastAPI Server: Running<br>
        • Database: Connected<br>
        • GCS: Available<br>
        • GPU: Not detected
    </div>
    """

@router.get("/validate-config", response_class=HTMLResponse)
async def quick_validate_config(request: Request):
    """Quick configuration validation."""
    time.sleep(0.4)  # Simulate validation
    return """
    <div class="alert alert-warning">
        <i class="bi bi-exclamation-triangle"></i> <strong>Configuration Check</strong><br>
        Config file loaded successfully.<br>
        <small>Some optional settings may need adjustment.</small>
    </div>
    """

@router.get("/reload-datasets", response_class=HTMLResponse)
async def quick_reload_datasets(request: Request):
    """Quick dataset reload action."""
    time.sleep(0.7)  # Simulate reload
    return """
    <div class="alert alert-success">
        <i class="bi bi-arrow-clockwise"></i> <strong>Datasets Reloaded</strong><br>
        Found 3 datasets in storage.
    </div>
    """
