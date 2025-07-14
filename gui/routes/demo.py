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
