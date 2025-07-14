"""
FastAPI application for deepEMIA web interface.

This replaces the Streamlit GUI with a more flexible and performant web interface
using FastAPI + HTMX for modern web interactions.
"""

import os
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import uvicorn

from src.utils.config import get_config
from src.utils.logger_utils import system_logger

# Initialize FastAPI app
app = FastAPI(
    title="deepEMIA Control Panel",
    description="Deep Learning Image Analysis Web Interface",
    version="2.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# Setup static files and templates
static_dir = Path(__file__).parent / "static"
templates_dir = Path(__file__).parent / "templates"

# Create directories if they don't exist
static_dir.mkdir(exist_ok=True)
templates_dir.mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
templates = Jinja2Templates(directory=str(templates_dir))

# Load configuration
try:
    config = get_config()
    system_logger.info("Configuration loaded successfully")
except Exception as e:
    system_logger.error(f"Failed to load configuration: {e}")
    config = {"bucket": "not-configured"}

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """
    Home page - Hello World for now, will expand to full dashboard later.
    """
    return templates.TemplateResponse(
        "index.html", 
        {
            "request": request,
            "title": "deepEMIA Control Panel",
            "config": config
        }
    )

@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring."""
    return {"status": "healthy", "service": "deepEMIA-GUI", "timestamp": time.time()}

# Include demo routes
from gui.routes import demo
app.include_router(demo.router)

def run_server():
    """Run the FastAPI server."""
    uvicorn.run(
        "gui.fastapi_app:app",
        host="0.0.0.0",  # Listen on all interfaces for external access
        port=8888,
        reload=True,
        log_level="info"
    )

if __name__ == "__main__":
    print("Starting deepEMIA FastAPI GUI on http://0.0.0.0:8888")
    print("Access via VM external IP: http://[VM_EXTERNAL_IP]:8888")
    run_server()
