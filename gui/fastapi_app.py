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

try:
    from fastapi import FastAPI, Request
    from fastapi.staticfiles import StaticFiles
    from fastapi.templating import Jinja2Templates
    from fastapi.responses import HTMLResponse
    import uvicorn
except ImportError as e:
    print(f"❌ FastAPI dependencies not installed: {e}")
    print("Please run: pip install fastapi uvicorn jinja2 python-multipart")
    sys.exit(1)

try:
    from src.utils.config import get_config
    from src.utils.logger_utils import system_logger
except ImportError as e:
    print(f"⚠️  Could not import deepEMIA modules: {e}")
    print("Using fallback configuration...")
    
    # Fallback configuration
    def get_config():
        return {"bucket": "not-configured"}
    
    class FallbackLogger:
        def info(self, msg): print(f"INFO: {msg}")
        def error(self, msg): print(f"ERROR: {msg}")
    
    system_logger = FallbackLogger()

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
    Home page - Full dashboard interface
    """
    return templates.TemplateResponse(
        "dashboard.html", 
        {
            "request": request,
            "title": "deepEMIA Control Panel",
            "config": config
        }
    )

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    """
    Dashboard page - same as home
    """
    return templates.TemplateResponse(
        "dashboard.html", 
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

# Include routes with error handling
try:
    from gui.routes import demo
    app.include_router(demo.router)
    system_logger.info("Demo routes loaded successfully")
except ImportError as e:
    print(f"⚠️  Could not load demo routes: {e}")
    
    # Add a simple demo route directly
    @app.get("/demo/hello", response_class=HTMLResponse)
    async def demo_hello_fallback(request: Request):
        current_time = time.strftime("%H:%M:%S")
        return f"""
        <div class="alert alert-success fade-in">
            <strong>Success!</strong> FastAPI is working correctly. 
            The backend responded at {current_time}.
            <br><small>This is a fallback demo endpoint.</small>
        </div>
        """

try:
    from gui.routes import datasets
    app.include_router(datasets.router, prefix="/api/datasets", tags=["datasets"])
    system_logger.info("Dataset routes loaded successfully")
except ImportError as e:
    print(f"⚠️  Could not load dataset routes: {e}")

try:
    from gui.routes import tasks
    app.include_router(tasks.router, prefix="/api/tasks", tags=["tasks"])
    system_logger.info("Task routes loaded successfully")
except ImportError as e:
    print(f"⚠️  Could not load task routes: {e}")

def run_server():
    """Run the FastAPI server."""
    uvicorn.run(
        "gui.fastapi_app:app",
        host="0.0.0.0",  # Listen on all interfaces for external access
        port=8505,
        reload=True,
        log_level="info"
    )

if __name__ == "__main__":
    print("Starting deepEMIA FastAPI GUI on http://0.0.0.0:8505")
    print("Access via VM external IP: http://[VM_EXTERNAL_IP]:8505")
    run_server()
