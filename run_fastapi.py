#!/usr/bin/env python3
"""
Simple FastAPI startup script with dependency checking.
"""

import sys
import subprocess
from pathlib import Path

def check_and_install_dependencies():
    """Check if FastAPI dependencies are installed, install if missing."""
    required_packages = [
        "fastapi==0.104.1",
        "uvicorn[standard]==0.24.0", 
        "jinja2==3.1.2",
        "python-multipart==0.0.6"
    ]
    
    print("🔍 Checking FastAPI dependencies...")
    
    missing_packages = []
    for package in required_packages:
        package_name = package.split("==")[0].split("[")[0]
        try:
            __import__(package_name.replace("-", "_"))
            print(f"✅ {package_name} - OK")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package_name} - MISSING")
    
    if missing_packages:
        print(f"\n📦 Installing missing packages: {', '.join(missing_packages)}")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install"
            ] + missing_packages)
            print("✅ Dependencies installed successfully!")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install dependencies: {e}")
            return False
    
    return True

def start_server():
    """Start the FastAPI server."""
    print("🚀 Starting deepEMIA FastAPI GUI...")
    print("=" * 60)
    print("📡 Server will be accessible at:")
    print("   - Local:    http://localhost:8505")
    print("   - Network:  http://0.0.0.0:8505") 
    print("   - External: http://[VM_EXTERNAL_IP]:8505")
    print("📚 API Documentation:")
    print("   - Swagger:  http://[VM_EXTERNAL_IP]:8505/api/docs")
    print("   - ReDoc:    http://[VM_EXTERNAL_IP]:8505/api/redoc")
    print("=" * 60)
    print("Press Ctrl+C to stop the server")
    print("=" * 60)
    
    # Import and run
    try:
        import uvicorn
        uvicorn.run(
            "gui.fastapi_app:app",
            host="0.0.0.0",
            port=8505,
            reload=False,  # Disable reload for production
            log_level="info"
        )
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        return False
    
    return True

if __name__ == "__main__":
    try:
        # Check dependencies first
        if not check_and_install_dependencies():
            print("❌ Dependency check failed. Exiting.")
            sys.exit(1)
        
        # Start server
        if not start_server():
            print("❌ Server startup failed. Exiting.")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\n🛑 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)
