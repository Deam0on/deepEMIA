#!/usr/bin/env python3
"""
Launch script for deepEMIA Gradio interface.

This script provides an easy way to launch the new Gradio-based GUI
with proper environment setup and configuration.
"""
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

def setup_environment():
    """Setup environment variables and checks."""
    print("🔧 Setting up environment...")
    
    # Check for admin password hash
    if not os.environ.get('ADMIN_PASSWORD_HASH'):
        print("⚠️  Warning: ADMIN_PASSWORD_HASH not set")
        print("   Admin features will be disabled")
        print("   To enable admin features:")
        print("   1. Set your password: export ADMIN_PASSWORD_HASH=$(echo -n 'your_password' | sha256sum | cut -d' ' -f1)")
        print()
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Error: Python 3.8+ required")
        sys.exit(1)
    
    print("✅ Environment setup complete")


def check_dependencies():
    """Check if required dependencies are installed."""
    print("📦 Checking dependencies...")
    
    required_packages = [
        "gradio",
        "google-cloud-storage", 
        "pillow",
        "pandas",
        "numpy"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        print("   Please install with: pip install -r requirements.txt")
        sys.exit(1)
    
    print("✅ All dependencies found")


def main():
    """Main launcher function."""
    print("🔬 deepEMIA - Modern Gradio Interface")
    print("=" * 50)
    
    setup_environment()
    check_dependencies()
    
    # Import and launch
    try:
        from gui.gradio_app import launch_app
        
        print("🚀 Launching Gradio interface...")
        print("   Access the interface at: http://127.0.0.1:7860")
        print("   Press Ctrl+C to stop")
        print()
        
        launch_app(
            server_name="127.0.0.1",
            server_port=7860,
            share=False,
            debug=False
        )
        
    except KeyboardInterrupt:
        print("\n👋 Shutting down...")
    except Exception as e:
        print(f"❌ Error launching interface: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
