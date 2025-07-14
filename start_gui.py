#!/usr/bin/env python3
"""
Start script for the deepEMIA web interface.

This script provides options to start either the new FastAPI GUI 
or the legacy Streamlit GUI.
"""

import argparse
import sys
from pathlib import Path

def start_fastapi():
    """Start the FastAPI GUI."""
    from gui.fastapi_app import run_server
    print("=" * 60)
    print("    deepEMIA FastAPI GUI Starting")
    print("=" * 60)
    print("🚀 Starting deepEMIA FastAPI GUI...")
    print("📡 Server will be accessible at:")
    print("   - Local:    http://localhost:8080")
    print("   - Network:  http://0.0.0.0:8080")
    print("   - External: http://[VM_EXTERNAL_IP]:8080")
    print("📚 API Documentation:")
    print("   - Swagger:  http://[VM_EXTERNAL_IP]:8080/api/docs")
    print("   - ReDoc:    http://[VM_EXTERNAL_IP]:8080/api/redoc")
    print("=" * 60)
    print("Press Ctrl+C to stop the server")
    print("=" * 60)
    run_server()

def start_streamlit():
    """Start the legacy Streamlit GUI."""
    import subprocess
    gui_path = Path(__file__).parent / "gui" / "streamlit_gui.py"
    print("=" * 60)
    print("    deepEMIA Streamlit GUI Starting")
    print("=" * 60)
    print("🚀 Starting deepEMIA Streamlit GUI...")
    print("📡 Server will be accessible at:")
    print("   - Default:  http://localhost:8501")
    print("   - External: Use --server.address 0.0.0.0 for external access")
    print("=" * 60)
    print("Press Ctrl+C to stop the server")
    print("=" * 60)
    subprocess.run([
        "streamlit", "run", str(gui_path), 
        "--server.address", "0.0.0.0",
        "--server.port", "8501"
    ])

def main():
    parser = argparse.ArgumentParser(
        description="Start deepEMIA Web Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python start_gui.py                    # Start FastAPI GUI (default)
  python start_gui.py --interface fastapi
  python start_gui.py --interface streamlit

The FastAPI interface is the new modern GUI with better performance.
The Streamlit interface is kept for backward compatibility.
        """
    )
    parser.add_argument(
        "--interface", 
        choices=["fastapi", "streamlit"], 
        default="fastapi",
        help="Choose web interface (default: fastapi)"
    )
    
    args = parser.parse_args()
    
    try:
        if args.interface == "fastapi":
            start_fastapi()
        else:
            start_streamlit()
    except KeyboardInterrupt:
        print("\n\n🛑 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Error starting {args.interface} GUI: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
