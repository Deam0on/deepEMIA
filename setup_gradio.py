#!/usr/bin/env python3
"""
Setup script for deepEMIA Gradio interface.

This script helps users set up the new Gradio interface including
dependency installation and environment configuration.
"""
import os
import sys
import subprocess
import hashlib
from pathlib import Path

def print_banner():
    """Print setup banner."""
    print("🔬 deepEMIA - Gradio Interface Setup")
    print("=" * 50)
    print()

def check_python_version():
    """Check Python version compatibility."""
    print("🐍 Checking Python version...")
    
    if sys.version_info < (3, 8):
        print("❌ Error: Python 3.8+ required")
        print(f"   Current version: {sys.version}")
        return False
    
    print(f"✅ Python {sys.version.split()[0]} detected")
    return True

def install_dependencies():
    """Install required dependencies."""
    print("\n📦 Installing dependencies...")
    
    try:
        # First install core packages
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "gradio>=4.15.0",
            "google-cloud-storage>=2.11.0",
            "psutil>=5.9.8",
            "pandas>=1.4.4",
            "pillow>=9.1.0",
            "numpy>=1.23.5"
        ])
        
        print("✅ Core dependencies installed")
        
        # Check if full requirements should be installed
        install_full = input("\n❓ Install full requirements (ML dependencies)? [y/N]: ").lower().strip()
        
        if install_full in ['y', 'yes']:
            print("📦 Installing full requirements (this may take a while)...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
            ])
            print("✅ Full requirements installed")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def setup_admin_password():
    """Setup admin password hash."""
    print("\n🔐 Setting up admin access...")
    
    current_hash = os.environ.get('ADMIN_PASSWORD_HASH')
    if current_hash:
        print("✅ Admin password already configured")
        update = input("❓ Update admin password? [y/N]: ").lower().strip()
        if update not in ['y', 'yes']:
            return True
    
    print("\n📝 Please set an admin password for dataset management features")
    print("   This password will be used to create/delete datasets")
    
    while True:
        password = input("Enter admin password: ").strip()
        if len(password) < 6:
            print("❌ Password must be at least 6 characters")
            continue
        
        confirm = input("Confirm password: ").strip()
        if password != confirm:
            print("❌ Passwords don't match")
            continue
        
        break
    
    # Create hash
    password_hash = hashlib.sha256(password.encode()).hexdigest()
    
    # Show instructions for setting environment variable
    print(f"\n📋 Add this to your environment:")
    print(f"   Windows: set ADMIN_PASSWORD_HASH={password_hash}")
    print(f"   Linux/Mac: export ADMIN_PASSWORD_HASH={password_hash}")
    
    print("\n💡 To make this permanent:")
    print("   Windows: Add to system environment variables")
    print("   Linux/Mac: Add to ~/.bashrc or ~/.zshrc")
    
    # Try to set for current session
    os.environ['ADMIN_PASSWORD_HASH'] = password_hash
    print("✅ Admin password configured for current session")
    
    return True

def test_installation():
    """Test the installation."""
    print("\n🧪 Testing installation...")
    
    try:
        # Test gradio import
        import gradio as gr
        print("✅ Gradio import successful")
        
        # Test google cloud storage import
        from google.cloud import storage
        print("✅ Google Cloud Storage import successful")
        
        # Test other imports
        import pandas as pd
        import numpy as np
        from PIL import Image
        print("✅ Core dependencies working")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def create_launch_script():
    """Create a convenient launch script."""
    print("\n📝 Creating launch scripts...")
    
    # Windows batch file
    batch_content = """@echo off
echo Starting deepEMIA Gradio Interface...
python launch_gradio.py
pause
"""
    
    try:
        with open("launch_gradio.bat", "w") as f:
            f.write(batch_content)
        print("✅ Windows launch script: launch_gradio.bat")
    except:
        pass
    
    # Unix shell script
    shell_content = """#!/bin/bash
echo "Starting deepEMIA Gradio Interface..."
python3 launch_gradio.py
"""
    
    try:
        with open("launch_gradio.sh", "w") as f:
            f.write(shell_content)
        os.chmod("launch_gradio.sh", 0o755)
        print("✅ Unix launch script: launch_gradio.sh")
    except:
        pass

def print_next_steps():
    """Print next steps for the user."""
    print("\n🎉 Setup complete!")
    print("\n📋 Next steps:")
    print("1. Configure your Google Cloud credentials")
    print("2. Update config/config.yaml with your GCS bucket")
    print("3. Launch the interface:")
    print("   • Python: python launch_gradio.py")
    print("   • Windows: double-click launch_gradio.bat")
    print("   • Unix: ./launch_gradio.sh")
    print()
    print("🌐 The interface will be available at: http://127.0.0.1:7860")
    print()
    print("📚 For help, see the README.md or run with --help")

def main():
    """Main setup function."""
    print_banner()
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        print("\n❌ Setup failed during dependency installation")
        sys.exit(1)
    
    # Setup admin password
    if not setup_admin_password():
        print("\n❌ Setup failed during password configuration")
        sys.exit(1)
    
    # Test installation
    if not test_installation():
        print("\n⚠️  Installation test failed - interface may not work properly")
    
    # Create launch scripts
    create_launch_script()
    
    # Print next steps
    print_next_steps()

if __name__ == "__main__":
    main()
