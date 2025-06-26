#!/usr/bin/env python3
"""
Tmux CLI Manager for UW Computer Vision Pipeline

This script launches the CLI wizard in a tmux session and monitors its progress,
allowing you to use your terminal while the process runs in the background.
"""

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


def run_command(cmd, capture_output=True):
    """Run a shell command and return the result."""
    try:
        result = subprocess.run(
            cmd, 
            shell=True, 
            capture_output=capture_output, 
            text=True,
            check=False
        )
        return result
    except Exception as e:
        print(f"Error running command '{cmd}': {e}")
        return None


def check_tmux_available():
    """Check if tmux is available on the system."""
    result = run_command("which tmux")
    if result and result.returncode == 0:
        return True
    
    result = run_command("tmux -V")
    return result and result.returncode == 0


def create_tmux_session(session_name, command):
    """Create a new tmux session with the given command."""
    # Kill existing session if it exists
    run_command(f"tmux kill-session -t {session_name} 2>/dev/null")
    
    # Create new session
    tmux_cmd = f"tmux new-session -d -s {session_name} '{command}'"
    result = run_command(tmux_cmd)
    
    if result and result.returncode == 0:
        print(f"✅ Created tmux session '{session_name}'")
        return True
    else:
        print(f"❌ Failed to create tmux session '{session_name}'")
        return False


def get_tmux_output(session_name, lines=10):
    """Get the last N lines from a tmux session."""
    cmd = f"tmux capture-pane -t {session_name} -p"
    result = run_command(cmd)
    
    if result and result.returncode == 0:
        lines_list = result.stdout.strip().split('\n')
        return lines_list[-lines:] if len(lines_list) > lines else lines_list
    return []


def is_session_running(session_name):
    """Check if a tmux session is still running."""
    result = run_command(f"tmux list-sessions")
    if result and result.returncode == 0:
        return session_name in result.stdout
    return False


def monitor_session(session_name, check_interval=5):
    """Monitor the tmux session and print updates."""
    print(f"\n🔍 Monitoring tmux session '{session_name}'...")
    print(f"📱 Check interval: {check_interval} seconds")
    print("💡 Use Ctrl+C to stop monitoring (process will continue in tmux)")
    print("🔧 Attach to session manually: tmux attach-session -t " + session_name)
    print("-" * 60)
    
    last_output = []
    
    try:
        while is_session_running(session_name):
            current_output = get_tmux_output(session_name, 15)
            
            # Only print new lines
            if current_output != last_output:
                # Clear screen and show timestamp
                os.system('clear' if os.name == 'posix' else 'cls')
                print(f"📊 Session: {session_name} | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print("-" * 60)
                
                # Print the output
                for line in current_output:
                    if line.strip():  # Only print non-empty lines
                        print(line)
                
                print("-" * 60)
                print(f"💡 Session is running. Next update in {check_interval}s...")
                print("🔧 Manual attach: tmux attach-session -t " + session_name)
                
                last_output = current_output
            
            time.sleep(check_interval)
            
    except KeyboardInterrupt:
        print(f"\n⏹️  Stopped monitoring. Session '{session_name}' is still running.")
        print(f"🔧 To attach: tmux attach-session -t {session_name}")
        print(f"🗑️  To kill: tmux kill-session -t {session_name}")
        return
    
    print(f"\n✅ Session '{session_name}' has completed.")
    
    # Show final output
    final_output = get_tmux_output(session_name, 20)
    if final_output:
        print("\n📋 Final output:")
        print("-" * 40)
        for line in final_output:
            if line.strip():
                print(line)


def main():
    parser = argparse.ArgumentParser(
        description="Launch UW Computer Vision CLI in tmux session",
        epilog="""
Examples:
  # Launch CLI wizard in tmux and monitor
  python tmux_cli.py

  # Launch with custom session name
  python tmux_cli.py --session my_cv_session

  # Launch with custom check interval
  python tmux_cli.py --interval 10

  # Just create session without monitoring
  python tmux_cli.py --no-monitor

Tmux Commands:
  # Attach to session manually
  tmux attach-session -t cv_pipeline

  # List all sessions
  tmux list-sessions

  # Kill the session
  tmux kill-session -t cv_pipeline
""",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--session", "-s",
        type=str,
        default="cv_pipeline",
        help="Name of the tmux session (default: cv_pipeline)"
    )
    
    parser.add_argument(
        "--interval", "-i",
        type=int,
        default=5,
        help="Monitoring check interval in seconds (default: 5)"
    )
    
    parser.add_argument(
        "--no-monitor", "-n",
        action="store_true",
        help="Create session but don't monitor (detach immediately)"
    )
    
    parser.add_argument(
        "--command", "-c",
        type=str,
        default="python cli_main.py",
        help="Command to run in tmux (default: python cli_main.py)"
    )
    
    args = parser.parse_args()
    
    # Check if tmux is available
    if not check_tmux_available():
        print("❌ tmux is not available on this system.")
        print("📦 Install tmux with:")
        print("   Ubuntu/Debian: sudo apt-get install tmux")
        print("   CentOS/RHEL: sudo yum install tmux")
        print("   macOS: brew install tmux")
        sys.exit(1)
    
    # Check if the CLI script exists
    cli_script = Path.home() / "deepEMIA" / "cli_main.py"
    if not cli_script.exists():
        print(f"❌ CLI script not found: {cli_script.absolute()}")
        print("🔍 Make sure the deepEMIA directory exists in your home folder")
        sys.exit(1)
    
    print("🚀 UW Computer Vision Tmux Manager")
    print(f"📁 Working directory: {Path.cwd()}")
    print(f"🎯 Session name: {args.session}")
    print(f"⚡ Command: {args.command}")
    
    # Create the tmux session
    if not create_tmux_session(args.session, args.command):
        sys.exit(1)
    
    if args.no_monitor:
        print(f"✅ Session '{args.session}' created and detached")
        print(f"🔧 To attach: tmux attach-session -t {args.session}")
        print(f"🗑️  To kill: tmux kill-session -t {args.session}")
    else:
        # Give the session a moment to start
        time.sleep(2)
        
        # Monitor the session
        monitor_session(args.session, args.interval)


if __name__ == "__main__":
    main()
