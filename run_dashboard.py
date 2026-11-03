#!/usr/bin/env python
"""
Persistent Dashboard Runner - Keeps Streamlit running with automatic restart
"""
import subprocess
import sys
import time
import os
import platform

def run_streamlit():
    """Run Streamlit dashboard with restart capability"""
    script_path = os.path.join(os.path.dirname(__file__), "dashboard.py")
    
    # Get the venv python executable
    if platform.system() == "Windows":
        venv_python = os.path.join(os.path.dirname(__file__), "venv", "Scripts", "python.exe")
    else:
        venv_python = os.path.join(os.path.dirname(__file__), "venv", "bin", "python")
    
    print("=" * 60)
    print("🚀 Starting Dashboard Server (Auto-restart enabled)")
    print("=" * 60)
    print(f"📍 Dashboard will be available at: http://localhost:8501")
    print(f"📁 Script: {script_path}")
    print(f"🐍 Using Python: {venv_python}")
    print("=" * 60)
    
    retry_count = 0
    max_retries = 10
    
    while retry_count < max_retries:
        try:
            # Run Streamlit in subprocess using venv python
            process = subprocess.Popen(
                [venv_python, "-m", "streamlit", "run", script_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            # Print output in real-time
            print("\n[Server Output]")
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    print(output.rstrip())
            
            # Check exit code
            exit_code = process.poll()
            if exit_code == 0:
                print("✅ Dashboard stopped cleanly")
                break
            else:
                print(f"⚠️  Dashboard crashed with exit code: {exit_code}")
                retry_count += 1
                if retry_count < max_retries:
                    print(f"🔄 Restarting in 5 seconds... (Attempt {retry_count}/{max_retries})")
                    time.sleep(5)
        
        except KeyboardInterrupt:
            print("\n👋 Dashboard stopped by user")
            process.terminate()
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            retry_count += 1
            if retry_count < max_retries:
                print(f"🔄 Retrying in 5 seconds... (Attempt {retry_count}/{max_retries})")
                time.sleep(5)
    
    if retry_count >= max_retries:
        print(f"❌ Dashboard failed after {max_retries} attempts")
        sys.exit(1)

if __name__ == "__main__":
    run_streamlit()

