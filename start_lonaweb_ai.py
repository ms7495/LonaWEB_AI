#!/usr/bin/env python3
"""
LonaWEB AI Application Launcher - FIXED VERSION
Run this script to start the LonaWEB AI Streamlit application.
"""

import os
import sys
from pathlib import Path


def print_banner():
    """Print LonaWEB AI banner"""
    banner = """
    ██╗      ██████╗ ███╗   ██╗ █████╗ ██╗    ██╗███████╗██████╗      █████╗ ██╗
    ██║     ██╔═══██╗████╗  ██║██╔══██╗██║    ██║██╔════╝██╔══██╗    ██╔══██╗██║
    ██║     ██║   ██║██╔██╗ ██║███████║██║ █╗ ██║█████╗  ██████╔╝    ███████║██║
    ██║     ██║   ██║██║╚██╗██║██╔══██║██║███╗██║██╔══╝  ██╔══██╗    ██╔══██║██║
    ███████╗╚██████╔╝██║ ╚████║██║  ██║╚███╔███╔╝███████╗██████╔╝    ██║  ██║██║
    ╚══════╝ ╚═════╝ ╚═╝  ╚═══╝╚═╝  ╚═╝ ╚══╝╚══╝ ╚══════╝╚═════╝     ╚═╝  ╚═╝╚═╝

    """
    print(banner)


def main():
    """Main launcher function"""
    print_banner()
    print("=" * 80)

    # Get project root directory
    project_root = Path(__file__).parent.absolute()
    main_dir = project_root / "main"

    # Verify main directory exists
    if not main_dir.exists():
        print("❌ Error: 'main' directory not found!")
        print(f"Expected location: {main_dir}")
        return 1

    # Add directories to Python path
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    if str(main_dir) not in sys.path:
        sys.path.insert(0, str(main_dir))

    # Change to main directory
    os.chdir(str(main_dir))

    # FIX: Correct model path - check both locations
    possible_model_paths = [
        main_dir / "models" / "Llama-3.2-1B-Instruct-f16.gguf",  # main/models/
        project_root / "models" / "Llama-3.2-1B-Instruct-f16.gguf",  # project/models/
    ]

    model_path = None
    for path in possible_model_paths:
        if path.exists():
            model_path = path
            break

    if model_path:
        os.environ["LLAMA_MODEL_PATH"] = str(model_path)
        print(f"✅ GGUF model found: {model_path}")
    else:
        print("⚠️  GGUF model not found at expected locations:")
        for path in possible_model_paths:
            print(f"   - {path}")
        print("   AI responses may be limited")

    print(f"📁 Working directory: {main_dir}")
    print(f"🤖 Model path: {model_path or 'Not found'}")
    print("🔧 Python path configured")

    # Check for Qdrant
    print("🔍 Checking Qdrant connection...")
    try:
        import subprocess
        result = subprocess.run(["curl", "-s", "http://localhost:6333"],
                                capture_output=True, timeout=5)
        if result.returncode == 0:
            print("✅ Qdrant server is running")
        else:
            print("⚠️  Qdrant server not detected - starting with in-memory mode")
    except:
        print("⚠️  Could not check Qdrant status - will attempt connection anyway")

    # Try to run streamlit
    try:
        import streamlit.web.cli as stcli

        # Use the no-threading app if available, otherwise fallback
        app_options = [
            main_dir / "ui" / "streamlit_app_no_threading.py",
            main_dir / "ui" / "streamlit_app.py"
        ]

        app_path = None
        for path in app_options:
            if path.exists():
                app_path = path
                break

        if not app_path:
            print("❌ Error: No Streamlit app found")
            print("Expected locations:")
            for path in app_options:
                print(f"   - {path}")
            return 1

        print(f"📱 Starting LonaWEB AI: {app_path.name}")
        print("🌐 App will open in your browser automatically")
        print("⏹  Press Ctrl+C to stop the application")
        print("=" * 80)

        # Run streamlit
        sys.argv = [
            "streamlit", "run", str(app_path),
            "--server.address", "localhost",
            "--server.port", "8501",
            "--server.headless", "true",
            "--server.runOnSave", "true"
        ]
        sys.exit(stcli.main())

    except ImportError:
        print("❌ Streamlit not installed!")
        print("📦 Install with: pip install streamlit")
        print("📋 Or install all requirements: pip install -r requirements.txt")
        return 1
    except KeyboardInterrupt:
        print("\n👋 LonaWEB AI stopped by user")
        return 0
    except Exception as e:
        print(f"❌ Error starting application: {e}")
        print("\n🔧 Troubleshooting:")
        print("1. Check if all dependencies are installed")
        print("2. Verify the main/ui/ directory structure")
        print("3. Try running: pip install -r requirements.txt")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
