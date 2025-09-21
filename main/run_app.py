#!/usr/bin/env python3
"""
Launcher script for the DocuChat Streamlit application.
This script ensures proper Python path setup before running the app.
"""

import sys
from pathlib import Path


def setup_python_path():
    """Setup Python path for proper imports"""
    # Get the directory containing this script (main/)
    current_dir = Path(__file__).parent.absolute()

    # Add the main directory to Python path if not already there
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))

    # Also add the parent directory (PythonProject/) for broader imports
    parent_dir = current_dir.parent
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))

    print(f"Python path setup complete:")
    print(f"  Main dir: {current_dir}")
    print(f"  Parent dir: {parent_dir}")


def run_streamlit_app():
    """Run the Streamlit application"""
    try:
        import streamlit.web.cli as stcli
        import sys

        # Setup the path
        setup_python_path()

        # Path to the streamlit app
        app_path = Path(__file__).parent / "ui" / "streamlit_app.py"

        # Run streamlit
        sys.argv = ["streamlit", "run", str(app_path)]
        sys.exit(stcli.main())

    except ImportError:
        print("❌ Streamlit not installed. Install with: pip install streamlit")
        return False
    except Exception as e:
        print(f"❌ Error running Streamlit app: {e}")
        return False


if __name__ == "__main__":
    print("🚀 Starting DocuChat application...")
    run_streamlit_app()
