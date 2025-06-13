#!/usr/bin/env python3
"""
MuseQuill Book Planner Setup Script
Sets up and runs the complete book planner system
"""

import os
import sys
import subprocess
from pathlib import Path
import shutil


def check_python_version():
    """Check if Python version is 3.7+"""
    if sys.version_info < (3, 7):
        print("❌ Python 3.7 or higher is required")
        sys.exit(1)
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")


def create_project_structure():
    """Create the project directory structure"""
    print("📁 Creating project structure...")
    
    dirs = [
        "musequill_planner",
        "musequill_planner/static",
        "musequill_planner/templates",
        "musequill_planner/api",
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    print("✅ Project structure created")


def create_requirements_file():
    """Create requirements.txt file"""
    requirements = """fastapi==0.104.1
uvicorn==0.24.0
pydantic==2.5.0
python-multipart==0.0.6
jinja2==3.1.2
python-jose==3.3.0
passlib==1.7.4
bcrypt==4.0.1
"""
    
    with open("requirements.txt", "w") as f:
        f.write(requirements)
    
    print("✅ Requirements file created")
