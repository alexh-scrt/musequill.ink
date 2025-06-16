#!/usr/bin/env python3
"""
MuseQuill.ink Installation Script
Automated installation of dependencies with conda environment setup.
"""

import os
import sys
import subprocess
import argparse
import json
from pathlib import Path
from typing import List, Optional


class MuseQuillInstaller:
    """MuseQuill installation manager with conda support."""
    
    def __init__(self, project_root: Path = None, conda_env: str = "musequill"):
        self.project_root = project_root or Path(__file__).parent
        self.conda_env = conda_env
        self.python_executable = sys.executable
        
    def check_conda_available(self) -> bool:
        """Check if conda is available."""
        try:
            result = subprocess.run(
                ["conda", "--version"],
                capture_output=True,
                text=True,
                check=True
            )
            print(f"✅ Conda available: {result.stdout.strip()}")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("❌ conda not available - please install Anaconda or Miniconda")
            print("   Download from: https://conda.io/miniconda.html")
            return False
    
    def get_conda_envs(self) -> List[str]:
        """Get list of conda environments."""
        try:
            result = subprocess.run(
                ["conda", "env", "list", "--json"],
                capture_output=True,
                text=True,
                check=True
            )
            env_data = json.loads(result.stdout)
            env_names = []
            for env_path in env_data.get("envs", []):
                env_name = Path(env_path).name
                env_names.append(env_name)
            return env_names
        except (subprocess.CalledProcessError, json.JSONDecodeError):
            print("⚠️  Could not list conda environments")
            return []
    
    def conda_env_exists(self) -> bool:
        """Check if the conda environment exists."""
        envs = self.get_conda_envs()
        exists = self.conda_env in envs
        if exists:
            print(f"✅ Conda environment '{self.conda_env}' exists")
        else:
            print(f"❌ Conda environment '{self.conda_env}' does not exist")
        return exists
    
    def create_conda_environment(self, python_version: str = "3.13") -> bool:
        """Create a new conda environment with specified Python version."""
        print(f"🔄 Creating conda environment '{self.conda_env}' with Python {python_version}...")
        try:
            subprocess.run(
                ["conda", "create", "-n", self.conda_env, f"python={python_version}", "-y"],
                check=True
            )
            print(f"✅ Conda environment '{self.conda_env}' created successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to create conda environment: {e}")
            return False
    
    def get_conda_python_executable(self) -> Optional[str]:
        """Get the Python executable path for the conda environment."""
        try:
            # Get conda info for the environment
            result = subprocess.run(
                ["conda", "info", "--envs"],
                capture_output=True,
                text=True,
                check=True
            )
            
            # Parse the output to find the environment path
            for line in result.stdout.split('\n'):
                if self.conda_env in line and not line.startswith('#'):
                    parts = line.split()
                    if len(parts) >= 2:
                        env_path = parts[-1]  # Last part is the path
                        if os.name == "nt":  # Windows
                            python_exe = os.path.join(env_path, "python.exe")
                        else:  # Unix/Linux/macOS
                            python_exe = os.path.join(env_path, "bin", "python")
                        
                        if os.path.exists(python_exe):
                            return python_exe
            
            print(f"❌ Could not find Python executable for environment '{self.conda_env}'")
            return None
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to get conda environment info: {e}")
            return None
    
    def check_python_version_in_env(self, python_exe: str) -> bool:
        """Check Python version in the conda environment."""
        try:
            result = subprocess.run(
                [python_exe, "--version"],
                capture_output=True,
                text=True,
                check=True
            )
            version_str = result.stdout.strip()
            print(f"✅ Python version in '{self.conda_env}': {version_str}")
            
            # Extract version numbers
            version_parts = version_str.split()[1].split('.')
            major, minor = int(version_parts[0]), int(version_parts[1])
            
            if major != 3 or minor < 11:
                print(f"❌ Python 3.11+ required, found {major}.{minor}")
                return False
                
            return True
        except (subprocess.CalledProcessError, ValueError, IndexError) as e:
            print(f"❌ Failed to check Python version: {e}")
            return False
    
    def check_pip_in_env(self, python_exe: str) -> bool:
        """Check if pip is available in the conda environment."""
        try:
            result = subprocess.run(
                [python_exe, "-m", "pip", "--version"],
                capture_output=True,
                text=True,
                check=True
            )
            print(f"✅ pip available in '{self.conda_env}': {result.stdout.strip()}")
            return True
        except subprocess.CalledProcessError:
            print(f"❌ pip not available in '{self.conda_env}'")
            return False
    
    def upgrade_pip_in_env(self, python_exe: str) -> bool:
        """Upgrade pip in the conda environment."""
        print(f"🔄 Upgrading pip in '{self.conda_env}'...")
        try:
            subprocess.run(
                [python_exe, "-m", "pip", "install", "--upgrade", "pip"],
                check=True
            )
            print("✅ pip upgraded successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to upgrade pip: {e}")
            return False
    
    
    def install_requirements(self, requirements_file: str, python_exe: str) -> bool:
        """Install requirements from a file in the conda environment."""
        requirements_path = self.project_root / requirements_file
        
        if not requirements_path.exists():
            print(f"❌ Requirements file not found: {requirements_file}")
            return False
        
        print(f"🔄 Installing requirements from {requirements_file} in '{self.conda_env}'...")
        try:
            subprocess.run(
                [python_exe, "-m", "pip", "install", "-r", str(requirements_path)],
                check=True
            )
            print(f"✅ Requirements installed from {requirements_file}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install requirements: {e}")
            return False
    
    def install_editable_package(self, python_exe: str) -> bool:
        """Install the package in editable mode in the conda environment."""
        print(f"🔄 Installing MuseQuill in editable mode in '{self.conda_env}'...")
        try:
            subprocess.run(
                [python_exe, "-m", "pip", "install", "-e", "."],
                cwd=self.project_root,
                check=True
            )
            print("✅ MuseQuill installed in editable mode")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install MuseQuill: {e}")
            return False
    
    def create_env_file(self) -> bool:
        """Create .env file from example."""
        env_file = self.project_root / ".env"
        env_example = self.project_root / ".env.example"
        
        if env_file.exists():
            print("✅ .env file already exists")
            return True
        
        if not env_example.exists():
            print("❌ .env.example file not found")
            return False
        
        print("🔄 Creating .env file from .env.example...")
        try:
            # Copy .env.example to .env
            with open(env_example, "r") as src, open(env_file, "w") as dst:
                content = src.read()
                # Add warning comment
                dst.write("# Created by install.py - Please update with your actual values\n")
                dst.write(content)
            
            print("✅ .env file created")
            print("⚠️  Please update .env file with your actual API keys and configuration")
            return True
        except Exception as e:
            print(f"❌ Failed to create .env file: {e}")
            return False
    
    def create_directories(self) -> bool:
        """Create necessary directories."""
        directories = [
            "data",
            "logs", 
            "uploads",
            "cache",
        ]
        
        print("🔄 Creating directories...")
        for directory in directories:
            dir_path = self.project_root / directory
            dir_path.mkdir(exist_ok=True)
            print(f"  📁 {directory}/")
        
        print("✅ Directories created")
        return True
    
    def run_tests(self, python_exe: str) -> bool:
        """Run basic tests to verify installation."""
        print("🔄 Running installation verification tests...")
        
        # Test 1: Import test
        try:
            result = subprocess.run(
                [python_exe, "-c", "import musequill; print('✅ MuseQuill module imports successfully')"],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                check=True
            )
            print(result.stdout.strip())
        except subprocess.CalledProcessError:
            print("❌ MuseQuill module import failed")
            return False
        
        # Test 2: Basic functionality test
        test_script = self.project_root / "test_foundation.py"
        if test_script.exists():
            try:
                subprocess.run([python_exe, str(test_script)], check=True)
                print("✅ Foundation tests passed")
            except subprocess.CalledProcessError:
                print("⚠️  Some foundation tests failed (this might be okay)")
        
        return True
    
    def generate_activation_script(self) -> None:
        """Generate conda activation script."""
        script_content = f"""#!/bin/bash
# MuseQuill.ink Conda Environment Activation Script
# Usage: source activate_musequill.sh

echo "🚀 Activating MuseQuill conda environment..."
conda activate {self.conda_env}

if [ $? -eq 0 ]; then
    echo "✅ Environment '{self.conda_env}' activated successfully"
    echo "🔧 Python: $(python --version)"
    echo "📦 Pip: $(pip --version)"
    echo ""
    echo "📋 Quick commands:"
    echo "  python main.py --mode config    # Test configuration"
    echo "  python main.py --mode api       # Start API server"
    echo "  make test                       # Run tests"
    echo "  make run-api                    # Development server"
    echo ""
    echo "To deactivate: conda deactivate"
else
    echo "❌ Failed to activate environment '{self.conda_env}'"
fi
"""
        
        # Write activation script
        script_path = self.project_root / "activate_musequill.sh"
        with open(script_path, "w") as f:
            f.write(script_content)
        
        # Make it executable on Unix-like systems
        if os.name != "nt":
            os.chmod(script_path, 0o755)
        
        print(f"✅ Activation script created: {script_path}")
    
    def install(
        self,
        mode: str = "dev",
        python_version: str = "3.13",
        upgrade_pip: bool = True,
        run_tests: bool = True,
        force_recreate: bool = False
    ) -> bool:
        """Run the complete conda-based installation process."""
        print("🚀 Starting MuseQuill.ink Installation (Conda)")
        print("=" * 50)
        
        # Check conda availability
        if not self.check_conda_available():
            return False
        
        # Handle conda environment
        if force_recreate and self.conda_env_exists():
            print(f"🔄 Removing existing environment '{self.conda_env}'...")
            try:
                subprocess.run(["conda", "env", "remove", "-n", self.conda_env, "-y"], check=True)
                print(f"✅ Environment '{self.conda_env}' removed")
            except subprocess.CalledProcessError as e:
                print(f"❌ Failed to remove environment: {e}")
                return False
        
        # Create environment if it doesn't exist
        if not self.conda_env_exists():
            if not self.create_conda_environment(python_version):
                return False
        
        # Get Python executable from conda environment
        python_exe = self.get_conda_python_executable()
        if not python_exe:
            return False
        
        # Verify Python version in environment
        if not self.check_python_version_in_env(python_exe):
            return False
        
        # Check pip in environment
        if not self.check_pip_in_env(python_exe):
            return False
        
        # Upgrade pip if requested
        if upgrade_pip and not self.upgrade_pip_in_env(python_exe):
            return False
        
        # Install requirements based on mode
        requirements_files = {
            "base": "requirements-base.txt",
            "ai": "requirements-ai.txt", 
            "dev": "requirements-dev.txt",
            "prod": "requirements-prod.txt",
            "full": "requirements.txt"
        }
        
        requirements_file = requirements_files.get(mode, "requirements-dev.txt")
        if not self.install_requirements(requirements_file, python_exe):
            return False
        
        # Install package in editable mode
        if not self.install_editable_package(python_exe):
            return False
        
        # Setup environment
        if not self.create_env_file():
            return False
        
        if not self.create_directories():
            return False
        
        # Generate activation script
        self.generate_activation_script()
        
        # Run tests
        if run_tests and not self.run_tests(python_exe):
            print("⚠️  Tests failed, but installation might still be functional")
        
        print("\n" + "=" * 50)
        print("🎉 MuseQuill.ink Installation Complete!")
        
        # Print activation instructions
        print(f"\n📋 Next Steps:")
        print(f"1. Activate conda environment:")
        print(f"   conda activate {self.conda_env}")
        print(f"   # or source activate_musequill.sh")
        print(f"")
        print(f"2. Update .env file with your API keys:")
        print(f"   OPENAI_API_KEY=sk-your-actual-key-here")
        print(f"")
        print(f"3. Test installation:")
        print(f"   python main.py --mode config --verbose")
        print(f"")
        print(f"4. Start development:")
        print(f"   python main.py --mode api --reload")
        print(f"   # or make run-api")
        print(f"")
        print(f"💡 Pro tip: Use 'make quickstart' for guided setup")
        
        return True


def main():
    """Main installation script."""
    parser = argparse.ArgumentParser(description="MuseQuill.ink Installation Script (Conda)")
    
    parser.add_argument(
        "--mode",
        choices=["base", "ai", "dev", "prod", "full"],
        default="dev",
        help="Installation mode (default: dev)"
    )
    
    parser.add_argument(
        "--env-name",
        default="musequill",
        help="Conda environment name (default: musequill)"
    )
    
    parser.add_argument(
        "--python-version",
        default="3.13",
        help="Python version for conda environment (default: 3.13)"
    )
    
    parser.add_argument(
        "--no-pip-upgrade",
        action="store_true", 
        help="Skip pip upgrade"
    )
    
    parser.add_argument(
        "--no-tests",
        action="store_true",
        help="Skip running tests"
    )
    
    parser.add_argument(
        "--force-recreate",
        action="store_true",
        help="Force recreate conda environment if it exists"
    )
    
    args = parser.parse_args()
    
    installer = MuseQuillInstaller(conda_env=args.env_name)
    
    success = installer.install(
        mode=args.mode,
        python_version=args.python_version,
        upgrade_pip=not args.no_pip_upgrade,
        run_tests=not args.no_tests,
        force_recreate=args.force_recreate
    )
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())