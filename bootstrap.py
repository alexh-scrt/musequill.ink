"""
Bootstrap the project
"""
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# pylint: disable=unused-import,wrong-import-position
import env
