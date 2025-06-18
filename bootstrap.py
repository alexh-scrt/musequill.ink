"""
Bootstrap the project
"""
import sys
from pathlib import Path
import colorama
# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# pylint: disable=unused-import,wrong-import-position
import env
from musequill.config.settings import Settings
from musequill.config.logging import setup_logging

colorama.init()

# Create settings
settings:Settings = Settings()
settings.LOG_LEVEL = "DEBUG"
settings.STRUCTURED_LOGGING = True
settings.LOG_FORMAT = "json"
settings.LOG_FILE_PATH = None  # No file logging for test
        
# Setup logging
setup_logging(settings)

