"""
Configuration module for CookMoney application
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base directories
BASE_DIR = Path(__file__).parent.parent
REPORTS_DIR = BASE_DIR / "reports"
UPLOADED_DIR = REPORTS_DIR / "uploaded"
ANALYSIS_DIR = REPORTS_DIR / "analysis"

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

# Application Settings
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "10"))
SUPPORTED_IMAGE_FORMATS = os.getenv("SUPPORTED_IMAGE_FORMATS", "jpg,jpeg,png,gif,webp").split(",")
SUPPORTED_TEXT_FORMATS = os.getenv("SUPPORTED_TEXT_FORMATS", "txt,md,pdf,html").split(",")

# Create directories if they don't exist
UPLOADED_DIR.mkdir(parents=True, exist_ok=True)
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
