"""
File processor module for handling different file types
"""
import base64
from pathlib import Path
from typing import Dict, List, Optional
import mimetypes

from PIL import Image
from pypdf import PdfReader
import requests
from bs4 import BeautifulSoup
import chardet

from .config import SUPPORTED_IMAGE_FORMATS, SUPPORTED_TEXT_FORMATS, MAX_FILE_SIZE_MB


class FileProcessor:
    """Process different types of files for AI analysis"""
    
    def __init__(self):
        self.max_file_size = MAX_FILE_SIZE_MB * 1024 * 1024  # Convert to bytes
    
    def validate_file(self, file_path: Path) -> bool:
        """Validate file size and format"""
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Check file size
        if file_path.stat().st_size > self.max_file_size:
            raise ValueError(f"File size exceeds maximum allowed size of {MAX_FILE_SIZE_MB}MB")
        
        return True
    
    def get_file_type(self, file_path: Path) -> str:
        """Determine the type of file"""
        extension = file_path.suffix.lower().lstrip('.')
        
        if extension in SUPPORTED_IMAGE_FORMATS:
            return "image"
        elif extension in SUPPORTED_TEXT_FORMATS:
            return "text"
        else:
            return "unknown"
    
    def process_image(self, file_path: Path) -> Dict[str, str]:
        """Process image file and return base64 encoded data"""
        self.validate_file(file_path)
        
        # Open and validate image
        img = Image.open(file_path)
        
        # Convert to base64
        with open(file_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")
        
        # Get MIME type
        mime_type = mimetypes.guess_type(file_path)[0] or "image/jpeg"
        
        return {
            "type": "image",
            "format": file_path.suffix.lower().lstrip('.'),
            "mime_type": mime_type,
            "data": image_data,
            "width": img.width,
            "height": img.height
        }
    
    def process_pdf(self, file_path: Path) -> Dict[str, str]:
        """Extract text from PDF file"""
        self.validate_file(file_path)
        
        text_content = []
        with open(file_path, "rb") as f:
            pdf_reader = PdfReader(f)
            for page_num, page in enumerate(pdf_reader.pages):
                text = page.extract_text()
                if text.strip():
                    text_content.append(f"--- Page {page_num + 1} ---\n{text}")
        
        return {
            "type": "text",
            "format": "pdf",
            "content": "\n\n".join(text_content),
            "pages": len(pdf_reader.pages)
        }
    
    def _detect_encoding(self, file_path: Path) -> str:
        """Detect file encoding"""
        with open(file_path, "rb") as f:
            raw_data = f.read(10000)  # Read first 10KB for detection
            result = chardet.detect(raw_data)
            return result['encoding'] or 'utf-8'
    
    def process_text_file(self, file_path: Path) -> Dict[str, str]:
        """Read text file content"""
        self.validate_file(file_path)
        
        # Try UTF-8 first, then detect encoding if it fails
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
        except UnicodeDecodeError:
            # Fallback to encoding detection
            encoding = self._detect_encoding(file_path)
            with open(file_path, "r", encoding=encoding, errors='replace') as f:
                content = f.read()
        
        return {
            "type": "text",
            "format": file_path.suffix.lower().lstrip('.'),
            "content": content
        }
    
    def process_html(self, file_path: Path) -> Dict[str, str]:
        """Extract text from HTML file"""
        self.validate_file(file_path)
        
        # Try UTF-8 first, then detect encoding if it fails
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                html_content = f.read()
        except UnicodeDecodeError:
            # Fallback to encoding detection
            encoding = self._detect_encoding(file_path)
            with open(file_path, "r", encoding=encoding, errors='replace') as f:
                html_content = f.read()
        
        soup = BeautifulSoup(html_content, "html.parser")
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Get text
        text = soup.get_text()
        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)
        
        return {
            "type": "text",
            "format": "html",
            "content": text,
            "title": soup.title.string if soup.title else ""
        }
    
    def process_file(self, file_path: Path) -> Dict[str, str]:
        """Process file based on its type"""
        file_type = self.get_file_type(file_path)
        
        if file_type == "image":
            return self.process_image(file_path)
        elif file_path.suffix.lower() == '.pdf':
            return self.process_pdf(file_path)
        elif file_path.suffix.lower() in ['.html', '.htm']:
            return self.process_html(file_path)
        elif file_type == "text":
            return self.process_text_file(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_path.suffix}")
    
    def process_multiple_files(self, file_paths: List[Path]) -> List[Dict[str, str]]:
        """Process multiple files"""
        results = []
        for file_path in file_paths:
            try:
                result = self.process_file(file_path)
                result["file_name"] = file_path.name
                result["file_path"] = str(file_path)
                results.append(result)
            except Exception as e:
                results.append({
                    "file_name": file_path.name,
                    "file_path": str(file_path),
                    "error": str(e)
                })
        return results
