"""
Basic tests for CookMoney application
"""
import unittest
from pathlib import Path
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.file_processor import FileProcessor
from src.config import UPLOADED_DIR, ANALYSIS_DIR, BASE_DIR


class TestFileProcessor(unittest.TestCase):
    """Test FileProcessor class"""
    
    def setUp(self):
        self.processor = FileProcessor()
        self.test_file = BASE_DIR / "reports" / "example_report.md"
    
    def test_file_exists(self):
        """Test that example file exists"""
        self.assertTrue(self.test_file.exists())
    
    def test_get_file_type(self):
        """Test file type detection"""
        self.assertEqual(self.processor.get_file_type(self.test_file), "text")
        
        # Test image detection
        image_path = Path("test.jpg")
        self.assertEqual(self.processor.get_file_type(image_path), "image")
        
        # Test PDF detection
        pdf_path = Path("test.pdf")
        self.assertEqual(self.processor.get_file_type(pdf_path), "text")
    
    def test_process_text_file(self):
        """Test processing text file"""
        result = self.processor.process_text_file(self.test_file)
        
        self.assertEqual(result["type"], "text")
        self.assertEqual(result["format"], "md")
        self.assertIn("content", result)
        self.assertGreater(len(result["content"]), 0)
        self.assertIn("XX科技", result["content"])
    
    def test_process_file(self):
        """Test general file processing"""
        result = self.processor.process_file(self.test_file)
        
        self.assertEqual(result["type"], "text")
        self.assertIn("content", result)


class TestConfiguration(unittest.TestCase):
    """Test configuration"""
    
    def test_directories_exist(self):
        """Test that required directories exist"""
        self.assertTrue(UPLOADED_DIR.exists())
        self.assertTrue(ANALYSIS_DIR.exists())
        self.assertTrue(BASE_DIR.exists())
    
    def test_directory_structure(self):
        """Test directory structure"""
        self.assertTrue((BASE_DIR / "reports").exists())
        self.assertTrue((BASE_DIR / "src").exists())


class TestCLI(unittest.TestCase):
    """Test CLI functionality"""
    
    def test_main_module_import(self):
        """Test that main module can be imported"""
        import main
        self.assertTrue(hasattr(main, 'main'))
        self.assertTrue(hasattr(main, 'analyze_command'))
        self.assertTrue(hasattr(main, 'list_command'))


if __name__ == '__main__':
    # Run tests
    unittest.main()
