"""
Aviation RAG Chat - OCR and PDF Text Extraction Module
Handles PDF loading, text extraction, and OCR for scanned documents
"""

import os
import re
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass
from loguru import logger

import fitz  # PyMuPDF

# OCR imports (optional)
try:
    import pytesseract
    from pdf2image import convert_from_path
    from PIL import Image
    OCR_AVAILABLE = True
    
    # Set Tesseract path for Windows
    import platform
    if platform.system() == "Windows":
        tesseract_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        if os.path.exists(tesseract_path):
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
            
except ImportError:
    OCR_AVAILABLE = False
    logger.warning("OCR libraries not available. Install pytesseract and pdf2image for OCR support.")


@dataclass
class PageContent:
    """Content extracted from a single page."""
    page_number: int
    text: str
    char_count: int


class PDFExtractor:
    """
    Extract text from PDF documents.
    
    Features:
    - Clean text extraction with formatting preservation
    - OCR support for scanned/image-based PDFs
    - Page-by-page extraction
    """
    
    def __init__(
        self,
        min_text_length: int = 50,
        use_ocr: bool = True,
    ):
        """
        Initialize the PDF extractor.
        
        Args:
            min_text_length: Minimum text length to consider valid
            use_ocr: Whether to use OCR for image-based PDFs
        """
        self.min_text_length = min_text_length
        self.use_ocr = use_ocr and OCR_AVAILABLE
        
        logger.info(f"PDFExtractor initialized: OCR available={OCR_AVAILABLE}")
    
    def _clean_text(self, text: str) -> str:
        """
        Clean extracted text while preserving meaningful structure.
        
        Operations:
        - Remove excessive whitespace
        - Fix common PDF extraction issues
        - Preserve paragraph breaks
        - Handle special characters
        """
        if not text:
            return ""
        
        # Replace multiple spaces with single space
        text = re.sub(r'[ \t]+', ' ', text)
        
        # Replace multiple newlines with double newline (paragraph break)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Fix hyphenated line breaks (word-\nbreak -> wordbreak)
        text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)
        
        # Remove page headers/footers patterns (common in textbooks)
        # This is a simplified pattern - adjust based on actual documents
        text = re.sub(r'^\d+\s*$', '', text, flags=re.MULTILINE)
        
        # Clean up leading/trailing whitespace on each line
        lines = [line.strip() for line in text.split('\n')]
        text = '\n'.join(lines)
        
        # Final cleanup
        text = text.strip()
        
        return text
    
    def _extract_page_text(self, page: fitz.Page) -> str:
        """
        Extract text from a single PDF page.
        
        Uses PyMuPDF's text extraction with layout preservation.
        """
        # Extract text with layout preservation
        text = page.get_text("text", sort=True)
        return self._clean_text(text)
    
    def extract_all_pages(self, pdf_path: str) -> List[PageContent]:
        """
        Extract text from all pages of a PDF.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of PageContent objects
        """
        pages = []
        
        try:
            doc = fitz.open(pdf_path)
            total_text_length = 0
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = self._extract_page_text(page)
                
                if text:  # Only add pages with content
                    pages.append(PageContent(
                        page_number=page_num + 1,  # 1-indexed
                        text=text,
                        char_count=len(text)
                    ))
                    total_text_length += len(text)
            
            doc.close()
            
            # If no text extracted and OCR is available, try OCR
            if total_text_length < 100 and self.use_ocr:
                logger.info(f"No text found in {pdf_path}, attempting OCR...")
                pages = self._extract_pages_with_ocr(pdf_path)
            
            logger.debug(f"Extracted {len(pages)} pages from {pdf_path}")
            
        except Exception as e:
            logger.error(f"Error extracting pages from {pdf_path}: {e}")
            raise
        
        return pages
    
    def _extract_pages_with_ocr(self, pdf_path: str) -> List[PageContent]:
        """
        Extract text from PDF using OCR (for scanned/image-based PDFs).
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of PageContent objects
        """
        if not OCR_AVAILABLE:
            logger.warning("OCR not available - install pytesseract and pdf2image")
            return []
        
        pages = []
        
        try:
            # Find poppler path for Windows
            poppler_path = None
            import platform
            if platform.system() == "Windows":
                import glob
                # Check common installation paths
                poppler_patterns = [
                    os.path.expanduser("~\\AppData\\Local\\Microsoft\\WinGet\\Packages\\*Poppler*\\**\\bin"),
                    "C:\\Program Files\\poppler*\\bin",
                    "C:\\poppler*\\bin",
                ]
                for pattern in poppler_patterns:
                    matches = glob.glob(pattern, recursive=True)
                    if matches:
                        poppler_path = matches[0]
                        break
            
            # Convert PDF pages to images
            logger.info(f"Converting PDF to images for OCR: {pdf_path}")
            if poppler_path:
                logger.debug(f"Using poppler from: {poppler_path}")
                images = convert_from_path(pdf_path, dpi=200, poppler_path=poppler_path)
            else:
                images = convert_from_path(pdf_path, dpi=200)
            
            for page_num, image in enumerate(images, 1):
                logger.info(f"OCR processing page {page_num}/{len(images)}")
                
                # Perform OCR on the image
                text = pytesseract.image_to_string(image, lang='eng')
                text = self._clean_text(text)
                
                if text and len(text) > self.min_text_length:
                    pages.append(PageContent(
                        page_number=page_num,
                        text=text,
                        char_count=len(text)
                    ))
            
            logger.info(f"OCR completed: {len(pages)} pages with text extracted")
            
        except Exception as e:
            logger.error(f"OCR failed for {pdf_path}: {e}")
        
        return pages
    
    def get_document_metadata(self, pdf_path: str) -> dict:
        """
        Extract metadata from a PDF document.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary with document metadata
        """
        try:
            doc = fitz.open(pdf_path)
            metadata = {
                "total_pages": len(doc),
                "title": doc.metadata.get("title", ""),
                "author": doc.metadata.get("author", ""),
                "subject": doc.metadata.get("subject", ""),
                "keywords": doc.metadata.get("keywords", ""),
                "creator": doc.metadata.get("creator", ""),
                "producer": doc.metadata.get("producer", ""),
            }
            doc.close()
            return metadata
        except Exception as e:
            logger.error(f"Failed to extract metadata from {pdf_path}: {e}")
            return {}


# Singleton instance
pdf_extractor = PDFExtractor()
