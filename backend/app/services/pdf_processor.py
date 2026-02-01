"""
Aviation RAG Chat - PDF Processing Module
Handles PDF loading, text extraction, and chunking with page awareness
Supports OCR for scanned/image-based PDFs
"""

import os
import re
import hashlib
import time
from pathlib import Path
from typing import List, Tuple, Optional, Generator
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import fitz  # PyMuPDF
from loguru import logger

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

from app.core.config import settings
from app.models.schemas import DocumentMetadata, TextChunk, ProcessedDocument


@dataclass
class PageContent:
    """Content extracted from a single page."""
    page_number: int
    text: str
    char_count: int


class PDFProcessor:
    """
    Process PDF documents for RAG ingestion.
    
    Features:
    - Clean text extraction with formatting preservation
    - Page-aware chunking with overlap
    - Parallel processing for multiple documents
    - Metadata extraction
    - OCR support for scanned/image-based PDFs
    """
    
    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None,
        min_chunk_size: int = 50,
        use_ocr: bool = True,
    ):
        """
        Initialize the PDF processor.
        
        Args:
            chunk_size: Maximum characters per chunk
            chunk_overlap: Number of overlapping characters between chunks
            min_chunk_size: Minimum characters for a valid chunk
            use_ocr: Whether to use OCR for image-based PDFs
        """
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.use_ocr = use_ocr and OCR_AVAILABLE
        
        logger.info(
            f"PDFProcessor initialized: chunk_size={self.chunk_size}, "
            f"overlap={self.chunk_overlap}"
        )
    
    def _generate_document_id(self, filepath: str) -> str:
        """Generate a unique document ID from filepath."""
        return hashlib.md5(filepath.encode()).hexdigest()[:12]
    
    def _generate_chunk_id(self, doc_id: str, chunk_index: int) -> str:
        """Generate a unique chunk ID."""
        return f"{doc_id}_chunk_{chunk_index:04d}"
    
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
    
    def _extract_all_pages(self, pdf_path: str) -> List[PageContent]:
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
                
                if text and len(text) > self.min_chunk_size:
                    pages.append(PageContent(
                        page_number=page_num,
                        text=text,
                        char_count=len(text)
                    ))
            
            logger.info(f"OCR completed: {len(pages)} pages with text extracted")
            
        except Exception as e:
            logger.error(f"OCR failed for {pdf_path}: {e}")
        
        return pages
    
    def _chunk_text_with_pages(
        self,
        pages: List[PageContent],
        doc_id: str,
        filename: str,
    ) -> List[TextChunk]:
        """
        Split text into chunks while tracking page numbers.
        Chunking Strategy:
        1. Process pages sequentially, building chunks up to chunk_size
        2. Try to break at sentence boundaries
        3. Include overlap from previous chunk
        4. Track which page(s) each chunk comes from
        
        This ensures citations can point to specific pages.
        """
        chunks = []
        current_chunk = ""
        current_page = 1
        chunk_index = 0
        
        for page in pages:
            page_text = page.text
            page_num = page.page_number
            
            # Add page marker for tracking
            remaining_text = page_text
            
            while remaining_text:
                # Calculate how much space we have
                space_available = self.chunk_size - len(current_chunk)
                
                if space_available <= 0:
                    # Current chunk is full, save it
                    if len(current_chunk.strip()) >= self.min_chunk_size:
                        chunks.append(TextChunk(
                            chunk_id=self._generate_chunk_id(doc_id, chunk_index),
                            document_id=doc_id,
                            content=current_chunk.strip(),
                            page_number=current_page,
                            chunk_index=chunk_index,
                            metadata={
                                "filename": filename,
                                "source": filename,
                            }
                        ))
                        chunk_index += 1
                    
                    # Start new chunk with overlap
                    overlap_text = self._get_overlap_text(current_chunk)
                    current_chunk = overlap_text
                    current_page = page_num
                    space_available = self.chunk_size - len(current_chunk)
                
                # Take what fits
                if len(remaining_text) <= space_available:
                    current_chunk += " " + remaining_text
                    remaining_text = ""
                else:
                    # Find a good break point
                    break_point = self._find_break_point(
                        remaining_text, 
                        space_available
                    )
                    current_chunk += " " + remaining_text[:break_point]
                    remaining_text = remaining_text[break_point:].strip()
        
        # Don't forget the last chunk
        if len(current_chunk.strip()) >= self.min_chunk_size:
            chunks.append(TextChunk(
                chunk_id=self._generate_chunk_id(doc_id, chunk_index),
                document_id=doc_id,
                content=current_chunk.strip(),
                page_number=current_page,
                chunk_index=chunk_index,
                metadata={
                    "filename": filename,
                    "source": filename,
                }
            ))
        
        return chunks
    
    def _get_overlap_text(self, text: str) -> str:
        """Get the overlap portion from the end of text."""
        if len(text) <= self.chunk_overlap:
            return text
        
        # Try to start at a sentence boundary within the overlap region
        overlap_region = text[-self.chunk_overlap:]
        
        # Find the first sentence start in the overlap
        sentence_starts = [m.start() for m in re.finditer(r'[.!?]\s+[A-Z]', overlap_region)]
        
        if sentence_starts:
            # Start from the sentence boundary
            return overlap_region[sentence_starts[0] + 2:]
        
        # Fall back to word boundary
        word_boundary = overlap_region.find(' ')
        if word_boundary > 0:
            return overlap_region[word_boundary + 1:]
        
        return overlap_region
    
    def _find_break_point(self, text: str, max_length: int) -> int:
        """
        Find a good point to break text, preferring sentence boundaries.
        """
        if max_length >= len(text):
            return len(text)
        
        search_region = text[:max_length]
        
        # Prefer sentence boundaries
        sentence_ends = [m.end() for m in re.finditer(r'[.!?]\s+', search_region)]
        if sentence_ends:
            return sentence_ends[-1]
        
        # Fall back to paragraph breaks
        para_breaks = [m.end() for m in re.finditer(r'\n\n', search_region)]
        if para_breaks:
            return para_breaks[-1]
        
        # Fall back to word boundary
        last_space = search_region.rfind(' ')
        if last_space > 0:
            return last_space
        
        return max_length
    
    def process_document(self, pdf_path: str) -> ProcessedDocument:
        """
        Process a single PDF document.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            ProcessedDocument with all chunks
        """
        start_time = time.time()
        pdf_path = str(pdf_path)
        filename = os.path.basename(pdf_path)
        
        logger.info(f"Processing document: {filename}")
        
        # Generate document ID
        doc_id = self._generate_document_id(pdf_path)
        
        # Extract pages
        pages = self._extract_all_pages(pdf_path)
        
        if not pages:
            raise ValueError(f"No text content extracted from {pdf_path}")
        
        # Get document metadata
        file_stat = os.stat(pdf_path)
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        doc.close()
        
        metadata = DocumentMetadata(
            filename=filename,
            filepath=pdf_path,
            file_size=file_stat.st_size,
            total_pages=total_pages,
            category=self._infer_category(pdf_path),
        )
        
        # Chunk the document
        chunks = self._chunk_text_with_pages(pages, doc_id, filename)
        
        processing_time = time.time() - start_time
        
        logger.info(
            f"Processed {filename}: {len(chunks)} chunks, "
            f"{processing_time:.2f}s"
        )
        
        return ProcessedDocument(
            document_id=doc_id,
            metadata=metadata,
            chunks=chunks,
            total_chunks=len(chunks),
            processing_time_seconds=processing_time,
        )
    
    def _infer_category(self, filepath: str) -> Optional[str]:
        """Infer document category from filepath."""
        path_lower = filepath.lower()
        
        if "navigation" in path_lower:
            return "Air Navigation"
        elif "meteorology" in path_lower:
            return "Meteorology"
        elif "regulation" in path_lower:
            return "Air Regulations"
        elif "instrument" in path_lower:
            return "Instruments"
        elif "flight-planning" in path_lower or "flight planning" in path_lower:
            return "Flight Planning"
        elif "mass" in path_lower or "balance" in path_lower:
            return "Mass and Balance"
        elif "radio" in path_lower:
            return "Radio Navigation"
        
        return None
    
    def process_directory(
        self,
        directory: str,
        max_workers: int = 4,
    ) -> List[ProcessedDocument]:
        """
        Process all PDFs in a directory (including subdirectories).
        
        Uses parallel processing for efficiency.
        
        Args:
            directory: Path to directory containing PDFs
            max_workers: Number of parallel workers
            
        Returns:
            List of processed documents
        """
        directory = Path(directory)
        pdf_files = list(directory.rglob("*.pdf"))
        
        if not pdf_files:
            logger.warning(f"No PDF files found in {directory}")
            return []
        
        logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        processed_docs = []
        errors = []
        
        # Process in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_path = {
                executor.submit(self.process_document, str(pdf_path)): pdf_path
                for pdf_path in pdf_files
            }
            
            for future in as_completed(future_to_path):
                pdf_path = future_to_path[future]
                try:
                    result = future.result()
                    processed_docs.append(result)
                except Exception as e:
                    logger.error(f"Failed to process {pdf_path}: {e}")
                    errors.append((str(pdf_path), str(e)))
        
        logger.info(
            f"Processed {len(processed_docs)} documents successfully, "
            f"{len(errors)} errors"
        )
        
        return processed_docs


# Singleton instance
pdf_processor = PDFProcessor()
