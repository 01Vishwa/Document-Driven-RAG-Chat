"""
Aviation RAG Chat - Text Chunking Module
Page-aware text chunking with overlap and sentence boundary breaking
"""

import re
import hashlib
from typing import List, Optional
from loguru import logger

from app.core.config import settings
from app.core.schemas import TextChunk
from app.rag.ocr_extract import PageContent


class TextChunker:
    """
    Split text into chunks while tracking page numbers.
    
    Chunking Strategy:
    ==================
    We use a page-aware sliding window approach with sentence boundary breaking:
    
    1. **Chunk Size (512 chars)**: Balances context completeness vs retrieval precision.
       - Too small: loses context, fragments concepts
       - Too large: reduces retrieval precision, wastes tokens
       - 512 chars ≈ 100-120 words ≈ 3-5 sentences - optimal for dense technical content
    
    2. **Overlap (128 chars)**: Ensures continuity across chunk boundaries.
       - ~25% overlap prevents important information from being split
       - Helps maintain context for concepts that span chunk boundaries
    
    3. **Page Tracking**: Each chunk knows its source page for accurate citations.
       - Aviation documents require precise page references
       - Enables regulatory compliance citation requirements
    
    4. **Sentence Boundary Breaking**: Chunks break at sentence boundaries when possible.
       - Prevents mid-sentence splits that lose meaning
       - Improves retrieval relevance
    
    Rationale for Aviation Documents:
    - Technical aviation content is dense with definitions and procedures
    - Regulations often reference specific sections/pages
    - Exam questions often target specific facts that fit in 512-char chunks
    """
    
    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None,
        min_chunk_size: int = 50,
    ):
        """
        Initialize the text chunker.
        
        Args:
            chunk_size: Maximum characters per chunk
            chunk_overlap: Number of overlapping characters between chunks
            min_chunk_size: Minimum characters for a valid chunk
        """
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        self.min_chunk_size = min_chunk_size
        
        logger.info(
            f"TextChunker initialized: chunk_size={self.chunk_size}, "
            f"overlap={self.chunk_overlap}"
        )
    
    def _generate_document_id(self, filepath: str) -> str:
        """Generate a unique document ID from filepath."""
        return hashlib.md5(filepath.encode()).hexdigest()[:12]
    
    def _generate_chunk_id(self, doc_id: str, chunk_index: int) -> str:
        """Generate a unique chunk ID."""
        return f"{doc_id}_chunk_{chunk_index:04d}"
    
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
    
    def chunk_pages(
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
        
        Args:
            pages: List of PageContent objects
            doc_id: Document ID
            filename: Source filename
            
        Returns:
            List of TextChunk objects
        """
        chunks = []
        current_chunk = ""
        current_page = pages[0].page_number if pages else 1
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
    
    def chunk_text(
        self,
        text: str,
        doc_id: str,
        filename: str,
        page_number: int = 1,
    ) -> List[TextChunk]:
        """
        Chunk a single text string (without page tracking).
        
        Args:
            text: Text to chunk
            doc_id: Document ID
            filename: Source filename
            page_number: Page number to assign
            
        Returns:
            List of TextChunk objects
        """
        # Create a single page content and delegate
        pages = [PageContent(
            page_number=page_number,
            text=text,
            char_count=len(text)
        )]
        return self.chunk_pages(pages, doc_id, filename)


# Singleton instance
text_chunker = TextChunker()
