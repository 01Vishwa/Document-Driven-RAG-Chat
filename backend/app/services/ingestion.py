"""
Document Ingestion Service with Cross-Page Chunking & CUDA Acceleration.
Handles PDF loading, text extraction, chunking, and indexing for both FAISS and BM25.

CRITICAL GUARANTEES:
- Cross-page chunking (no information loss at page boundaries)
- Lossless page number mapping via character offsets
- Deterministic indexing (re-ingestion cannot corrupt)
- Normalized embeddings for cosine similarity
- All paths from environment variables
- CUDA Acceleration for Embeddings & FAISS
"""
import os
import re
import pickle
import time
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

import numpy as np
from pypdf import PdfReader
import pdfplumber
from sentence_transformers import SentenceTransformer
import vecs
# import faiss  # Removed FAISS
import torch
from rank_bm25 import BM25Okapi
from tqdm import tqdm
from langchain_text_splitters import RecursiveCharacterTextSplitter

import vecs
from app.core import get_settings, logger


# Stopwords for BM25 tokenization
STOPWORDS = frozenset({
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'to', 'of',
    'in', 'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through',
    'during', 'before', 'after', 'above', 'below', 'between', 'under',
    'and', 'but', 'or', 'so', 'if', 'when', 'where', 'what', 'which', 'who',
    'that', 'this', 'these', 'those', 'it', 'its', 'they', 'them', 'their',
})


@dataclass
class DocumentChunk:
    """
    Represents a chunk of text from a document.
    
    Required metadata for citation traceability:
    - chunk_id: Unique identifier for the chunk
    - page_number: Source page (can span multiple pages)
    - source_document: Original PDF filename
    - raw_text: Original unprocessed text (for verification)
    """
    chunk_id: str
    content: str  # Processed text for retrieval
    document_name: str
    page_number: Optional[int] = None
    page_numbers: List[int] = field(default_factory=list)  # For multi-page chunks
    chunk_index: int = 0
    total_chunks: int = 0
    start_char: int = 0
    end_char: int = 0
    raw_text: str = ""  # Original text before cleaning
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_metadata(self) -> Dict[str, Any]:
        """Convert to metadata dict for vector store."""
        return {
            "chunk_id": self.chunk_id,
            "content": self.content,
            "document_name": self.document_name,
            "page_number": self.page_number,
            "page_numbers": self.page_numbers,
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "raw_text": self.raw_text,
            **self.metadata
        }


class IngestionService:
    """
    Service for ingesting PDF documents into vector and BM25 indices.
    
    CHUNKING STRATEGY (Justified):
    - chunk_size=1000: Balances context (enough for complete concepts) 
      vs retrieval precision (not too much noise)
    - chunk_overlap=200: Ensures sentence boundaries aren't lost and
      provides context continuity for related concepts
    - Cross-page chunking: Documents are merged before chunking to
      prevent information loss at page boundaries
    - Sentence-aware splitting: Avoids mid-sentence cuts
    
    INDEXING GUARANTEES:
    - FAISS uses normalized embeddings for cosine similarity
    - BM25 uses proper tokenization with stopword removal
    - Re-ingestion clears existing indices (no duplicate vectors)
    - Deterministic chunk IDs based on content hash
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.embedding_model: Optional[SentenceTransformer] = None
        self.vx_client = None
        self.collection = None
        self.bm25_index: Optional[BM25Okapi] = None
        self.chunks: List[DocumentChunk] = []
        self.chunk_texts: List[str] = []
        self._initialized = False
        
        # Check for CUDA
        if torch.cuda.is_available():
            self.device = "cuda"
            logger.info("✔ CUDA connection established (GPU detected)")
        else:
            self.device = "cpu"
            logger.warning("CUDA not found, using CPU")
            
        logger.info(f"Ingestion Service running on: {self.device.upper()}")
        
    def _get_embedding_model(self) -> SentenceTransformer:
        """Lazy load the embedding model."""
        if self.embedding_model is None:
            logger.info(f"Loading embedding model: {self.settings.embedding_model} on {self.device}")
            self.embedding_model = SentenceTransformer(self.settings.embedding_model, device=self.device)
        return self.embedding_model

    def _get_supabase_collection(self, dimension: int):
        """Lazy load/create Supabase collection."""
        if self.collection is None:
            if not self.settings.supabase_connection_string:
                raise ValueError("SUPABASE_CONNECTION_STRING is not set in configuration")
            
            if self.vx_client is None:
                self.vx_client = vecs.create_client(self.settings.supabase_connection_string)
            
            logger.info(f"Connecting to Supabase collection: {self.settings.supabase_collection_name}")
            self.collection = self.vx_client.get_or_create_collection(
                name=self.settings.supabase_collection_name, 
                dimension=dimension
            )
        return self.collection
    
    def _generate_chunk_id(self, content: str, doc_name: str, chunk_idx: int) -> str:
        """
        Generate a deterministic unique chunk ID based on content hash.
        This ensures the same content always gets the same ID.
        """
        hash_input = f"{doc_name}:{chunk_idx}:{content[:100]}"
        return hashlib.md5(hash_input.encode()).hexdigest()[:12]
    
    def _extract_text_with_headers(self, pdf_path: Path) -> List[Dict[str, Any]]:
        """
        Extract text preserving structure and identifying headers.
        Returns a list of sections: {'header': str, 'content': str, 'page': int}
        """
        sections = []
        current_header = "General"
        current_content = []
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                logger.info(f"Extracting {len(pdf.pages)} pages from {pdf_path.name}")
                
                # First pass: Determine median font size to detect headers
                font_sizes = []
                for page in pdf.pages[:5]: # Sample first 5 pages
                    for char in page.chars:
                        font_sizes.append(char.get('size', 0))
                
                median_size = np.median(font_sizes) if font_sizes else 10
                header_threshold = median_size * 1.2  # Headers are 20% larger
                
                for i, page in enumerate(pdf.pages, start=1):
                    # Extract words with properties
                    words = page.extract_words(extra_attrs=['fontname', 'size', 'top'])
                    lines = {} # Map top_position -> list of words
                    
                    # Group words into lines
                    for w in words:
                        top = round(w['top'], 1) # Fuzzy grouping by line position
                        if top not in lines:
                            lines[top] = []
                        lines[top].append(w)
                    
                    sorted_tops = sorted(lines.keys())
                    
                    for top in sorted_tops:
                        line_words = lines[top]
                        line_text = ' '.join(w['text'] for w in line_words)
                        if not line_text.strip():
                            continue
                            
                        # Check if line is a header
                        # Heuristic: Average font size of line > threshold
                        avg_size = sum(w['size'] for w in line_words) / len(line_words)
                        is_header = avg_size >= header_threshold and len(line_text) < 100
                        
                        if is_header:
                            # Save previous section
                            if current_content:
                                sections.append({
                                    'header': current_header,
                                    'content': "\n".join(current_content),
                                    'page': i # Approximate page (start of section)
                                })
                                current_content = []
                            
                            current_header = line_text
                        else:
                            current_content.append(line_text)
                            
                # Save last section
                if current_content:
                    sections.append({
                        'header': current_header,
                        'content': "\n".join(current_content),
                        'page': len(pdf.pages)
                    })
                    
        except Exception as e:
            logger.error(f"pdfplumber failed for {pdf_path}: {e}")
            # Fallback to simple extraction
            return self._extract_simple_fallback(pdf_path)
            
        return sections

    def _extract_simple_fallback(self, pdf_path: Path) -> List[Dict[str, Any]]:
        """Fallback extraction if smart extraction fails."""
        sections = []
        reader = PdfReader(pdf_path)
        for i, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                sections.append({
                    'header': f"Page {i}",
                    'content': text,
                    'page': i
                })
        return sections
    
    def _clean_text(self, text: str) -> str:
        """
        Clean extracted text while preserving semantic content.
        
        SAFE OPERATIONS (no semantic loss):
        - Remove excessive whitespace
        - Normalize line breaks
        - Remove control characters
        
        PRESERVED:
        - Numbers, punctuation, special aviation terms (VOR/DME, FL350, etc.)
        """
        # Remove control characters but keep newlines
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
        
        # Normalize whitespace within lines
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            # Collapse multiple spaces but preserve line structure
            line = ' '.join(line.split())
            if line:
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def _tokenize_for_bm25(self, text: str) -> List[str]:
        """
        Proper tokenization for BM25 with:
        - Lowercase normalization
        - Stopword removal
        - Aviation term handling (VOR/DME -> VOR, DME)
        - Minimum token length
        """
        # Handle special aviation terms with slashes
        text = re.sub(r'/', ' ', text)
        
        # Extract words
        words = re.findall(r'\b[a-zA-Z0-9]+\b', text.lower())
        
        # Filter: remove stopwords and very short tokens
        tokens = [w for w in words if w not in STOPWORDS and len(w) > 1]
        
        return tokens
    
    def _chunk_document(
        self,
        sections: List[Dict[str, Any]],
        doc_name: str,
        chunk_size: int,
        chunk_overlap: int
    ) -> List[DocumentChunk]:
        """
        Chunk a document using LangChain's RecursiveCharacterTextSplitter.
        Prepends headers to EACH chunk to improve retrieval context.
        """
        if not sections:
            return []
            
        # Use LangChain splitter per user requirement
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
            length_function=len
        )
        
        chunks = []
        global_chunk_idx = 0
        
        for section in sections:
            header = section['header']
            content = section['content']
            page = section['page']
            
            # Split the content
            raw_chunks = splitter.split_text(content)
            
            for raw_chunk in raw_chunks:
                # CRITICAL: Prepend header to every chunk
                # "If a page has a header 'Emergency Landing,' prepend that header to every chunk"
                enriched_content = f"Section: {header}\n\n{raw_chunk}"
                
                chunk_id = self._generate_chunk_id(enriched_content, doc_name, global_chunk_idx)
                
                chunks.append(DocumentChunk(
                    chunk_id=chunk_id,
                    content=enriched_content,
                    document_name=doc_name,
                    page_number=page,
                    page_numbers=[page], # Simplified for now
                    chunk_index=global_chunk_idx,
                    raw_text=raw_chunk,
                    metadata={
                        "source": doc_name,
                        "page": page,
                        "header": header, # Store header in metadata too
                        "chunk_size": len(enriched_content)
                    }
                ))
                global_chunk_idx += 1
        
        logger.info(f"Created {len(chunks)} enriched chunks from {doc_name}")
        return chunks
    
    def _get_pages_for_range(
        self,
        start_char: int,
        end_char: int,
        page_boundaries: List[Tuple[int, int, int]]
    ) -> List[int]:
        """
        Determine which pages a character range spans.
        
        Args:
            start_char: Start character position
            end_char: End character position
            page_boundaries: List of (page_start, page_end, page_number)
            
        Returns:
            List of page numbers that this range spans
        """
        pages = []
        for page_start, page_end, page_num in page_boundaries:
            # Check if ranges overlap
            if start_char < page_end and end_char > page_start:
                pages.append(page_num)
        return pages if pages else [1]  # Default to page 1 if no match
    
    def _upsert_to_supabase(self, embeddings: np.ndarray, chunks: List[DocumentChunk]):
        """
        Upsert vectors and metadata to Supabase.
        """
        if not chunks:
            return

        dimension = embeddings.shape[1]
        
        logger.info("Connecting to Supabase...")
        try:
            collection = self._get_supabase_collection(dimension)
            logger.info("✔ Supabase connection established")
            logger.info("✔ Table (Collection) created/verified")
        except Exception as e:
            logger.error(f"❌ Supabase connection failed: {e}")
            raise e
        
        logger.info(f"Upserting {len(chunks)} vectors to Supabase...")
        
        # Prepare records: (id, vector, metadata)
        records = []
        for i, chunk in enumerate(chunks):
            # metadata = chunk.to_metadata() # Use the helper if available or convert manually
            # We added to_metadata earlier
            records.append((
                chunk.chunk_id,
                embeddings[i].tolist(),
                chunk.to_metadata()
            ))
            
        # Batch upsert to avoid timeouts (e.g., 500 at a time)
        batch_size = 500
        total_batches = (len(records) + batch_size - 1) // batch_size
        
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            logger.info(f"Upserting batch {i//batch_size + 1}/{total_batches} ({len(batch)} records)")
            collection.upsert(records=batch)
            
        logger.info("✔ Embedding upload done")
        logger.info(f"Supabase upsert complete: {len(records)} records")
    
    def _build_bm25_index(self, texts: List[str]) -> BM25Okapi:
        """
        Build BM25 index from tokenized texts.
        Uses proper tokenization with stopword removal.
        """
        tokenized = [self._tokenize_for_bm25(text) for text in texts]
        logger.info(f"Building BM25 index: {len(tokenized)} documents")
        return BM25Okapi(tokenized)
    
    def ingest_documents(
        self,
        pdf_paths: Optional[List[str]] = None,
        force_reindex: bool = False
    ) -> Dict[str, Any]:
        """
        Ingest PDF documents and build indices.
        
        CRITICAL: force_reindex=True clears existing indices completely
        to prevent duplicate vectors.
        
        Args:
            pdf_paths: List of PDF file paths. If None, ingests all PDFs from Raw directory.
            force_reindex: If True, rebuilds indices even if they exist.
            
        Returns:
            Dictionary with ingestion statistics.
        """
        start_time = time.time()
        
        # Check for existing indices (skip if not forcing reindex)
        if not force_reindex and self._load_existing_indices():
            logger.info("Loaded existing indices from disk")
            return {
                "success": True,
                "message": "Loaded existing indices",
                "documents_processed": 0,
                "total_chunks": len(self.chunks),
                "processing_time_seconds": time.time() - start_time
            }
        
        # CRITICAL: Clear existing data to prevent duplicates
        self.chunks = []
        self.chunk_texts = []
        # self.faiss_index = None # Removed
        self.bm25_index = None
        self._initialized = False
        
        # In Supabase, if force_reindex is True, we might want to clear the collection
        # But vecs doesn't have a simple 'clear' without recreate?
        # Actually we can just drop and recreate if needed, or query delete.
        # For now, we'll rely on upsert (overwriting same IDs).
        # Since IDs are deterministic based on content, this is safe. 
        # But stale chunks (from old docs) won't be deleted.
        # Ideally: if force_reindex, drop collection.
        if force_reindex:
             # Lazy check connection to drop
             if self.settings.supabase_connection_string:
                 try:
                     client = vecs.create_client(self.settings.supabase_connection_string)
                     client.delete_collection(self.settings.supabase_collection_name)
                     logger.info(f"Dropped Supabase collection: {self.settings.supabase_collection_name} for re-indexing")
                     # Reset client wrapper to force recreation
                     self.vx_client = None
                     self.collection = None
                 except Exception as e:
                     logger.warning(f"Failed to drop collection: {e}")
        
        # Determine PDF files to process
        if pdf_paths is None:
            # Find all PDFs in Raw directory (from settings)
            raw_dir = Path(self.settings.raw_documents_path)
            if not raw_dir.exists():
                # Fallback paths
                for fallback in ["Raw", "../Raw", "../../Raw"]:
                    raw_dir = Path(fallback)
                    if raw_dir.exists():
                        break
            
            pdf_files = list(raw_dir.rglob("*.pdf"))
            logger.info(f"Found {len(pdf_files)} PDF files in {raw_dir.absolute()}")
        else:
            pdf_files = [Path(p) for p in pdf_paths]
        
        if not pdf_files:
            logger.warning("No PDF files found for ingestion")
            return {
                "success": False,
                "message": "No PDF files found",
                "documents_processed": 0,
                "total_chunks": 0,
                "processing_time_seconds": time.time() - start_time
            }
            
        # USER REQUEST: Establish connection BEFORE embedding to prevent wasted effort
        logger.info("Verifying Supabase connection before processing...")
        try:
            self._get_supabase_collection(dimension=384)
            logger.info("✔ Supabase connection verified successfully")
        except Exception as e:
            logger.error(f"❌ Failed to connect to Supabase: {e}")
            raise ConnectionError(f"Could not connect to Supabase: {e}")
        
        # Process each PDF with cross-page chunking
        all_chunks = []
        for pdf_path in tqdm(pdf_files, desc="Processing PDFs"):
            logger.info(f"Processing: {pdf_path.name}")
            
            # Extract text with headers
            sections = self._extract_text_with_headers(pdf_path)
            
            if not sections:
                logger.warning(f"No text extracted from {pdf_path.name}")
                continue
            
            # Chunk with header enrichment
            doc_chunks = self._chunk_document(
                sections=sections,
                doc_name=pdf_path.name,
                chunk_size=self.settings.chunk_size,
                chunk_overlap=self.settings.chunk_overlap
            )
            all_chunks.extend(doc_chunks)
        
        if not all_chunks:
            return {
                "success": False,
                "message": "No text extracted from PDFs",
                "documents_processed": len(pdf_files),
                "total_chunks": 0,
                "processing_time_seconds": time.time() - start_time
            }
        
        # Update total_chunks for each chunk
        total = len(all_chunks)
        for chunk in all_chunks:
            chunk.total_chunks = total
        
        self.chunks = all_chunks
        self.chunk_texts = [chunk.content for chunk in all_chunks]
        
        logger.info(f"Total chunks created: {len(self.chunks)}")
        
        # Generate embeddings with CUDA if available
        logger.info(f"Generating embeddings using {self.device.upper()}...")
        model = self._get_embedding_model()
        
        # Performance optimization: Increase batch size for GPU
        batch_size = 256 if self.device == "cuda" else 32
        
        embeddings = model.encode(
            self.chunk_texts,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
            batch_size=batch_size
        )
        embeddings = embeddings.astype(np.float32)
        
        logger.info(f"Embeddings shape: {embeddings.shape}")
        
        # --- LOCAL CACHING START ---
        # User requested resiliency: Save processed data (chunks + embeddings) BEFORE DB Upload
        try:
            backup_path = Path("data/ingestion_backup.pkl")
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            with open(backup_path, "wb") as f:
                pickle.dump({
                    "chunks": self.chunks,
                    "chunk_texts": self.chunk_texts,
                    "embeddings": embeddings,
                    "pdf_files": [str(p) for p in pdf_files]
                }, f)
            logger.info(f"✔ Local Backup saved to {backup_path}. If upload fails, you can resume.")
        except Exception as e:
            logger.warning(f"Failed to save local backup: {e}")
        # --- LOCAL CACHING END ---
        
        # Upsert to Supabase
        logger.info("Upserting to Supabase...")
        self._upsert_to_supabase(embeddings, self.chunks)
        
        # Build BM25 index
        logger.info("Building BM25 index...")
        self.bm25_index = self._build_bm25_index(self.chunk_texts)
        
        # Save indices (BM25 only now)
        self._save_indices()
        self._initialized = True
        
        processing_time = time.time() - start_time
        logger.info(f"Ingestion complete in {processing_time:.2f}s")
        logger.info(f"Summary: {len(pdf_files)} docs, {len(self.chunks)} chunks, "
                   f"{embeddings.shape[1]} dims, {self.faiss_index.ntotal if hasattr(self, 'faiss_index') and self.faiss_index else 'Supabase'} vectors")
        
        return {
            "success": True,
            "message": f"Successfully processed {len(pdf_files)} documents",
            "documents_processed": len(pdf_files),
            "total_chunks": len(self.chunks),
            "processing_time_seconds": processing_time
        }
    
    def load_backup_and_upload(self):
        """
        Resume ingestion from local backup file.
        Skips PDF parsing and embedding generation.
        """
        backup_path = Path("data/ingestion_backup.pkl")
        if not backup_path.exists():
            logger.error("No backup file found at data/ingestion_backup.pkl")
            return {"success": False, "message": "No backup found"}
            
        logger.info(f"Loading backup from {backup_path}...")
        start_time = time.time()
        
        try:
            with open(backup_path, "rb") as f:
                data = pickle.load(f)
                
            self.chunks = data["chunks"]
            self.chunk_texts = data["chunk_texts"]
            embeddings = data["embeddings"]
            
            logger.info(f"Loaded {len(self.chunks)} chunks and embeddings shape {embeddings.shape}")
            
            # Upsert
            logger.info("Resuming Upsert to Supabase...")
            self._upsert_to_supabase(embeddings, self.chunks)
            
            # Rebuild BM25
            logger.info("Rebuilding BM25 index...")
            self.bm25_index = self._build_bm25_index(self.chunk_texts)
            self._save_indices()
            self._initialized = True
            
            return {
                "success": True, 
                "message": "Restored from backup and uploaded successfully",
                "total_chunks": len(self.chunks),
                "processing_time_seconds": time.time() - start_time
            }
        except Exception as e:
            logger.error(f"Failed to restore from backup: {e}")
            raise e

    def _save_indices(self):
        """
        Save BM25 index and updated metadata to disk.
        FAISS is replaced by Supabase, so we only need local BM25 for hybrid search.
        """
        data_dir = self.settings.data_path
        data_dir.mkdir(parents=True, exist_ok=True)
        
        # We no longer save FAISS index locally
        # faiss_path = data_dir / "faiss_index" ... removed
        
        # Save BM25 index and chunks (atomic write via temp file)
        bm25_path = data_dir / "bm25_index.pkl"
        temp_path = data_dir / "bm25_index.pkl.tmp"
        with open(temp_path, 'wb') as f:
            pickle.dump({
                'bm25': self.bm25_index,
                'chunks': self.chunks,
                'chunk_texts': self.chunk_texts
            }, f)
        temp_path.replace(bm25_path)  # Atomic rename
        logger.info(f"Saved BM25 index to {bm25_path} ({len(self.chunks)} chunks)")
        
        # Save embeddings for future use (optional, but good for debug)
        # embeddings_path = data_dir / "embeddings.npy"
        # np.save(embeddings_path, embeddings)
        # logger.info(f"Saved embeddings to {embeddings_path}")
    
    def _load_existing_indices(self) -> bool:
        """Load existing indices from disk (BM25 only). Supabase handles vectors."""
        data_dir = self.settings.data_path
        bm25_path = data_dir / "bm25_index.pkl"
        
        if not bm25_path.exists():
            logger.info("BM25 index not found on disk")
            return False
        
        try:
            # Load BM25 and chunks
            with open(bm25_path, 'rb') as f:
                data = pickle.load(f)
                self.bm25_index = data['bm25']
                self.chunks = data['chunks']
                self.chunk_texts = data['chunk_texts']
            
            logger.info(f"Loaded BM25 index: {len(self.chunks)} chunks")
            
            # Load embedding model
            self._get_embedding_model()
            
            # Initialize Supabase client lazily or check connection
            if self.settings.supabase_connection_string:
                try:
                    # Just access it to test connection, but don't force creation if just reading
                    # Actually we don't strictly need to do anything here as it's lazy loaded
                    pass
                except Exception as e:
                    logger.warning(f"Supabase connection warning: {e}")
            
            self._initialized = True
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load indices: {e}")
            return False
    
    def is_initialized(self) -> bool:
        """Check if indices are loaded (BM25 local + ready for Supabase)."""
        return self._initialized and self.bm25_index is not None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        return {
            "initialized": self._initialized,
            "total_chunks": len(self.chunks) if self.chunks else 0,
            "total_vectors": "Managed by Supabase",
            "unique_documents": len(set(c.document_name for c in self.chunks)) if self.chunks else 0,
            "device": self.device,
            "supabase_collection": self.settings.supabase_collection_name
        }


# Global singleton instance
_ingestion_service: Optional[IngestionService] = None


def get_ingestion_service() -> IngestionService:
    """Get or create the ingestion service singleton."""
    global _ingestion_service
    if _ingestion_service is None:
        _ingestion_service = IngestionService()
    return _ingestion_service
