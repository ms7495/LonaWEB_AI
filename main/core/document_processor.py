# core/document_processor.py - Simplified from your file_processing.py
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import pandas as pd

# Use your existing file extraction logic but simplified
try:
    import pdfplumber

    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False

try:
    import docx

    HAS_DOCX = True
except ImportError:
    HAS_DOCX = False

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Document processing - simplified from your file_processing.py"""

    def __init__(self):
        self.chunk_size = 500  # Simplified chunking
        self.chunk_overlap = 50

    def process_file(self, file_path: str, original_filename: str = None):
        """Process file - adapted from your load_and_chunk_file"""

        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = file_path.suffix.lower()
        source_name = original_filename or file_path.name

        # Extract text based on file type (use your existing logic)
        if ext == '.pdf':
            pages = self._extract_pdf_text(file_path)
        elif ext in ['.docx', '.doc']:
            pages = self._extract_docx_text(file_path)
        elif ext == '.txt':
            pages = self._extract_txt_text(file_path)
        elif ext in ['.xlsx', '.xls', '.csv']:
            pages = self._extract_excel_text(file_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")

        if not pages:
            raise ValueError("No content could be extracted")

        # Create chunks (simplified from your chunking logic)
        chunks = self._create_chunks(pages, source_name)

        # Create metadata
        metadata = {
            "filename": source_name,
            "file_type": ext,
            "total_pages": len(pages),
            "total_chunks": len(chunks),
            "processed_at": datetime.now().isoformat()
        }

        return chunks, metadata

    def _extract_pdf_text(self, file_path: Path):
        """Extract PDF text - simplified from your extract_text_from_pdf"""
        pages = []

        if not HAS_PDFPLUMBER:
            raise ImportError("pdfplumber not available for PDF processing")

        try:
            with pdfplumber.open(file_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    page_num = i + 1
                    text = page.extract_text() or ""

                    # Basic cleanup
                    text = re.sub(r'\s+', ' ', text).strip()

                    if text:
                        pages.append((page_num, text))
        except Exception as e:
            logger.error(f"PDF extraction failed: {e}")
            raise

        return pages

    def _extract_docx_text(self, file_path: Path):
        """Extract DOCX text - simplified from your extract_text_from_docx"""
        if not HAS_DOCX:
            raise ImportError("python-docx not available for DOCX processing")

        try:
            doc = docx.Document(file_path)
            text_parts = []

            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    text_parts.append(paragraph.text.strip())

            # Combine all text into single "page"
            full_text = '\n\n'.join(text_parts)
            return [(1, full_text)] if full_text else []

        except Exception as e:
            logger.error(f"DOCX extraction failed: {e}")
            raise

    def _extract_txt_text(self, file_path: Path):
        """Extract text file - from your extract_text_from_txt"""
        encodings = ['utf-8', 'utf-8-sig', 'utf-16', 'cp1252', 'iso-8859-1', 'latin1']

        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    text = f.read().strip()

                if text:
                    return [(1, text)]

            except UnicodeDecodeError:
                continue
            except Exception as e:
                logger.error(f"TXT extraction failed: {e}")
                break

        raise ValueError(f"Could not decode {file_path} with any encoding")

    def _extract_excel_text(self, file_path: Path):
        """Extract Excel/CSV text - simplified from your logic"""
        try:
            if file_path.suffix.lower() == '.csv':
                df = pd.read_csv(file_path)
            else:
                df = pd.read_excel(file_path, sheet_name=None)  # All sheets

                # If multiple sheets, combine them
                if isinstance(df, dict):
                    combined_text = ""
                    for sheet_name, sheet_df in df.items():
                        combined_text += f"\n\nSheet: {sheet_name}\n"
                        combined_text += sheet_df.to_string(index=False)
                    return [(1, combined_text)]
                else:
                    df = df  # Single sheet

            # Convert DataFrame to text
            text = f"Data from {file_path.name}:\n\n"
            text += df.to_string(index=False)

            return [(1, text)]

        except Exception as e:
            logger.error(f"Excel extraction failed: {e}")
            raise

    def _create_chunks(self, pages: List[Tuple], source_name: str):
        """Create chunks - simplified from your chunking logic"""
        chunks = []
        chunk_index = 0

        for page_num, text in pages:
            # Simple text splitting (remove complex chunking strategies)
            sentences = self._split_into_sentences(text)

            current_chunk = ""
            for sentence in sentences:
                # Check if adding sentence would exceed chunk size
                if len(current_chunk) + len(sentence) > self.chunk_size:
                    if current_chunk:
                        # Save current chunk
                        chunk_data = {
                            "text": current_chunk.strip(),
                            "original_filename": source_name,
                            "page_number": page_num,
                            "chunk_index": chunk_index
                        }
                        chunks.append(chunk_data)
                        chunk_index += 1

                        # Start new chunk with overlap
                        current_chunk = sentence
                    else:
                        # Sentence too long, force it
                        chunk_data = {
                            "text": sentence[:self.chunk_size],
                            "original_filename": source_name,
                            "page_number": page_num,
                            "chunk_index": chunk_index
                        }
                        chunks.append(chunk_data)
                        chunk_index += 1
                else:
                    current_chunk += " " + sentence if current_chunk else sentence

            # Add final chunk if any
            if current_chunk:
                chunk_data = {
                    "text": current_chunk.strip(),
                    "original_filename": source_name,
                    "page_number": page_num,
                    "chunk_index": chunk_index
                }
                chunks.append(chunk_data)
                chunk_index += 1

        return chunks

    def _split_into_sentences(self, text: str):
        """Simple sentence splitting"""
        # Basic sentence splitting (remove complex NLP)
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
