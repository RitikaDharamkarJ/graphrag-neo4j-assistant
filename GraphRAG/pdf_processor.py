"""
PDF processing module using Docling (simplified).
Handles PDF extraction and chunking without OCR.
"""

from pathlib import Path
from typing import List, Dict, Any
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions

class PDFProcessor:
    """Class for processing PDF documents and extracting chunks."""
    
    def __init__(self, min_chunk_length: int = 50):
        """
        Initialize PDF processor with simplified Docling configuration.
        
        Args:
            min_chunk_length: Minimum length of text chunks to extract
        """
        self.min_chunk_length = min_chunk_length
        
        # Configure Docling without OCR - simple text extraction only
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = False  # Disable OCR to avoid RapidOCR
        pipeline_options.do_table_structure = False  # Simplify - no table processing
        
        # Initialize document converter with simple configuration
        self.doc_converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options
                )
            }
        )
        
        print("📄 Initialized Docling PDF Processor (OCR disabled)")
    
    def extract_chunks(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        Extract text chunks from a PDF file using Docling.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of dictionaries containing text chunks and metadata
        """
        print(f"📄 Extracting content from: {pdf_path}")
        
        try:
            # Convert PDF document using simplified Docling
            result = self.doc_converter.convert(pdf_path)
            
            chunks = []
            chunk_id = 0
            
            # Method 1: Try using export_to_markdown for full text
            try:
                full_text = result.document.export_to_markdown()
                print(f"📝 Extracted {len(full_text)} characters via markdown export")
                
                # Split into paragraphs by double newlines
                paragraphs = full_text.split('\n\n')
                
                for para in paragraphs:
                    # Clean up the text
                    text = para.strip()
                    
                    # Only store meaningful chunks (longer than minimum length)
                    if text and len(text) > self.min_chunk_length:
                        chunks.append({
                            "id": f"{Path(pdf_path).stem}_chunk_{chunk_id}",
                            "text": text,
                            "source": Path(pdf_path).name,
                            "chunk_index": chunk_id,
                            "metadata": {
                                "length": len(text),
                                "extraction_method": "markdown"
                            }
                        })
                        chunk_id += 1
            
            except Exception as e:
                print(f"⚠️  Markdown export failed: {e}")
                print("📝 Trying alternative extraction method...")
                
                # Method 2: Fallback to iterate_items
                for element in result.document.iterate_items():
                    text = element.text.strip() if hasattr(element, 'text') else ""
                    
                    if text and len(text) > self.min_chunk_length:
                        chunks.append({
                            "id": f"{Path(pdf_path).stem}_chunk_{chunk_id}",
                            "text": text,
                            "source": Path(pdf_path).name,
                            "chunk_index": chunk_id,
                            "metadata": {
                                "element_type": element.__class__.__name__,
                                "length": len(text),
                                "extraction_method": "iterate_items"
                            }
                        })
                        chunk_id += 1
            
            print(f"✅ Extracted {len(chunks)} chunks from PDF")
            
            # Debug info if no chunks extracted
            if len(chunks) == 0:
                print("⚠️  WARNING: No chunks extracted!")
                print(f"   - Min chunk length: {self.min_chunk_length}")
                print(f"   - Try reducing min_chunk_length in config.py")
                print(f"   - Or check if PDF has extractable text")
            
            return chunks
            
        except Exception as e:
            print(f"❌ Error extracting PDF: {str(e)}")
            raise

    
    def get_chunk_statistics(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get statistics about extracted chunks.
        
        Args:
            chunks: List of chunk dictionaries
            
        Returns:
            Dictionary with statistics
        """
        if not chunks:
            return {"total_chunks": 0, "total_chars": 0, "avg_chunk_size": 0}
        
        total_chars = sum(chunk["metadata"]["length"] for chunk in chunks)
        avg_chunk_size = total_chars / len(chunks)
        
        return {
            "total_chunks": len(chunks),
            "total_chars": total_chars,
            "avg_chunk_size": round(avg_chunk_size, 2),
            "min_chunk_size": min(chunk["metadata"]["length"] for chunk in chunks),
            "max_chunk_size": max(chunk["metadata"]["length"] for chunk in chunks)
        }