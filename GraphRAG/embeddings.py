"""
Embedding generation module using Google Gemini.
Handles text-to-vector conversion with retry logic.
"""

import time
import random
from typing import List
import google.generativeai as genai

class EmbeddingGenerator:
    """Class for generating embeddings using Gemini API."""
    
    def __init__(self, api_key: str, model: str = "models/text-embedding-004", max_retries: int = 3):
        """
        Initialize embedding generator.
        
        Args:
            api_key: Gemini API key
            model: Embedding model to use
            max_retries: Maximum number of retries for rate limiting
        """
        self.api_key = api_key
        self.model = model
        self.max_retries = max_retries
        
        # Configure Gemini API
        genai.configure(api_key=self.api_key)
        
        print(f"🤖 Initialized Gemini Embedding Model: {self.model}")
    
    def generate(self, text: str, task_type: str = "retrieval_document") -> List[float]:
        """
        Generate embedding vector for text with retry logic.
        
        Args:
            text: Input text to embed
            task_type: Type of task (retrieval_document or retrieval_query)
            
        Returns:
            Embedding vector as list of floats
        """
        for attempt in range(self.max_retries):
            try:
                # Generate embedding using Gemini API
                result = genai.embed_content(
                    model=self.model,
                    content=text,
                    task_type=task_type
                )
                return result['embedding']
                
            except Exception as e:
                # Handle rate limiting with exponential backoff
                if "429" in str(e) or "quota" in str(e).lower():
                    if attempt < self.max_retries - 1:
                        wait_time = (2 ** attempt) + random.uniform(0, 1)
                        print(f"⏳ Rate limit hit. Waiting {wait_time:.2f}s before retry...")
                        time.sleep(wait_time)
                    else:
                        raise Exception("❌ Rate limit exceeded. Please wait and try again.")
                else:
                    print(f"❌ Error generating embedding: {str(e)}")
                    raise
    
    def generate_query_embedding(self, query: str) -> List[float]:
        """
        Generate embedding for a search query.
        
        Args:
            query: Query text
            
        Returns:
            Embedding vector
        """
        return self.generate(query, task_type="retrieval_query")
    
    def generate_document_embedding(self, document: str) -> List[float]:
        """
        Generate embedding for a document.
        
        Args:
            document: Document text
            
        Returns:
            Embedding vector
        """
        return self.generate(document, task_type="retrieval_document")
