"""
Response generation module using Google Gemini.
Handles generating responses from retrieved context.
"""

import time
import random
from typing import List, Dict, Any
import google.generativeai as genai

class ResponseGenerator:
    """Class for generating responses using Gemini LLM."""
    
    def __init__(self, api_key: str, model: str = "gemini-1.5-flash", max_retries: int = 3):
        """
        Initialize response generator.
        
        Args:
            api_key: Gemini API key
            model: Generation model to use
            max_retries: Maximum number of retries for rate limiting
        """
        self.api_key = api_key
        self.model_name = model
        self.max_retries = max_retries
        
        # Configure Gemini API
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(self.model_name)
        
        print(f"🤖 Initialized Gemini Generation Model: {self.model_name}")
    
    def generate(self, query: str, context_chunks: List[Dict[str, Any]]) -> str:
        """
        Generate response based on query and retrieved context.
        
        Args:
            query: User query
            context_chunks: Retrieved relevant chunks
            
        Returns:
            Generated response text
        """
        # Build context from retrieved chunks
        context = self._build_context(context_chunks)
        
        # Create prompt with context and query
        prompt = self._create_prompt(query, context)
        
        # Generate response with retry logic
        for attempt in range(self.max_retries):
            try:
                response = self.model.generate_content(prompt)
                return response.text
                
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
                    print(f"❌ Error generating response: {str(e)}")
                    raise
    
    def _build_context(self, chunks: List[Dict[str, Any]]) -> str:
        """
        Build formatted context from chunks.
        
        Args:
            chunks: List of chunk dictionaries
            
        Returns:
            Formatted context string
        """
        if not chunks:
            return "No relevant context found."
        
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            context_parts.append(
                f"[Context {i} - Source: {chunk['source']}, Chunk {chunk['chunk_index']}]\n"
                f"{chunk['text']}\n"
            )
        
        return "\n".join(context_parts)
    
    def _create_prompt(self, query: str, context: str) -> str:
        """
        Create prompt for the LLM.
        
        Args:
            query: User query
            context: Retrieved context
            
        Returns:
            Formatted prompt
        """
        prompt = f"""You are a helpful AI assistant answering questions based on provided context.

Context Information:
{context}

User Question: {query}

Instructions:
- Answer the question using ONLY the information provided in the context above
- Be accurate and specific
- If the context doesn't contain enough information to fully answer the question, acknowledge this
- Cite which context sources you used in your answer
- Keep your answer clear and concise

Answer:"""
        
        return prompt
