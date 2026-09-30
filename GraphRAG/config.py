"""
Configuration module for RAG system.
Handles loading and validating environment variables.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Configuration class for RAG system."""
    
    def __init__(self):
        """Initialize and validate configuration."""
        # Neo4j Configuration
        self.neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.neo4j_user = os.getenv("NEO4J_USER", "neo4j")
        self.neo4j_password = os.getenv("NEO4J_PASSWORD")
        self.neo4j_database = os.getenv("NEO4J_DATABASE", "graphragdb")
        self.neo4j_instance = os.getenv("NEO4J_INSTANCE", "My_instance")
        
        # Gemini API Configuration
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        
        # Model Configuration
        self.embedding_model = "models/text-embedding-004"
        self.generation_model = "gemini-2.0-flash"  # Latest Gemini 2.0 Flash model
        
        # RAG Configuration
        self.chunk_min_length = 20  # Minimum chunk length in characters (reduced from 50)
        self.top_k_results = 5  # Default number of results to retrieve
        self.max_retries = 3  # Max retries for API calls
        
        # Validate required settings
        self._validate()
    
    def _validate(self):
        """Validate that required configuration values are set."""
        if not self.neo4j_password:
            raise ValueError("NEO4J_PASSWORD not found in environment variables")
        if not self.gemini_api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
    
    def display_config(self):
        """Display current configuration (without sensitive data)."""
        print("="*60)
        print("RAG System Configuration")
        print("="*60)
        print(f"Neo4j URI: {self.neo4j_uri}")
        print(f"Neo4j User: {self.neo4j_user}")
        print(f"Neo4j Database: {self.neo4j_database}")
        print(f"Neo4j Instance: {self.neo4j_instance}")
        print(f"Embedding Model: {self.embedding_model}")
        print(f"Generation Model: {self.generation_model}")
        print(f"Chunk Min Length: {self.chunk_min_length}")
        print(f"Top K Results: {self.top_k_results}")
        print("="*60)