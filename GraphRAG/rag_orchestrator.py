"""
RAG Orchestrator - Main module that coordinates all RAG components.
"""

from typing import List, Dict, Any
from config import Config
from pdf_processor import PDFProcessor
from embeddings import EmbeddingGenerator
from vector_store import VectorStore
from response_generator import ResponseGenerator

class RAGOrchestrator:
    """Main orchestrator class that coordinates all RAG components."""
    
    def __init__(self, config: Config = None):
        """
        Initialize RAG orchestrator with all components.
        
        Args:
            config: Configuration object (creates default if None)
        """
        # Load configuration
        self.config = config if config else Config()
        
        print("\n" + "="*60)
        print("🚀 Initializing RAG System Components")
        print("="*60)
        
        # Initialize all components
        self.pdf_processor = PDFProcessor(
            min_chunk_length=self.config.chunk_min_length
        )
        
        self.embedding_generator = EmbeddingGenerator(
            api_key=self.config.gemini_api_key,
            model=self.config.embedding_model,
            max_retries=self.config.max_retries
        )
        
        self.vector_store = VectorStore(
            uri=self.config.neo4j_uri,
            user=self.config.neo4j_user,
            password=self.config.neo4j_password,
            database=self.config.neo4j_database
        )
        
        self.response_generator = ResponseGenerator(
            api_key=self.config.gemini_api_key,
            model=self.config.generation_model,
            max_retries=self.config.max_retries
        )
        
        print("="*60)
        print("✅ RAG System Initialized Successfully!")
        print("="*60 + "\n")
    
    def process_and_store_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """
        Complete pipeline: Extract PDF, generate embeddings, store in Neo4j.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Dictionary with processing statistics
        """
        print(f"\n📚 Processing PDF: {pdf_path}")
        print("-" * 60)
        
        # Step 1: Extract chunks from PDF
        chunks = self.pdf_processor.extract_chunks(pdf_path)
        
        if not chunks:
            return {
                "success": False,
                "message": "No chunks extracted from PDF",
                "chunks_processed": 0
            }
        
        # Get statistics
        # stats = self.pdf_processor.get_chunk_statistics(chunks)
        # print(f"📊 Statistics: {stats}")
        
        # Step 2: Generate embeddings for all chunks
        print(f"\n🔄 Generating embeddings for {len(chunks)} chunks...")
        embeddings = []
        for i, chunk in enumerate(chunks, 1):
            embedding = self.embedding_generator.generate_document_embedding(chunk["text"])
            embeddings.append(embedding)
            if i % 10 == 0:
                print(f"   Progress: {i}/{len(chunks)} embeddings generated")
        
        # Step 3: Store chunks with embeddings in Neo4j
        print(f"\n💾 Storing chunks in Neo4j...")
        self.vector_store.store_chunks_batch(chunks, embeddings)
        
        print("-" * 60)
        print(f"✅ PDF Processing Complete!")
        
        return {
            "success": True,
            "message": f"Successfully processed {len(chunks)} chunks",
            "chunks_processed": len(chunks),
            "statistics": stats
        }
    
    def query(self, question: str, top_k: int = None) -> Dict[str, Any]:
        """
        Query the RAG system: retrieve relevant chunks and generate response.
        
        Args:
            question: User question
            top_k: Number of chunks to retrieve (uses config default if None)
            
        Returns:
            Dictionary with question, answer, and source chunks
        """
        if top_k is None:
            top_k = self.config.top_k_results
        
        print(f"\n🔍 Processing Query: {question}")
        print("-" * 60)
        
        # Step 1: Generate query embedding
        print("🔄 Generating query embedding...")
        query_embedding = self.embedding_generator.generate_query_embedding(question)
        
        # Step 2: Retrieve relevant chunks from Neo4j
        print(f"🔍 Searching for top {top_k} relevant chunks...")
        relevant_chunks = self.vector_store.search_similar(query_embedding, top_k)
        
        if not relevant_chunks:
            print("⚠️  No relevant chunks found")
            return {
                "question": question,
                "answer": "I couldn't find any relevant information in the knowledge base to answer your question.",
                "sources": []
            }
        
        print(f"✅ Found {len(relevant_chunks)} relevant chunks")
        
        # Step 3: Generate response using LLM
        print("🤖 Generating response...")
        answer = self.response_generator.generate(question, relevant_chunks)
        
        print("-" * 60)
        print("✅ Query Processing Complete!")
        
        return {
            "question": question,
            "answer": answer,
            "sources": relevant_chunks
        }
    
    def get_database_info(self) -> Dict[str, Any]:
        """
        Get information about the current database state.
        
        Returns:
            Dictionary with database statistics
        """
        chunk_count = self.vector_store.get_chunk_count()
        sources = self.vector_store.get_all_sources()
        
        return {
            "total_chunks": chunk_count,
            "total_documents": len(sources),
            "documents": sources
        }
    
    def delete_document(self, source: str):
        """
        Delete all chunks from a specific document.
        
        Args:
            source: Source document name
        """
        self.vector_store.delete_by_source(source)
    
    def close(self):
        """Close all connections."""
        self.vector_store.close()
        print("👋 RAG System shut down successfully")
