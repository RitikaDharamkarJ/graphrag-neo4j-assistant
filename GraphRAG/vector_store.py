"""
Neo4j vector database module.
Handles storing and retrieving embeddings from Neo4j.
"""

from typing import List, Dict, Any
from neo4j import GraphDatabase

class VectorStore:
    """Class for managing vector storage in Neo4j."""
    
    def __init__(self, uri: str, user: str, password: str, database: str = "graphragdb"):
        """
        Initialize Neo4j vector store.
        
        Args:
            uri: Neo4j connection URI
            user: Neo4j username
            password: Neo4j password
            database: Database name
        """
        self.uri = uri
        self.user = user
        self.password = password
        self.database = database
        
        # Initialize Neo4j driver
        self.driver = GraphDatabase.driver(
            self.uri,
            auth=(self.user, self.password)
        )
        
        print(f"🗄️  Connected to Neo4j database: {self.database}")
        
        # Setup database schema
        self._setup_schema()
    
    def _setup_schema(self):
        """Create Neo4j schema with constraints and vector index."""
        with self.driver.session(database=self.database) as session:
            # Create unique constraint for chunk IDs
            session.run("""
                CREATE CONSTRAINT chunk_id IF NOT EXISTS
                FOR (c:Chunk) REQUIRE c.id IS UNIQUE
            """)
            
            # Create vector index for similarity search
            # Note: Requires Neo4j 5.11+ with vector search support
            try:
                session.run("""
                    CREATE VECTOR INDEX chunk_embedding IF NOT EXISTS
                    FOR (c:Chunk) ON (c.embedding)
                    OPTIONS {indexConfig: {
                        `vector.dimensions`: 768,
                        `vector.similarity_function`: 'cosine'
                    }}
                """)
                print("✅ Vector index created/verified")
            except Exception as e:
                print(f"⚠️  Vector index note: {e}")
                print("   Manual similarity calculation will be used")
    
    def store_chunk(self, chunk_id: str, text: str, embedding: List[float], 
                   source: str, chunk_index: int, metadata: Dict[str, Any]):
        """
        Store a single chunk with embedding in Neo4j.
        
        Args:
            chunk_id: Unique chunk identifier
            text: Chunk text content
            embedding: Embedding vector
            source: Source document name
            chunk_index: Index of chunk in document
            metadata: Additional metadata
        """
        with self.driver.session(database=self.database) as session:
            session.run("""
                MERGE (c:Chunk {id: $id})
                SET c.text = $text,
                    c.source = $source,
                    c.chunk_index = $chunk_index,
                    c.embedding = $embedding,
                    c.metadata = $metadata
            """, {
                "id": chunk_id,
                "text": text,
                "source": source,
                "chunk_index": chunk_index,
                "embedding": embedding,
                "metadata": str(metadata)
            })
    
    def store_chunks_batch(self, chunks: List[Dict[str, Any]], embeddings: List[List[float]]):
        """
        Store multiple chunks with embeddings in batch.
        
        Args:
            chunks: List of chunk dictionaries
            embeddings: List of embedding vectors
        """
        print(f"💾 Storing {len(chunks)} chunks in Neo4j...")
        
        for chunk, embedding in zip(chunks, embeddings):
            self.store_chunk(
                chunk_id=chunk["id"],
                text=chunk["text"],
                embedding=embedding,
                source=chunk["source"],
                chunk_index=chunk["chunk_index"],
                metadata=chunk["metadata"]
            )
            print(f"   ✓ Stored: {chunk['id']}")
        
        print(f"✅ All {len(chunks)} chunks stored successfully!")
    
    def search_similar(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar chunks using vector similarity.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of top results to return
            
        Returns:
            List of similar chunks with similarity scores
        """
        with self.driver.session(database=self.database) as session:
            # Perform cosine similarity search using dot product
            result = session.run("""
                MATCH (c:Chunk)
                WITH c, 
                     reduce(dot = 0.0, i IN range(0, size(c.embedding)-1) | 
                        dot + c.embedding[i] * $query_embedding[i]) AS similarity
                RETURN c.id AS id, 
                       c.text AS text, 
                       c.source AS source,
                       c.chunk_index AS chunk_index,
                       similarity
                ORDER BY similarity DESC
                LIMIT $top_k
            """, {
                "query_embedding": query_embedding,
                "top_k": top_k
            })
            
            chunks = []
            for record in result:
                chunks.append({
                    "id": record["id"],
                    "text": record["text"],
                    "source": record["source"],
                    "chunk_index": record["chunk_index"],
                    "similarity": float(record["similarity"])
                })
            
            return chunks
    
    def get_all_sources(self) -> List[str]:
        """
        Get list of all document sources in the database.
        
        Returns:
            List of unique source names
        """
        with self.driver.session(database=self.database) as session:
            result = session.run("""
                MATCH (c:Chunk)
                RETURN DISTINCT c.source AS source
                ORDER BY source
            """)
            return [record["source"] for record in result]
    
    def get_chunk_count(self) -> int:
        """
        Get total number of chunks in database.
        
        Returns:
            Total chunk count
        """
        with self.driver.session(database=self.database) as session:
            result = session.run("MATCH (c:Chunk) RETURN count(c) AS count")
            return result.single()["count"]
    
    def delete_by_source(self, source: str):
        """
        Delete all chunks from a specific source.
        
        Args:
            source: Source document name
        """
        with self.driver.session(database=self.database) as session:
            result = session.run("""
                MATCH (c:Chunk {source: $source})
                DELETE c
                RETURN count(c) AS deleted_count
            """, {"source": source})
            deleted = result.single()["deleted_count"]
            print(f"🗑️  Deleted {deleted} chunks from source: {source}")
    
    def close(self):
        """Close Neo4j database connection."""
        self.driver.close()
        print("🔌 Neo4j connection closed")
