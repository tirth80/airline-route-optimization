"""
ChromaDB Vector Store for RAG System

Stores document chunks as embeddings for semantic search.
"""

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List, Dict
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FlightKnowledgeStore:
    """
    Vector store for flight delay knowledge base.
    """
    
    def __init__(
        self,
        persist_dir: str = "./chroma_db",
        collection_name: str = "flight_knowledge",
        embedding_model: str = "all-MiniLM-L6-v2"
    ):
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        
        # Create directory if not exists
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize embedding model
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embedder = SentenceTransformer(embedding_model)
        
        # Initialize ChromaDB
        logger.info(f"Initializing ChromaDB at {persist_dir}")
        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "Flight delay knowledge base"}
        )
        
        logger.info(f"Collection '{collection_name}' ready with {self.collection.count()} documents")
    
    def add_chunks(self, chunks: List[Dict]):
        """
        Add chunks to the vector store.
        """
        if not chunks:
            logger.warning("No chunks to add")
            return
        
        logger.info(f"Adding {len(chunks)} chunks to vector store...")
        
        # Extract data
        ids = [chunk["id"] for chunk in chunks]
        documents = [chunk["content"] for chunk in chunks]
        metadatas = [chunk["metadata"] for chunk in chunks]
        
        # Generate embeddings
        logger.info("Generating embeddings...")
        embeddings = self.embedder.encode(documents).tolist()
        
        # Add to ChromaDB
        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )
        
        logger.info(f"Successfully added {len(chunks)} chunks")
        logger.info(f"Total documents in store: {self.collection.count()}")
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Search for relevant chunks.
        """
        # Generate query embedding
        query_embedding = self.embedder.encode(query).tolist()
        
        # Search
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        # Format results
        formatted = []
        for i in range(len(results["ids"][0])):
            formatted.append({
                "id": results["ids"][0][i],
                "content": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i] if "distances" in results else None
            })
        
        return formatted
    
    def clear(self):
        """
        Clear all documents from the collection.
        """
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "Flight delay knowledge base"}
        )
        logger.info("Collection cleared")
    
    def count(self) -> int:
        """
        Return number of documents in store.
        """
        return self.collection.count()


if __name__ == "__main__":
    from rag.chunking.text_chunker import TextChunker
    
    print("\n🗄️ Testing Vector Store...\n")
    
    # Create chunks
    chunker = TextChunker()
    chunks = chunker.chunk_directory("knowledge_base/historical")
    
    # Initialize store
    store = FlightKnowledgeStore()
    
    # Clear and add chunks
    store.clear()
    store.add_chunks(chunks)
    
    # Test search
    print("\n🔍 Testing search...")
    query = "Which airline has the best on-time performance?"
    results = store.search(query, top_k=3)
    
    print(f"\nQuery: {query}")
    print(f"Found {len(results)} results:\n")
    
    for i, result in enumerate(results, 1):
        print(f"{i}. {result['metadata']['section']}")
        print(f"   Source: {result['metadata']['source_file']}")
        print(f"   Preview: {result['content'][:100]}...")
        print()

