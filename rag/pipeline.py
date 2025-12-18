"""
RAG Pipeline for Flight Delay AI Assistant

Connects retrieval, context building, and LLM response generation.
"""

import os
from typing import Dict, List
from dotenv import load_dotenv
import logging

from rag.chunking.text_chunker import TextChunker
from rag.vectorstore.chroma_store import FlightKnowledgeStore

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FlightDelayRAG:
    def __init__(self, rebuild_index: bool = False):
        logger.info("Initializing Flight Delay RAG...")
        
        self.chunker = TextChunker()
        self.store = FlightKnowledgeStore()
        
        if rebuild_index or self.store.count() == 0:
            self._build_index()
        
        logger.info(f"RAG ready with {self.store.count()} documents")
    
    def _build_index(self):
        logger.info("Building vector index...")
        
        all_chunks = []
        
        historical_chunks = self.chunker.chunk_directory("knowledge_base/historical")
        all_chunks.extend(historical_chunks)
        
        current_chunks = self.chunker.chunk_directory("knowledge_base/current")
        all_chunks.extend(current_chunks)
        
        trend_chunks = self.chunker.chunk_directory("knowledge_base/trends")
        all_chunks.extend(trend_chunks)
        
        self.store.clear()
        self.store.add_chunks(all_chunks)
        
        logger.info(f"Index built with {len(all_chunks)} total chunks")
    
    def refresh_index(self):
        self._build_index()
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        return self.store.search(query, top_k=top_k)
    
    def build_context(self, results: List[Dict]) -> str:
        context_parts = []
        for i, result in enumerate(results, 1):
            section = result["metadata"]["section"]
            content = result["content"]
            context_parts.append(f"[Source {i}: {section}]\n{content}")
        return "\n\n".join(context_parts)
    
    def generate_response(self, query: str, context: str) -> str:
        groq_key = os.getenv("GROQ_API_KEY")
        
        if not groq_key or groq_key == "your_groq_key_here":
            return self._generate_simple_response(query, context)
        
        try:
            from groq import Groq
            client = Groq(api_key=groq_key)
            
            prompt = f"""You are a helpful AI assistant for flight delay analysis.
Use the following context to answer the user's question.
Be specific and include numbers when available.
If there is current/today's data, prioritize that over historical data.

CONTEXT:
{context}

USER QUESTION: {query}

Provide a helpful, concise answer based on the context above."""

            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.7
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"LLM error: {e}")
            return self._generate_simple_response(query, context)
    
    def _generate_simple_response(self, query: str, context: str) -> str:
        return f"""Based on the flight delay knowledge base:

{context}

---
Note: For AI-powered responses, add your GROQ_API_KEY to .env file."""
    
    def ask(self, query: str, top_k: int = 5) -> Dict:
        logger.info(f"Processing query: {query}")
        
        results = self.retrieve(query, top_k=top_k)
        context = self.build_context(results)
        answer = self.generate_response(query, context)
        sources = [r["metadata"]["source_file"] for r in results]
        
        return {
            "query": query,
            "answer": answer,
            "sources": list(set(sources)),
            "num_sources": len(results)
        }


if __name__ == "__main__":
    print("\nTesting RAG Pipeline with Groq...\n")
    rag = FlightDelayRAG()
    result = rag.ask("What is the current flight status at JFK?")
    print(f"Answer: {result['answer']}")
