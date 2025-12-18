"""
Text Chunking Pipeline for RAG System

Splits documents into smaller, meaningful chunks for embedding and retrieval.
"""

import re
from pathlib import Path
from typing import List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextChunker:
    """
    Chunks documents by headers and sections for better retrieval.
    """
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk_markdown(self, content: str, source_file: str) -> List[Dict]:
        """
        Chunk markdown content by headers.
        """
        chunks = []
        sections = re.split(r'\n(?=##)', content)
        
        for i, section in enumerate(sections):
            if not section.strip():
                continue
            
            lines = section.strip().split('\n')
            header = lines[0].replace('#', '').strip() if lines else "Introduction"
            body = '\n'.join(lines[1:]).strip() if len(lines) > 1 else ""
            
            chunk_content = f"{header}\n\n{body}" if body else header
            
            if len(chunk_content) < 50:
                continue
            
            chunk = {
                "id": f"{Path(source_file).stem}_{i}",
                "content": chunk_content,
                "metadata": {
                    "source_file": source_file,
                    "section": header,
                    "chunk_index": i,
                    "char_count": len(chunk_content)
                }
            }
            chunks.append(chunk)
        
        return chunks
    
    def chunk_directory(self, directory: str) -> List[Dict]:
        """
        Chunk all markdown files in a directory.
        """
        all_chunks = []
        dir_path = Path(directory)
        
        if not dir_path.exists():
            logger.error(f"Directory not found: {directory}")
            return []
        
        md_files = list(dir_path.glob("*.md"))
        logger.info(f"Found {len(md_files)} markdown files in {directory}")
        
        for md_file in sorted(md_files):
            logger.info(f"Processing: {md_file.name}")
            content = md_file.read_text(encoding='utf-8')
            chunks = self.chunk_markdown(content, str(md_file))
            all_chunks.extend(chunks)
            logger.info(f"  Created {len(chunks)} chunks")
        
        logger.info(f"Total chunks created: {len(all_chunks)}")
        return all_chunks


if __name__ == "__main__":
    chunker = TextChunker()
    print("\n Testing Text Chunker...\n")
    chunks = chunker.chunk_directory("knowledge_base/historical")
    print(f"\n Total chunks: {len(chunks)}")
