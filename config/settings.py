"""
Configuration settings for Airline Route Optimization - Phase 2
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ═══════════════════════════════════════════════════════════
# PROJECT PATHS
# ═══════════════════════════════════════════════════════════

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
KNOWLEDGE_BASE_DIR = PROJECT_ROOT / "knowledge_base"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"

# ═══════════════════════════════════════════════════════════
# API KEYS
# ═══════════════════════════════════════════════════════════

AVIATIONSTACK_API_KEY = os.getenv("AVIATIONSTACK_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# ═══════════════════════════════════════════════════════════
# RAG SETTINGS
# ═══════════════════════════════════════════════════════════

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K_RESULTS = 5

# ═══════════════════════════════════════════════════════════
# TRACKED AIRPORTS
# ═══════════════════════════════════════════════════════════

TRACKED_AIRPORTS = ["ATL", "LAX", "ORD", "DFW", "DEN", "JFK", "SFO"]

# ═══════════════════════════════════════════════════════════
# APPLICATION SETTINGS
# ═══════════════════════════════════════════════════════════

ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
DEBUG = os.getenv("DEBUG", "True").lower() == "true"

