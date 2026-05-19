import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Paths
    PDF_PATH = "data/raw/Union_Budget_Analysis-2026-27.pdf"
    CHROMA_PATH = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db")
    
    # Models
    #os.environ['HUGGINGFACE_API_KEY'] = os.getenv('HUGGINGFACE_API_KEY')
    LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    
    # Chunking
    CHUNK_SIZE = 1024
    CHUNK_OVERLAP = 200
    
    # Retrieval
    TOP_K = 5
    SIMILARITY_THRESHOLD = 0.7
    
    # Budget-specific keywords for enhanced retrieval
    BUDGET_KEYWORDS = [
        "fiscal deficit", "capex", "receipts", "expenditure", 
        "gst", "income tax", "customs duty", "allocation",
        "subsidy", "disinvestment", "fiscal consolidation"
    ]