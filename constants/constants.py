
"""
Constants used throughout the RAG Chatbot application.
"""

# Vector database settings
CHROMA_PATH = "./chromadb_store"
COLLECTION_NAME = "Test-Print-Report"

# Model settings
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_LLM = "llama3.2"
AVAILABLE_MODELS = ["llama3.2", "granite3-dense", "mistral","deepseek-r1"]

# Query settings
MAX_SEARCH_RESULTS = 20
MAX_DISPLAY_RESULTS = 10
MAX_RAG_CONTEXT = 3

# UI settings
APP_TITLE = "RAG Chatbot"
APP_DESCRIPTION = "Upload files and perform vector-based retrieval and generation using AI."

# Search patterns for query classification
SEARCH_PATTERNS = [
    r'find\s+.*(document|file|print job|event)',
    r'search\s+for',
    r'looking\s+for',
    r'(show|get|display|retrieve)\s+.*(document|file|print job|event)',
    r'with\s+(heavy|medium|light|coated|uncoated|silk|matte|gloss|\d+\s*gsm)',
    r'(filter|where|that has)',
]

# Question patterns for query classification
QUESTION_PATTERNS = [
    r'^(what|how|why|when|where|who|can you|could you|tell me)',
    r'\?$',
    r'(explain|describe|elaborate|detail|info about)',
    r'(configuration|setting|parameter|specification|property)',
]

# Stop words for keyword extraction
STOP_WORDS = ['with', 'and', 'the', 'for', 'find', 'documents']