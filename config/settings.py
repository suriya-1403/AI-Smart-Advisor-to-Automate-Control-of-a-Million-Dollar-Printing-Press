"""
Configuration settings for the RAG Chatbot application.
"""

import os
from constants.constants import (
    CHROMA_PATH,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    DEFAULT_LLM,
    AVAILABLE_MODELS,
    MAX_SEARCH_RESULTS,
    MAX_DISPLAY_RESULTS,
    MAX_RAG_CONTEXT,
    APP_TITLE,
    APP_DESCRIPTION
)

# Environment-specific overrides
ENV_CHROMA_PATH = os.environ.get("CHROMA_PATH", CHROMA_PATH)
ENV_COLLECTION_NAME = os.environ.get("COLLECTION_NAME", COLLECTION_NAME)
ENV_EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", EMBEDDING_MODEL)
ENV_DEFAULT_LLM = os.environ.get("DEFAULT_LLM", DEFAULT_LLM)
ENV_MAX_SEARCH_RESULTS = int(os.environ.get("MAX_SEARCH_RESULTS", MAX_SEARCH_RESULTS))
ENV_MAX_DISPLAY_RESULTS = int(os.environ.get("MAX_DISPLAY_RESULTS", MAX_DISPLAY_RESULTS))
ENV_MAX_RAG_CONTEXT = int(os.environ.get("MAX_RAG_CONTEXT", MAX_RAG_CONTEXT))

# Pytorch CUDA settings
DISABLE_CUDA = os.environ.get("DISABLE_CUDA", "1") == "1"