"""
Configuration package for the MCP Server.
"""

from mcp_server.config.settings import (
    DATA_DIR,
    DOCUMENTS_DIR,
    GROQ_API,
    LLM_MODEL,
    PDF_DIR,
    RULESET_FILES,
    RULESETS_DIR,
    SERVER_HOST,
    SERVER_NAME,
    SERVER_PORT,
    SERVER_TRANSPORT,
)

__all__ = [
    "SERVER_NAME",
    "SERVER_HOST",
    "SERVER_PORT",
    "SERVER_TRANSPORT",
    "DATA_DIR",
    "DOCUMENTS_DIR",
    "PDF_DIR",
    "RULESETS_DIR",
    "LLM_MODEL",
    "RULESET_FILES",
    "GROQ_API",
]
