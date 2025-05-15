"""
Configuration package for the MCP Server.
"""

from mcp_server.config.settings import (
    DATA_DIR,
    DOCUMENTS_DIR,
    LLM_MODEL,
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
    "RULESETS_DIR",
    "LLM_MODEL",
    "RULESET_FILES",
]
