"""
Configuration settings for the MCP Server.
"""

import os

# Server configuration
SERVER_NAME = "PrintSystem"
SERVER_HOST = os.getenv("HOST", "0.0.0.0")
SERVER_PORT = int(os.getenv("PORT", 8050))
SERVER_TRANSPORT = "sse"

# Path configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "mcp_server/data")
DOCUMENTS_DIR = os.path.join(DATA_DIR, "documents")
RULESETS_DIR = os.path.join(DATA_DIR, "rulesets")

# LLM configuration
LLM_MODEL = "llama3.2"

# Ruleset files
RULESET_FILES = {
    "media treatment class": "mediatreatment_class.html",
    "ink coverage class": "Ink_coverage-converted.html",
    "media weight class": "mediaweight_class.html",
}
