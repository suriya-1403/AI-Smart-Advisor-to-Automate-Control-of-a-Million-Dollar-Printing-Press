"""
Configuration settings for the MCP Server.
"""

import os

from dotenv import load_dotenv

load_dotenv()

# Server configuration
SERVER_NAME = "PrintSystem"
SERVER_HOST = os.getenv("HOST", "0.0.0.0")
SERVER_PORT = int(os.getenv("PORT", 8050))
SERVER_TRANSPORT = "sse"
GROQ_API = os.getenv("groq")

# Path configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "mcp_server/data")
DOCUMENTS_DIR = os.path.join(DATA_DIR, "documents")
RULESETS_DIR = os.path.join(DATA_DIR, "rulesets")
PDF_DIR = os.path.join(DATA_DIR, "PDF")

# LLM configuration
LLM_MODEL = "meta-llama/llama-4-maverick-17b-128e-instruct"

# Ruleset files
RULESET_FILES = {
    "media treatment class": "mediatreatment_class.html",
    "ink coverage class": "Ink_coverage-converted.html",
    "media weight class": "mediaweight_class.html",
}
