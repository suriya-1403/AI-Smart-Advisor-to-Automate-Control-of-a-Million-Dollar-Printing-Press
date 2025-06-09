"""
Configuration settings for the MCP Server.

This module centralizes all configuration parameters for the MCP (Model Context Protocol)
Server that controls HP PageWide printing press automation. It handles environment
variable loading, path configuration, and system-wide settings.

Author: Suriyakrishnan Sathish & Rujula More
Version: 1.0.0
Last Modified: 8th Jun 2025
"""

import os

from dotenv import load_dotenv

# Load environment variables from .env file
# This should be called early in the application lifecycle
load_dotenv()

# Server configuration
# These settings control the MCP server instance behavior
SERVER_NAME = "PrintSystem"  # Identifier for this MCP server instance
SERVER_HOST = os.getenv("HOST", "0.0.0.0")  # Host address (0.0.0.0 for all interfaces)
SERVER_PORT = int(os.getenv("PORT", 8050))  # Port for MCP server (default: 8050)
SERVER_TRANSPORT = "sse"  # Server-Sent Events for real-time communication
GROQ_API = os.getenv("groq")  # API key for Groq LLM services (required)

# Path configuration
# Dynamic path resolution for cross-platform compatibility
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "mcp_server/data")
DOCUMENTS_DIR = os.path.join(DATA_DIR, "documents")
RULESETS_DIR = os.path.join(DATA_DIR, "rulesets")
PDF_DIR = os.path.join(DATA_DIR, "PDF")

# LLM configuration
LLM_MODEL = "meta-llama/llama-4-maverick-17b-128e-instruct"

# Ruleset files
# Maps logical ruleset types to their corresponding HTML files
# Used by configuration agents to load appropriate rule tables
RULESET_FILES = {
    "media treatment class": "mediatreatment_class.html",
    "ink coverage class": "Ink_coverage-converted.html",
    "media weight class": "mediaweight_class.html",
}
