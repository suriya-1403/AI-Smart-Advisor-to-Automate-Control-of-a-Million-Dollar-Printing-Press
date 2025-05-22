"""
Tools package for the MCP Server.
"""

from mcp_server.tools.document_tools import setup_document_tools
from mcp_server.tools.eventRetrival_tools import setup_event_tools
from mcp_server.tools.generalKnowledge_tools import setup_general_knowledge_tools
from mcp_server.tools.ruleset_tools import setup_ruleset_tools

__all__ = [
    "setup_document_tools",
    "setup_ruleset_tools",
    "setup_event_tools",
    "setup_general_knowledge_tools",
]
