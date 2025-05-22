"""
MCP Server implementation.
"""

from mcp.server.fastmcp import FastMCP

from mcp_server.config import SERVER_HOST, SERVER_NAME, SERVER_PORT, SERVER_TRANSPORT
from mcp_server.tools import (
    setup_document_tools,
    setup_event_tools,
    setup_general_knowledge_tools,
    setup_ruleset_tools,
)


def create_server():
    """
    Create and configure the MCP server.

    Returns:
        Configured FastMCP server instance.
    """
    # Create a single MCP server
    mcp = FastMCP(SERVER_NAME, host=SERVER_HOST, port=SERVER_PORT)

    # Add document search tools and resources
    setup_document_tools(mcp)

    # Add ruleset evaluation tools and resources
    setup_ruleset_tools(mcp)

    # Add event information tools and resources
    setup_event_tools(mcp)

    # Add general knowledge tools and resources
    setup_general_knowledge_tools(mcp)

    return mcp


def run_server():
    """
    Run the MCP server.
    """
    mcp = create_server()
    mcp.run(transport=SERVER_TRANSPORT)


if __name__ == "__main__":
    run_server()
