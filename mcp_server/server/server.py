"""
MCP (Model Context Protocol) Server implementation.

This module implements the core MCP server that provides AI tool capabilities
for HP PageWide printing press automation. It coordinates between different
tool modules and manages client sessions through Server-Sent Events (SSE).

The server supports four main categories of tools:
- Document search tools for finding printing event records
- Ruleset evaluation tools for configuration optimization
- Event information tools for detailed event analysis
- General knowledge tools for educational responses

Author: Suriyakrishnan Sathish & Rujula More
Version: 1.0.0
Last Modified: 2025
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
    Create and configure the complete MCP server instance.

    This function orchestrates the entire server creation process including:
    1. Server instance creation
    2. Tool module registration
    3. Error handling setup
    4. Middleware configuration
    5. Lifecycle hook registration

    Returns:
        FastMCP: Fully configured MCP server ready for startup

    Raises:
        EnvironmentError: If server configuration is invalid
        ImportError: If required dependencies are missing

    Example:
        >>> server = create_server()
        >>> server.run()
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
