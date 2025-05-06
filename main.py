import os
import sys

from mcp_server.server.server import create_server

# Add the current directory to the Python path so imports work
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
# Get the server instance
mcp_server = create_server()

# This variable name is important for MCP dev to find it
mcp = mcp_server

if __name__ == "__main__":
    mcp.run(transport="sse")
