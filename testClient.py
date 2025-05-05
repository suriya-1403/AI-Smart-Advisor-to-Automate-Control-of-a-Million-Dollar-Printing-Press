import asyncio

from mcp import ClientSession
from mcp.client.sse import sse_client


async def main():
    # Connect to MCP server via SSE
    async with sse_client("http://localhost:8050/sse") as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            # Initialize session
            await session.initialize()

            # List available tools
            tools_result = await session.list_tools()
            print("🔧 Available tools:")
            for tool in tools_result.tools:
                print(f"  - {tool.name}: {tool.description}")
            print(
                "Query: Find documents with heavy ink coverage on coated media around 220 gsm"
            )
            # Call your tool: find_document
            result = await session.call_tool(
                "find_document",
                arguments={
                    "query": "Find documents with heavy ink coverage on coated media around 220 gsm"
                },
            )
            print("\n📄 Tool Response:")
            print(result.content[0].text)


if __name__ == "__main__":
    asyncio.run(main())
