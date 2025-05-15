"""
Application entry point.
"""

import asyncio

from langchain_core.tracers import ConsoleCallbackHandler
from langsmith import Client
from mcp import ClientSession
from mcp.client.sse import sse_client

from mcp_server.config import SERVER_HOST, SERVER_PORT
from mcp_server.core import create_router
from mcp_server.workflows import create_document_workflow, create_ruleset_workflow

# Initialize client
client = Client()

# Initialize workflows with callbacks
router = create_router(callbacks=[ConsoleCallbackHandler()])
doc_workflow = create_document_workflow(callbacks=[ConsoleCallbackHandler()])
ruleset_workflow = create_ruleset_workflow(callbacks=[ConsoleCallbackHandler()])


async def process_query(query: str):
    """
    Process a user query through the appropriate workflow.

    Args:
        query: User query string.

    Returns:
        String with processing results.
    """
    # Connect to the MCP server
    async with sse_client(f"http://{SERVER_HOST}:{SERVER_PORT}/sse") as (r, w):
        async with ClientSession(r, w) as session:
            await session.initialize()

            # Route the query to the appropriate workflow
            result = router.invoke({"query": query})
            task_type = result["task_type"]

            if task_type == "document_search":
                # Process document search query
                doc_result = await doc_workflow.ainvoke(
                    {"query": query, "session": session}
                )
                print("\n🔧 LangGraph Workflow Structure:")
                print(doc_workflow.get_graph().print_ascii())
                full_doc = doc_result.get("document", "No document found")
                summary = doc_result.get("summary", "No summary generated")

                return f"""
==================== 📄 Document Search Results ====================
{full_doc}

==================== ✨ Summary ====================
{summary}
"""
            else:
                # Process ruleset evaluation query
                ruleset_result = await ruleset_workflow.ainvoke(
                    {"query": query, "session": session}
                )
                print("\n🔧 LangGraph Workflow Structure:")
                print(ruleset_workflow.get_graph().print_ascii())
                explanation = ruleset_result.get(
                    "explanation", "No explanation generated"
                )
                return explanation


def main():
    """
    Main entry point for the application.
    """
    print("🔍 MCP Client - Document and Ruleset Assistant")
    print("==============================================")

    while True:
        try:
            user_query = input("\n📝 Enter your query (or 'exit' to quit): ")

            if user_query.lower() in ("exit", "quit", "q"):
                break

            print("\nProcessing query...")
            output = asyncio.run(process_query(user_query))
            print(output)

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\nError: {str(e)}")


if __name__ == "__main__":
    main()
