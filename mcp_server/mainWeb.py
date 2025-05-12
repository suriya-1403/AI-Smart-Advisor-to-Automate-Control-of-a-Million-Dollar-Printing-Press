import traceback

import uvicorn
from fastapi import Body, FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import your existing functions
from langchain_core.tracers import ConsoleCallbackHandler
from langsmith import Client
from mcp import ClientSession
from mcp.client.sse import sse_client

from mcp_server.config import SERVER_HOST, SERVER_PORT
from mcp_server.core import create_router
from mcp_server.workflows import create_document_workflow, create_ruleset_workflow

# Initialize client and workflows
client = Client()
router = create_router(callbacks=[ConsoleCallbackHandler()])
doc_workflow = create_document_workflow(callbacks=[ConsoleCallbackHandler()])
ruleset_workflow = create_ruleset_workflow(callbacks=[ConsoleCallbackHandler()])

# Create FastAPI app
app = FastAPI(title="MCP Query API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Modified process_query function with better error handling
async def process_query(query: str):
    """
    Process a user query through the appropriate workflow with improved error handling.

    Args:
        query: User query string.

    Returns:
        Dictionary with processing results.
    """
    try:
        # Connect to the MCP server
        async with sse_client(f"http://{SERVER_HOST}:{SERVER_PORT}/sse") as (r, w):
            try:
                async with ClientSession(r, w) as session:
                    try:
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

                            return {
                                "type": "document_search",
                                "document": full_doc,
                                "summary": summary,
                            }
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
                            return {
                                "type": "ruleset_evaluation",
                                "explanation": explanation,
                            }
                    except Exception as e:
                        print(f"Error during workflow execution: {str(e)}")
                        print(traceback.format_exc())
                        raise Exception(f"Workflow error: {str(e)}")
            except Exception as e:
                print(f"Error in client session: {str(e)}")
                print(traceback.format_exc())
                raise Exception(f"Session error: {str(e)}")
    except Exception as e:
        print(f"Error in SSE client connection: {str(e)}")
        print(traceback.format_exc())
        raise Exception(f"Connection error: {str(e)}")


# Create API endpoint with better error handling
@app.post("/query")
async def handle_query(query: str = Body(..., embed=True)):
    """Process a query and return the results with detailed error messages."""
    try:
        result = await process_query(query)
        return {"status": "success", "result": result}
    except Exception as e:
        error_detail = str(e)
        print(f"API Error: {error_detail}")
        print(traceback.format_exc())
        return {
            "status": "error",
            "message": error_detail,
            "detail": traceback.format_exc(),
        }


# Main function to run the server
def main():
    """Run the FastAPI server."""
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="debug")


if __name__ == "__main__":
    main()
