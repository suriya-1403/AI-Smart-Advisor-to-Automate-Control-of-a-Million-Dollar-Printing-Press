import json
import traceback
import os
from pathlib import Path

import uvicorn
from fastapi import Body, FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import your existing functions
from langchain_core.tracers import ConsoleCallbackHandler
from langsmith import Client
from mcp import ClientSession
from mcp.client.sse import sse_client

from mcp_server.config import SERVER_HOST, SERVER_PORT, LLM_MODEL
from mcp_server.core import create_router
from mcp_server.workflows import (
    create_document_workflow,
    create_event_workflow,
    create_general_knowledge_workflow,
    create_ruleset_workflow,
)

# Initialize client and workflows
client = Client()
router = create_router(callbacks=[ConsoleCallbackHandler()])
doc_workflow = create_document_workflow(callbacks=[ConsoleCallbackHandler()])
ruleset_workflow = create_ruleset_workflow(callbacks=[ConsoleCallbackHandler()])
event_workflow = create_event_workflow(callbacks=[ConsoleCallbackHandler()])
knowledge_workflow = create_general_knowledge_workflow(
    callbacks=[ConsoleCallbackHandler()]
)

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

# Add new endpoint to check for JSON files
@app.get("/check-json-files")
async def check_json_files():
    """Check if JSON files exist in the documents directory."""
    try:
        documents_dir = Path("/app/mcp_server/data/documents")
        if not documents_dir.exists():
            return {"hasJsonFiles": False}
        
        json_files = list(documents_dir.glob("*.json"))
        return {"hasJsonFiles": len(json_files) > 0}
    except Exception as e:
        print(f"Error checking JSON files: {str(e)}")
        print(traceback.format_exc())
        return {"hasJsonFiles": False}

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
        SSE_URL = f"http://{os.getenv('SSE_HOST')}:{os.getenv('SSE_PORT')}/sse"
        # Connect to the MCP server
        async with sse_client(SSE_URL) as (r, w):
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
                                "model": LLM_MODEL
                            }
                        elif task_type == "event_information":
                            # Process event information query
                            event_result = await event_workflow.ainvoke(
                                {"query": query, "session": session}
                            )
                            print("\n🔧 LangGraph Workflow Structure:")
                            print(event_workflow.get_graph().print_ascii())

                            event_identifier = event_result.get(
                                "event_identifier", "Unknown"
                            )
                            found_documents = event_result.get(
                                "found_documents", "No documents found"
                            )
                            structured_data = event_result.get(
                                "structured_data", "No data structured"
                            )
                            summary = event_result.get(
                                "summary", "No summary generated"
                            )

                            return {
                                "type": "event_information",
                                "event_identifier": event_identifier,
                                "documents": found_documents,
                                "structured_data": structured_data,
                                "summary": summary,
                                "model": LLM_MODEL
                            }
                        elif task_type == "general_knowledge":
                            # Process general knowledge query
                            knowledge_result = await knowledge_workflow.ainvoke(
                                {"query": query, "session": session}
                            )
                            print("\n🔧 LangGraph Workflow Structure:")
                            print(knowledge_workflow.get_graph().print_ascii())

                            formatted_query = knowledge_result.get(
                                "formatted_query", query
                            )
                            knowledge_response = knowledge_result.get(
                                "knowledge_response", "No response generated"
                            )
                            final_answer = knowledge_result.get(
                                "final_answer", "No final answer generated"
                            )

                            return {
                                "type": "general_knowledge",
                                "formatted_query": formatted_query,
                                "knowledge_response": knowledge_response,
                                "final_answer": final_answer,
                                "model": LLM_MODEL
                            }
                        else:
                            # Process ruleset evaluation query
                            ruleset_result = await ruleset_workflow.ainvoke(
                                {"query": query, "session": session}
                            )
                            print("\n🔧 LangGraph Workflow Structure:")
                            print(ruleset_workflow.get_graph().print_ascii())
                            explanation_raw = ruleset_result.get(
                                "explanation", "No explanation generated"
                            )
                            print("📦 RAW RULESET RESULT:", ruleset_result)
                            # parsed_insights = json.loads(ruleset_result['evaluation']['llm_insights'])
                            # llm_insights_str = ruleset_result["evaluation"]["llm_insights"]
                            llm_insights_raw = ruleset_result["evaluation"][
                                "llm_insights"
                            ]
                            report = ruleset_result.get("evaluation", {}).get("report")

                            if report:
                                print("🧪 Report:", report)
                            else:
                                print("⚠️ No report found.")
                                # If it's a valid JSON string, parse it. Otherwise, treat it as plain text.
                            # if isinstance(llm_insights_raw, str) and llm_insights_raw.strip().startswith("{"):
                            #     llm_insights = json.loads(llm_insights_raw)
                            #     report = llm_insights.get("report", "No report")
                            #     llm_response = llm_insights.get("llm_insights", "No explanation")
                            #     print("🧪 Report:", llm_insights.get("report", "No report"))
                            #     print("🧪 Explanation:", llm_insights.get("llm_insights", "No explanation"))
                            # else:
                            #     llm_insights_raw= print("🧪 Plain Explanation:", llm_insights_raw)
                            # print("🧪 Type of explanation:", type(ruleset_result.get("explanation")))
                            # report = "No report generated"
                            # llm_response = "No LLM response"

                            # if isinstance(explanation_raw, dict) and isinstance(explanation_raw.get("llm_insights"), str):
                            #     try:
                            #         # llm_insights is still a stringified JSON
                            #         parsed = json.loads(explanation_raw["llm_insights"])
                            #         report = parsed.get("report", "No report found")
                            #         llm_response = parsed.get("llm_insights", "No LLM response")
                            #     except Exception as e:
                            #         print(f"⚠️ Failed to parse inner LLM JSON: {e}")
                            return {
                                "type": "ruleset_evaluation",
                                "report": report,
                                "explanation": llm_insights_raw,
                                "model": LLM_MODEL
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
