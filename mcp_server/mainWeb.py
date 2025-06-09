"""
Main FastAPI web server for the MCP (Model Context Protocol) Server.

This module provides the web API layer for the AI Smart Advisor printing system.
It handles HTTP requests, manages client sessions, coordinates workflows, and
provides additional endpoints for logging and system monitoring.

Key Features:
- RESTful API endpoints for query processing
- Real-time logging dashboard with geolocation
- JSON file validation for data integrity
- Comprehensive error handling and debugging
- CORS support for cross-origin requests

Author: Suriyakrishnan Sathish & Rujula More
Version: 1.0.0
Last Modified: 2025
"""
import json
import os
import traceback
from pathlib import Path

import requests
import uvicorn
from fastapi import Body, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

# Import workflow and session management
from langchain_core.tracers import ConsoleCallbackHandler
from langsmith import Client
from mcp import ClientSession
from mcp.client.sse import sse_client

# Import local modules
from mcp_server.config import DOCUMENTS_DIR, LLM_MODEL, SERVER_HOST, SERVER_PORT
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

LOG_FILE_PATH = "/logs/access.log"


geo_cache = {}


def is_local_ip(ip):
    """
    Check if an IP address is from a local/private network.

    Args:
        ip (str): IP address to check

    Returns:
        bool: True if IP is local/private, False if public

    Example:
        >>> is_local_ip("192.168.1.1")
        True
        >>> is_local_ip("8.8.8.8")
        False
    """
    return (
        ip.startswith("192.")
        or ip.startswith("127.")
        or ip.startswith("10.")
        or ip.startswith("172.")
    )


# ipapi.co with error handling
def ipapi_fallback(ip):
    """
    Get city location for IP address using ipapi.co service.

    This function provides a fallback geolocation service with error handling
    and rate limit detection.

    Args:
        ip (str): IP address to geolocate

    Returns:
        Optional[str]: City name if successful, None if failed

    Example:
        >>> city = ipapi_fallback("8.8.8.8")
        >>> print(city)  # "Mountain View" (Google's location)
    """
    try:
        res = requests.get(f"https://ipapi.co/{ip}/city/", timeout=2)
        if res.status_code == 200:
            if "rapid request" not in res.text.lower():
                return res.text.strip()
    except:
        pass
    return None


# GeoIP provider functions
GEOIP_PROVIDERS = [
    ipapi_fallback,
    lambda ip: requests.get(f"https://ipwho.is/{ip}", timeout=2).json().get("city"),
    lambda ip: requests.get(f"https://ipinfo.io/{ip}/json", timeout=2)
    .json()
    .get("city"),
    lambda ip: requests.get(f"https://ipapi.de/ip/{ip}.json", timeout=2)
    .json()
    .get("city"),
]


# Master function to get location
def get_ip_location(ip: str) -> str:
    """
    Get geographic location for an IP address using multiple providers.

    This function tries multiple geolocation providers for reliability and
    caches results to avoid repeated API calls.

    Args:
        ip (str): IP address to geolocate

    Returns:
        str: City name or "Local Network"/"Unknown" if cannot determine

    Example:
        >>> location = get_ip_location("8.8.8.8")
        >>> print(location)  # "Mountain View"
    """
    if ip in geo_cache:
        return geo_cache[ip]

    if is_local_ip(ip):
        geo_cache[ip] = "Local Network"
        return geo_cache[ip]

    for provider in GEOIP_PROVIDERS:
        try:
            city = provider(ip)
            if (
                city
                and isinstance(city, str)
                and "please try again" not in city.lower()
            ):
                geo_cache[ip] = city
                return city
        except:
            continue

    geo_cache[ip] = "Unknown"
    return "Unknown"


def extract_real_ip(entry):
    """
    Extract the real client IP address from log entry.

    Handles various proxy configurations and forwarded headers to determine
    the actual client IP address.

    Args:
        entry (Dict[str, Any]): Log entry dictionary from Caddy server

    Returns:
        str: Real client IP address or "Unknown" if cannot determine

    Example:
        >>> log_entry = {"request": {"headers": {"X-Forwarded-For": ["1.2.3.4"]}}}
        >>> ip = extract_real_ip(log_entry)
        >>> print(ip)  # "1.2.3.4"
    """
    try:
        xff = entry.get("request", {}).get("headers", {}).get("X-Forwarded-For")
        if xff and isinstance(xff, list) and xff[0]:
            return xff[0]
        return entry.get("request", {}).get("remote_ip", "Unknown")
    except:
        return "Unknown"


@app.get("/logzz", response_class=JSONResponse)
async def get_caddy_logs():
    """
    Retrieve and process Caddy server access logs with geolocation data.

    This endpoint reads the Caddy access log file, parses JSON entries,
    extracts client IP addresses, and enriches each entry with geographic
    location information.

    Returns:
        Dict containing logs with geolocation data added

    Raises:
        HTTPException: If log file cannot be read or processed

    Example Response:
        {
            "data": [
                {
                    "request": {...},
                    "remote_ip": "8.8.8.8",
                    "location": "Mountain View"
                }
            ]
        }
    """
    logs = []
    if os.path.exists(LOG_FILE_PATH):
        with open(LOG_FILE_PATH, "r") as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    ip = extract_real_ip(entry)
                    entry["remote_ip"] = ip
                    entry["location"] = get_ip_location(ip)
                    logs.append(entry)
                except Exception as e:
                    print("❌ Failed to parse log line:", e)
                    continue
    return {"data": logs}


@app.get("/logdashz", response_class=HTMLResponse)
async def serve_dashboard():
    """
    Serve the logging dashboard HTML interface.

    This endpoint provides a web-based dashboard for viewing access logs
    with geographic visualization and filtering capabilities.

    Returns:
        str: HTML content for the dashboard

    Raises:
        HTTPException: If dashboard file cannot be found or read

    Example:
        Access via browser: http://localhost:8000/logdashz
    """
    with open("/app/mcp_server/logDash.html") as f:
        return f.read()


# Add new endpoint to check for JSON files
@app.get("/check-json-files")
async def check_json_files():
    """
    Check if JSON document files exist in the documents directory.

    This endpoint validates that the system has access to printing event
    documents required for document search functionality.

    Returns:
        Dict[str, bool]: Dictionary with hasJsonFiles boolean indicator

    Example Response:
        {"hasJsonFiles": true}
    """
    try:
        print(DOCUMENTS_DIR)
        documents_dir = Path(DOCUMENTS_DIR)
        if not documents_dir.exists():
            print("→ directory does not exist")

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
    Process a user query through the appropriate workflow with comprehensive error handling.

    This is the core function that coordinates query processing by:
    1. Establishing connection to MCP server
    2. Routing query to appropriate workflow
    3. Executing workflow and gathering results
    4. Formatting response for client

    Args:
        query (str): User query string to process

    Returns:
        Dict[str, Any]: Processed results with type-specific structure

    Raises:
        Exception: Various exceptions for connection, processing, or workflow errors

    Example:
        >>> result = await process_query("Find heavy ink coverage documents")
        >>> print(result["type"])  # "document_search"
    """
    try:
        SSE_URL = f"http://{os.getenv('SSE_HOST')}:{os.getenv('SSE_PORT')}/sse"
        # SSE_URL = "http://localhost:8050/sse"
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
                                "model": LLM_MODEL,
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
                                "model": LLM_MODEL,
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
                                "model": LLM_MODEL,
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
                                "model": LLM_MODEL,
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
    """
    Main API endpoint for processing user queries.

    This endpoint accepts natural language queries about printing operations
    and routes them through appropriate AI workflows for processing.

    Args:
        query (str): User query string embedded in request body

    Returns:
        Dict[str, Any]: Processing results with status and workflow-specific data

    Raises:
        HTTPException: For client errors (400) or server errors (500)

    Example Request:
        POST /query
        {"query": "Find documents with heavy ink coverage on glossy media"}

    Example Response:
        {
            "status": "success",
            "result": {
                "type": "document_search",
                "document": "...",
                "summary": "..."
            }
        }
    """
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
    """
    Main function to run the FastAPI server.

    Configures and starts the uvicorn ASGI server with appropriate settings
    for development and production environments.

    Environment Variables:
        HOST: Server host address (default: 0.0.0.0)
        PORT: Server port (default: 8000)
        DEBUG: Enable debug mode (default: False)
    """
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="debug")


if __name__ == "__main__":
    main()
