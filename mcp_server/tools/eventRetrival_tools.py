"""
Event retrieval and analysis tools for the MCP Server.

This module provides comprehensive event information processing capabilities,
including event ID extraction, JSON metadata retrieval, PDF content analysis,
and intelligent summarization using LLM technology.

Key Features:
- Event ID pattern extraction from natural language queries
- JSON metadata file discovery and loading
- PDF content extraction and processing
- LLM-powered content summarization
- Comprehensive event metadata formatting
- MCP tool registration for event-based workflows

Author: AI Smart Advisor Team
Version: 1.0.0
Last Modified: 2024
"""

import json
import os
import re
from pathlib import Path
from typing import Optional

from langchain_groq import ChatGroq
from mcp.server.fastmcp import FastMCP
from PyPDF2 import PdfReader

from mcp_server.config import DOCUMENTS_DIR, GROQ_API, LLM_MODEL, PDF_DIR

# Initialize LLM for content summarization
llm = ChatGroq(model=LLM_MODEL, api_key=GROQ_API)

# === UTILS ===


def extract_event_identifier(query: str) -> Optional[str]:
    """
    Extract event identifier from user query using pattern matching.

    This function searches for event IDs in various formats within user queries,
    including explicit event references and standalone numbers that could
    represent event identifiers.

    Args:
        query (str): User query string to analyze for event identifiers

    Returns:
        Optional[str]: Extracted event ID as string, or None if not found

    Pattern Matching:
        - "event 42", "event_42", "event-42" → "42"
        - "event id 128" → "128"
        - Standalone numbers as fallback

    Example:
        >>> extract_event_identifier("Tell me about event 71")
        '71'
        >>> extract_event_identifier("What happened in event_id 42?")
        '42'
        >>> extract_event_identifier("Show me details for 156")
        '156'
        >>> extract_event_identifier("How does printing work?")
        None
    """
    patterns = [r"event[_\s-]?(\d+)", r"(\d+)"]
    for pattern in patterns:
        match = re.search(pattern, query.lower())
        if match:
            return match.group(1)
    return None


def find_event_json_by_id(event_id: str) -> Optional[tuple[str, dict]]:
    """
    Find JSON file containing data for a specific event ID.

    This function searches through all JSON files in the documents directory
    to find the file containing metadata for the specified event ID.

    Args:
        event_id (str): Event identifier to search for

    Returns:
        Optional[Tuple[str, Dict[str, Any]]]: Tuple of (filename, event_data)
                                            or None if not found

    Raises:
        None: All exceptions are caught and logged

    Example:
        >>> result = find_event_json_by_id("42")
        >>> if result:
        ...     filename, data = result
        ...     print(f"Found in {filename}: {data['event_id']}")
        ... else:
        ...     print("Event not found")
    """
    for filename in os.listdir(DOCUMENTS_DIR):
        if filename.endswith(".json"):
            path = os.path.join(DOCUMENTS_DIR, filename)
            try:
                with open(path, "r") as f:
                    data = json.load(f)
                    if str(data.get("event_id")) == str(event_id):
                        return filename, data
            except Exception as e:
                print(f"⚠️ Failed reading {filename}: {e}")
    return None


def read_pdf_matching_filename(json_filename: str) -> Optional[str]:
    """
    Read PDF content for a file matching the given JSON filename.

    This function looks for PDF files with the same base name as the JSON file
    and extracts all text content from the PDF for processing.

    Args:
        json_filename (str): Name of the JSON file (used to find matching PDF)

    Returns:
        Optional[str]: Extracted text content from PDF, or None if not found/failed

    PDF Processing:
        - Matches filenames by base name (without extension)
        - Extracts text from all pages
        - Filters out empty pages
        - Joins content with newlines

    Example:
        >>> content = read_pdf_matching_filename("event_42.json")
        >>> if content:
        ...     print(f"PDF contains {len(content)} characters")
        ... else:
        ...     print("No PDF found or read failed")
    """
    basename = Path(json_filename).stem
    for pdf_path in Path(PDF_DIR).glob("*.pdf"):
        if Path(pdf_path).stem == basename:
            try:
                reader = PdfReader(pdf_path)
                return "\n".join(
                    [
                        page.extract_text()
                        for page in reader.pages
                        if page.extract_text()
                    ]
                )
            except Exception as e:
                print(f"❌ PDF read failed: {e}")
    return None


def summarize_pdf_content(content: str) -> str:
    """
    Generate an intelligent summary of PDF content using LLM.

    This function uses a language model to create concise, technical summaries
    of printing event reports, focusing on key operational details and outcomes.

    Args:
        content (str): Raw text content extracted from PDF

    Returns:
        str: LLM-generated summary of the content

    Raises:
        Exception: If LLM service is unavailable or processing fails

    Summary Focus Areas:
        - Key technical specifications and parameters
        - Equipment configuration and settings
        - Performance results and outcomes
        - Notable issues or recommendations

    Example:
        >>> pdf_text = "Long technical report about printing job..."
        >>> summary = summarize_pdf_content(pdf_text)
        >>> print(summary)
        'Technical summary highlighting key findings...'
    """
    prompt = f"""
You are an expert summarizer in paragraphs. Summarize the key technical details from the following print event report:

--- BEGIN REPORT ---
{content}
--- END REPORT ---

Return a concise summary for a technical user.
"""
    return llm.invoke(prompt).content.strip()


def format_event_metadata(event: dict) -> str:
    """
    Format event metadata into a readable summary display.

    This function creates a standardized, formatted presentation of event
    metadata for consistent display across the application.

    Args:
        event (Dict[str, Any]): Event data dictionary containing metadata fields

    Returns:
        str: Formatted metadata summary with emoji indicators

    Formatted Fields:
        - Event ID and basic identification
        - Location and date information
        - Press model and equipment details
        - Media specifications (coating, weight)
        - Ink coverage information

    Example:
        >>> event_data = {
        ...     "event_id": "42",
        ...     "location": "Las Vegas Expo",
        ...     "Press Model": "HP T490"
        ... }
        >>> formatted = format_event_metadata(event_data)
        >>> print(formatted)
        📋 **Metadata Summary**
        - Event ID: 42
        - Location: Las Vegas Expo
        - Press Model: HP T490
        ...
    """
    return f"""📋 **Metadata Summary**
- Event ID: {event.get("event_id", "Unknown")}
- Location: {event.get("location", "Unknown")}
- Date: {event.get("publish_date", "Unknown")}
- Press Model: {event.get("Press Model", "Unknown")}
- Ink Coverage: {event.get("Ink Coverage", "Unknown")}
- Media Coating: {event.get("Media Coating", "Unknown")}
- Weight (GSM): {event.get("Media Weight GSM", "Unknown")}
"""


# === MCP Tool Setup ===


def setup_event_tools(mcp: FastMCP):
    """
    Set up event information tools for the MCP server.

    This function registers all event-related tools and resources with the
    MCP server, providing comprehensive event analysis capabilities.

    Args:
        mcp (FastMCP): FastMCP server instance to register tools with

    Raises:
        Exception: If tool registration fails

    Registered Tools:
        - get_event_summary: Complete event analysis with PDF processing

    Registered Resources:
        - events://list: List all available events
        - events://stats: Event corpus statistics

    Example:
        >>> server = FastMCP("PrintSystem")
        >>> setup_event_tools(server)
        >>> print("Event tools registered successfully")
    """

    @mcp.tool()
    def get_event_summary(query: str) -> str:
        """
        Extract event information and generate comprehensive summary with PDF analysis.

        This tool provides complete event analysis by:
        1. Extracting event ID from natural language query
        2. Loading JSON metadata for the event
        3. Finding and processing matching PDF content
        4. Generating intelligent LLM summary of technical details

        Args:
            query (str): Natural language query containing event reference

        Returns:
            str: Comprehensive event summary with metadata and PDF analysis

        Query Examples:
            - "Tell me about event 71"
            - "What happened in event 42?"
            - "Show me details for event_id 128"
            - "Analyze the Vegas expo event"

        Response Format:
            - Event metadata summary (ID, location, date, specifications)
            - PDF content analysis (if available)
            - LLM-generated technical insights
            - Data validation results
        """
        event_id = extract_event_identifier(query)
        if not event_id:
            return "❌ Could not extract event ID from your query."

        found = find_event_json_by_id(event_id)
        if not found:
            return f"❌ No JSON found for event ID {event_id}."

        json_filename, event_data = found
        metadata_summary = format_event_metadata(event_data)

        pdf_text = read_pdf_matching_filename(json_filename)
        if not pdf_text:
            return metadata_summary + "\n\n⚠️ PDF file not found for content summary."

        llm_summary = summarize_pdf_content(pdf_text)
        return metadata_summary + "\n\n🧠 **LLM Summary:**\n" + llm_summary

    @mcp.resource("events://list")
    def list_all_events() -> str:
        """
        List all available event IDs from JSON files in the documents directory.

        This resource provides a comprehensive overview of all events available
        in the system, including metadata about file availability and status.

        Returns:
            str: Formatted list of available events with statistics

        Example:
            >>> events = list_all_events()
            >>> print(events)
        """
        event_ids = []
        for filename in os.listdir(DOCUMENTS_DIR):
            if filename.endswith(".json"):
                try:
                    with open(os.path.join(DOCUMENTS_DIR, filename), "r") as f:
                        data = json.load(f)
                        if "event_id" in data:
                            event_ids.append(str(data["event_id"]))
                except:
                    continue
        return (
            "📂 Available Event IDs:\n" + "\n".join(sorted(set(event_ids)))
            if event_ids
            else "❌ No events found."
        )
