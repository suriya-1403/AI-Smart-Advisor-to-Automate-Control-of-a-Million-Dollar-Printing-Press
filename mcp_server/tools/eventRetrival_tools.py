import os
import json
import re
from typing import Optional
from pathlib import Path
from PyPDF2 import PdfReader

from langchain_groq import ChatGroq
from mcp.server.fastmcp import FastMCP
from mcp_server.config import DOCUMENTS_DIR, GROQ_API, LLM_MODEL, PDF_DIR

llm = ChatGroq(model=LLM_MODEL, api_key=GROQ_API)

# === UTILS ===

def extract_event_identifier(query: str) -> Optional[str]:
    patterns = [r"event[_\s-]?(\d+)", r"(\d+)"]
    for pattern in patterns:
        match = re.search(pattern, query.lower())
        if match:
            return match.group(1)
    return None

def find_event_json_by_id(event_id: str) -> Optional[tuple[str, dict]]:
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
    basename = Path(json_filename).stem
    for pdf_path in Path(PDF_DIR).glob("*.pdf"):
        if Path(pdf_path).stem == basename:
            try:
                reader = PdfReader(pdf_path)
                return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
            except Exception as e:
                print(f"❌ PDF read failed: {e}")
    return None

def summarize_pdf_content(content: str) -> str:
    prompt = f"""
You are an expert summarizer in paragraphs. Summarize the key technical details from the following print event report:

--- BEGIN REPORT ---
{content}
--- END REPORT ---

Return a concise summary for a technical user.
"""
    return llm.invoke(prompt).content.strip()

def format_event_metadata(event: dict) -> str:
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
    @mcp.tool()
    def get_event_summary(query: str) -> str:
        """
        Given a query, extract the event ID, load the matching JSON and PDF,
        and return a metadata summary and LLM-generated content summary.
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
        List all available event IDs from JSON files.
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
        return "📂 Available Event IDs:\n" + "\n".join(sorted(set(event_ids))) if event_ids else "❌ No events found."