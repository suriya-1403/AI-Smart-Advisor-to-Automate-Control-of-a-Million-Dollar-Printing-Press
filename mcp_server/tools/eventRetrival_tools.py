"""
Tools for event information retrieval and analysis.
"""

import json
import os
import re
import PyPDF2
from typing import Dict, List, Optional, Tuple

from langchain_groq import ChatGroq
from mcp.server.fastmcp import FastMCP

from mcp_server.config import DOCUMENTS_DIR, GROQ_API, LLM_MODEL

# Define the PDF directory path
PDF_DIR = "/Users/suriya/Documents/Github/AI-Smart-Advisor-to-Automate-Control-of-a-Million-Dollar-Printing-Press/mcp_server/data/PDF"


def load_json_data(file_path):
    """
    Load a JSON document from file.

    Args:
        file_path: Path to the JSON file.

    Returns:
        Loaded JSON data or None if failed.
    """
    try:
        with open(file_path, "r") as file:
            return json.load(file)
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        return None


def extract_event_identifier(query: str) -> Dict[str, str]:
    """
    Extract event identifier from user query.

    Args:
        query: User query string.

    Returns:
        Dictionary containing extracted identifiers.
    """
    identifiers = {}

    # Extract event ID patterns
    event_id_patterns = [
        r"event[\s_-]*(\d+)",
        r"event_id[\s_-]*(\d+)",
        r"(\d+)",  # Just numbers
    ]

    for pattern in event_id_patterns:
        match = re.search(pattern, query.lower())
        if match:
            identifiers["event_id"] = match.group(1)
            break

    # Extract location keywords
    location_keywords = ["vegas", "las vegas", "expo", "chicago", "atlanta", "dallas"]
    for keyword in location_keywords:
        if keyword in query.lower():
            identifiers["location"] = keyword
            break

    return identifiers


def find_pdf_for_event(json_filename: str) -> Optional[str]:
    """
    Find the corresponding PDF file for a JSON file.

    Args:
        json_filename: Filename of the JSON file.

    Returns:
        Path to the PDF file if found, None otherwise.
    """
    if not os.path.exists(PDF_DIR):
        print(f"PDF directory not found: {PDF_DIR}")
        return None

    # Extract the base name without extension
    base_name = os.path.splitext(os.path.basename(json_filename))[0]
    
    # Look for PDF with matching name
    pdf_path = os.path.join(PDF_DIR, f"{base_name}.pdf")
    if os.path.exists(pdf_path):
        return pdf_path
    
    # If not found, try to match with event_id
    try:
        json_data = load_json_data(json_filename)
        if json_data and "event_id" in json_data:
            event_id = str(json_data["event_id"])
            
            # Look for PDF files containing the event_id in their names
            for filename in os.listdir(PDF_DIR):
                if filename.endswith(".pdf") and event_id in filename:
                    return os.path.join(PDF_DIR, filename)
    except Exception as e:
        print(f"Error matching PDF with event_id: {e}")
    
    return None


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text content from a PDF file.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        Extracted text content.
    """
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page_num in range(len(reader.pages)):
                text += reader.pages[page_num].extract_text() + "\n"
            return text
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return f"Error extracting text from PDF: {str(e)}"


def find_event_documents(identifiers: Dict[str, str]) -> List[Dict]:
    """
    Find documents matching the event identifiers.

    Args:
        identifiers: Dictionary of search criteria.

    Returns:
        List of matching document records with PDF paths.
    """
    matching_docs = []

    if not os.path.exists(DOCUMENTS_DIR):
        return matching_docs

    json_files = [f for f in os.listdir(DOCUMENTS_DIR) if f.endswith(".json")]

    for file_name in json_files:
        file_path = os.path.join(DOCUMENTS_DIR, file_name)
        record = load_json_data(file_path)

        if not record:
            continue

        # Check for event_id match
        if "event_id" in identifiers:
            event_id = str(record.get("event_id", ""))
            if event_id == identifiers["event_id"]:
                # Find corresponding PDF
                pdf_path = find_pdf_for_event(file_path)
                if pdf_path:
                    record["pdf_path"] = pdf_path
                matching_docs.append(record)
                continue

        # Check for location match
        if "location" in identifiers:
            location = str(record.get("location", "")).lower()
            if identifiers["location"] in location:
                # Find corresponding PDF
                pdf_path = find_pdf_for_event(file_path)
                if pdf_path:
                    record["pdf_path"] = pdf_path
                matching_docs.append(record)

    return matching_docs


def structure_event_data(documents: List[Dict]) -> Dict:
    """
    Structure event data from documents.

    Args:
        documents: List of document records.

    Returns:
        Structured event information.
    """
    if not documents:
        return {}

    # Use the first document as primary, but could aggregate multiple
    primary_doc = documents[0]

    structured_data = {
        "event_metadata": {
            "event_id": primary_doc.get("event_id"),
            "publish_date": primary_doc.get("publish_date"),
            "location": primary_doc.get("location"),
            "total_documents": len(documents),
        },
        "printing_specifications": {
            "press_model": primary_doc.get("Press Model"),
            "ink_coverage": primary_doc.get("Ink Coverage"),
            "media_weight_gsm": primary_doc.get("Media Weight GSM"),
            "media_coating": primary_doc.get("Media Coating"),
            "media_finish": primary_doc.get("Media Finish"),
            "optical_density": primary_doc.get("Optical Density"),
            "press_quality_mode": primary_doc.get("Press Quality Mode"),
        },
        "equipment_settings": {
            "dryer_configuration": primary_doc.get("Dryer Configuration"),
            "moisturizer_model": primary_doc.get("Moisturizer Model"),
            "surfactant": primary_doc.get("Surfactant"),
            "winders_brand": primary_doc.get("Winders Brand"),
        },
        "performance_data": {
            "target_press_speed": primary_doc.get("Target Press Speed"),
            "target_dryer_power": primary_doc.get("Target Dryer Power"),
            "tensions": {
                "dryer_zone": primary_doc.get("Dryer Zone Tension"),
                "print_zone": primary_doc.get("Print Zone Tension"),
                "unwinder_zone": primary_doc.get("Unwinder Zone Tension"),
                "rewinder_zone": primary_doc.get("Rewinder Zone Tension"),
            },
        },
        "pdf_data": {
            "path": primary_doc.get("pdf_path"),
            "available": "pdf_path" in primary_doc,
        },
        "raw_documents": documents,
    }

    return structured_data


def generate_event_summary(structured_data: Dict) -> str:
    """
    Generate a narrative summary of the event.

    Args:
        structured_data: Structured event information.

    Returns:
        Formatted summary string.
    """
    if not structured_data:
        return "No event information found."

    metadata = structured_data.get("event_metadata", {})
    specs = structured_data.get("printing_specifications", {})
    equipment = structured_data.get("equipment_settings", {})
    performance = structured_data.get("performance_data", {})
    pdf_data = structured_data.get("pdf_data", {})

    summary_parts = []

    # Event header
    event_id = metadata.get("event_id", "Unknown")
    location = metadata.get("location", "Unknown location")
    date = metadata.get("publish_date", "Unknown date")

    summary_parts.append(f"# Event {event_id} Summary")
    summary_parts.append(f"**Location:** {location}")
    summary_parts.append(f"**Date:** {date}")
    summary_parts.append("")

    # Printing specifications
    summary_parts.append("## Printing Specifications")
    if specs.get("press_model"):
        summary_parts.append(f"**Press Model:** {specs['press_model']}")
    if specs.get("ink_coverage"):
        summary_parts.append(f"**Ink Coverage:** {specs['ink_coverage']}")
    if specs.get("media_weight_gsm"):
        summary_parts.append(f"**Media Weight:** {specs['media_weight_gsm']} GSM")
    if specs.get("media_coating") and specs.get("media_finish"):
        summary_parts.append(
            f"**Media Type:** {specs['media_coating']} {specs['media_finish']}"
        )
    if specs.get("optical_density"):
        summary_parts.append(f"**Optical Density:** {specs['optical_density']}")
    if specs.get("press_quality_mode"):
        summary_parts.append(f"**Print Quality Mode:** {specs['press_quality_mode']}")
    summary_parts.append("")

    # Equipment configuration
    summary_parts.append("## Equipment Configuration")
    if equipment.get("dryer_configuration"):
        summary_parts.append(
            f"**Dryer Configuration:** {equipment['dryer_configuration']}"
        )
    if equipment.get("moisturizer_model"):
        summary_parts.append(f"**Moisturizer Model:** {equipment['moisturizer_model']}")
    if equipment.get("surfactant"):
        summary_parts.append(f"**Surfactant:** {equipment['surfactant']}")
    if equipment.get("winders_brand"):
        summary_parts.append(f"**Winders Brand:** {equipment['winders_brand']}")
    summary_parts.append("")

    # Performance results
    summary_parts.append("## Performance Results")
    if performance.get("target_press_speed"):
        summary_parts.append(
            f"**Target Press Speed:** {performance['target_press_speed']}"
        )
    if performance.get("target_dryer_power"):
        summary_parts.append(
            f"**Target Dryer Power:** {performance['target_dryer_power']}"
        )

    tensions = performance.get("tensions", {})
    if any(tensions.values()):
        summary_parts.append("**Tension Settings:**")
        for zone, value in tensions.items():
            if value:
                summary_parts.append(f"  - {zone.replace('_', ' ').title()}: {value}")
    
    # PDF information
    if pdf_data.get("available"):
        summary_parts.append("\n## Additional Documentation")
        summary_parts.append("**PDF Report:** Available")
        summary_parts.append(f"**PDF Path:** {pdf_data.get('path')}")
    
    return "\n".join(summary_parts)


def setup_event_tools(mcp: FastMCP):
    """
    Set up event information tools for the MCP server.

    Args:
        mcp: FastMCP server instance.
    """

    @mcp.tool()
    def get_event_information(query: str) -> str:
        """
        Retrieve detailed information about a specific printing event.

        Args:
            query: Query containing event identifier (e.g., "event 71", "Vegas expo").

        Returns:
            Detailed event information and summary.
        """
        # Extract event identifiers
        identifiers = extract_event_identifier(query)

        if not identifiers:
            return "Could not identify event from query. Please specify an event ID or location."

        # Find matching documents
        documents = find_event_documents(identifiers)

        if not documents:
            identifier_str = ", ".join([f"{k}: {v}" for k, v in identifiers.items()])
            return f"No documents found for event with {identifier_str}."

        # Structure the data
        structured_data = structure_event_data(documents)

        # Generate summary
        summary = generate_event_summary(structured_data)

        # Extract PDF content if available
        pdf_content = ""
        if structured_data.get("pdf_data", {}).get("available"):
            pdf_path = structured_data["pdf_data"]["path"]
            pdf_content = extract_text_from_pdf(pdf_path)
            
            if pdf_content:
                summary += "\n\n## PDF Content Summary\n"
                
                # Use a language model to summarize the PDF content
                llm = ChatGroq(model=LLM_MODEL, api_key=GROQ_API)
                try:
                    # Truncate PDF content if it's too long
                    truncated_content = pdf_content[:10000] if len(pdf_content) > 10000 else pdf_content
                    
                    prompt = f"""
                    Summarize the following PDF content from a printing event report:
                    
                    {truncated_content}
                    
                    Focus on key findings, metrics, and conclusions. Keep the summary concise.
                    """
                    
                    result = llm.invoke(prompt)
                    pdf_summary = result.content if hasattr(result, "content") else str(result)
                    
                    summary += f"\n{pdf_summary}\n\n"
                    summary += "\n**Note:** Full PDF content is available for more detailed information."
                except Exception as e:
                    summary += f"\nError summarizing PDF content: {str(e)}\n"
                    summary += f"\nRaw PDF content (first 500 chars):\n{pdf_content[:500]}...\n"
            else:
                summary += "\n\n**Note:** PDF file found but content could not be extracted."

        return summary

    @mcp.tool()
    def get_pdf_content(event_id: str) -> str:
        """
        Extract and return the content of a PDF file associated with an event.

        Args:
            event_id: Event ID to look for.

        Returns:
            Extracted text from the PDF.
        """
        if not os.path.exists(DOCUMENTS_DIR):
            return "Documents directory not found."

        json_files = [f for f in os.listdir(DOCUMENTS_DIR) if f.endswith(".json")]
        
        for file_name in json_files:
            file_path = os.path.join(DOCUMENTS_DIR, file_name)
            record = load_json_data(file_path)
            
            if record and str(record.get("event_id", "")) == event_id:
                pdf_path = find_pdf_for_event(file_path)
                
                if pdf_path:
                    content = extract_text_from_pdf(pdf_path)
                    return f"PDF Content for Event {event_id}:\n\n{content}"
                else:
                    return f"No PDF file found for Event {event_id}."
                
        return f"No event found with ID {event_id}."

    @mcp.tool()
    def find_event_by_criteria(
        event_id: str = None, location: str = None, press_model: str = None
    ) -> str:
        """
        Find events by specific criteria.

        Args:
            event_id: Specific event ID to search for.
            location: Location to search for.
            press_model: Press model to filter by.

        Returns:
            List of matching events with basic information.
        """
        if not os.path.exists(DOCUMENTS_DIR):
            return "Documents directory not found."

        json_files = [f for f in os.listdir(DOCUMENTS_DIR) if f.endswith(".json")]
        matching_events = []

        for file_name in json_files:
            file_path = os.path.join(DOCUMENTS_DIR, file_name)
            record = load_json_data(file_path)

            if not record:
                continue

            # Apply filters
            if event_id and str(record.get("event_id", "")) != str(event_id):
                continue
            if (
                location
                and location.lower() not in str(record.get("location", "")).lower()
            ):
                continue
            if (
                press_model
                and press_model.lower()
                not in str(record.get("Press Model", "")).lower()
            ):
                continue

            # Check for PDF
            pdf_path = find_pdf_for_event(file_path)
            has_pdf = "Yes" if pdf_path else "No"

            matching_events.append(
                {
                    "event_id": record.get("event_id"),
                    "location": record.get("location"),
                    "date": record.get("publish_date"),
                    "press_model": record.get("Press Model"),
                    "has_pdf": has_pdf,
                    "pdf_path": pdf_path,
                }
            )

        if not matching_events:
            return "No events found matching the specified criteria."

        # Format results
        result_lines = ["Found matching events:"]
        for event in matching_events:
            result_lines.append(
                f"Event {event['event_id']} - {event['press_model']} at {event['location']} ({event['date']}) - PDF: {event['has_pdf']}"
            )

        return "\n".join(result_lines)

    @mcp.resource("events://list")
    def list_all_events() -> str:
        """
        List all available events.

        Returns:
            List of all events with basic information.
        """
        if not os.path.exists(DOCUMENTS_DIR):
            return "Documents directory not found."

        json_files = [f for f in os.listdir(DOCUMENTS_DIR) if f.endswith(".json")]
        events = []

        for file_name in json_files:
            file_path = os.path.join(DOCUMENTS_DIR, file_name)
            record = load_json_data(file_path)

            if record:
                # Check for PDF
                pdf_path = find_pdf_for_event(file_path)
                has_pdf = "Yes" if pdf_path else "No"
                
                events.append(
                    {
                        "event_id": record.get("event_id", "Unknown"),
                        "location": record.get("location", "Unknown"),
                        "date": record.get("publish_date", "Unknown"),
                        "press_model": record.get("Press Model", "Unknown"),
                        "has_pdf": has_pdf,
                    }
                )

        if not events:
            return "No events found."

        # Sort by event_id
        events.sort(key=lambda x: str(x["event_id"]))

        result_lines = ["Available Events:"]
        for event in events:
            result_lines.append(
                f"Event {event['event_id']} - {event['press_model']} at {event['location']} ({event['date']}) - PDF: {event['has_pdf']}"
            )

        return "\n".join(result_lines)

    @mcp.resource("events://pdf_list")
    def list_pdf_documents() -> str:
        """
        List all available PDF documents.

        Returns:
            List of all PDF documents.
        """
        if not os.path.exists(PDF_DIR):
            return "PDF directory not found."

        pdf_files = [f for f in os.listdir(PDF_DIR) if f.endswith(".pdf")]
        
        if not pdf_files:
            return "No PDF files found."

        # Sort alphabetically
        pdf_files.sort()

        result_lines = ["Available PDF Documents:"]
        for i, pdf_file in enumerate(pdf_files, 1):
            result_lines.append(f"{i}. {pdf_file}")

        return "\n".join(result_lines)