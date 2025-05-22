"""
Tools for event information retrieval and analysis.
"""

import json
import os
import re
from typing import Dict, List, Optional

from langchain_groq import ChatGroq
from mcp.server.fastmcp import FastMCP

from mcp_server.config import DOCUMENTS_DIR, GROQ_API, LLM_MODEL


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


def find_event_documents(identifiers: Dict[str, str]) -> List[Dict]:
    """
    Find documents matching the event identifiers.

    Args:
        identifiers: Dictionary of search criteria.

    Returns:
        List of matching document records.
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
                matching_docs.append(record)
                continue

        # Check for location match
        if "location" in identifiers:
            location = str(record.get("location", "")).lower()
            if identifiers["location"] in location:
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

        return summary

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

            matching_events.append(
                {
                    "event_id": record.get("event_id"),
                    "location": record.get("location"),
                    "date": record.get("publish_date"),
                    "press_model": record.get("Press Model"),
                }
            )

        if not matching_events:
            return "No events found matching the specified criteria."

        # Format results
        result_lines = ["Found matching events:"]
        for event in matching_events:
            result_lines.append(
                f"Event {event['event_id']} - {event['press_model']} at {event['location']} ({event['date']})"
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
                events.append(
                    {
                        "event_id": record.get("event_id", "Unknown"),
                        "location": record.get("location", "Unknown"),
                        "date": record.get("publish_date", "Unknown"),
                        "press_model": record.get("Press Model", "Unknown"),
                    }
                )

        if not events:
            return "No events found."

        # Sort by event_id
        events.sort(key=lambda x: str(x["event_id"]))

        result_lines = ["Available Events:"]
        for event in events:
            result_lines.append(
                f"Event {event['event_id']} - {event['press_model']} at {event['location']} ({event['date']})"
            )

        return "\n".join(result_lines)
