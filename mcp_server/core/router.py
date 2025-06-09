"""
Query router for determining task type and routing requests to appropriate workflows.

This module implements an intelligent query classification system that analyzes user
input and determines which processing pipeline should handle the request. It uses
LLM-based classification combined with pattern matching for reliable routing.

The router supports four main task types:
- Document search: Finding relevant printing event documents
- Ruleset evaluation: Calculating optimal printer configurations
- Event information: Retrieving specific event details and reports
- General knowledge: Educational responses about printing concepts

Author: Suriyakrishnan Sathish & Rujula More
Version: 1.0.0
Last Modified: 2025
"""

from langchain_core.messages import HumanMessage, SystemMessage

# from langchain_ollama import OllamaLLM
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from mcp_server.config import GROQ_API, LLM_MODEL


# Define state
class State(TypedDict):
    """
    State type for the query router workflow.

    This TypedDict defines the structure of data that flows through the
    router's state graph during query processing.

    Attributes:
        query (str): The original user query to be classified
        task_type (str): The determined task type after classification
        result (str): Final routing result or error message
    """

    query: str
    task_type: str
    result: str


def create_router(callbacks=None):
    """
    Create a router graph to determine query type and route to appropriate workflow.

    This function builds a LangGraph state machine that uses LLM-based classification
    combined with pattern matching to reliably route user queries to the correct
    processing pipeline.

    Args:
        callbacks: Optional callbacks for graph execution monitoring and debugging

    Returns:
        Compiled router graph ready for query processing

    Raises:
        ConnectionError: If unable to connect to Groq LLM service
        ValueError: If invalid configuration parameters provided

    """

    # Setup LLM
    # llm = OllamaLLM(model=LLM_MODEL)
    llm = ChatGroq(model=LLM_MODEL, api_key=GROQ_API)

    # Define classifier node
    def classify_query(state: State):
        """
        Classify user query into one of four task types using LLM and pattern matching.

        This function combines LLM-based semantic understanding with rule-based
        pattern matching to provide reliable query classification.

        Args:
            state (State): Current state containing the user query

        Returns:
            Dict[str, str]: Updated state with determined task_type

        Classification Types:
            - document_search: Finding relevant documents from collections
            - ruleset_evaluation: Calculating printer settings based on rules
            - event_information: Retrieving specific event details
            - general_knowledge: Educational responses about printing concepts

        """

        system_prompt = """
        You are a classifier that determines whether a user query is about:

        1. Document search: Finding relevant documents from a collection based on user query criteria. This involves matching the query against document metadata or content to retrieve the most appropriate document(s). Examples:
        - "Find prints with heavy ink coverage on glossy media"
        - "Show me documents about T250 printers using 75GSM paper"
        - "Get all reports from the Las Vegas expo"
        - "Which documents mention coated inkjet with 266 GSM?"

        2. Ruleset evaluation: Determining appropriate printer settings or evaluating printing parameters based on established rules. This involves applying printing rules to specific scenarios to recommend proper configurations. Examples:
        - "I want to print on uncoated eggshell paper with an ink coverage of 96% and optical density of 88%. The paper weight is 372 GSM. Use the Performance HDK print mode, with a 1-Zone dryer, Eltex moisturizer, Silicon surfactant, and EMT winders give final values.”
        - "Give target press speed and target dryer power I'm working with coated glossy media, ink coverage of 45%, and optical density of 60%. The weight is 70 GSM, and the print mode is Quality. The dryer is set to DEM, using Weko moisturizer, Water as surfactant, and Hunkeler winders."
        - "The job uses coated inkjet paper with a smooth finish, ink coverage at 96%, optical density 97%, and media weight 385 GSM. We’re printing in Performance mode, with a 3-Zone dryer, Eltex moisturizer, Silicon surfactant, and EMT winders give target dryer power and other values.”
        - “Give final values for printing on uncoated satin paper (weight: 210 GSM) with 50% ink and 85% density, in Performance mode, using Default dryer, Eltex, Water surfactant, and EMT winders.”

        3. Event information: Asking for specific details about a particular printing event, including what happened, the parameters used, and outcomes. These queries focus on retrieving and summarizing information about specific events. Examples:
        - "Tell me about event 71"
        - "What happened in event 42?"
        - "Give me details about event_id 128"
        - "Show me information about the Vegas expo event"
        - "Describe event 205"
        - "What were the results of event 89?"
        - "Tell me what occurred during event 156"

        4. General knowledge: Educational questions about printing concepts, media types, and technology explanations. These are "how", "what", "why" questions that seek to understand concepts rather than find specific documents or calculate settings. Examples:
        - "Can you explain the difference between coated and uncoated media?"
        - "How does heavy ink coverage affect media requirements?"
        - "What's the relationship between ink coverage and media weight?"
        - "Why would someone choose silk finish over matte?"
        - "How do dryer configurations work?"
        - "What are the benefits of moisturizers in printing?"
        - "How does media weight impact print quality?"

        Return ONLY the task type as either "document_search", "ruleset_evaluation", "event_information" or "general_knowledge".
        """

        try:
            result = llm.invoke(
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=state["query"]),
                ]
            )

            # Handle both object with content attribute and string responses
            content = result
            if hasattr(result, "content"):
                content = result.content

            content_lower = str(content).lower()

            print(f"DEBUG: LLM Classifier: {content_lower}")

            query_lower = state["query"].lower()

            # Check for event-related patterns first
            event_patterns = [
                "event",
                "event id",
                "event_id",
                r"event\s*\d+",
                r"event_id\s*\d+",
            ]

            event_keywords = [
                "tell",
                "about",
                "what",
                "describe",
                "show",
                "information",
                "details",
                "happened",
                "occurred",
            ]

            # More explicit event detection
            is_event_query = False
            if any(pattern in query_lower for pattern in event_patterns):
                is_event_query = True
            elif "event" in query_lower and any(
                keyword in query_lower for keyword in event_keywords
            ):
                is_event_query = True
            elif "event_information" in content_lower:
                is_event_query = True

            if is_event_query:
                print(f"🎯 Event information detected")
                return {"task_type": "event_information"}

            # Check LLM classification result
            if "general_knowledge" in content_lower:
                print(f"🤖 LLM classified as general_knowledge")
                return {"task_type": "general_knowledge"}
            elif "event_information" in content_lower:
                print(f"🤖 LLM classified as event_information")
                return {"task_type": "event_information"}
            elif (
                "document" in content_lower
                or "find" in content_lower
                or "search" in content_lower
                or "report" in content_lower
                or "pdf" in content_lower
                or "document_search" in content_lower
            ):
                print(f"🤖 LLM classified as document_search")
                return {"task_type": "document_search"}
            else:
                print(f"🤖 LLM classified as ruleset_evaluation")
                return {"task_type": "ruleset_evaluation"}
        except Exception as e:
            print(f"Error in classify_query: {str(e)}")
            # Default to general knowledge for educational queries
            query_lower = state["query"].lower()
            if any(
                indicator in query_lower
                for indicator in ["explain", "how", "what", "why"]
            ):
                return {"task_type": "general_knowledge"}
            return {"task_type": "document_search"}

    # Define graph
    workflow = StateGraph(State)
    workflow.add_node("classifier", classify_query)

    # Connect nodes
    workflow.add_edge(START, "classifier")
    workflow.add_conditional_edges(
        "classifier",
        lambda x: x["task_type"],
        {
            "document_search": END,
            "ruleset_evaluation": END,
            "event_information": END,
            "general_knowledge": END,
        },
    )

    # Apply callbacks if provided
    compiled = workflow.compile()
    if callbacks:
        compiled.callbacks = callbacks

    return compiled
