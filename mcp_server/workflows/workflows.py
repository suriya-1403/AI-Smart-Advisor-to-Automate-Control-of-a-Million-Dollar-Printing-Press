"""
LangGraph workflows for coordinating complex AI processing pipelines.

This module implements sophisticated state-based workflows using LangGraph for
orchestrating multi-step AI processing tasks. Each workflow handles a specific
type of user query through a series of coordinated processing nodes.

Key Features:
- Type-safe state management with TypedDict schemas
- Async processing with proper error handling
- LangSmith tracing for observability and debugging
- Modular node architecture for maintainability
- Comprehensive response formatting and validation

Workflow Types:
- Document Search: Enhanced query processing and result summarization
- Ruleset Evaluation: Parameter formatting and configuration processing
- Event Information: Event ID extraction and comprehensive analysis
- General Knowledge: Educational response generation and formatting

Author: AI Smart Advisor Team
Version: 1.0.0
Last Modified: 2024
"""


import json

# from langchain_ollama import OllamaLLM
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph
from langsmith import traceable
from typing_extensions import TypedDict

from mcp_server.config import GROQ_API, LLM_MODEL

# from mcp_server.eventRetrival_tools import extract_event_identifier, get_document_chunks_by_event_id
# Initialize LLM
# llm = OllamaLLM(model=LLM_MODEL)
llm = ChatGroq(model=LLM_MODEL, api_key=GROQ_API)


class DocSearchState(TypedDict):
    """
    State schema for document search workflow.

    This TypedDict defines the structure of data that flows through the
    document search workflow, ensuring type safety and clear interfaces.

    Attributes:
        query (str): Original user search query
        enhanced_query (str): AI-enhanced query for better retrieval
        document (str): Retrieved document content
        summary (str): AI-generated summary of results
        session (object): MCP ClientSession for tool communication
    """

    query: str
    enhanced_query: str
    document: str
    summary: str
    session: object  # MCP ClientSession


@traceable(name="DocumentSearchWorkflow")
def create_document_workflow(callbacks=None):
    """
    Create a comprehensive document search workflow using LangGraph.

    This workflow implements a three-stage process for intelligent document
    retrieval: query enhancement, document search, and result summarization.

    Args:
        callbacks (Optional[List]): Callbacks for workflow monitoring and debugging

    Returns:
        StateGraph: Compiled workflow ready for execution

    Workflow Stages:
        1. Query Enhancement: Optimize search terms using AI
        2. Document Search: Execute enhanced search via MCP tools
        3. Result Summarization: Generate concise summary of findings

    Example:
        >>> workflow = create_document_workflow()
        >>> result = await workflow.ainvoke({
        ...     "query": "heavy ink coverage documents",
        ...     "session": mcp_session
        ... })
        >>> print(result["summary"])
    """

    async def enhance_query(state: DocSearchState):
        """
        Enhance the user's search query for improved document retrieval.

        This node applies AI-powered query optimization to improve search
        relevance by expanding terms, adding context, and refining language.

        Args:
            state (DocSearchState): Current workflow state

        Returns:
            Dict[str, str]: State update with enhanced_query

        Enhancement Techniques:
            - Term expansion for technical concepts
            - Context addition for printing terminology
            - Query restructuring for better matching
        """

        prompt = (
            "You are an expert search assistant. Transform the user's document search query "
            "into an optimized query for internal document retrieval."
        )
        result = await llm.ainvoke([prompt, state["query"]])
        return {
            "enhanced_query": str(result.content)
            if hasattr(result, "content")
            else str(result)
        }

    async def search_document(state: DocSearchState):
        """
        Execute document search using the enhanced query via MCP tools.

        This node calls the MCP document search tool with the enhanced query
        to retrieve relevant printing event documents and specifications.

        Args:
            state (DocSearchState): Current workflow state

        Returns:
            Dict[str, str]: State update with document search results
        """

        result = await state["session"].call_tool(
            "find_document", {"query": state["enhanced_query"]}
        )
        return {"document": result.content[0].text}

    async def summarize(state: DocSearchState):
        """
        Generate an intelligent summary of the document search results.

        This node creates a concise, informative summary of the retrieved
        documents, highlighting key findings and relevant details.

        Args:
            state (DocSearchState): Current workflow state

        Returns:
            Dict[str, str]: State update with summary content
        """
        prompt = "Summarize the following document search results in a concise manner."
        result = await llm.ainvoke([prompt, state["document"]])
        return {
            "summary": str(result.content)
            if hasattr(result, "content")
            else str(result)
        }

    workflow = StateGraph(DocSearchState)
    workflow.add_node("enhance_query", enhance_query)
    workflow.add_node("search_document", search_document)
    workflow.add_node("summarize", summarize)

    workflow.add_edge(START, "enhance_query")
    workflow.add_edge("enhance_query", "search_document")
    workflow.add_edge("search_document", "summarize")
    workflow.add_edge("summarize", END)

    compiled = workflow.compile()
    if callbacks:
        compiled.callbacks = callbacks
    return compiled


class RulesetState(TypedDict):
    """
    State schema for ruleset evaluation workflow.

    Attributes:
        query (str): Original user configuration query
        formatted_parameters (str): AI-formatted parameter structure
        evaluation (str): Ruleset evaluation results
        explanation (str): Human-readable explanation of results
        session (object): MCP ClientSession for tool communication
    """

    query: str
    formatted_parameters: str
    evaluation: str
    explanation: str
    session: object  # MCP ClientSession


@traceable(name="RulesetEvaluationWorkflow")
def create_ruleset_workflow(callbacks=None):
    """
    Create a comprehensive ruleset evaluation workflow for printer configuration.

    This workflow processes user printing requirements through a multi-stage
    pipeline: parameter formatting, ruleset evaluation, and result explanation.

    Args:
        callbacks (Optional[List]): Callbacks for workflow monitoring

    Returns:
        StateGraph: Compiled workflow for configuration processing

    Workflow Stages:
        1. Parameter Formatting: Structure user input for rule processing
        2. Ruleset Evaluation: Apply configuration rules via multi-agent system
        3. Result Explanation: Generate human-readable explanations

    Example:
        >>> workflow = create_ruleset_workflow()
        >>> result = await workflow.ainvoke({
        ...     "query": "media coating - Coated\\nink coverage - 85",
        ...     "session": mcp_session
        ... })
        >>> print(result["explanation"])
    """

    async def format_parameters(state: RulesetState):
        """
        Format and structure printing parameters from user query.

        This node converts various input formats into a standardized structure
        suitable for ruleset processing, ensuring consistency and completeness.

        Args:
            state (RulesetState): Current workflow state

        Returns:
            Dict[str, str]: State update with formatted_parameters
        """
        prompt = (
            "Convert the following printing query into a structured parameter list "
            "with key-value pairs."
        )
        result = await llm.ainvoke([prompt, state["query"]])
        return {
            "formatted_parameters": str(result.content)
            if hasattr(result, "content")
            else str(result)
        }

    # async def evaluate_ruleset(state: RulesetState):
    #     """
    #     Evaluate ruleset using the formatted parameters.
    #     """
    #     result = await state["session"].call_tool(
    #         "evaluate_ruleset", {"query": state["formatted_parameters"]}
    #     )
    #     return {"evaluation": result.content[0].value}
    import json  # make sure this is imported at the top

    async def evaluate_ruleset(state: RulesetState):
        """
        Evaluate printing configuration using the multi-agent ruleset system.

        This node executes the core configuration logic by calling the MCP
        ruleset evaluation tool and processing the structured results.

        Args:
            state (RulesetState): Current workflow state

        Returns:
            Dict[str, Any]: State update with evaluation results
        """
        result = await state["session"].call_tool(
            "evaluate_ruleset", {"query": state["formatted_parameters"]}
        )

        raw_text = result.content[0].text
        extracted_explanation = ""
        if 'content="' in raw_text:
            extracted_explanation = (
                raw_text.split('content="', 1)[1].split('"', 1)[0].replace("\\n", "\n")
            )

        try:
            parsed = json.loads(raw_text)
            return {
                "evaluation": {
                    "report": parsed.get("report", ""),
                    "llm_insights": parsed.get("llm_insights", ""),
                }
            }
        except Exception:
            # Fallback to manually extract `report` and `llm_insights`
            return {
                "evaluation": {
                    "report": raw_text.split("## Explanation")[0].strip(),
                    "llm_insights": extracted_explanation.strip(),
                }
            }

    async def explain_results(state: RulesetState):
        """
        Extract and format explanation from ruleset evaluation results.

        This node processes the evaluation results to create clear, actionable
        explanations for users about the recommended configuration.

        Args:
            state (RulesetState): Current workflow state

        Returns:
            Dict[str, str]: State update with explanation text
        """
        evaluation = state.get("evaluation", {})
        raw_insight = evaluation.get("llm_insights", "")

        if hasattr(raw_insight, "content"):
            explanation_text = raw_insight.content
        else:
            explanation_text = str(raw_insight)

        return {"explanation": evaluation.get("llm_insights", "").strip()}

    workflow = StateGraph(RulesetState)
    workflow.add_node("format_parameters", format_parameters)
    workflow.add_node("evaluate_ruleset", evaluate_ruleset)
    workflow.add_node("explain_results", explain_results)

    workflow.add_edge(START, "format_parameters")
    workflow.add_edge("format_parameters", "evaluate_ruleset")
    workflow.add_edge("evaluate_ruleset", "explain_results")
    workflow.add_edge("explain_results", END)

    compiled = workflow.compile()
    if callbacks:
        compiled.callbacks = callbacks
    return compiled


class EventInfoState(TypedDict):
    """
    State schema for event information workflow.

    Attributes:
        query (str): Original user query about specific events
        event_identifier (str): Extracted event ID or identifier
        summary (str): Comprehensive event summary with analysis
        session (object): MCP ClientSession for tool communication
    """

    query: str
    event_identifier: str
    summary: str
    session: object


@traceable(name="EventInformationWorkflow")
def create_event_workflow(callbacks=None):
    """
    Create a specialized workflow for comprehensive event information processing.

    This workflow handles queries about specific printing events by extracting
    event identifiers and retrieving detailed information including metadata,
    PDF content analysis, and LLM-generated insights.

    Args:
        callbacks (Optional[List]): Callbacks for workflow monitoring

    Returns:
        StateGraph: Compiled workflow for event information processing

    Workflow Stages:
        1. Event ID Extraction: Identify target event from user query
        2. Event Summary Retrieval: Get comprehensive event information

    Example:
        >>> workflow = create_event_workflow()
        >>> result = await workflow.ainvoke({
        ...     "query": "tell me about event 42",
        ...     "session": mcp_session
        ... })
        >>> print(result["summary"])
    """

    async def extract_event_id(state: EventInfoState):
        """
        Extract event identifier from user query using pattern matching.

        This node analyzes the user query to identify specific event references
        and extract the event ID for subsequent processing.

        Args:
            state (EventInfoState): Current workflow state

        Returns:
            Dict[str, str]: State update with event_identifier

        Raises:
            ValueError: If no valid event ID can be extracted
        """
        result = await state["session"].call_tool(
            "extract_event_identifier", {"query": state["query"]}
        )

        event_id = result.content[0].text.strip()

        if not event_id:
            raise ValueError("❌ No valid event ID found in the query.")

        return {"event_identifier": event_id}

    async def fetch_event_summary(state: EventInfoState):
        """
        Retrieve comprehensive event summary including metadata and PDF analysis.

        This node calls the event summary tool to get detailed information
        about the identified event, including JSON metadata and PDF content analysis.

        Args:
            state (EventInfoState): Current workflow state

        Returns:
            Dict[str, str]: State update with comprehensive summary
        """
        result = await state["session"].call_tool(
            "get_event_summary", {"query": state["query"]}
        )
        return {"summary": result.content[0].text}

    workflow = StateGraph(EventInfoState)
    workflow.add_node("extract_event_id", extract_event_id)
    workflow.add_node("fetch_event_summary", fetch_event_summary)
    workflow.add_edge(START, "extract_event_id")
    workflow.add_edge("extract_event_id", "fetch_event_summary")
    workflow.add_edge("fetch_event_summary", END)

    compiled = workflow.compile()
    if callbacks:
        compiled.callbacks = callbacks
    return compiled


class GeneralKnowledgeState(TypedDict):
    """
    State schema for general knowledge workflow.

    Attributes:
        query (str): Original user question about printing concepts
        formatted_query (str): AI-optimized query for knowledge retrieval
        knowledge_response (str): Raw response from knowledge system
        final_answer (str): Formatted final answer for user presentation
        session (object): MCP ClientSession for tool communication
    """

    query: str
    formatted_query: str
    knowledge_response: str
    final_answer: str
    session: object  # MCP ClientSession


@traceable(name="GeneralKnowledgeWorkflow")
def create_general_knowledge_workflow(callbacks=None):
    """
    Create a comprehensive workflow for educational printing knowledge responses.

    This workflow processes educational questions about printing concepts through
    a three-stage pipeline: query formatting, knowledge retrieval, and answer formatting.

    Args:
        callbacks (Optional[List]): Callbacks for workflow monitoring

    Returns:
        StateGraph: Compiled workflow for knowledge processing

    Workflow Stages:
        1. Query Formatting: Optimize question for knowledge retrieval
        2. Knowledge Retrieval: Get expert-level response via MCP tools
        3. Answer Formatting: Structure response for optimal user experience

    Example:
        >>> workflow = create_general_knowledge_workflow()
        >>> result = await workflow.ainvoke({
        ...     "query": "How does ink coverage affect media requirements?",
        ...     "session": mcp_session
        ... })
        >>> print(result["final_answer"])
    """

    async def format_knowledge_query(state: GeneralKnowledgeState):
        """
        Format and optimize the user's question for better knowledge retrieval.

        This node enhances educational questions by adding context, clarifying
        technical terms, and structuring queries for optimal knowledge system response.

        Args:
            state (GeneralKnowledgeState): Current workflow state

        Returns:
            Dict[str, str]: State update with formatted_query
        """
        prompt = (
            "Reformat this printing-related question to be clear and comprehensive "
            "for educational response generation."
        )
        result = await llm.ainvoke([prompt, state["query"]])
        return {
            "formatted_query": str(result.content)
            if hasattr(result, "content")
            else str(result)
        }

    async def get_knowledge_response(state: GeneralKnowledgeState):
        """
        Retrieve comprehensive knowledge response using the MCP knowledge tool.

        This node calls the specialized general knowledge tool to get expert-level
        educational responses about printing concepts and technologies.

        Args:
            state (GeneralKnowledgeState): Current workflow state

        Returns:
            Dict[str, str]: State update with knowledge_response
        """
        result = await state["session"].call_tool(
            "answer_general_question", {"query": state["formatted_query"]}
        )

        # Extract the response from the MCP tool result
        raw_response = (
            result.content[0].text
            if hasattr(result, "content") and result.content
            else str(result)
        )

        return {"knowledge_response": raw_response}

    async def format_final_answer(state: GeneralKnowledgeState):
        """
        Format the final answer for optimal user presentation.

        This node processes the raw knowledge response to create a well-structured,
        comprehensive answer that's easy to read and understand.

        Args:
            state (GeneralKnowledgeState): Current workflow state

        Returns:
            Dict[str, str]: State update with final_answer
        """
        try:
            # Try to parse the JSON response
            knowledge_data = json.loads(state["knowledge_response"])

            # Format as a comprehensive response
            formatted_answer = f"""
**Answer:**
{knowledge_data.get('answer', 'No answer provided')}

**Key Points:**
{chr(10).join([f"• {point}" for point in knowledge_data.get('key_points', [])])}

**Technical Details:**
{knowledge_data.get('technical_details', 'No technical details provided')}

**Best Practices:**
{knowledge_data.get('best_practices', 'No best practices provided')}

**Related Concepts:**
{', '.join(knowledge_data.get('related_concepts', []))}
"""

            return {"final_answer": formatted_answer}

        except json.JSONDecodeError:
            # If not JSON, use the raw response
            return {"final_answer": state["knowledge_response"]}
        except Exception as e:
            return {"final_answer": f"Error formatting response: {str(e)}"}

    workflow = StateGraph(GeneralKnowledgeState)
    workflow.add_node("format_knowledge_query", format_knowledge_query)
    workflow.add_node("get_knowledge_response", get_knowledge_response)
    workflow.add_node("format_final_answer", format_final_answer)

    workflow.add_edge(START, "format_knowledge_query")
    workflow.add_edge("format_knowledge_query", "get_knowledge_response")
    workflow.add_edge("get_knowledge_response", "format_final_answer")
    workflow.add_edge("format_final_answer", END)

    compiled = workflow.compile()
    if callbacks:
        compiled.callbacks = callbacks
    return compiled
