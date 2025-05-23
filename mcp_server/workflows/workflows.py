"""
Workflows for document search and ruleset evaluation.
"""

import json

# from langchain_ollama import OllamaLLM
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph
from langsmith import traceable
from typing_extensions import TypedDict

from mcp_server.config import GROQ_API, LLM_MODEL

# Initialize LLM
# llm = OllamaLLM(model=LLM_MODEL)
llm = ChatGroq(model=LLM_MODEL, api_key=GROQ_API)


class DocSearchState(TypedDict):
    """
    State type for document search workflow.
    """

    query: str
    enhanced_query: str
    document: str
    summary: str
    session: object  # MCP ClientSession


@traceable(name="DocumentSearchWorkflow")
def create_document_workflow(callbacks=None):
    """
    Create a workflow for document search.

    Args:
        callbacks: Callbacks for the graph.

    Returns:
        Compiled workflow graph.
    """

    async def enhance_query(state: DocSearchState):
        """
        Enhance the user's query for better document retrieval.
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
        Search for documents using the enhanced query.
        """
        result = await state["session"].call_tool(
            "find_document", {"query": state["enhanced_query"]}
        )
        return {"document": result.content[0].text}

    async def summarize(state: DocSearchState):
        """
        Summarize the document search results.
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
    State type for ruleset evaluation workflow.
    """

    query: str
    formatted_parameters: str
    evaluation: str
    explanation: str
    session: object  # MCP ClientSession


@traceable(name="RulesetEvaluationWorkflow")
def create_ruleset_workflow(callbacks=None):
    """
    Create a workflow for ruleset evaluation.

    Args:
        callbacks: Callbacks for the graph.

    Returns:
        Compiled workflow graph.
    """

    async def format_parameters(state: RulesetState):
        """
        Format parameters from the user's query.
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
    async def evaluate_ruleset(state: RulesetState):
        """
        Evaluate ruleset using the formatted parameters.
        """
        result = await state["session"].call_tool(
            "evaluate_ruleset", {"query": state["formatted_parameters"]}
        )

        raw_text = result.content[0].text  # TextContent — you must access .text

        try:
            parsed = json.loads(raw_text)  # Only if the tool returns JSON
        except Exception as e:
            parsed = {"report": None, "llm_insights": raw_text}  # fallback

        return {"evaluation": parsed}

    async def explain_results(state: RulesetState):
        """
        Explain the results of the ruleset evaluation.
        """
        # prompt = "Explain the following ruleset evaluation results in simple terms."
        # result = await llm.ainvoke([prompt, state["evaluation"]])
        return {
            # "explanation": str(result.content)
            # if hasattr(result, "content")
            # else str(result)
            "explanation": state["evaluation"]
        }

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
    State type for event information workflow.
    """

    query: str
    event_identifier: str
    found_documents: str
    pdf_content: str
    structured_data: str
    summary: str
    session: object  # MCP ClientSession


@traceable(name="EventInformationWorkflow")
def create_event_workflow(callbacks=None):
    """
    Create a workflow for event information retrieval.

    Args:
        callbacks: Callbacks for the graph.

    Returns:
        Compiled workflow graph.
    """

    async def extract_event_identifier(state: EventInfoState):
        """
        Extract event identifier from the user's query.
        """
        prompt = (
            "Extract the event identifier from this query. Look for event IDs, "
            "event numbers, location names, or other identifying information. "
            "Return just the key identifier (e.g., '71', 'Vegas expo', 'event_128')."
        )
        result = await llm.ainvoke([prompt, state["query"]])
        return {
            "event_identifier": str(result.content)
            if hasattr(result, "content")
            else str(result)
        }

    async def search_event_documents(state: EventInfoState):
        """
        Search for documents related to the specific event.
        """
        result = await state["session"].call_tool(
            "get_event_information", {"query": state["query"]}
        )
        return {"found_documents": result.content[0].text}

    async def extract_pdf_content(state: EventInfoState):
        """
        Extract PDF content if event identifier is a specific event ID.
        """
        # Check if the event identifier looks like an event ID (number)
        if state["event_identifier"].isdigit():
            try:
                result = await state["session"].call_tool(
                    "get_pdf_content", {"event_id": state["event_identifier"]}
                )
                return {"pdf_content": result.content[0].text}
            except Exception as e:
                return {"pdf_content": f"Error retrieving PDF content: {str(e)}"}
        return {"pdf_content": ""}

    async def structure_event_data(state: EventInfoState):
        """
        Structure the event data for analysis.
        """
        combined_data = state["found_documents"]
        
        # Include PDF content if available
        if state["pdf_content"] and "PDF Content for Event" in state["pdf_content"]:
            prompt = (
                "Extract the most important insights from this PDF content. "
                "Focus on key metrics, results, and technical details. "
                "Limit your response to 3-5 key findings."
            )
            result = await llm.ainvoke([prompt, state["pdf_content"]])
            pdf_insights = str(result.content) if hasattr(result, "content") else str(result)
            
            combined_data += f"\n\nPDF Key Insights:\n{pdf_insights}"
            
        prompt = (
            "Structure the following event information into key categories: "
            "event metadata, printing specifications, equipment settings, performance data, "
            "and PDF insights if available. "
            "Extract and organize all relevant details."
        )
        result = await llm.ainvoke([prompt, combined_data])
        return {
            "structured_data": str(result.content)
            if hasattr(result, "content")
            else str(result)
        }

    async def generate_event_summary(state: EventInfoState):
        """
        Generate a comprehensive summary of the event.
        """
        prompt = (
            "Create a comprehensive narrative summary of this printing event. "
            "Include what happened, the printing specifications used, equipment configuration, "
            "and any notable outcomes or results from both JSON data and PDF content if available. "
            "Make it informative and easy to understand."
        )
        result = await llm.ainvoke([prompt, state["structured_data"]])
        return {
            "summary": str(result.content)
            if hasattr(result, "content")
            else str(result)
        }

    workflow = StateGraph(EventInfoState)
    workflow.add_node("extract_event_identifier", extract_event_identifier)
    workflow.add_node("search_event_documents", search_event_documents)
    workflow.add_node("extract_pdf_content", extract_pdf_content)
    workflow.add_node("structure_event_data", structure_event_data)
    workflow.add_node("generate_event_summary", generate_event_summary)

    workflow.add_edge(START, "extract_event_identifier")
    workflow.add_edge("extract_event_identifier", "search_event_documents")
    workflow.add_edge("search_event_documents", "extract_pdf_content")
    workflow.add_edge("extract_pdf_content", "structure_event_data")
    workflow.add_edge("structure_event_data", "generate_event_summary")
    workflow.add_edge("generate_event_summary", END)

    compiled = workflow.compile()
    if callbacks:
        compiled.callbacks = callbacks
    return compiled

class GeneralKnowledgeState(TypedDict):
    """
    State type for general knowledge workflow.
    """

    query: str
    formatted_query: str
    knowledge_response: str
    final_answer: str
    session: object  # MCP ClientSession


@traceable(name="GeneralKnowledgeWorkflow")
def create_general_knowledge_workflow(callbacks=None):
    """
    Create a workflow for general knowledge responses.

    Args:
        callbacks: Callbacks for the graph.

    Returns:
        Compiled workflow graph.
    """

    async def format_knowledge_query(state: GeneralKnowledgeState):
        """
        Format the user's query for better knowledge retrieval.
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
        Get comprehensive knowledge response using the MCP tool.
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
        Format the final answer for user presentation.
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
