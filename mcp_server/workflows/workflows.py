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
llm = ChatGroq(model="mistral-saba-24b", api_key=GROQ_API)


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
