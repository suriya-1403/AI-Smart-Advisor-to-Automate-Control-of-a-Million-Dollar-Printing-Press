"""
Query router for determining task type.
"""

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import OllamaLLM
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from mcp_server.config import LLM_MODEL


# Define state
class State(TypedDict):
    """
    State type for the query router.
    """

    query: str
    task_type: str
    result: str


def create_router(callbacks=None):
    """
    Create a router graph to determine query type.

    Args:
        callbacks: Callbacks for the graph.

    Returns:
        Compiled router graph.
    """
    # Setup LLM
    llm = OllamaLLM(model=LLM_MODEL)

    # Define classifier node
    def classify_query(state: State):
        """
        Classify if this is a document search or ruleset evaluation.
        """
        system_prompt = """
        You are a classifier that determines whether a user query is about:
        1. Document search: Finding or retrieving documents based on content
        2. Ruleset evaluation: Evaluating printing parameters using rulesets

        Return ONLY the task type as either "document_search" or "ruleset_evaluation".
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

            # Simple classification logic
            if (
                "document" in content_lower
                or "find" in content_lower
                or "search" in content_lower
                or "report" in content_lower
                or "pdf" in content_lower
            ):
                return {"task_type": "document_search"}
            else:
                return {"task_type": "ruleset_evaluation"}
        except Exception as e:
            print(f"Error in classify_query: {str(e)}")
            # Default to document search if classification fails
            return {"task_type": "document_search"}

    # Define graph
    workflow = StateGraph(State)
    workflow.add_node("classifier", classify_query)

    # Connect nodes
    workflow.add_edge(START, "classifier")
    workflow.add_conditional_edges(
        "classifier",
        lambda x: x["task_type"],
        {"document_search": END, "ruleset_evaluation": END},
    )

    # Apply callbacks if provided
    compiled = workflow.compile()
    if callbacks:
        compiled.callbacks = callbacks

    return compiled
