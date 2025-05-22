"""
Tools for general knowledge responses about printing concepts.
"""

import json
from typing import Any, Dict

from langchain_groq import ChatGroq
from mcp.server.fastmcp import FastMCP

from mcp_server.config import GROQ_API, LLM_MODEL


def setup_general_knowledge_tools(mcp: FastMCP):
    """
    Set up general knowledge tools for the MCP server.

    Args:
        mcp: FastMCP server instance.
    """

    @mcp.tool()
    def answer_general_question(query: str) -> str:
        """
        Answer general knowledge questions about printing concepts, media types,
        ink coverage, press operations, and HP PageWide technology.

        Args:
            query: User's question about printing concepts.

        Returns:
            Comprehensive educational response in JSON format.
        """

        llm = ChatGroq(model=LLM_MODEL, api_key=GROQ_API)

        system_prompt = """
        You are an expert HP PageWide printing consultant and educator. Your role is to provide comprehensive, educational responses about printing concepts, media types, ink coverage, press operations, and HP PageWide technology.

        When answering questions, provide detailed explanations that include:
        1. Clear definitions and concepts
        2. Technical details and specifications
        3. Practical applications and real-world examples
        4. Comparisons when relevant
        5. Best practices and recommendations
        6. Impact on print quality and efficiency

        Focus on topics such as:
        - Media types (coated vs uncoated, finishes like matte, silk, gloss)
        - Ink coverage and its effects on media requirements
        - Press speed and dryer power relationships
        - Print quality modes and their applications
        - Media weight (GSM) considerations
        - Dryer configurations and their purposes
        - Moisturizer and surfactant effects
        - Winder systems and tension settings

        Provide your response in JSON format with the following structure:
        {
            "answer": "Main comprehensive answer",
            "key_points": ["Point 1", "Point 2", "Point 3"],
            "technical_details": "Additional technical information",
            "best_practices": "Recommended practices",
            "related_concepts": ["Related concept 1", "Related concept 2"]
        }

        Make your responses educational, accurate, and practical for print operators and technicians.
        """

        user_prompt = f"""
        Please provide a comprehensive educational response to this printing-related question:

        Question: {query}

        Provide detailed information that would help someone understand the concepts involved.
        """

        try:
            result = llm.invoke(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
            )

            response_content = (
                result.content if hasattr(result, "content") else str(result)
            )

            # Try to parse as JSON first
            try:
                parsed_response = json.loads(response_content)
                return json.dumps(parsed_response, indent=2)
            except json.JSONDecodeError:
                # If not valid JSON, create a structured response
                structured_response = {
                    "answer": response_content,
                    "key_points": ["Comprehensive explanation provided"],
                    "technical_details": "See main answer for technical details",
                    "best_practices": "Refer to main answer for best practices",
                    "related_concepts": [
                        "Contact HP support for additional information"
                    ],
                }
                return json.dumps(structured_response, indent=2)

        except Exception as e:
            error_response = {
                "answer": f"I apologize, but I encountered an error while processing your question: {str(e)}",
                "key_points": ["Error occurred during processing"],
                "technical_details": "Please try rephrasing your question",
                "best_practices": "Contact technical support if the issue persists",
                "related_concepts": ["Error handling", "System troubleshooting"],
            }
            return json.dumps(error_response, indent=2)

    @mcp.resource("knowledge://topics")
    def list_knowledge_topics() -> str:
        """
        List available knowledge topics that can be explained.

        Returns:
            List of topics the system can explain.
        """
        topics = [
            "Media Types and Coatings",
            "- Coated vs Uncoated media",
            "- Media finishes (Matte, Silk, Gloss)",
            "- Media weight (GSM) considerations",
            "",
            "Ink and Coverage",
            "- Ink coverage effects on media",
            "- Optical density relationships",
            "- Color management",
            "",
            "Press Operations",
            "- Press speed optimization",
            "- Dryer power and configurations",
            "- Print quality modes",
            "",
            "Equipment and Settings",
            "- Moisturizer systems",
            "- Surfactant types and effects",
            "- Winder systems and tension",
            "- Dryer zone configurations",
            "",
            "Quality and Troubleshooting",
            "- Print quality factors",
            "- Media compatibility",
            "- Performance optimization",
        ]

        return "\n".join(topics)

    @mcp.resource("knowledge://examples")
    def get_example_questions() -> str:
        """
        Get example questions that can be answered.

        Returns:
            List of example questions.
        """
        examples = [
            "How does heavy ink coverage affect media requirements?",
            "Can you explain the difference between coated and uncoated media?",
            "What's the relationship between ink coverage and media weight?",
            "Why would someone choose silk finish over matte?",
            "How do dryer configurations affect print quality?",
            "What are the benefits of using moisturizers in printing?",
            "When should I use different press quality modes?",
            "How does media weight impact press speed?",
            "What's the purpose of different surfactant types?",
            "How do winder tensions affect print quality?",
        ]

        return "\n".join([f"- {example}" for example in examples])
