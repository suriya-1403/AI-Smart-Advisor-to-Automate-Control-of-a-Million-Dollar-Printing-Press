"""
General knowledge tools for HP PageWide printing concepts and educational responses.

This module provides comprehensive educational support for printing concepts,
media types, equipment operations, and HP PageWide technology. It uses advanced
LLM processing to deliver expert-level explanations tailored for technical users.

Key Features:
- Expert-level printing concept explanations
- Technical specification guidance
- Best practices and recommendations
- Troubleshooting and optimization advice
- Comprehensive response formatting and cleanup
- Educational content organization and presentation

Author: AI Smart Advisor Team
Version: 1.0.0
Last Modified: 2024
"""


import re
from typing import Any, Dict

from langchain_groq import ChatGroq
from mcp.server.fastmcp import FastMCP

from mcp_server.config import GROQ_API, LLM_MODEL


def clean_response(response_text: str) -> str:
    """
    Clean up LLM knowledge responses by removing duplicate or generic sections.

    This function post-processes LLM responses to remove common artifacts like
    redundant headers, generic placeholder text, and formatting inconsistencies
    that can occur in automated response generation.

    Args:
        response_text (str): Raw response text from the LLM

    Returns:
        str: Cleaned and formatted response text

    Cleaning Operations:
        - Remove redundant "Knowledge Response" headers
        - Strip generic fallback content
        - Clean up formatting prefixes
        - Normalize whitespace and line breaks

    Example:
        >>> raw_response = "Knowledge Response\\n\\n**Answer:**\\nExplanation here..."
        >>> cleaned = clean_response(raw_response)
        >>> print(cleaned)
        'Explanation here...'
    """
    # Remove any "Knowledge Response" header that might be present
    response_text = re.sub(
        r"^Knowledge Response\s*\n+", "", response_text, flags=re.IGNORECASE
    )

    # Remove the generic fallback sections at the end if they appear after proper content
    fallback_pattern = (
        r"\*\*Key Points:\*\*\s*•\s*Comprehensive explanation provided[\s\S]*$"
    )
    response_text = re.sub(fallback_pattern, "", response_text)

    # Clean up any "**Answer:**" prefix if present
    response_text = re.sub(r"^\s*\*\*Answer:\*\*\s*", "", response_text)

    # Clean up extra whitespace
    response_text = re.sub(r"\n{3,}", "\n\n", response_text)

    return response_text.strip()


def setup_general_knowledge_tools(mcp: FastMCP):
    """
    Set up general knowledge tools for the MCP server.

    This function registers comprehensive educational tools that provide expert-level
    responses about HP PageWide printing concepts, technical specifications, and
    operational guidance.

    Args:
        mcp (FastMCP): FastMCP server instance to register tools with

    Raises:
        Exception: If tool registration fails

    Registered Tools:
        - answer_general_question: Comprehensive educational responses
        - validate_printing_question: Query validation utility
        - get_topic_category: Query categorization utility

    Registered Resources:
        - knowledge://topics: Available knowledge topics
        - knowledge://examples: Example questions
        - knowledge://categories: Topic categorization guide

    Example:
        >>> server = FastMCP("PrintSystem")
        >>> setup_general_knowledge_tools(server)
        >>> print("Knowledge tools registered successfully")
    """

    @mcp.tool()
    def answer_general_question(query: str) -> str:
        """
        Answer general knowledge questions about printing concepts, media types,
        ink coverage, press operations, and HP PageWide technology.

        This tool provides comprehensive educational responses covering all aspects
        of HP PageWide printing technology, from basic concepts to advanced
        technical specifications and optimization strategies.

        Args:
            query (str): User's question about printing concepts

        Returns:
            str: Comprehensive educational response with proper formatting

        Capabilities:
            - Technical concept explanations
            - Equipment operation guidance
            - Media specification details
            - Best practices and recommendations
            - Troubleshooting and optimization advice
            - Comparative analysis and decision support

        Query Examples:
            - "Explain the difference between coated and uncoated media"
            - "How does heavy ink coverage affect media requirements?"
            - "What's the relationship between press speed and dryer power?"
            - "When should I use different quality modes?"
            - "How do winder tensions affect print quality?"
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

        Format your response with these sections:
        1. Start with a comprehensive main answer (no heading needed)
        2. Key Points (bullet list of important points)
        3. Technical Details (technical specifications and information)
        4. Best Practices (recommendations for optimal results)
        5. Related Concepts (bullet list of related topics)

        Make your responses educational, accurate, and practical for print operators and technicians.
        Important:
            - Make responses technically accurate, practically useful, and educational for print operators and technicians.
            - If a question is too general or unrelated to HP PageWide printing (e.g., "How do airplanes work?"), respond with:
            "Sorry, I specialize in HP PageWide printing systems and cannot assist with that topic."
            - Do **not** use generic placeholder phrases like "Comprehensive explanation provided.".
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

            # Clean up the response to remove any duplicate or generic sections
            cleaned_response = clean_response(response_content)

            return cleaned_response

        except Exception as e:
            error_message = f"""
            ## Error Processing Request

            I apologize, but I encountered an error while processing your question: {str(e)}

            ### Troubleshooting
            - Please try rephrasing your question
            - Ensure your question is related to printing concepts
            - Contact technical support if the issue persists
            """
            return error_message

    @mcp.resource("knowledge://topics")
    def list_knowledge_topics() -> str:
        """
        List comprehensive knowledge topics that can be explained by the system.

        This resource provides a detailed overview of all printing concepts,
        technologies, and operational areas covered by the knowledge system.

        Returns:
            str: Organized list of available topics with examples

        Example:
            >>> topics = list_knowledge_topics()
            >>> print(topics)
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
        Get comprehensive example questions organized by topic category.

        This resource provides users with specific examples of questions
        that can be effectively answered by the knowledge system.

        Returns:
            str: Categorized list of example questions

        Example:
            >>> examples = get_example_questions()
            >>> print(examples)
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
