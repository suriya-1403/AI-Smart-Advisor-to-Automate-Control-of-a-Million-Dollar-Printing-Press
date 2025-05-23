"""
Tools for general knowledge responses about printing concepts.
"""

import re
from typing import Any, Dict

from langchain_groq import ChatGroq
from mcp.server.fastmcp import FastMCP

from mcp_server.config import GROQ_API, LLM_MODEL


def clean_response(response_text: str) -> str:
    """
    Clean up a knowledge response by removing duplicate or generic sections.
    
    Args:
        response_text: Raw response text from the LLM.
        
    Returns:
        Cleaned response text.
    """
    # Remove any "Knowledge Response" header that might be present
    response_text = re.sub(r'^Knowledge Response\s*\n+', '', response_text, flags=re.IGNORECASE)
    
    # Remove the generic fallback sections at the end if they appear after proper content
    fallback_pattern = r'\*\*Key Points:\*\*\s*•\s*Comprehensive explanation provided[\s\S]*$'
    response_text = re.sub(fallback_pattern, '', response_text)
    
    # Clean up any "**Answer:**" prefix if present
    response_text = re.sub(r'^\s*\*\*Answer:\*\*\s*', '', response_text)
    
    # Clean up extra whitespace
    response_text = re.sub(r'\n{3,}', '\n\n', response_text)
    
    return response_text.strip()


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
            Comprehensive educational response with proper formatting.
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
        Do not include any generic placeholders like "Comprehensive explanation provided" or "See main answer for technical details".
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