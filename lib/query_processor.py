"""
Query processing for the RAG Chatbot.
"""

import json
import re
from typing import Any, Dict, List, Tuple

from langchain_ollama import OllamaLLM
from sentence_transformers import SentenceTransformer

from constants.constants import QUESTION_PATTERNS, SEARCH_PATTERNS
from log import logger


class QueryProcessor:
    """
    Handles processing and classification of user queries.

    This class is responsible for determining query intent,
    expanding queries into structured filters, and generating
    embeddings for vector search.
    """

    def __init__(self, embedding_model: SentenceTransformer, llm: OllamaLLM):
        """Initialize the query processor."""
        self.embedding_model = embedding_model
        self.llm = llm

    def classify_query(self, user_query: str) -> Tuple[str, str]:
        """
        Determine if the query is a search or a question.

        Args:
            user_query: The user's input query

        Returns:
            Tuple of (query_type, processed_query) where query_type is
            either "search" or "question"
        """
        logger.debug(f"Classifying query: '{user_query}'")
        # Check if any search pattern matches
        for pattern in SEARCH_PATTERNS:
            if re.search(pattern, user_query.lower(), re.IGNORECASE):
                logger.debug(f"Query matched search pattern: {pattern}")
                return "search", user_query

        # Check if any question pattern matches
        for pattern in QUESTION_PATTERNS:
            if re.search(pattern, user_query.lower(), re.IGNORECASE):
                logger.debug(f"Query matched question pattern: {pattern}")
                return "question", user_query

        # If it contains a filename, likely asking about a specific file
        if ".json" in user_query or ".pdf" in user_query:
            logger.debug("Query contains filename, treating as question")
            return "question", user_query

        # Default to search if we can't determine
        logger.debug("Query classification defaulted to search")
        return "search", user_query

    def expand_query(self, user_query: str) -> Dict[str, Any]:
        """
        Extract structured filters from user query using LLM.

        Args:
            user_query: The user's input query

        Returns:
            Dictionary of extracted parameters and values
        """
        logger.debug(f"Expanding query: '{user_query}'")
        prompt = f"""
        Analyze this user query: "{user_query}"

        The user is searching for printing documents with specific characteristics.
        Based ONLY on what's explicitly mentioned in the query, extract these parameters:

        1. Ink Coverage: [Painted, Heavy, Medium, Light]
        2. Media Weight GSM: [numerical value]
        3. Media Coating: [Coated, Uncoated]
        4. Media Finish: [Silk, Matte, Gloss]

        Format your response EXACTLY like this example:
        {{
            "Ink Coverage": "...",
            "Media Weight GSM": ..,
            "Media Coating": "..",
            "Media Finish": "..."
        }}

        Only include parameters that are EXPLICITLY mentioned in the query.
        If a parameter is not mentioned, DO NOT include it in the JSON.
        The response must be ONLY the JSON object - no other text, explanations or code blocks.
        """

        response = self.llm.invoke(prompt)
        logger.debug("Got structured query response from LLM")

        try:
            # Clean up the response - remove any markdown and extra text
            cleaned_response = response.strip()

            # Try to find JSON between curly braces
            json_start = cleaned_response.find("{")
            json_end = cleaned_response.rfind("}") + 1

            if json_start != -1 and json_end != -1:
                json_text = cleaned_response[json_start:json_end]
                structured_query = json.loads(json_text)
                logger.debug(
                    f"Successfully parsed query parameters: {structured_query}"
                )
            else:
                # If no curly braces found, try removing markdown blocks
                if "```json" in cleaned_response and "```" in cleaned_response:
                    json_start = cleaned_response.find("```json") + 7
                    json_end = cleaned_response.rfind("```")
                    json_text = cleaned_response[json_start:json_end].strip()
                    structured_query = json.loads(json_text)
                    logger.debug(
                        f"Successfully parsed query parameters: {structured_query}"
                    )
                else:
                    # Last resort: try the whole response
                    structured_query = json.loads(cleaned_response)
                    logger.debug(
                        f"Successfully parsed query parameters with errors: {structured_query}"
                    )

        except json.JSONDecodeError:
            logger.warning(
                "Failed to parse LLM response as JSON, using manual extraction"
            )
            # Create a basic structured query by simple text analysis
            structured_query = {}

            # Manual extraction
            if "heavy" in user_query.lower():
                structured_query["Ink Coverage"] = "Heavy"
            elif "medium" in user_query.lower():
                structured_query["Ink Coverage"] = "Medium"
            elif "light" in user_query.lower():
                structured_query["Ink Coverage"] = "Light"

            if "coated" in user_query.lower():
                structured_query["Media Coating"] = "Coated"
            elif "uncoated" in user_query.lower():
                structured_query["Media Coating"] = "Uncoated"

            if "silk" in user_query.lower():
                structured_query["Media Finish"] = "Silk"
            elif "matte" in user_query.lower():
                structured_query["Media Finish"] = "Matte"
            elif "gloss" in user_query.lower():
                structured_query["Media Finish"] = "Gloss"

            # Try to extract GSM value
            gsm_match = re.search(r"(\d+)\s*gsm", user_query.lower())
            if gsm_match:
                structured_query["Media Weight GSM"] = int(gsm_match.group(1))
            logger.debug(f"Manually extracted parameters: {structured_query}")
        return structured_query

    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a text string."""
        return self.embedding_model.encode(text).tolist()
