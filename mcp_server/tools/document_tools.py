"""
Tools for document search and management in the MCP Server.

This module implements advanced document search capabilities using BM25 ranking
algorithm combined with intelligent filtering and LLM-enhanced query processing.
It manages JSON printing event documents and provides sophisticated search
functionality for finding relevant printing records.

Key Features:
- BM25 text ranking algorithm for relevance scoring
- Condition-based filtering with structured parameters
- GSM (media weight) proximity reranking
- LLM-enhanced condition extraction from natural language
- Comprehensive result formatting and presentation

Author: Suriyakrishnan Sathish & Rujula More
Version: 1.0.0
Last Modified: 2024
"""

import json
import math
import os
import re
from collections import Counter

import pandas as pd

# from langchain_ollama import OllamaLLM
from langchain_groq import ChatGroq
from mcp.server.fastmcp import FastMCP

from mcp_server.config import DOCUMENTS_DIR, GROQ_API, LLM_MODEL

llm = ChatGroq(model=LLM_MODEL, api_key=GROQ_API)


def load_json_data(file_path):
    """
    Load a JSON document from file with comprehensive error handling.

    This function safely loads JSON documents and handles various file
    system and parsing errors that might occur during document loading.

    Args:
        file_path (str): Absolute path to the JSON file to load

    Returns:
        Optional[Dict[str, Any]]: Loaded JSON data or None if loading failed

    Raises:
        None: All exceptions are caught and logged, returns None on failure

    Example:
        >>> data = load_json_data("/path/to/event_42.json")
        >>> if data:
        ...     print(f"Event ID: {data.get('event_id')}")
        ... else:
        ...     print("Failed to load document")
    """
    try:
        with open(file_path, "r") as file:
            return json.load(file)
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        return None


def create_document_text(record):
    """
    Convert JSON record to searchable text representation.

    This function creates a comprehensive text representation of a printing
    event record that can be effectively searched using text-based algorithms.

    Args:
        record (Dict[str, Any]): Document record containing printing event data

    Returns:
        str: Formatted searchable text representation

    Example:
        >>> record = {
        ...     "event_id": "42",
        ...     "location": "Las Vegas Expo",
        ...     "Ink Coverage": "Heavy",
        ...     "Media Weight GSM": "250"
        ... }
        >>> text = create_document_text(record)
        >>> print(text)
        'Event 42 on unknown at Las Vegas Expo. Printing with Heavy ink coverage...'
    """
    return (
        f"Event {record.get('event_id', 'unknown')} on {record.get('publish_date', 'unknown')} "
        f"at {record.get('location', 'unknown')}. "
        f"Printing with {record.get('Ink Coverage', 'unknown')} ink coverage on "
        f"{record.get('Media Weight GSM', 'unknown')} GSM "
        f"{record.get('Media Coating', 'unknown')} paper with {record.get('Media Finish', 'unknown')} finish "
        f"using {record.get('Press Model', 'unknown')}."
    )


def tokenize(text):
    """
    Convert text to tokens for BM25 processing.

    This function performs text preprocessing including normalization,
    punctuation removal, and tokenization for search algorithms.

    Args:
        text (str): Input text to tokenize

    Returns:
        List[str]: List of normalized tokens

    Example:
        >>> tokens = tokenize("Heavy ink coverage on 250 GSM paper")
        >>> print(tokens)
        ['heavy', 'ink', 'coverage', 'on', '250', 'gsm', 'paper']
    """
    return re.sub(r"[^\w\s]", " ", text.lower()).split()


def extract_gsm(gsm_value):
    """
    Extract GSM (media weight) value from various input formats.

    This function handles different representations of GSM values including
    strings with units, plain numbers, and mixed formats.

    Args:
        gsm_value (Union[str, int, float]): GSM value in various formats

    Returns:
        Optional[int]: Extracted integer GSM value or None if extraction failed

    Example:
        >>> extract_gsm("250 GSM")
        250
        >>> extract_gsm(175.5)
        175
        >>> extract_gsm("unknown")
        None
    """
    if isinstance(gsm_value, str):
        match = re.search(r"\d+", gsm_value)
        return int(match.group()) if match else None
    elif isinstance(gsm_value, (int, float)):
        return int(gsm_value)
    return None


def is_strict_match(record, conditions):
    """
    Check if a record matches all specified conditions exactly.

    This function performs strict matching against all provided conditions,
    requiring exact matches for categorical data and range matching for
    numerical data where appropriate.

    Args:
        record (Dict[str, Any]): Document record to check
        conditions (Dict[str, Any]): Dictionary of conditions to match

    Returns:
        bool: True if record matches all conditions, False otherwise

    Example:
        >>> record = {"Media Coating": "Coated", "Ink Coverage": "Heavy"}
        >>> conditions = {"Media Coating": "Coated"}
        >>> is_strict_match(record, conditions)
        True
    """
    for key, expected in conditions.items():
        value = record.get(key)
        if value is None:
            return False
        value_str = str(value).strip().lower()

        if isinstance(expected, list):
            if all(str(val).strip().lower() != value_str for val in expected):
                return False
        else:
            if str(expected).strip().lower() != value_str:
                return False
    return True


def extract_conditions_with_llm(query: str) -> dict:
    """
    Extract structured printing conditions from a user query using LLM.

    This function uses a language model to parse natural language queries
    and extract structured printing parameters that can be used for filtering.

    Args:
        query (str): User query string containing printing specifications

    Returns:
        Dict[str, Any]: Dictionary containing extracted printing conditions

    Raises:
        Exception: If LLM service is unavailable or response parsing fails

    Example:
        >>> conditions = extract_conditions_with_llm(
        ...     "Find heavy ink coverage on glossy coated media"
        ... )
        >>> print(conditions)
        {'Ink Coverage': 'Heavy', 'Media Coating': 'Coated', 'Media Finish': 'Glossy'}
    """
    prompt = f"""You are an assistant that extracts structured printing conditions from user queries.
     From the given query, identify any of the following conditions (if present):
    - Ink Coverage: Heavy, Medium, or Light
    - Media Coating: Coated or Uncoated
    - Media Finish: Glossy or Matte
    - Press Model: HP T250 or HP T490

    Only include fields mentioned or implied in the query. Return a JSON object.
    Query: "{query}"
    JSON: """

    output = llm.invoke(prompt)

    print(f"DEBUG: LLM output: {output}")
    print(f"DEBUG: LLM output type: {type(output)}")

    # Handle AIMessage response format
    if hasattr(output, "content"):
        content_str = str(output.content)
    else:
        content_str = str(output)

    print(f"DEBUG: Extracted content string: {content_str}")

    # Clean up the content string to extract only the JSON
    content_str = content_str.strip()

    # Remove code block markers if present
    if content_str.startswith("```"):
        # Find the end of the starting marker
        start_idx = content_str.find("\n")
        if start_idx > 0:
            # Find the end marker
            end_idx = content_str.rfind("```")
            if end_idx > start_idx:
                # Extract just the content between markers
                content_str = content_str[start_idx + 1 : end_idx].strip()
            else:
                # Only remove the starting marker
                content_str = content_str[start_idx + 1 :].strip()

    print(f"DEBUG: Cleaned content string: {content_str}")

    # Try to parse JSON from the cleaned content string
    try:
        extracted = json.loads(content_str)
        print(f"DEBUG: Successfully parsed JSON: {extracted}")
        return extracted
    except json.JSONDecodeError as json_err:
        print(f"ERROR: Failed to parse JSON: {json_err}")

        # Try to extract JSON using regex as fallback
        import re

        json_match = re.search(r"(\{.*\})", content_str, re.DOTALL)
        if json_match:
            try:
                json_str = json_match.group(1)
                print(f"DEBUG: Matched JSON string: {json_str}")
                extracted = json.loads(json_str)
                print(f"DEBUG: Successfully parsed JSON with regex: {extracted}")
                return extracted
            except json.JSONDecodeError:
                print("ERROR: Failed to parse JSON even with regex")

    # try:
    #     extracted = json.loads(output)
    #     return extracted
    # except json.JSONDecodeError:
    #     print("LLM response was not valid JSON:", output)
    #     return {}


class BM25:
    """
    BM25 ranking algorithm implementation for document relevance scoring.

    BM25 (Best Matching 25) is a probabilistic ranking function used to
    estimate the relevance of documents to a given search query. This
    implementation includes optimizations for printing document search.

    Attributes:
        corpus (List[List[str]]): Tokenized documents corpus
        k1 (float): Term frequency scaling parameter (default: 1.5)
        b (float): Document length normalization parameter (default: 0.75)
        corpus_size (int): Number of documents in corpus
        avg_doc_len (float): Average document length in tokens
        doc_freqs (List[Counter]): Term frequencies for each document
        idf (Dict[str, float]): Inverse document frequency for each term
        doc_lens (List[int]): Length of each document in tokens
    """

    def __init__(self, corpus, k1=1.5, b=0.75):
        """
        Initialize BM25 with a tokenized corpus.

        Args:
            corpus (List[List[str]]): List of tokenized documents
            k1 (float): Controls term frequency saturation (default: 1.5)
            b (float): Controls document length normalization (default: 0.75)

        Example:
            >>> tokenized_docs = [['heavy', 'ink'], ['light', 'coverage']]
            >>> bm25 = BM25(tokenized_docs)
        """
        self.corpus = corpus
        self.k1 = k1
        self.b = b
        self.corpus_size = len(corpus)
        self.avg_doc_len = (
            sum(len(doc) for doc in corpus) / self.corpus_size if corpus else 0
        )
        self.doc_freqs = []
        self.idf = {}
        self.doc_lens = []
        self._initialize()

    def _initialize(self):
        """
        Initialize document frequencies and IDF (Inverse Document Frequency) values.

        This method computes term frequencies for each document and calculates
        IDF values for all terms in the corpus.
        """
        for doc in self.corpus:
            self.doc_lens.append(len(doc))
            freqs = Counter(doc)
            self.doc_freqs.append(freqs)
            for term in freqs:
                self.idf[term] = self.idf.get(term, 0) + 1

        for term, df in self.idf.items():
            self.idf[term] = math.log((self.corpus_size - df + 0.5) / (df + 0.5) + 1)

    def get_scores(self, query):
        """
        Calculate BM25 scores for all documents given a query.

        This method computes relevance scores for each document in the corpus
        based on the provided query terms using the BM25 algorithm.

        Args:
            query (List[str]): Tokenized query terms

        Returns:
            List[float]: BM25 scores for each document (same order as corpus)

        Example:
            >>> query_tokens = ['heavy', 'ink', 'coverage']
            >>> scores = bm25.get_scores(query_tokens)
            >>> best_doc_index = scores.index(max(scores))
        """
        scores = [0] * self.corpus_size
        for term in query:
            if term not in self.idf:
                continue
            for i, freqs in enumerate(self.doc_freqs):
                if term in freqs:
                    tf = freqs[term]
                    doc_len = self.doc_lens[i]
                    scores[i] += self.idf[term] * (
                        (tf * (self.k1 + 1))
                        / (
                            tf
                            + self.k1
                            * (1 - self.b + self.b * doc_len / self.avg_doc_len)
                        )
                    )
        return scores


# Load documents at module level for reuse
documents = []
document_records = []

if os.path.exists(DOCUMENTS_DIR):
    json_files = [f for f in os.listdir(DOCUMENTS_DIR) if f.endswith(".json")]
    for file_name in json_files:
        file_path = os.path.join(DOCUMENTS_DIR, file_name)
        record = load_json_data(file_path)
        if record:
            document_text = create_document_text(record)
            documents.append(
                {
                    "id": record.get("event_id", file_name),
                    "text": document_text,
                    "record": record,
                }
            )
            document_records.append(record)

# Create document dataframe
df = (
    pd.DataFrame(documents)
    if documents
    else pd.DataFrame(columns=["id", "text", "record"])
)

# Tokenize corpus for BM25
tokenized_corpus = [tokenize(doc["text"]) for doc in documents]
bm25 = BM25(tokenized_corpus) if tokenized_corpus else None


def setup_document_tools(mcp: FastMCP):
    """
    Set up document search tools for the MCP server.

    This function registers all document-related tools and resources with the
    MCP server, including search functionality and document management tools.

    Args:
        mcp (FastMCP): FastMCP server instance to register tools with

    Raises:
        Exception: If tool registration fails or documents cannot be loaded

    Example:
        >>> server = FastMCP("PrintSystem")
        >>> setup_document_tools(server)
        >>> print("Document tools registered successfully")
    """
    # Simple BM25 cache to store filtered BM25 instances
    bm25_cache = {}
    print(f"DEBUG: Loaded {len(documents)} documents from {DOCUMENTS_DIR}")
    if documents:
        print(f"DEBUG: First document ID: {documents[0]['id']}")
        print(
            f"DEBUG: First document GSM: {documents[0]['record'].get('Media Weight GSM')}"
        )
        print(
            f"DEBUG: First document Ink Coverage: {documents[0]['record'].get('Ink Coverage')}"
        )
        print(
            f"DEBUG: First document Media Coating: {documents[0]['record'].get('Media Coating')}"
        )
    else:
        print(f"DEBUG: No documents loaded from {DOCUMENTS_DIR}")
        print(f"DEBUG: Directory exists: {os.path.exists(DOCUMENTS_DIR)}")
        print(
            f"DEBUG: JSON files found: {[f for f in os.listdir(DOCUMENTS_DIR) if f.endswith('.json')]}"
        )

    @mcp.tool()
    def find_document(
        query: str, target_gsm: int = None, conditions: dict = None
    ) -> str:
        """
        Find relevant documents using BM25 ranking with optional filtering and reranking.

        This tool provides sophisticated document search capabilities including:
        - BM25 relevance scoring for text matching
        - Condition-based filtering for structured parameters
        - GSM proximity reranking for media weight optimization
        - LLM-enhanced condition extraction from natural language

        Args:
            query (str): Natural language search query
            target_gsm (Optional[int]): Target GSM value for proximity reranking
            conditions (Optional[Dict[str, Any]]): Explicit filtering conditions

        Returns:
            str: Formatted search results with document details

        Example Usage:
            >>> result = find_document("heavy ink coverage on glossy media")
            >>> print(result)

        Query Examples:
            - "Find prints with heavy ink coverage on glossy media"
            - "Show me T490 printer events using 250 GSM paper"
            - "Las Vegas expo printing events with coated media"
        """
        if not documents:
            return "No documents available for search"

        # overall_start = time.time()
        print(f"DEBUG: Search query: '{query}'")
        print(f"DEBUG: Documents available: {len(documents)}")

        # Step 1: Parse target GSM from query if not provided
        if target_gsm is None and "gsm" in query.lower():
            gsm_match = re.search(r"(\d+)\s*gsm", query.lower())
            if gsm_match:
                target_gsm = int(gsm_match.group(1))

        # Step 2: Extract conditions
        extracted_conditions = {}
        if conditions is None:
            extracted_conditions = extract_conditions_with_llm(query)
        else:
            extracted_conditions = conditions

        # Create a cache key based on extracted conditions
        conditions_key = frozenset(extracted_conditions.items())

        # Step 3: Strict Filtering
        filtered_docs = [
            doc
            for doc in documents
            if is_strict_match(doc["record"], extracted_conditions)
        ]

        # Step 4: BM25 Scoring

        if filtered_docs:
            if conditions_key in bm25_cache:
                bm25_filtered = bm25_cache[conditions_key]
            else:
                tokenized_filtered = [tokenize(doc["text"]) for doc in filtered_docs]
                bm25_filtered = BM25(tokenized_filtered)
                bm25_cache[conditions_key] = bm25_filtered

            scores = bm25_filtered.get_scores(tokenize(query))
            results = [
                {"document": filtered_docs[i], "score": s} for i, s in enumerate(scores)
            ]
        else:
            # Fallback to global search
            scores = bm25.get_scores(tokenize(query))
            results = [
                {"document": documents[i], "score": s} for i, s in enumerate(scores)
            ]

        # Step 5: GSM Reranking
        if target_gsm:
            results = sorted(
                results,
                key=lambda res: abs(
                    extract_gsm(res["document"]["record"].get("Media Weight GSM", 0))
                    - target_gsm
                )
                if extract_gsm(res["document"]["record"].get("Media Weight GSM", 0))
                is not None
                else float("inf"),
            )

        # Step 6: Format Results
        top_results = results[:5]
        if not top_results or all(res["score"] == 0 for res in top_results):
            return "No matching documents found"

        output = [f"Found {len(top_results)} document(s) matching your query:"]
        for i, res in enumerate(top_results):
            doc = res["document"]
            output.append(f"\nResult {i + 1}:")
            output.append(f"Event ID: {doc['record'].get('event_id', 'Unknown')}")
            output.append(f"Date: {doc['record'].get('publish_date', 'Unknown')}")
            output.append(f"Location: {doc['record'].get('location', 'Unknown')}")
            output.append(f"Press Model: {doc['record'].get('Press Model', 'Unknown')}")
            output.append(
                f"Ink Coverage: {doc['record'].get('Ink Coverage', 'Unknown')}"
            )
            output.append(
                f"Media: {doc['record'].get('Media Weight GSM', 'Unknown')} GSM, "
                + f"{doc['record'].get('Media Coating', 'Unknown')}, "
                + f"{doc['record'].get('Media Finish', 'Unknown')}"
            )
            if i < len(top_results) - 1:
                output.append("---")

        return "\n".join(output)

    @mcp.resource("documents://list")
    def list_documents() -> str:
        """
        List all available documents that can be searched.

        This resource provides an overview of all loaded printing event documents
        including basic metadata for each document.

        Returns:
            str: Formatted list of available documents

        Example:
            >>> docs = list_documents()
            >>> print(docs)
        """
        if not documents:
            return "No documents available"

        output = ["Available Print Events:"]
        for i, doc in enumerate(documents):
            record = doc["record"]
            output.append(
                f"{i + 1}. Event {record.get('event_id', 'Unknown')} - "
                + f"{record.get('Press Model', 'Unknown')} at {record.get('location', 'Unknown')}"
            )

        return "\n".join(output)

    @mcp.resource("documents://parameters")
    def list_parameters() -> str:
        """
        List all available parameters that can be used for filtering documents.

        This resource shows all unique parameter values found in the document
        corpus, which can be used for targeted searches and filtering.

        Returns:
            str: Formatted list of searchable parameters and their values

        Example:
            >>> params = list_parameters()
            >>> print(params)
        """
        param_values = {}

        for record in document_records:
            for key, value in record.items():
                if key not in param_values:
                    param_values[key] = set()
                param_values[key].add(str(value))

        output = ["Available Parameters for Filtering:"]
        for param, values in param_values.items():
            if len(values) < 10:  # Only show reasonable number of values
                output.append(f"{param}: {', '.join(sorted(values))}")
            else:
                output.append(f"{param}: {len(values)} unique values")

        return "\n".join(output)
