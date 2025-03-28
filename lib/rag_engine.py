"""
RAG Engine for the RAG Chatbot.
"""

import re
import json
import streamlit as st
from typing import Dict, List, Any

from lib.vector_db import VectorDBManager
from lib.query_processor import QueryProcessor
from lib.document_scorer import DocumentScorer
from config import MAX_RAG_CONTEXT, MAX_SEARCH_RESULTS
from log import logger


class RAGEngine:
    """
    Handles Retrieval Augmented Generation operations.

    This class coordinates the retrieval of documents and the
    generation of answers using the language model.
    """

    def __init__(
            self,
            vector_db: VectorDBManager,
            query_processor: QueryProcessor,
            document_scorer: DocumentScorer,
            llm
    ):
        """Initialize the RAG Engine."""
        self.vector_db = vector_db
        self.query_processor = query_processor
        self.document_scorer = document_scorer
        self.llm = llm

    def answer_question(self, user_query: str) -> str:
        """
        Generate an answer to a user question using RAG.

        Args:
            user_query: The user's question

        Returns:
            String containing the generated answer
        """
        # Generate query embedding
        query_embedding = self.query_processor.generate_embedding(user_query)

        try:
            # Check for special cases (direct lookups)
            file_match = re.search(r'["\']?([^"\']+\.json)["\']?', user_query)
            event_id_match = re.search(r'event\s+id\s*[:\s]\s*([a-zA-Z0-9-_]+)', user_query, re.IGNORECASE)

            if file_match:
                filename = file_match.group(1)
                st.info(f"Looking for information about file: {filename}")

                # Query by filename
                results = self.vector_db.query_by_embedding(
                    query_embedding,
                    where_clause={"filename": filename},
                    n_results=MAX_RAG_CONTEXT
                )
            elif event_id_match:
                event_id = event_id_match.group(1)
                st.info(f"Looking for information about event ID: {event_id}")

                # Query by event ID
                results = self.vector_db.query_by_embedding(
                    query_embedding,
                    where_clause={"event_id": event_id},
                    n_results=MAX_RAG_CONTEXT
                )
            else:
                # Standard semantic search
                results = self.vector_db.query_by_embedding(
                    query_embedding,
                    n_results=MAX_RAG_CONTEXT
                )

            # Check if we found any documents
            if not results["documents"] or not results["documents"][0]:
                return "I couldn't find specific information about that document or configuration. Could you try rephrasing your question?"

            # Build context from retrieved documents with their metadata
            context_sections = []

            for i in range(min(MAX_RAG_CONTEXT, len(results["documents"][0]))):
                if i < len(results["documents"][0]) and i < len(results["metadatas"][0]):
                    document_text = results["documents"][0][i]
                    metadata = results["metadatas"][0][i]

                    context_sections.append(f"Document {i + 1}:\n{document_text}")
                    context_sections.append(f"Document {i + 1} Metadata:\n{json.dumps(metadata, indent=2)}")

            # Join all context sections
            full_context = "\n\n".join(context_sections)

            # Create RAG prompt
            rag_prompt = f"""
            Given the following context information retrieved from a document database:

            {full_context}

            Answer this question from the user: "{user_query}"

            Provide a detailed and helpful response focusing on the printing configuration and settings.
            Include specific details about ink coverage, media weight, coating, finish, press model, etc.
            Format your answer in a clear, easy-to-read way.
            Base your answer on the provided context. If certain information isn't available in the data, mention that.
            If the context doesn't contain relevant information to answer the question, please state that.
            """

            # Generate response using LLM with retrieved context
            response = self.llm.invoke(rag_prompt)
            return response

        except Exception as e:
            logger.error(f"Error in RAG pipeline: {e}")
            return f"Sorry, I encountered an error while trying to answer your question: {str(e)}"

    def search_documents(self, user_query: str) -> List[Dict[str, Any]]:
        """
        Search for documents based on the user query.

        Args:
            user_query: The user's search query

        Returns:
            List of document dictionaries matching the query
        """
        # Expand query to get structured filters
        query_filters = self.query_processor.expand_query(user_query)

        # Generate embedding for semantic search
        query_embedding = self.query_processor.generate_embedding(user_query)

        # Show extracted parameters
        st.write(f"📊 **Extracted Search Parameters:**")
        for key, value in query_filters.items():
            st.write(f"- {key}: {value}")

        # Build where clause for metadata filtering
        where_clause = {"$and": []}

        # Only add filters that were extracted from the query
        if "Ink Coverage" in query_filters and query_filters["Ink Coverage"]:
            where_clause["$and"].append({"Ink Coverage": query_filters["Ink Coverage"]})

        if "Media Coating" in query_filters and query_filters["Media Coating"]:
            where_clause["$and"].append({"Media Coating": query_filters["Media Coating"]})

        if "Media Finish" in query_filters and query_filters["Media Finish"]:
            where_clause["$and"].append({"Media Finish": query_filters["Media Finish"]})

        # Handle location separately since it might be part of a string
        if "location" in query_filters and query_filters["location"]:
            where_clause["$and"].append({"location": query_filters["location"]})

        try:
            # If no filters were extracted, perform a pure semantic search
            if not where_clause["$and"]:
                st.info("ℹ️ Using semantic search without filters")
                query_results = self.vector_db.query_by_embedding(
                    query_embedding,
                    n_results=MAX_SEARCH_RESULTS
                )
            else:
                # Apply filters with semantic search
                st.info("ℹ️ Using semantic search with filters")
                query_results = self.vector_db.query_by_embedding(
                    query_embedding,
                    where_clause=where_clause,
                    n_results=MAX_SEARCH_RESULTS
                )

            # Get the documents retrieved by vector similarity
            retrieved_docs = query_results["metadatas"][0] if query_results["metadatas"] else []

            if retrieved_docs:
                # Rank documents based on relevance to query
                ranked_docs = self.document_scorer.rank_documents(retrieved_docs, user_query)
                return ranked_docs

            return []

        except Exception as e:
            logger.error(f"❌ Error during search: {e}")
            # Fallback to simpler search
            try:
                st.info("ℹ️ Trying fallback search")
                query_results = self.vector_db.query_by_embedding(query_embedding, n_results=5)
                return query_results["metadatas"][0] if query_results["metadatas"] else []
            except Exception as fallback_error:
                logger.error(f"❌ Fallback search failed: {fallback_error}")
                return []

    def analyze_documents(self, user_query: str, documents: List[Dict[str, Any]]) -> str:
        """
        Generate an analysis of the top retrieved documents.

        Args:
            user_query: The user's search query
            documents: List of document dictionaries to analyze

        Returns:
            String containing the analysis
        """
        docs_json = json.dumps(documents[:3], indent=2)
        analysis_prompt = f"""
        Based on the user query: "{user_query}"

        Analyze these top documents that were retrieved by similarity:
        {docs_json}

        For each document, explain in one clear sentence why it's relevant to the query.
        Focus on matching attributes and key information.
        """

        return self.llm.invoke(analysis_prompt)