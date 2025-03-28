"""
Main application module for the RAG Chatbot.

This module contains the main Streamlit application and UI components.
"""

import os
import json
import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from langchain_ollama import OllamaLLM

from lib import VectorDBManager, QueryProcessor, DocumentScorer, RAGEngine
from config import (
    DISABLE_CUDA, CHROMA_PATH, COLLECTION_NAME, EMBEDDING_MODEL,
    DEFAULT_LLM, AVAILABLE_MODELS, APP_TITLE, APP_DESCRIPTION
)
from log import logger
from constants import STOP_WORDS

# Disable PyTorch CUDA if configured
if DISABLE_CUDA:
    os.environ["PYTORCH_NO_CUDA"] = "1"

logger.info("Starting RAG Chatbot application")


class RAGChatbotApp:
    """Main RAG Chatbot application class."""

    def __init__(self):
        """Initialize the application."""
        # Configure Streamlit page
        self.configure_streamlit()

        # Initialize components
        self.initialize_components()

        # Set up UI
        self.build_ui()

        # Handle events
        self.handle_events()

    def configure_streamlit(self):
        """Configure Streamlit settings."""
        st.set_page_config(page_title=APP_TITLE, layout="wide")

    def initialize_components(self):
        """Initialize application components."""
        # Vector database
        self.vector_db = VectorDBManager(
            chroma_path=CHROMA_PATH,
            collection_name=COLLECTION_NAME,
            embedding_model_name=EMBEDDING_MODEL
        )

        # Embedding model
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)

        # Get model selection from sidebar (or use default)
        self.model_name = st.session_state.get("model_name", DEFAULT_LLM)

        # Language model
        self.llm = OllamaLLM(model=self.model_name)

        # Query processor
        self.query_processor = QueryProcessor(self.embedding_model, self.llm)

        # Document scorer
        self.document_scorer = DocumentScorer()

        # RAG engine
        self.rag_engine = RAGEngine(
            self.vector_db,
            self.query_processor,
            self.document_scorer,
            self.llm
        )

    def build_ui(self):
        """Build the user interface."""
        # Header
        st.markdown(f"### 📄 {APP_TITLE}")
        st.write(APP_DESCRIPTION)

        # Sidebar
        self.build_sidebar()

        # File Display Section
        self.display_uploaded_files()

        # Only show the Store JSON button if files are uploaded
        if self.uploaded_files:
            if st.button("🚀 Store JSON in Vector Store"):
                self.handle_store_json()

        # Query Input
        st.subheader("🔍 Ask a question:")
        self.user_query = st.text_input("Enter your search query or ask a question:")

        # Debug Options
        self.show_debug = st.checkbox("Show Debug Options")
        if self.show_debug:
            st.subheader("🛠️ Debug Tools")
            if st.button("📊 Show Collection Stats"):
                self.show_collection_stats()

    def build_sidebar(self):
        """Build the sidebar UI."""
        st.sidebar.header("⚙️ Settings")

        # Model Selection
        self.selected_model = st.sidebar.selectbox(
            "Select Model",
            AVAILABLE_MODELS,
            index=AVAILABLE_MODELS.index(self.model_name) if self.model_name in AVAILABLE_MODELS else 0
        )
        # Update model if changed
        if self.selected_model != self.model_name:
            self.model_name = self.selected_model
            st.session_state["model_name"] = self.model_name
            # Reinitialize LLM if needed
            self.llm = OllamaLLM(model=self.model_name)

        # File Type Selection
        self.file_type = st.sidebar.selectbox("Select File Type", ["PDF", "JSON"], index=0)

        # File Upload
        self.uploaded_files = st.sidebar.file_uploader(
            "Upload PDF/JSON files",
            type=["pdf", "json"],
            accept_multiple_files=True
        )

        if self.uploaded_files:
            st.sidebar.success(f"✅ Loaded {len(self.uploaded_files)} file(s) into memory.")

    def display_uploaded_files(self):
        """Display information about uploaded files."""
        st.subheader("📂 Uploaded Files")
        if self.uploaded_files:
            for file in self.uploaded_files:
                st.write(f"📄 {file.name} - {file.size / 1024:.1f} KB ")
        else:
            st.info("No files uploaded yet. Please upload JSON or PDF files.")

    def show_collection_stats(self):
        """Show statistics about the vector collection."""
        try:
            collection_count = self.vector_db.get_collection_count()
            st.write(f"Documents in collection: {collection_count}")
            st.success("ChromaDB is working correctly!")
        except Exception as e:
            st.error(f"ChromaDB error: {str(e)}")

    def handle_events(self):
        """Handle UI events."""
        # Search/Answer button
        if st.button("🔎 Search/Answer"):
            self.handle_search()

    def handle_store_json(self):
        """Handle storing JSON documents in the vector database."""
        if not self.uploaded_files:
            st.warning("⚠️ Please upload JSON files first.")
            return

        json_list = []

        for file in self.uploaded_files:
            try:
                file_content = file.read()
                json_data = json.loads(file_content)
                # Add filename to the data
                json_data["filename"] = file.name
                json_list.append(json_data)
            except Exception as e:
                st.error(f"Error parsing file {file.name}: {str(e)}")

        if not json_list:
            st.error("No valid JSON files found.")
            return

        # Convert to DataFrame for easier processing
        df = pd.DataFrame(json_list)

        with st.spinner("Storing documents in vector database..."):
            try:
                # Store documents in vector database
                self.vector_db.store_documents(df.to_dict('records'))
                st.success("✅ JSON data successfully stored in vector database with embeddings!")
            except Exception as e:
                st.error(f"Error storing documents: {str(e)}")

    def extract_keywords(self, query):
        """Extract keywords from a query string."""
        # Convert to lowercase and split
        words = query.lower().split()

        # Filter by length and exclude stop words
        keywords = [word for word in words
                    if len(word) > 3 and word not in STOP_WORDS]

        return keywords

    def handle_search(self):
        """Handle search or question answering."""
        if not self.user_query:
            st.warning("⚠️ Please enter a search query or question.")
            return

        # Create a placeholder for search results
        results_placeholder = st.empty()

        # Determine if this is a search or a question
        query_type, processed_query = self.query_processor.classify_query(self.user_query)

        if query_type == "question":
            # Handle as a specific question about a document using RAG
            with st.spinner("🧠 Using RAG to answer your question..."):
                answer = self.rag_engine.answer_question(processed_query)

                with results_placeholder.container():
                    st.success("✅ Answer generated using RAG")
                    st.markdown("### 📝 Answer")
                    st.markdown(answer)
                    st.info(
                        "If you want to search for documents instead, try using keywords like 'find documents with' or 'search for'.")
        else:
            # Handle as a search query
            with st.spinner("🔍 Searching documents with vector embeddings..."):
                # Search for documents
                retrieved_docs = self.rag_engine.search_documents(self.user_query)

                if not retrieved_docs:
                    # If no results, try a more flexible search approach
                    st.info("Trying alternative search method...")
                    # Get the main keywords from the query
                    keywords = self.extract_keywords(self.user_query)

                    if keywords:
                        keyword_query = " ".join(keywords)
                        st.write(f"Searching with keywords: {keyword_query}")

                        # Create embedding for the keyword query
                        keyword_embedding = self.query_processor.generate_embedding(keyword_query)

                        query_results = self.vector_db.query_by_embedding(
                            keyword_embedding,
                            n_results=5
                        )
                        retrieved_docs = query_results["metadatas"][0] if query_results["metadatas"] else []

            if not retrieved_docs:
                st.error("❌ No matching documents found. Try a different query.")
                st.info(
                    "If you were asking a question instead of searching, try rephrasing with question words like 'what' or 'how'.")
            else:
                # Display results
                with results_placeholder.container():
                    st.success(f"🎯 Found {len(retrieved_docs)} matching documents")

                    # Convert results to DataFrame
                    results_df = pd.DataFrame(retrieved_docs)

                    # Select columns to display in main table
                    display_columns = ["event_id", "Ink Coverage", "Media Coating", "Media Finish",
                                       "Media Weight GSM", "Press Model", "location"]
                    # Make sure we only include columns that exist
                    available_columns = [col for col in display_columns if col in results_df.columns]

                    if available_columns:
                        display_df = results_df[available_columns]

                        # Display Table
                        st.subheader("🔹 Retrieved Documents")
                        st.dataframe(display_df)

                        # Generate explanations using RAG
                        st.subheader("🔹 Document Analysis")

                        with st.spinner("Generating document analysis..."):
                            analysis = self.rag_engine.analyze_documents(self.user_query, retrieved_docs)
                            st.markdown(analysis)
                    else:
                        st.warning("Retrieved documents are missing expected columns")
                        st.json(retrieved_docs)

                st.sidebar.success("✅ **Vector Search Complete!**")
                st.info(
                    "If you were asking a specific question about a document instead of searching, try rephrasing with question words like 'what' or 'how'.")


def main():
    """Main entry point for the application."""
    app = RAGChatbotApp()
    logger.info("Initializing RAG Chatbot")


if __name__ == "__main__":
    main()