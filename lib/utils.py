import os
import json
import streamlit as st
from io import BytesIO
from PyPDF2 import PdfReader
from langchain.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document


class Chatbot:
    def __init__(self, file_type="PDF", model_name="llama3.2"):
        """Initialize RAG system without vector store (uses direct context)."""
        self.model_name = model_name
        self.file_type = file_type
        self.model = Ollama(model=self.model_name)
        self.embeddings = OllamaEmbeddings(model=self.model_name)
        self.context = ""  # Store extracted document context

        self.parser = StrOutputParser()

    def process_uploaded_files(self, uploaded_files):
        """Process uploaded PDFs and JSON files and store them as direct context."""
        extracted_texts = []

        for uploaded_file in uploaded_files:
            if uploaded_file.type == "application/pdf":  # Process PDF
                pdf_reader = PdfReader(BytesIO(uploaded_file.getvalue()))
                text = "\n".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
                extracted_texts.append(text)

            elif uploaded_file.type == "application/json":  # Process JSON
                json_data = json.load(uploaded_file)  # Load JSON data
                flattened_data = self.flatten_dict(json_data)  # Flatten nested JSON

                # Convert flattened dictionary to a single formatted text string
                json_text = "\n".join([f"{key}: {value}" for key, value in flattened_data.items()])
                extracted_texts.append(json_text)  # Ensure it's a string

        self.context = "\n".join(extracted_texts)  # Store extracted text as context
        print(self.context)
        return f"✅ Loaded {len(uploaded_files)} file(s) into memory."

    def flatten_dict(self, data, parent_key=''):
        """Flatten nested JSON into a single-level dictionary for easier processing."""
        items = []

        if isinstance(data, dict):  # Only call .items() if it's a dictionary
            for k, v in data.items():
                new_key = f"{parent_key}.{k}" if parent_key else k

                if isinstance(v, dict):  # Recursive call for nested dictionaries
                    items.extend(self.flatten_dict(v, new_key).items())
                elif isinstance(v, list):  # Convert lists into space-separated strings
                    items.append((new_key, " ".join(map(str, v))))
                elif isinstance(v, (str, int, float, bool)):  # Store primitive values
                    items.append((new_key, str(v)))
        elif isinstance(data, (str, int, float, bool)):  # If input is a primitive, return as a dict
            items.append((parent_key, str(data)))

        return dict(items)

    def generate_response(self, query):
        """Generate response using uploaded document context if available."""
        context = self.context if self.context else "No relevant context found."
        prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
        response = self.model.invoke(prompt)
        return response
