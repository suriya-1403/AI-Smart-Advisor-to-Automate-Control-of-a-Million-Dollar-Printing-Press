# import os
# import json
# import streamlit as st
# from io import BytesIO
# from PyPDF2 import PdfReader
# from langchain.vectorstores import FAISS
# from langchain_community.llms import Ollama
# from langchain_community.embeddings import OllamaEmbeddings
# from langchain_core.documents import Document
#
# class Chatbot:
#     def __init__(self, file_type="PDF", model_name="llama3.2", vector_store_path="vector_db"):
#         """Initialize RAG system with FAISS vector store, including metadata filtering."""
#         self.model_name = model_name
#         self.file_type = file_type
#         self.model = Ollama(model=self.model_name)
#         self.embeddings = OllamaEmbeddings(model=self.model_name)
#         self.vector_store_path = vector_store_path
#
#         if os.path.exists(vector_store_path):
#             self.vector_store = FAISS.load_local(vector_store_path, self.embeddings,
#                                                  allow_dangerous_deserialization=True)
#         else:
#             self.vector_store = None
#
#     def process_uploaded_files(self, uploaded_files):
#         """Extracts and stores PDFs & JSON data in FAISS, preserving numerical and categorical fields."""
#         extracted_documents = []
#
#         for uploaded_file in uploaded_files:
#             filename = uploaded_file.name
#
#             if uploaded_file.type == "application/pdf":  # Process PDF
#                 pdf_reader = PdfReader(BytesIO(uploaded_file.getvalue()))
#                 text = "\n".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
#                 doc = Document(page_content=text, metadata={"file_name": filename, "type": "PDF"})
#                 extracted_documents.append(doc)
#
#             elif uploaded_file.type == "application/json":  # Process JSON
#                 json_data = json.load(uploaded_file)
#
#                 # Extract important fields
#                 ink_coverage = json_data.get("Ink Coverage", "Unknown")
#                 media_weight_gsm = float(json_data.get("Media Weight GSM", -1))
#                 media_coating = json_data.get("Media Coating", "Unknown")
#                 press_model = json_data.get("Press Model", "Unknown")
#                 file_name = json_data.get("file_name", filename)  # Ensure file name is stored
#
#                 # Convert JSON to text for embedding
#                 json_text = "\n".join([f"{key}: {value}" for key, value in json_data.items()])
#
#                 doc = Document(
#                     page_content=json_text,
#                     metadata={
#                         "file_name": file_name,
#                         "type": "JSON",
#                         "ink_coverage": ink_coverage,
#                         "media_weight_gsm": media_weight_gsm,
#                         "media_coating": media_coating,
#                         "press_model": press_model
#                     }
#                 )
#                 extracted_documents.append(doc)
#
#         if extracted_documents:
#             if self.vector_store is None:
#                 self.vector_store = FAISS.from_documents(extracted_documents, self.embeddings)
#             else:
#                 self.vector_store.add_documents(extracted_documents)
#
#             # Save vector store
#             self.vector_store.save_local(self.vector_store_path)
#
#         return f"✅ Loaded {len(uploaded_files)} file(s) into vector store."
#
#     def flatten_dict(self, data, parent_key=""):
#         """Flattens nested JSON for better processing."""
#         items = []
#
#         if isinstance(data, dict):
#             for k, v in data.items():
#                 new_key = f"{parent_key}.{k}" if parent_key else k
#
#                 if isinstance(v, dict):
#                     items.extend(self.flatten_dict(v, new_key).items())
#                 elif isinstance(v, list):
#                     items.append((new_key, " ".join(map(str, v))))
#                 else:
#                     items.append((new_key, str(v)))
#         return dict(items)
#
#     # def generate_response(self, query):
#     #     """Retrieve and rank documents while ensuring unique selections."""
#     #     if self.vector_store is None:
#     #         return "No relevant documents found in the vector store."
#     #
#     #     # Retrieve multiple documents to ensure we have unique options
#     #     retrieved_docs = self.vector_store.similarity_search(query, k=10)
#     #
#     #     # Use a dictionary to filter out duplicates based on `file_name`
#     #     unique_docs = {}
#     #     for doc in retrieved_docs:
#     #         file_name = doc.metadata.get("file_name", "Unknown_File")
#     #
#     #         if file_name not in unique_docs:  # Only add if unique
#     #             unique_docs[file_name] = {
#     #                 "doc": doc,
#     #                 "ink_coverage": doc.metadata.get("ink_coverage", "Unknown"),
#     #                 "media_weight_gsm": doc.metadata.get("media_weight_gsm", -1),
#     #                 "media_coating": doc.metadata.get("media_coating", "Unknown"),
#     #                 "press_model": doc.metadata.get("press_model", "Unknown")
#     #             }
#     #
#     #         if len(unique_docs) >= 3:  # Stop once we have 3 unique documents
#     #             break
#     #
#     #     # Convert to a list and rank by closest paper weight
#     #     best_matches = sorted(unique_docs.values(), key=lambda x: abs(220 - x["media_weight_gsm"]))[:3]
#     #
#     #     # Construct formatted context with unique filenames
#     #     formatted_context = "\n\n".join([
#     #         f"File: {entry['doc'].metadata.get('file_name', 'Unknown_File')}\n"
#     #         f"Ink Coverage: {entry['ink_coverage']}\n"
#     #         f"Media Weight: {entry['media_weight_gsm']} GSM\n"
#     #         f"Coating: {entry['media_coating']}\n"
#     #         f"Press Model: {entry['press_model']}\n"
#     #         f"Extracted Content:\n{entry['doc'].page_content[:500]}..."  # Limit content for brevity
#     #         for entry in best_matches
#     #     ])
#     #
#     #     # Ensure LLM uses **only unique filenames**
#     #     prompt = f"""
#     #     Using the following structured JSON documents, generate a **ranked table** containing only the **top 3 best-matching unique documents** based on the given selection criteria.
#     #
#     #     **Documents:**
#     #     {formatted_context}
#     #
#     #     **Question:** {query}
#     #
#     #     **Answer:**
#     #     """
#     #
#     #     response = self.model.invoke(prompt)
#     #     return response
#     def generate_response(self, query):
#         """Retrieve and rank documents while ensuring structured text output (no Python code)."""
#         if self.vector_store is None:
#             return "No relevant documents found in the vector store."
#
#         # Retrieve multiple documents to ensure we have unique options
#         retrieved_docs = self.vector_store.similarity_search(query, k=10)
#
#         # Use a dictionary to filter out duplicates based on `file_name`
#         unique_docs = {}
#         for doc in retrieved_docs:
#             file_name = doc.metadata.get("file_name", "Unknown_File")
#
#             if file_name not in unique_docs:  # Only add if unique
#                 unique_docs[file_name] = {
#                     "doc": doc,
#                     "ink_coverage": doc.metadata.get("ink_coverage", "Unknown"),
#                     "media_weight_gsm": doc.metadata.get("media_weight_gsm", -1),
#                     "media_coating": doc.metadata.get("media_coating", "Unknown"),
#                     "press_model": doc.metadata.get("press_model", "Unknown")
#                 }
#
#             if len(unique_docs) >= 3:  # Stop once we have 3 unique documents
#                 break
#
#         # Convert to a list and rank by closest paper weight
#         best_matches = sorted(unique_docs.values(), key=lambda x: abs(220 - x["media_weight_gsm"]))[:3]
#
#         # Construct formatted context with unique filenames
#         formatted_context = "\n\n".join([
#             f"File: {entry['doc'].metadata.get('file_name', 'Unknown_File')}\n"
#             f"Ink Coverage: {entry['ink_coverage']}\n"
#             f"Media Weight: {entry['media_weight_gsm']} GSM\n"
#             f"Coating: {entry['media_coating']}\n"
#             f"Press Model: {entry['press_model']}\n"
#             f"Extracted Content:\n{entry['doc'].page_content[:500]}..."  # Limit content for brevity
#             for entry in best_matches
#         ])
#
#         # **Fix: Adjust Prompt to Force Text-Based Output**
#         prompt = f"""
#         Using the following structured JSON documents, generate a **ranked table** containing only the **top 3 best-matching unique documents** based on the given selection criteria.
#
#         **Documents:**
#         {formatted_context}
#
#         **Instructions:**
#         - Do NOT generate Python code or scripts.
#         - Return a structured response **in plain text**.
#         - Rank documents based on:
#           1. Ink Coverage
#           2. Media Weight
#           3. Coated Media Class
#         - Provide a **clear, text-based table** for ranking and an explanation.
#
#         **Expected Output Format:**
#         **Step 1: Ranked Table**
#         ```
#         | Rank | Document Name | Ink Coverage Class | Paper Weight GSM | Coated Media Class | Reason for Selection |
#         |------|--------------|--------------------|------------------|--------------------|----------------------|
#         | 1st  | [Document Name] | [Ink Coverage] | [Paper Weight] | [Coated Media] | [Reasoning] |
#         | 2nd  | [Document Name] | [Ink Coverage] | [Paper Weight] | [Coated Media] | [Reasoning] |
#         | 3rd  | [Document Name] | [Ink Coverage] | [Paper Weight] | [Coated Media] | [Reasoning] |
#         ```
#
#         **Step 2: Explanation of Ranking**
#         - Clearly explain why the **1st document** was chosen.
#         - Justify the ranking order based on the **closest match**.
#         - If two documents are very close, mention the **deciding factor**.
#         - If a document was **not selected**, briefly state why.
#
#         **Step 3: Press Details**
#         ```
#         | Filename | Press Model | Press Mode | Press Speed | Dryer Setting | Moisturizer Setting | Optimizer Usage | Tension Setting |
#         |----------|------------|------------|-------------|---------------|----------------------|-----------------|-----------------|
#         | [Filename] | [Press Model] | [Press Mode] | [Press Speed] | [Dryer Setting] | [Moisturizer] | [Optimizer] | [Tension] |
#         ```
#
#         **Answer (in plain text, NOT code):**
#         """
#
#         response = self.model.invoke(prompt)
#         return response
#
#
#
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