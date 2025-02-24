import os
import json
import streamlit as st
from langchain.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from config import VECTOR_STORE_PATH


class Chatbot:
    def __init__(self, pdf_folder, file_type="PDF", model_name="llama3.2", db_path=VECTOR_STORE_PATH):
        """Initialize RAG system with FAISS & In-Memory options and support for JSON/PDF files."""
        self.model_name = model_name
        self.pdf_folder = pdf_folder
        self.file_type = file_type  # Added support for file selection
        self.db_path = db_path  # FAISS storage path
        self.model = Ollama(model=self.model_name)  # Load selected model
        self.embeddings = OllamaEmbeddings(model=self.model_name)
        self.vectorstore_faiss = None
        self.vectorstore_memory = None
        self.selected_store = "FAISS"  # Default choice
        self.prompt_template = PromptTemplate(
            template="Context: {context}\n\nQuestion: {question}\n\nAnswer:",
            input_variables=["context", "question"],
        )
        self.parser = StrOutputParser()

    def update_model(self, new_model_name):
        """Update the chatbot model dynamically."""
        self.model_name = new_model_name
        self.model = Ollama(model=self.model_name)  # Reload with new model
        self.embeddings = OllamaEmbeddings(model=self.model_name)

    def flatten_dict(self, d, parent_key=''):
        """
        Recursively flattens a nested dictionary into a single-level dictionary.
        - Extracts all string, integer, and float values for text similarity.
        """
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}.{k}" if parent_key else k
            if isinstance(v, dict):  # Recursive flattening for nested dicts
                items.extend(self.flatten_dict(v, new_key).items())  # Use self.flatten_dict()
            elif isinstance(v, list):
                # Convert lists into space-separated strings
                items.append((new_key, " ".join(map(str, v))))
            elif isinstance(v, (str, int, float)):  # Keep text, numbers
                items.append((new_key, str(v)))
        return dict(items)

    def load_and_store(self):
        """Load PDFs or JSON files, create embeddings, and store in FAISS & In-Memory."""
        all_pages = []
        processed_files = []  # Store processed filenames
        print(f"Checking folder: {self.pdf_folder}")

        for file_name in os.listdir(self.pdf_folder):
            file_path = os.path.join(self.pdf_folder, file_name)
            print(f"Found file: {file_name}")

            if self.file_type == "PDF" and file_name.endswith(".pdf"):
                print(f"Processing PDF: {file_name}")
                loader = PyPDFLoader(file_path)
                pages = loader.load_and_split()
                all_pages.extend(pages)
                processed_files.append(file_name)  # Add file to processed list

            elif self.file_type == "JSON" and file_name.endswith(".json"):
                print(f"Processing JSON: {file_name}")
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                    # Call flatten_dict correctly using self.
                    flat_data = self.flatten_dict(data)

                    # Convert flattened dictionary to text
                    text = "\n".join([f"{key}: {value}" for key, value in flat_data.items()])
                    all_pages.append(Document(page_content=text))
                    processed_files.append(file_name)  # Add file to processed list

                    print(f"Processed JSON into document:\n{text}\n")
        st.session_state.vectorstore_files = processed_files
        print(f"Total loaded documents: {len(all_pages)}")
        if all_pages:
            self.vectorstore_faiss = FAISS.from_documents(all_pages, self.embeddings)
            self.vectorstore_faiss.save_local(self.db_path)  # Save FAISS index
            self.vectorstore_memory = DocArrayInMemorySearch.from_documents(all_pages, embedding=self.embeddings)
            return f"✅ Loaded {len(all_pages)} records from {len(os.listdir(self.pdf_folder))} {self.file_type} files."

        return f"⚠️ No {self.file_type} files found in the folder."

    def load_existing_store(self):
        """Load FAISS from disk if available and extract stored filenames."""
        index_path = os.path.join(self.db_path, "index.faiss")

        if os.path.exists(index_path):
            self.vectorstore_faiss = FAISS.load_local(
                self.db_path, self.embeddings, allow_dangerous_deserialization=True
            )

            # Extract filenames from FAISS document store
            stored_filenames = []
            if hasattr(self.vectorstore_faiss, "docstore"):
                stored_filenames = [
                    doc.metadata.get("source", "Unknown") for doc in self.vectorstore_faiss.docstore._dict.values()
                ]

            # Store filenames in session state
            import streamlit as st
            st.session_state.vectorstore_files = stored_filenames

            return f"✅ Loaded FAISS vector store from disk with {len(stored_filenames)} stored files."

        return "⚠️ No FAISS vector store found. Click 'Update Vector Space' to create it."

    def retrieve_documents(self, query):
        """Retrieve documents based on selected vector store."""
        results = []
        if self.selected_store == "FAISS" and self.vectorstore_faiss:
            results.extend(self.vectorstore_faiss.as_retriever().invoke(query))
        elif self.selected_store == "In-Memory" and self.vectorstore_memory:
            results.extend(self.vectorstore_memory.as_retriever().invoke(query))

        return results[:5]  # Return top 5 results

    def generate_response(self, query):
        """Generate response using retrieved documents and the model."""
        context_docs = self.retrieve_documents(query)
        context = "\n".join(doc.page_content for doc in context_docs) if context_docs else "No relevant context found."
        response = (self.prompt_template | self.model | self.parser).invoke({"context": context, "question": query})
        return response
