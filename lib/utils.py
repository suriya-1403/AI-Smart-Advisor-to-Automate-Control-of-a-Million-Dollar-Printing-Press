import os
from langchain.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config import VECTOR_STORE_PATH

class Chatbot:
    def __init__(self, pdf_folder, model_name="llama3.2", db_path=VECTOR_STORE_PATH):
        """Initialize RAG system with FAISS & In-Memory options and dynamic model selection."""
        self.model_name = model_name
        self.pdf_folder = pdf_folder
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

    def load_and_store(self):
        """Load PDFs, create embeddings, and store in FAISS & In-Memory."""
        all_pages = []
        for file_name in os.listdir(self.pdf_folder):
            if file_name.endswith(".pdf"):
                file_path = os.path.join(self.pdf_folder, file_name)
                loader = PyPDFLoader(file_path)
                pages = loader.load_and_split()
                all_pages.extend(pages)

        if all_pages:
            self.vectorstore_faiss = FAISS.from_documents(all_pages, self.embeddings)
            self.vectorstore_faiss.save_local(self.db_path)  # Save FAISS index
            self.vectorstore_memory = DocArrayInMemorySearch.from_documents(all_pages, embedding=self.embeddings)
            return f"✅ Loaded {len(all_pages)} pages from {len(os.listdir(self.pdf_folder))} PDFs."
        return "⚠️ No PDFs found in the folder."

    def load_existing_store(self):
        """Load FAISS from disk if available, otherwise initialize a new one."""
        index_path = os.path.join(self.db_path, "index.faiss")

        if os.path.exists(index_path):
            self.vectorstore_faiss = FAISS.load_local(
                self.db_path, self.embeddings, allow_dangerous_deserialization=True
            )
            return "✅ Loaded FAISS vector store from disk."
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
