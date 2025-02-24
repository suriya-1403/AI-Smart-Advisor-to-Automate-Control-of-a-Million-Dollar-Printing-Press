import os
import time
import warnings
import streamlit as st
from lib import Chatbot

warnings.filterwarnings("ignore", category=DeprecationWarning)

st.title("📚 RAG Chatbot")
st.sidebar.header("⚙️ Settings")
pdf_folder = "Dataset/Integrated Extraction"

# Available models
available_models = ["llama3.2", "granite3-dense", "mistral", "deepseek-r1"]

# Model selection dropdown
selected_model = st.sidebar.selectbox("🤖 Select Model", available_models)

# File type selection dropdown
file_type = st.sidebar.selectbox("📄 Select File Type", ["PDF", "JSON"])

# Drag-and-drop file uploader (Supports PDF and JSON)
uploaded_files = st.file_uploader(
    "📂 Upload PDF/JSON files",
    type=["pdf", "json"],
    accept_multiple_files=True
)

# Initialize RAG only once (store it in session state)
if "rag" not in st.session_state:
    st.session_state.rag = Chatbot(file_type=file_type, model_name=selected_model)
elif st.session_state.rag.model_name != selected_model or st.session_state.rag.file_type != file_type:
    st.session_state.rag = Chatbot(file_type=file_type, model_name=selected_model)  # Update model & file type

rag = st.session_state.rag  # Use stored RAG instance

# Extract text from uploaded files
if uploaded_files:
    with st.spinner("Processing uploaded files..."):
        file_processing_result = rag.process_uploaded_files(uploaded_files)
        st.session_state.document_context = rag.context  # Store extracted text
    st.sidebar.success(file_processing_result)

# User input for query
query = st.text_input("🔍 Ask a question:")

# Button to search
if st.button("🔍 Search"):
    if query:
        with st.spinner(f"Searching using {selected_model}..."):
            start_time = time.time()
            response = rag.generate_response(query)
            end_time = time.time()

        st.write("**Response:**")
        st.write(response)
        st.write(f"⏳ Time Taken: {end_time - start_time:.2f} seconds")
    else:
        st.warning("⚠️ Please enter a question before searching.")
