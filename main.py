# import os
# import time
# import warnings
# import streamlit as st
# from lib import Chatbot
#
# warnings.filterwarnings("ignore", category=DeprecationWarning)
#
# st.title("📚 RAG Chatbot")
# st.sidebar.header("⚙️ Settings")
# pdf_folder = "Dataset/Integrated Extraction"
#
# # Available models
# available_models = ["llama3.2", "granite3-dense", "mistral", "deepseek-r1"]
#
# # Model selection dropdown
# selected_model = st.sidebar.selectbox("🤖 Select Model", available_models)
#
# # File type selection dropdown
# file_type = st.sidebar.selectbox("📄 Select File Type", ["PDF", "JSON"])
#
# # Drag-and-drop file uploader (Supports PDF and JSON)
# uploaded_files = st.file_uploader(
#     "📂 Upload PDF/JSON files",
#     type=["pdf", "json"],
#     accept_multiple_files=True
# )
#
# # Initialize chatbot instance with vector store
# if "rag" not in st.session_state:
#     st.session_state.rag = Chatbot(file_type=file_type, model_name=selected_model, vector_store_path="vector_db")
# elif st.session_state.rag.model_name != selected_model:
#     st.session_state.rag = Chatbot(file_type=file_type, model_name=selected_model, vector_store_path="vector_db")
#
# rag = st.session_state.rag  # Use stored RAG instance
#
# # Process uploaded files if new ones are added
# if uploaded_files:
#     with st.spinner("Processing uploaded files..."):
#         file_processing_result = rag.process_uploaded_files(uploaded_files)
#         st.sidebar.success(file_processing_result)
#
# # User input query
# query = st.text_input("🔍 Ask a question:")
#
# # Button to search
# if st.button("🔍 Search"):
#     if query:
#         with st.spinner(f"Searching using {selected_model}..."):
#             start_time = time.time()
#             response = rag.generate_response(query)
#             end_time = time.time()
#
#         st.write("**Response:**")
#         st.write(response)
#         st.write(f"⏳ Time Taken: {end_time - start_time:.2f} seconds")
#     else:
#         st.warning("⚠️ Please enter a question before searching.")
#
# import os
# import time
# import warnings
# import streamlit as st
# import pandas as pd  # Import Pandas for tables
# from lib import Chatbot
#
# warnings.filterwarnings("ignore", category=DeprecationWarning)
#
# st.title("📚 RAG Chatbot")
# st.sidebar.header("⚙️ Settings")
#
# # Available models
# available_models = ["llama3.2", "granite3-dense", "mistral", "deepseek-r1"]
# selected_model = st.sidebar.selectbox("🤖 Select Model", available_models)
#
# # File type selection
# file_type = st.sidebar.selectbox("📄 Select File Type", ["PDF", "JSON"])
#
# # File uploader
# uploaded_files = st.file_uploader("📂 Upload PDF/JSON files", type=["pdf", "json"], accept_multiple_files=True)
#
# # Initialize chatbot instance with vector store
# if "rag" not in st.session_state:
#     st.session_state.rag = Chatbot(file_type=file_type, model_name=selected_model, vector_store_path="vector_db",embeddings_model_name="nomic-embed-text")
# elif st.session_state.rag.model_name != selected_model:
#     st.session_state.rag = Chatbot(file_type=file_type, model_name=selected_model, vector_store_path="vector_db",embeddings_model_name="nomic-embed-text")
#
# rag = st.session_state.rag  # Use stored RAG instance
#
# # Process uploaded files
# if uploaded_files:
#     with st.spinner("Processing uploaded files..."):
#         file_processing_result = rag.process_uploaded_files(uploaded_files)
#         st.sidebar.success(file_processing_result)
#
# # User provides **direct prompt** instead of a fixed query
# query = st.text_area("📝 Enter your prompt:", height=200,
#                      placeholder="Write a custom ranking prompt or retrieval request...")
#
# # Button to search
# if st.button("🔍 Submit Prompt"):
#     if query.strip():
#         st.subheader("👨🏻‍💻 Chat Query")
#         st.write(query)
#         with st.spinner(f"Processing query using {selected_model}..."):
#             start_time = time.time()
#             response = rag.generate_response(query)  # Use **user's raw prompt**
#             end_time = time.time()
#
#         # Display response
#         st.subheader("📜 AI Response")
#         st.write(response)
#
#         st.write(f"⏳ Time Taken: {end_time - start_time:.2f} seconds")
#     else:
#         st.warning("⚠️ Please enter a valid prompt before submitting.")

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