import os
import json
import streamlit as st
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_ollama import OllamaLLM
import os

os.environ["PYTORCH_NO_CUDA"] = "1"

# Initialize Streamlit UI
st.set_page_config(page_title="RAG Chatbot", layout="wide")

st.sidebar.header("⚙️ Settings")

# Model Selection
selected_model = st.sidebar.selectbox("Select Model", ["llama3.2", "gpt-4", "mistral"], index=0)

# File Type Selection
file_type = st.sidebar.selectbox("Select File Type", ["PDF", "JSON"], index=0)

# File Upload
uploaded_files = st.sidebar.file_uploader(
    "Upload PDF/JSON files",
    type=["pdf", "json"],
    accept_multiple_files=True
)

if uploaded_files:
    st.sidebar.success(f"✅ Loaded {len(uploaded_files)} file(s) into memory.")

# Title
st.markdown("### 📄 RAG Chatbot")
st.write("Upload files and perform **structured document retrieval** using AI.")

# File Display Section
st.subheader("📂 Uploaded Files")
if uploaded_files:
    for file in uploaded_files:
        st.write(f"📄 {file.name} - {file.size / 1024:.1f} KB ")
else:
    st.info("No files uploaded yet. Please upload JSON or PDF files.")

# ChromaDB Initialization
chroma_path = "./chromadb_store"
# Check if storage was deleted & reinitialize if needed
if not os.path.exists(chroma_path):
    os.makedirs(chroma_path, exist_ok=True)  # Recreate folder if missing

# Create a fresh ChromaDB instance
try:
    chroma_client = chromadb.PersistentClient(path=chroma_path)
    collection = chroma_client.get_or_create_collection("json_documents")
    print("✅ ChromaDB store initialized successfully!")
except ValueError as e:
    print("⚠️ Error initializing ChromaDB. Attempting reset...")
    os.system(f"rm -rf {chroma_path}")  # Force delete any corrupted files
    os.makedirs(chroma_path, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=chroma_path)
    collection = chroma_client.get_or_create_collection("json_documents")
    print("🔄 ChromaDB store reset and initialized!")

# Load Embedding Model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize LLM
llm = OllamaLLM(model=selected_model)

# Button: Store JSON in Vector Store
if st.button("🚀 Store JSON in Vector Store"):
    if not uploaded_files:
        st.warning("⚠️ Please upload JSON files first.")
    else:
        json_list = []

        for file in uploaded_files:
            json_data = json.load(file)
            json_list.append(json_data)

        df = pd.DataFrame(json_list)

        for _, row in df.iterrows():
            doc_text = f"Event {row['event_id']} at {row['location']} on {row['publish_date']} with {row['Ink Coverage']} ink and {row['Media Weight GSM']} GSM."

            # Compute embeddings
            embedding = embedding_model.encode(doc_text).tolist()

            # Add to ChromaDB
            collection.add(
                documents=[doc_text],
                metadatas=[{
                    "event_id": row["event_id"],
                    "publish_date": row["publish_date"],
                    "location": row["location"],
                    "Ink Coverage": row["Ink Coverage"],
                    "Media Weight GSM": row["Media Weight GSM"],
                    "Media Coating": row["Media Coating"],
                    "Media Finish": row["Media Finish"],
                    "Press Model": row["Press Model"]
                }],
                ids=[str(row["event_id"])]
            )

        st.success("✅ JSON data successfully stored in ChromaDB!")

# Query Input
st.subheader("🔍 Ask a question:")
user_query = st.text_input("Enter your search query:")


# Simple direct search function
def basic_search(query, n_results=5):
    """Perform a basic semantic search without complex filtering or ranking"""
    try:
        # Simple semantic search
        query_results = collection.query(
            query_texts=[query],
            n_results=n_results
        )

        return query_results["metadatas"][0] if query_results["metadatas"] else []
    except Exception as e:
        st.error(f"❌ Search error: {str(e)}")
        return []


# Debug Tools
if st.checkbox("Show Debug Options"):
    st.subheader("🛠️ Debug Tools")
    if st.button("📊 Show Collection Stats"):
        try:
            collection_count = collection.count()
            st.write(f"Documents in collection: {collection_count}")
            st.success("ChromaDB is working correctly!")
        except Exception as e:
            st.error(f"ChromaDB error: {str(e)}")

# Search Button & Output
if st.button("🔎 Search"):
    if not user_query:
        st.warning("⚠️ Please enter a search query.")
    else:
        # Create a placeholder for search results
        results_placeholder = st.empty()

        with st.spinner("🔍 Searching documents..."):
            # Perform basic search
            search_results = basic_search(user_query)

        if not search_results:
            st.error("❌ No matching documents found. Try a different query.")
        else:
            # Create a cleaner results display
            with results_placeholder.container():
                st.success(f"🎯 Found {len(search_results)} matching documents")

                # Convert results to DataFrame
                results_df = pd.DataFrame(search_results)

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

                    # Simple explanation for each result
                    st.subheader("🔹 Document Details")
                    for i, result in enumerate(search_results):
                        # Basic information display
                        st.markdown(f"""
                        <style>
                            .result-box {{
                                margin-bottom: 10px;
                                padding: 10px;
                                border-left: 4px solid blue;
                                background-color: rgba(248, 249, 250, 0.1);
                            }}
                        </style>

                        <div class="result-box">
                            <strong>Document {i + 1} (Event ID: {result.get('event_id', 'N/A')})</strong><br>
                            <span>Ink Coverage: {result.get('Ink Coverage', 'N/A')}</span><br>
                            <span>Media: {result.get('Media Weight GSM', 'N/A')} GSM, {result.get('Media Coating', 'N/A')}, {result.get('Media Finish', 'N/A')}</span><br>
                            <span>Press Model: {result.get('Press Model', 'N/A')}</span>
                        </div>
                        """, unsafe_allow_html=True)

                    # Additional Press Model Section for consistency with original UI
                    if "Press Model" in results_df.columns:
                        press_details = [{"Event ID": doc.get("event_id", "N/A"),
                                          "Press Model": doc.get("Press Model", "N/A")}
                                         for doc in search_results]
                        press_df = pd.DataFrame(press_details)

                        st.subheader("🔹 Press Model Details")
                        st.dataframe(press_df)
                else:
                    st.warning("Retrieved documents are missing expected columns")
                    st.json(search_results)

            st.sidebar.success("✅ **Search Complete!**")