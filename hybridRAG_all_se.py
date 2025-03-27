import os
import json
import re
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
st.write("Upload files and perform **vector-based retrieval and generation** using AI.")

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

    # Create an embedding function from our model
    embedding_function = SentenceTransformer("all-MiniLM-L6-v2")

    # Create collection with the embedding function
    collection = chroma_client.get_or_create_collection(
        name="json_documents"
    )
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

# Initialize Llama 3.2
llm = OllamaLLM(model=selected_model)

# **Button: Store JSON in Vector Store**
if st.button("🚀 Store JSON in Vector Store"):
    if not uploaded_files:
        st.warning("⚠️ Please upload JSON files first.")
    else:
        json_list = []

        for file in uploaded_files:
            file_content = file.read()
            json_data = json.loads(file_content)
            # Add filename to the data
            json_data["filename"] = file.name
            json_list.append(json_data)

        df = pd.DataFrame(json_list)

        for _, row in df.iterrows():
            doc_text = f"Event {row['event_id']} at {row['location']} on {row['publish_date']} with {row['Ink Coverage']} ink and {row['Media Weight GSM']} GSM."

            # Compute embeddings
            embedding = embedding_model.encode(doc_text).tolist()

            # Add to ChromaDB - now including the embeddings
            collection.add(
                documents=[doc_text],
                embeddings=[embedding],  # Include the embeddings explicitly
                metadatas=[{
                    "event_id": row["event_id"],
                    "publish_date": row["publish_date"],
                    "location": row["location"],
                    "Ink Coverage": row["Ink Coverage"],
                    "Media Weight GSM": row["Media Weight GSM"],
                    "Media Coating": row["Media Coating"],
                    "Media Finish": row["Media Finish"],
                    "Press Model": row["Press Model"],
                    "filename": row["filename"]  # Store filename in metadata
                }],
                ids=[str(row["event_id"])]
            )

        st.success("✅ JSON data successfully stored in vector database with embeddings!")

# Query Input
st.subheader("🔍 Ask a question:")
user_query = st.text_input("Enter your search query or ask a question:")


# Function to determine if the query is a search or a question
def process_query(user_query):
    """Process user query to determine if it's a search or a specific question"""

    # First check if it's likely a search query (looking for documents with specific attributes)
    search_patterns = [
        r'find\s+.*(document|file|print job|event)',
        r'search\s+for',
        r'looking\s+for',
        r'(show|get|display|retrieve)\s+.*(document|file|print job|event)',
        r'with\s+(heavy|medium|light|coated|uncoated|silk|matte|gloss|\d+\s*gsm)',
        r'(filter|where|that has)',
    ]

    # Check if any search pattern matches
    for pattern in search_patterns:
        if re.search(pattern, user_query.lower(), re.IGNORECASE):
            return "search", user_query

    # If it has question words or asking about specifics, treat as a question
    question_indicators = [
        r'^(what|how|why|when|where|who|can you|could you|tell me)',
        r'\?$',
        r'(explain|describe|elaborate|detail|info about)',
        r'(configuration|setting|parameter|specification|property)',
    ]

    for pattern in question_indicators:
        if re.search(pattern, user_query.lower(), re.IGNORECASE):
            return "question", user_query

    # If it contains a filename, likely asking about a specific file
    if ".json" in user_query or ".pdf" in user_query:
        return "question", user_query

    # Default to search if we can't determine
    return "search", user_query


# Enhanced RAG-based document answering function
def answer_document_question(user_query):
    """RAG pipeline for document questions"""

    # Generate query embedding for semantic search
    query_embedding = embedding_model.encode(user_query).tolist()

    try:
        # Special cases (direct lookups by ID or filename)
        file_match = re.search(r'["\']?([^"\']+\.json)["\']?', user_query)
        event_id_match = re.search(r'event\s+id\s*[:\s]\s*([a-zA-Z0-9-_]+)', user_query, re.IGNORECASE)

        if file_match:
            filename = file_match.group(1)
            st.info(f"Looking for information about file: {filename}")

            # First try metadata lookup
            results = collection.query(
                query_embeddings=[query_embedding],
                where={"filename": filename},
                n_results=3
            )
        elif event_id_match:
            event_id = event_id_match.group(1)
            st.info(f"Looking for information about event ID: {event_id}")

            # Try exact match by ID
            results = collection.query(
                query_embeddings=[query_embedding],
                where={"event_id": event_id},
                n_results=3
            )
        else:
            # Standard semantic search - use vector similarity to find relevant documents
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=3
            )

        # Check if we found any documents
        if not results["documents"] or not results["documents"][0]:
            return "I couldn't find specific information about that document or configuration. Could you try rephrasing your question?"

        # Build context from retrieved documents with their metadata
        context_sections = []

        for i in range(min(3, len(results["documents"][0]))):
            if i < len(results["documents"][0]) and i < len(results["metadatas"][0]):
                document_text = results["documents"][0][i]
                metadata = results["metadatas"][0][i]

                context_sections.append(f"Document {i + 1}:\n{document_text}")
                context_sections.append(f"Document {i + 1} Metadata:\n{json.dumps(metadata, indent=2)}")

        # Join all context sections
        full_context = "\n\n".join(context_sections)

        # Create RAG prompt with retrieved context
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
        response = llm.invoke(rag_prompt)
        return response

    except Exception as e:
        st.error(f"Error in RAG pipeline: {str(e)}")
        return "Sorry, I encountered an error while trying to answer your question. Please try again."


# Modified retrieve_documents function to use embeddings properly
def retrieve_documents(user_query):
    """Retrieves documents using vector similarity search, then re-ranks based on specific criteria"""

    # Get structured filters from the query
    query_filters = expand_query_with_llm(user_query)

    # Generate embedding for the query
    query_embedding = embedding_model.encode(user_query).tolist()

    # Print debugging info
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
        # If no filters were extracted, perform a pure semantic search without filtering
        if not where_clause["$and"]:
            st.info("ℹ️ Using semantic search without filters")
            query_results = collection.query(
                query_embeddings=[query_embedding],
                n_results=20  # Increased to get more candidates for re-ranking
            )
        else:
            # Apply filters with semantic search
            st.info("ℹ️ Using semantic search with filters")
            query_results = collection.query(
                query_embeddings=[query_embedding],
                n_results=20,  # Increased to get more candidates for re-ranking
                where=where_clause if where_clause and where_clause["$and"] else None
            )

        # Get the documents retrieved by vector similarity
        retrieved_docs = query_results["metadatas"][0] if query_results["metadatas"] else []

        if retrieved_docs:
            # Re-rank based on specific criteria
            # Define score_document function (copy from your original code)
            def score_document(doc, query):
                """Score a document based on how well it matches the query"""
                score = 0
                explanation = []

                # Extract key terms from query
                query_lower = query.lower()

                # Check for ink coverage
                if "ink" in query_lower:
                    if "heavy" in query_lower and doc.get("Ink Coverage") == "Heavy":
                        score += 10
                        explanation.append("Matches requested heavy ink coverage")
                    elif "medium" in query_lower and doc.get("Ink Coverage") == "Medium":
                        score += 10
                        explanation.append("Matches requested medium ink coverage")
                    elif "light" in query_lower and doc.get("Ink Coverage") == "Light":
                        score += 10
                        explanation.append("Matches requested light ink coverage")

                # Check for media coating
                if "coat" in query_lower:
                    if "coated" in query_lower and "coated" in str(doc.get("Media Coating", "")).lower():
                        score += 10
                        explanation.append("Matches requested coated media")
                    elif "uncoated" in query_lower and "uncoated" in str(doc.get("Media Coating", "")).lower():
                        score += 10
                        explanation.append("Matches requested uncoated media")

                # Check for media finish
                if "silk" in query_lower and "silk" in str(doc.get("Media Finish", "")).lower():
                    score += 10
                    explanation.append("Matches requested silk finish")
                elif "matte" in query_lower and "matte" in str(doc.get("Media Finish", "")).lower():
                    score += 10
                    explanation.append("Matches requested matte finish")
                elif "gloss" in query_lower and "gloss" in str(doc.get("Media Finish", "")).lower():
                    score += 10
                    explanation.append("Matches requested gloss finish")

                # Check for GSM weight
                import re
                gsm_matches = re.findall(r'(\d+)\s*gsm', query_lower)
                if gsm_matches and "Media Weight GSM" in doc:
                    requested_gsm = int(gsm_matches[0])
                    doc_gsm = int(doc["Media Weight GSM"]) if doc["Media Weight GSM"] else 0
                    # Score based on how close the document GSM is to the requested GSM
                    gsm_diff = abs(requested_gsm - doc_gsm)
                    if gsm_diff == 0:
                        score += 15
                        explanation.append(f"Exact match for requested {requested_gsm} GSM")
                    elif gsm_diff <= 20:
                        score += 10
                        explanation.append(f"Close match for GSM: {doc_gsm} vs requested {requested_gsm}")
                    elif gsm_diff <= 50:
                        score += 5
                        explanation.append(f"Approximate GSM match: {doc_gsm} vs requested {requested_gsm}")

                # Check for location if mentioned
                if "location" in doc and doc["location"]:
                    location = str(doc["location"]).lower()
                    if location in query_lower:
                        score += 10
                        explanation.append(f"Matches requested location: {doc['location']}")

                # Check for press model if mentioned
                if "Press Model" in doc and doc["Press Model"]:
                    press_model = str(doc["Press Model"]).lower()
                    # Check for partial matches in query
                    for word in press_model.split():
                        if len(word) > 3 and word in query_lower:  # Only check significant words
                            score += 10
                            explanation.append(f"Matches requested press model: {doc['Press Model']}")
                            break

                return score, ". ".join(explanation) if explanation else "Matched based on general relevance."

            # Apply scoring and ranking
            scored_docs = []
            for doc in retrieved_docs:
                score, explanation = score_document(doc, user_query)
                doc_copy = doc.copy()
                doc_copy["score"] = score
                doc_copy["Reason for Selection"] = explanation
                scored_docs.append(doc_copy)

            # Sort by score in descending order
            ranked_results = sorted(scored_docs, key=lambda x: x.get("score", 0), reverse=True)

            # Limit to top results
            top_ranked = ranked_results[:10] if len(ranked_results) >= 10 else ranked_results

            # Clean up score field before returning
            for doc in top_ranked:
                if "score" in doc:
                    del doc["score"]

            return top_ranked

        return retrieved_docs

    except Exception as e:
        st.error(f"❌ Error during search: {str(e)}")
        # Fallback to simpler search without where clause
        try:
            st.info("ℹ️ Trying fallback search")
            query_results = collection.query(
                query_embeddings=[query_embedding],
                n_results=5
            )
            return query_results["metadatas"][0] if query_results["metadatas"] else []
        except Exception as fallback_error:
            st.error(f"❌ Fallback search failed: {str(fallback_error)}")
            return []


# Query Expansion using Llama 3.2
def expand_query_with_llm(user_query):
    """ Uses Llama 3.2 in Ollama to extract structured metadata filters from user query """
    prompt = f"""
    Analyze this user query: "{user_query}"

    The user is searching for printing documents with specific characteristics.
    Based ONLY on what's explicitly mentioned in the query, extract these parameters:

    1. Ink Coverage: [Painted, Heavy, Medium, Light]
    2. Media Weight GSM: [numerical value]
    3. Media Coating: [Coated, Uncoated]
    4. Media Finish: [Silk, Matte, Gloss]

    Format your response EXACTLY like this example:
    {{
        "Ink Coverage": "Heavy",
        "Media Weight GSM": 220,
        "Media Coating": "Coated",
        "Media Finish": "Silk"
    }}

    Only include parameters that are EXPLICITLY mentioned in the query.
    If a parameter is not mentioned, DO NOT include it in the JSON.
    The response must be ONLY the JSON object - no other text, explanations or code blocks.
    """

    response = llm.invoke(prompt)

    # Try to extract JSON from the response
    try:
        # Clean up the response - remove any markdown and extra text
        cleaned_response = response.strip()

        # Try to find JSON between curly braces
        json_start = cleaned_response.find("{")
        json_end = cleaned_response.rfind("}") + 1

        if json_start != -1 and json_end != -1:
            json_text = cleaned_response[json_start:json_end]
            structured_query = json.loads(json_text)
        else:
            # If no curly braces found, try removing markdown blocks
            if "```json" in cleaned_response and "```" in cleaned_response:
                json_start = cleaned_response.find("```json") + 7
                json_end = cleaned_response.rfind("```")
                json_text = cleaned_response[json_start:json_end].strip()
                structured_query = json.loads(json_text)
            else:
                # Last resort: try the whole response
                structured_query = json.loads(cleaned_response)
    except json.JSONDecodeError:
        # Create a basic structured query by simple text analysis
        structured_query = {}

        # Manual extraction for common parameters
        if "heavy" in user_query.lower():
            structured_query["Ink Coverage"] = "Heavy"
        elif "medium" in user_query.lower():
            structured_query["Ink Coverage"] = "Medium"
        elif "light" in user_query.lower():
            structured_query["Ink Coverage"] = "Light"

        if "coated" in user_query.lower():
            structured_query["Media Coating"] = "Coated"
        elif "uncoated" in user_query.lower():
            structured_query["Media Coating"] = "Uncoated"

        if "silk" in user_query.lower():
            structured_query["Media Finish"] = "Silk"
        elif "matte" in user_query.lower():
            structured_query["Media Finish"] = "Matte"
        elif "gloss" in user_query.lower():
            structured_query["Media Finish"] = "Gloss"

        # Try to extract GSM value
        import re
        gsm_match = re.search(r'(\d+)\s*gsm', user_query.lower())
        if gsm_match:
            structured_query["Media Weight GSM"] = int(gsm_match.group(1))

        # Log the manually extracted query
        print(f"Manually extracted query: {structured_query}")

    # Log the structured query for debugging
    print(f"Structured query: {structured_query}")

    return structured_query


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
if st.button("🔎 Search/Answer"):
    if not user_query:
        st.warning("⚠️ Please enter a search query or question.")
    else:
        # Create a placeholder for search results
        results_placeholder = st.empty()

        # Determine if this is a search or a question
        query_type, processed_query = process_query(user_query)

        if query_type == "question":
            # Handle as a specific question about a document using RAG
            with st.spinner("🧠 Using RAG to answer your question..."):
                answer = answer_document_question(processed_query)

                with results_placeholder.container():
                    st.success("✅ Answer generated using RAG")
                    st.markdown("### 📝 Answer")
                    st.markdown(answer)
                    st.info(
                        "If you want to search for documents instead, try using keywords like 'find documents with' or 'search for'.")
        else:
            # Handle as a search query
            with st.spinner("🔍 Searching documents with vector embeddings..."):
                try:
                    # First try our normal search
                    retrieved_docs = retrieve_documents(user_query)

                    if not retrieved_docs:
                        # If no results, try a more flexible search approach
                        st.info("Trying alternative search method...")
                        # Get the main keywords from the query
                        keywords = [word for word in user_query.lower().split()
                                    if len(word) > 3 and word not in ['with', 'and', 'the', 'for', 'find', 'documents']]

                        if keywords:
                            keyword_query = " ".join(keywords)
                            st.write(f"Searching with keywords: {keyword_query}")

                            # Create embedding for the keyword query
                            keyword_embedding = embedding_model.encode(keyword_query).tolist()

                            query_results = collection.query(
                                query_embeddings=[keyword_embedding],
                                n_results=5
                            )
                            retrieved_docs = query_results["metadatas"][0] if query_results["metadatas"] else []
                except Exception as e:
                    st.error(f"Search error: {str(e)}")
                    retrieved_docs = []

            if not retrieved_docs:
                st.error("❌ No matching documents found. Try a different query.")
                st.info(
                    "If you were asking a question instead of searching, try rephrasing with question words like 'what' or 'how'.")
            else:
                # Display results directly
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

                        # Create an analysis prompt
                        docs_json = json.dumps(retrieved_docs[:3], indent=2)
                        analysis_prompt = f"""
                        Based on the user query: "{user_query}"

                        Analyze these top documents that were retrieved by similarity:
                        {docs_json}

                        For each document, explain in one clear sentence why it's relevant to the query.
                        Focus on matching attributes and key information.
                        """

                        with st.spinner("Generating document analysis..."):
                            analysis = llm.invoke(analysis_prompt)
                            st.markdown(analysis)
                    else:
                        st.warning("Retrieved documents are missing expected columns")
                        st.json(retrieved_docs)

                st.sidebar.success("✅ **Vector Search Complete!**")
                st.info(
                    "If you were asking a specific question about a document instead of searching, try rephrasing with question words like 'what' or 'how'.")