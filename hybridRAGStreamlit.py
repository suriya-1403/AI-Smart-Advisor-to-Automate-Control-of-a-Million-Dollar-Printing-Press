import os
import json
import streamlit as st
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_ollama import OllamaLLM
import os

os.environ["PYTORCH_NO_CUDA"] = "1"

# ✅ Initialize Streamlit UI
st.set_page_config(page_title="RAG Chatbot", layout="wide")

st.sidebar.header("⚙️ Settings")

# ✅ Model Selection
selected_model = st.sidebar.selectbox("Select Model", ["llama3.2", "gpt-4", "mistral"], index=0)

# ✅ File Type Selection
file_type = st.sidebar.selectbox("Select File Type", ["PDF", "JSON"], index=0)

# ✅ File Upload
uploaded_files = st.sidebar.file_uploader(
    "Upload PDF/JSON files",
    type=["pdf", "json"],
    accept_multiple_files=True
)

if uploaded_files:
    st.sidebar.success(f"✅ Loaded {len(uploaded_files)} file(s) into memory.")

# ✅ Title
st.markdown("### 📄 RAG Chatbot")
st.write("Upload files and perform **structured document retrieval** using AI.")

# ✅ File Display Section
st.subheader("📂 Uploaded Files")
if uploaded_files:
    for file in uploaded_files:
        st.write(f"📄 {file.name} - {file.size / 1024:.1f} KB ")
else:
    st.info("No files uploaded yet. Please upload JSON or PDF files.")

# ✅ ChromaDB Initialization
chroma_path = "./chromadb_store"
# Check if storage was deleted & reinitialize if needed
if not os.path.exists(chroma_path):
    os.makedirs(chroma_path, exist_ok=True)  # Recreate folder if missing

# ✅ Create a fresh ChromaDB instance
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

# ✅ Load Embedding Model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# ✅ Initialize Llama 3.2
llm = OllamaLLM(model=selected_model)

# ✅ **Button: Store JSON in Vector Store**
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

# ✅ Query Input
st.subheader("🔍 Ask a question:")
user_query = st.text_input("Enter your search query:")


# ✅ Step 3: Query Expansion using Llama 3.2
def expand_query_with_llm(user_query):
    """ Uses Llama 3.2 in Ollama to extract structured metadata filters from user query """
    prompt = f"""
    Analyze this user query: "{user_query}"

    The user is searching for printing documents with specific characteristics.
    Based ONLY on what's explicitly mentioned in the query, extract these parameters:

    1. Ink Coverage: [Heavy, Medium, Light]
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


# ✅ Step 4: Retrieve Relevant Documents
def retrieve_documents(user_query):
    """ Retrieves documents based on query expansion and semantic search """
    # Get structured filters from the query
    query_filters = expand_query_with_llm(user_query)

    # Print debugging info
    st.write(f"📊 **Extracted Search Parameters:**")
    for key, value in query_filters.items():
        st.write(f"- {key}: {value}")

    # Build where clause dynamically based on extracted filters
    where_clause = {"$and": []}

    # Only add filters that were extracted from the query
    if "Ink Coverage" in query_filters and query_filters["Ink Coverage"]:
        where_clause["$and"].append({"Ink Coverage": query_filters["Ink Coverage"]})

    if "Media Coating" in query_filters and query_filters["Media Coating"]:
        where_clause["$and"].append({"Media Coating": query_filters["Media Coating"]})

    if "Media Finish" in query_filters and query_filters["Media Finish"]:
        where_clause["$and"].append({"Media Finish": query_filters["Media Finish"]})

    if "Media Weight GSM" in query_filters and query_filters["Media Weight GSM"]:
        # For numerical values like GSM, we don't want exact matching
        # Instead, we'll handle this in the ranking function
        pass

    # Handle location separately since it might be part of a string
    if "location" in query_filters and query_filters["location"]:
        where_clause["$and"].append({"location": query_filters["location"]})

    try:
        # If no filters were extracted, perform a pure semantic search without filtering
        if not where_clause["$and"]:
            st.info("ℹ️ Using semantic search without filters")
            query_results = collection.query(
                query_texts=[user_query],
                n_results=10  # Get more results for better ranking
            )
        else:
            # Apply filters with semantic search
            st.info("ℹ️ Using semantic search with filters")
            query_results = collection.query(
                query_texts=[user_query],
                n_results=10,  # Get more results for better ranking
                where=where_clause
            )

        return query_results["metadatas"][0] if query_results["metadatas"] else []

    except Exception as e:
        st.error(f"❌ Error during search: {str(e)}")
        # Fallback to simpler search without where clause
        try:
            st.info("ℹ️ Trying fallback search")
            query_results = collection.query(
                query_texts=[user_query],
                n_results=5
            )
            return query_results["metadatas"][0] if query_results["metadatas"] else []
        except:
            return []


# ✅ Step 5: Rank & Explain Using Llama 3.2
def rank_and_explain_with_llm(user_query, top_results):
    """ Uses a more reliable approach to rank documents based on user query """

    # First, let's create a proper ranking function without relying on LLM for JSON
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

    # Score and rank all documents
    scored_docs = []
    for doc in top_results:
        score, explanation = score_document(doc, user_query)
        doc_copy = doc.copy()
        doc_copy["score"] = score
        doc_copy["Reason for Selection"] = explanation
        scored_docs.append(doc_copy)

    # Sort by score in descending order
    ranked_results = sorted(scored_docs, key=lambda x: x.get("score", 0), reverse=True)

    # Limit to top 3 results
    top_ranked = ranked_results[:3] if len(ranked_results) >= 3 else ranked_results

    # Try to get better explanations from LLM if possible
    try:
        # Create a simpler prompt for document explanations only
        docs_info = []
        for i, doc in enumerate(top_ranked):
            doc_info = {
                "rank": i + 1,
                "event_id": doc.get("event_id", "N/A"),
                "Ink Coverage": doc.get("Ink Coverage", "N/A"),
                "Media Weight GSM": doc.get("Media Weight GSM", "N/A"),
                "Media Coating": doc.get("Media Coating", "N/A"),
                "Media Finish": doc.get("Media Finish", "N/A")
            }
            docs_info.append(doc_info)

        prompt = f"""
        Given the user query: "{user_query}"

        Explain why each of these documents matches the query:
        {json.dumps(docs_info, indent=2)}

        For each document, provide a one-sentence explanation of why it matches the query.
        Format your response as:

        Document 1: Your explanation here.
        Document 2: Your explanation here.
        Document 3: Your explanation here.
        """

        explanations = llm.invoke(prompt)

        # Extract explanations line by line
        explanation_lines = explanations.strip().split('\n')

        # Update explanations if available
        for i, line in enumerate(explanation_lines):
            if i < len(top_ranked) and line.startswith(f"Document {i + 1}:"):
                explanation = line.split(':', 1)[1].strip()
                if explanation:
                    top_ranked[i]["Reason for Selection"] = explanation

    except Exception as e:
        print(f"Error getting explanations: {str(e)}")
        # Keep original explanations if LLM fails

    # Clean up score field before returning
    for doc in top_ranked:
        if "score" in doc:
            del doc["score"]

    return top_ranked

    return ranked_results


# ✅ Debug Tools
if st.checkbox("Show Debug Options"):
    st.subheader("🛠️ Debug Tools")
    if st.button("📊 Show Collection Stats"):
        try:
            collection_count = collection.count()
            st.write(f"Documents in collection: {collection_count}")
            st.success("ChromaDB is working correctly!")
        except Exception as e:
            st.error(f"ChromaDB error: {str(e)}")

# ✅ Step 6: Search Button & Output
if st.button("🔎 Search"):
    if not user_query:
        st.warning("⚠️ Please enter a search query.")
    else:
        # Create a placeholder for search results
        results_placeholder = st.empty()

        with st.spinner("🔍 Searching documents..."):
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
                        query_results = collection.query(
                            query_texts=[keyword_query],
                            n_results=5
                        )
                        retrieved_docs = query_results["metadatas"][0] if query_results["metadatas"] else []
            except Exception as e:
                st.error(f"Search error: {str(e)}")
                retrieved_docs = []

        if not retrieved_docs:
            st.error("❌ No matching documents found. Try a different query.")
        else:
            with st.spinner("🧠 Analyzing and ranking results..."):
                ranked_results = rank_and_explain_with_llm(user_query, retrieved_docs)

            # Create a cleaner results display
            with results_placeholder.container():
                st.success(f"🎯 Found {len(ranked_results)} matching documents")

                # Convert ranked results to DataFrame
                ranked_df = pd.DataFrame(ranked_results)

                # Select columns to display in main table
                display_columns = ["event_id", "Ink Coverage", "Media Coating", "Media Finish",
                                   "Media Weight GSM", "Press Model", "location"]
                # Make sure we only include columns that exist
                available_columns = [col for col in display_columns if col in ranked_df.columns]

                if available_columns:
                    display_df = ranked_df[available_columns]

                    # Display Ranked Table
                    st.subheader("🔹 Ranked Documents")
                    st.dataframe(display_df)

                    # Display reasoning for each result
                    st.subheader("🔹 Selection Reasoning")
                    for i, result in enumerate(ranked_results):
                        reason = result.get("Reason for Selection", "No explanation provided.")
                        # Format the explanation to look better
                        if reason == "Matched based on general relevance.":
                            reason_color = "orange"
                        else:
                            reason_color = "green"

                        st.markdown(f"""
                        <div style="margin-bottom: 10px; padding: 10px; border-left: 4px solid {reason_color}; background-color: #f8f9fa;">
                            <strong>Document {i + 1} (Event ID: {result.get('event_id', 'N/A')})</strong><br>
                            {reason}
                        </div>
                        """, unsafe_allow_html=True)

                    # Additional Details: Press Model for Top Documents
                    if "Press Model" in ranked_df.columns:
                        press_details = [{"Event ID": doc.get("event_id", "N/A"),
                                          "Press Model": doc.get("Press Model", "N/A")}
                                         for doc in ranked_results]
                        press_df = pd.DataFrame(press_details)

                        st.subheader("🔹 Press Model Details")
                        st.dataframe(press_df)
                else:
                    st.warning("Retrieved documents are missing expected columns")
                    st.json(ranked_results)

            st.sidebar.success("✅ **Ranking & Filtering Complete!**")