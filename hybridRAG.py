import os
import json
import chromadb
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from tabulate import tabulate
from langchain_ollama import OllamaLLM

# ✅ Step 1: Initialize ChromaDB (Local Vector Store)
chroma_client = chromadb.PersistentClient(path="./chromadb_store")
collection = chroma_client.get_or_create_collection("json_documents")

# ✅ Load Embedding Model (For Semantic Search)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# ✅ Initialize Llama 3.2 in Ollama
llm = OllamaLLM(model="llama3.2")

# ✅ Step 2: Load JSON Files & Store in ChromaDB
data_dir = "/Users/suriya/Documents/Github/Structured-Data-Similarity-Search-using-NLP/Dataset/ExtractedJSON/GT"  # Change to your JSON folder path
json_list = []

for file in os.listdir(data_dir):
    if file.endswith(".json"):
        with open(os.path.join(data_dir, file), "r") as f:
            json_data = json.load(f)
            json_list.append(json_data)

# Convert JSON Data to DataFrame
df = pd.DataFrame(json_list)

# Store in ChromaDB with metadata
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

print("✅ JSON data successfully indexed in ChromaDB.")


# ✅ Step 3: Use Llama 3.2 for Query Expansion
def expand_query_with_llm(user_query):
    """ Uses Llama 3.2 in Ollama to convert a natural language query into structured metadata filters """
    prompt = f"""
    Given the following user query: "{user_query}", extract and return structured search filters in JSON format.
    The search filters should include:
    - Ink Coverage (e.g., Heavy, Medium)
    - Paper Weight GSM 
    - Media Coating (e.g., Coated, Uncoated)
    Ensure that the response is a valid JSON object.
    """

    response = llm.invoke(prompt)
    try:
        structured_query = json.loads(response)
    except json.JSONDecodeError:
        structured_query = {"Ink Coverage": "Heavy", "Media Weight GSM": 100,
                            "Media Coating": "Uncoated Inkjet"}  # Default fallback

    return structured_query


# ✅ Step 4: Retrieve Relevant Documents
def retrieve_documents(user_query):
    """ Retrieves top 5 documents matching the LLM-expanded structured query """
    query_filters = expand_query_with_llm(user_query)

    query_results = collection.query(
        query_texts=[user_query],
        n_results=5,  # Get top 5 candidates
        where={
            "$and": [
                {"Ink Coverage": query_filters["Ink Coverage"]},
                {"Media Coating": query_filters["Media Coating"]}
            ]
        }
    )
    return query_results["metadatas"][0] if query_results["metadatas"] else []


# ✅ Step 5: Rank & Explain Using Llama 3.2
import json

import json

def rank_and_explain_with_llm(top_results):
    """ Uses Llama 3.2 in Ollama to rank documents & generate ranking explanations """
    prompt = f"""
    Using the following structured JSON documents, generate a **ranked table** containing only the **top 3 best-matching documents** based on the given selection criteria.

    ---
    **Ranking Criteria (Priority Order)**:
    1. **Ink Coverage Range** should be Heavy.
    2. **Paper Weight** must be closest to 100 GSM.
    3. **Coated Media** is Uncoated Inkjet.

    ---
    **Step 1: Rank the Documents**
    - Rank based on the criteria above.
    - Provide a brief explanation for each ranking decision.

    **Step 2: Output the Top 3 Matches**
    Return a structured JSON object **inside triple backticks (`json\n...\n`)**.

    ---
    Here are the retrieved documents:
    {json.dumps(top_results, indent=2)}

    ---
    **VERY IMPORTANT INSTRUCTIONS:**
    
    1. DO NOT generate any Python code in your response.
    2. ONLY respond with valid JSON format as shown below.
    3. Your entire response MUST be a valid JSON object wrapped in ```json and ``` markers.
    4. DO NOT include any explanations, comments, or text outside of the JSON structure.
    5. DO NOT provide any print statements, function definitions, or any other programming constructs.
    
    **Return the response in this exact JSON format:**
    ```json
    {{
        "ranked_documents": [
            {{
                "event_id": "XXX",
                "Ink Coverage": "Heavy",
                "Media Weight GSM": 220,
                "Media Coating": "Coated",
                "Reason for Selection": "Explain why this document ranked 1st."
            }},
            {{
                "event_id": "YYY",
                "Ink Coverage": "Heavy",
                "Media Weight GSM": 215,
                "Media Coating": "Coated",
                "Reason for Selection": "Explain why this document ranked 2nd."
            }},
            {{
                "event_id": "ZZZ",
                "Ink Coverage": "Heavy",
                "Media Weight GSM": 200,
                "Media Coating": "Coated",
                "Reason for Selection": "Explain why this document ranked 3rd."
            }}
        ]
    }}
    ```

    **I REPEAT: RETURN ONLY VALID JSON DATA. NO PYTHON CODE. NO EXPLANATIONS. NO ADDITIONAL TEXT.**
    """

    response = llm.invoke(prompt)
    print("Raw LLM Response:", response)

    # Extract JSON from response using markdown fencing
    json_start = response.find("```json")
    json_end = response.rfind("```")

    if json_start != -1 and json_end != -1:
        json_response = response[json_start+7:json_end].strip()  # Extract only JSON part
    else:
        json_response = response  # Try raw response if no JSON fencing

    try:
        ranked_results = json.loads(json_response)["ranked_documents"]
    except json.JSONDecodeError:
        print("⚠️ Warning: LLM response is still invalid. Using fallback ranking.")
        ranked_results = top_results[:3]  # Fallback: Use top 3 retrieved results

    # Ensure 'Reason for Selection' exists for every document
    for doc in ranked_results:
        if "Reason for Selection" not in doc:
            doc["Reason for Selection"] = f"Closest match based on Ink Coverage and Media Weight GSM ({doc['Media Weight GSM']})."

    return ranked_results




# ✅ Step 6: User Query and Output Results
user_query = "Find documents with heavy ink coverage, closest to 220 gsm, and coated media."
retrieved_docs = retrieve_documents(user_query)

if not retrieved_docs:
    print("❌ No matching documents found.")
else:
    ranked_results = rank_and_explain_with_llm(retrieved_docs)

    # Output Ranked Table
    ranked_table = []
    for i, doc in enumerate(ranked_results):
        ranked_table.append([
            i + 1, doc["event_id"], doc["Ink Coverage"], doc["Media Weight GSM"],
            doc["Media Coating"], doc["Reason for Selection"]
        ])

    print("\n🔹 **Ranked Documents Table:**")
    print(tabulate(ranked_table, headers=["Rank", "Document Name", "Ink Coverage", "Paper Weight GSM", "Coated Media",
                                          "Reason for Selection"], tablefmt="grid"))

    # # Output Press Model Details for Top 3 Documents
    # press_details_table = []
    # for doc in ranked_results:
    #     press_details_table.append([
    #         doc["event_id"], doc["Press Model"]
    #     ])
    #
    # print("\n🔹 **Press Model Details for Top 3 Documents:**")
    # print(tabulate(press_details_table, headers=["Filename", "Press Model"], tablefmt="grid"))

print("\n✅ **Ranking & Filtering Complete!**")
