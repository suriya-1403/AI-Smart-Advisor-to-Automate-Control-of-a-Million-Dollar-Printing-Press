import os
import json
import ollama

# Define the path to your folder containing JSON files
json_folder_path = "Dataset/Integrated Extraction"  # Change this to your actual folder path

# List to store extracted data
data = []

# Loop through all JSON files in the folder
for filename in os.listdir(json_folder_path):
    if filename.endswith(".json"):  # Process only JSON files
        file_path = os.path.join(json_folder_path, filename)

        # Open and read JSON file
        with open(file_path, "r", encoding="utf-8") as file:
            try:
                json_data = json.load(file)

                # Extract relevant attributes from the JSON structure
                extracted_info = {
                    "filename": filename,
                    "ink_coverage": json_data.get("focus", {}).get("Ink Coverage", "Unknown"),
                    "paper_weight": json_data.get("focus", {}).get("Media Weight GSM", "Unknown"),
                    "media_treatment": json_data.get("media_information", {}).get("name", "Unknown")
                }

                data.append(extracted_info)

            except json.JSONDecodeError:
                print(f"Error reading JSON file: {filename}")

# Convert extracted data to JSON format (raw input for Ollama)
json_input = json.dumps(data, indent=2)
print(json_input)
# #First query: Extract, sort, and format the data
# query_1 = f"""
# Your task is to analyze a collection of JSON documents and extract specific attributes, 
# including the filename, ink coverage, paper weight, and media treatment and make a table with these attributes

# After extracting this information from all the JSON files, generate a table that presents these details 
# in a structured format. The table should be sorted in ascending order:
#   - First by ink coverage,
#   - Then by paper weight,
#   - Finally by media treatment in alphabetical order.

# The output should be a **well-formatted table displaying the sorted data**, not code.

# Here is the extracted data:

# {json_input}

# Sort and format the data accordingly and present it as a table.
# """

# # Query Ollama for the first response
# response_1 = ollama.chat(model="llama3.2", messages=[{"role": "user", "content": query_1}])

# # Extract the formatted table from the first response
# sorted_table = response_1["message"]["content"]

# print("### Sorted Table ###")
# print(sorted_table)

#Second query: Get the best-matching three documents
query_2 = f"""
using following structured json:

{json_input}

Generate a second table that includes **up to two documents** that best match the following criteria:

- **Ink Coverage Class**: Corresponding to an **ink coverage of 42%**  
- **Paper Weight Class**: Corresponding to a **paper weight of 220 gsm**  
- **Coated Media Class**: The document must belong to a **coated media class**  

The best match should be determined based on the following priority order:
1. **First, select only those documents that match the ink coverage class.**  
2. **If multiple documents match, then filter based on the closest paper weight class.**  
3. **Finally, if further filtering is needed, prioritize documents based on the media treatment classification.**  

Present the final selection in a **structured table format** containing **up to two most relevant documents** based on the above criteria. Please dont give me code. give the structured table
"""

# Query Ollama for the second response
response_2 = ollama.generate(model="mistral",prompt=query_2)

# Extract the best-matching documents table
best_match_table = response_2["response"]

print("\n### Best-Matching Documents Table ###")
print(best_match_table)
