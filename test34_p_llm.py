import os
import json
import re
from langchain_ollama import OllamaLLM


class ImprovedLLMFilter:
    def __init__(self, model_name="llama3.2:3b-instruct-q5_K_M"):
        self.model = OllamaLLM(model=model_name)

    def process_json_folder(self, folder_path):
        """Load JSON documents as a list of dictionaries."""
        documents = []

        for filename in os.listdir(folder_path):
            if filename.lower().endswith(".json"):
                filepath = os.path.join(folder_path, filename)
                try:
                    with open(filepath, "r", encoding="utf-8") as file:
                        data = json.load(file)
                        # Ensure file_name is in the data
                        if "file_name" not in data:
                            data["file_name"] = filename
                        documents.append(data)
                except json.JSONDecodeError:
                    print(f"Error parsing {filename}")

        print(f"✅ Loaded {len(documents)} file(s) from {folder_path}.")
        return documents

    def list_document_fields(self, documents):
        """List all unique fields from documents and sample values."""
        field_info = {}
        for doc in documents:
            for field, value in doc.items():
                if field not in field_info:
                    field_info[field] = []
                if value and str(value) not in [str(v) for v in field_info[field]]:
                    field_info[field].append(value)
                if len(field_info[field]) >= 5:  # Limit samples to 5
                    continue

        # Print field summary
        print("\n📋 Document Fields:")
        for field, values in sorted(field_info.items()):
            print(f"- {field}: {', '.join(str(v) for v in values[:3])}{'...' if len(values) > 3 else ''}")

        return field_info

    def evaluate_document_with_llm(self, document, query):
        """Ask the LLM to evaluate whether the document matches the query criteria and determine its sort ranking."""
        # Convert document to JSON string
        doc_json = json.dumps(document, indent=2)

        # Create a more structured prompt that forces specific JSON output
        # and emphasizes the LLM's role in sorting
        prompt = f"""
        # TASK: Document Matching and Ranking Evaluation

        You are evaluating if a document matches specific query criteria AND determining its ranking.
        Return ONLY a valid JSON object directly - no code, no explanations, just the JSON.

        ## Query
        {query}

        ## Document
        ```json
        {doc_json}
        ```

        ## CRITERIA INTERPRETATION
        - "Ink Coverage" must exactly match "Heavy" 
        - "Media Coating" must contain the substring "Coated Inkjet"
        - "Media Weight GSM" should be evaluated based on numeric distance to target (110)
        - A document matches ONLY if it satisfies ALL THREE criteria

        ## SORTING INSTRUCTIONS
        - For "Media Weight GSM", calculate the absolute numeric difference: |document_gsm - 110|
        - Use this exact numeric difference as the "sort_value"
        - Documents with GSM closer to 110 should get LOWER sort values
        - For example: 100 GSM = sort_value of 10, 120 GSM = sort_value of 10, 150 GSM = sort_value of 40

        ## RESPONSE FORMAT (STRICT JSON ONLY)
        {{
            "matches": true or false,
            "match_score": "X/3 criteria matched",
            "match_reason": "Brief reason for matching or not",
            "sort_value": numeric_difference_from_target_gsm
        }}

        IMPORTANT:
        - I need ONLY the JSON object above
        - DO NOT write any code
        - DO NOT add explanations
        - DO NOT add comments
        - DO NOT add markdown formatting
        """

        # Invoke the LLM
        response = self.model.invoke(prompt)

        # Print raw response for debugging
        print(f"\n📌 Raw LLM Response for {document.get('file_name', 'unknown')}:\n{response}\n")

        # Extract JSON from the response
        try:
            return self.extract_json_from_response(response)
        except Exception as e:
            print(f"⚠️ Error evaluating document {document.get('file_name', 'unknown')}: {e}")
            # Fallback for when extraction fails
            return self.fallback_evaluation(document, query)

    def extract_json_from_response(self, response):
        """Extract JSON from LLM response with improved error handling."""
        # First, clean the response by removing common non-JSON content
        clean_response = response.strip()

        # Remove common prefixes that might appear before JSON
        clean_response = re.sub(r'^(.*?)(\{)', r'\2', clean_response, count=1)

        # Remove common suffixes that might appear after JSON
        clean_response = re.sub(r'(\})(.*?)

    def fallback_evaluation(self, document, query):
        """Create a fallback evaluation when LLM extraction fails.
        This is a fallback only - the LLM should be doing the actual sorting."""
        # Parse query to extract criteria
        ink_coverage_match = "Heavy" in query
        media_coating_match = "Coated Inkjet" in query

        # Extract target GSM weight from query
        gsm_match = re.search(r'(\d+)\s*GSM', query)
        target_gsm = int(gsm_match.group(1)) if gsm_match else 110  # Default to 110

        # Check document against criteria
        doc_ink = document.get("Ink Coverage", "")
        doc_coating = document.get("Media Coating", "")
        doc_gsm_str = document.get("Media Weight GSM", "0")
        # Convert GSM to integer for comparison
        try:
            doc_gsm = int(doc_gsm_str)
        except ValueError:
            doc_gsm = 0

        # Calculate matches
        matches = 0
        total_criteria = 3
        match_reasons = []

        if doc_ink == "Heavy":
            matches += 1
            match_reasons.append("Ink Coverage is Heavy")

        if "Coated Inkjet" in doc_coating:
            matches += 1
            match_reasons.append("Media Coating contains Coated Inkjet")

        # For GSM, calculate distance rather than exact match
        gsm_distance = abs(doc_gsm - target_gsm)

        # Even for GSM, we count it as a match if it's within a reasonable range
        # This is more lenient than the LLM's evaluation would be
        if gsm_distance <= 50:  # More lenient for fallback
            matches += 1
            match_reasons.append(f"Media Weight GSM ({doc_gsm}) is within range of target ({target_gsm})")

        # Determine overall match - the LLM would do this more intelligently
        # For fallback we're more lenient
        overall_match = matches >= 1  # At least one criterion must match

        # For sort value, use the raw GSM distance (LLM would be more sophisticated)
        # This preserves the LLM's role as the primary sorter when it works
        return {
            "matches": overall_match,
            "match_score": f"{matches}/{total_criteria} criteria matched",
            "match_reason": "; ".join(match_reasons) if match_reasons else "No criteria matched",
            "sort_value": gsm_distance  # Sort by GSM distance (lower is better)
        }

    def find_matching_documents(self, documents, query):
        """Find documents matching the query, letting the LLM handle ranking."""
        print(f"\n🔍 Evaluating {len(documents)} documents with LLM-based sorting and matching")

        results = []

        # Process documents in batches to show progress
        for i, doc in enumerate(documents):
            if i % 5 == 0:
                print(f"  Progress: {i}/{len(documents)} documents evaluated")

            # Include only the fields relevant to the query
            simplified_doc = {
                "file_name": doc.get("file_name", "unknown"),
                "Ink Coverage": doc.get("Ink Coverage", ""),
                "Media Coating": doc.get("Media Coating", ""),
                "Media Weight GSM": doc.get("Media Weight GSM", ""),
                "Media Finish": doc.get("Media Finish", "")
            }

            # Get evaluation result with LLM determining the sort value
            evaluation = self.evaluate_document_with_llm(simplified_doc, query)

            # Only include if it matches
            if evaluation and evaluation.get("matches", False):
                results.append({
                    "document": doc,
                    "evaluation": evaluation
                })

        print(f"  ✅ LLM found {len(results)} matching documents")

        # Sort by sort_value provided by the LLM (convert to float to ensure sorting works)
        sorted_results = sorted(
            results,
            key=lambda x: float(x["evaluation"].get("sort_value", float('inf')))
        )

        # Log the top sorting values to verify LLM is doing the sorting
        if sorted_results:
            print("  📊 Top 3 LLM sort values:")
            for i, result in enumerate(sorted_results[:3], 1):
                doc_name = result["document"].get("file_name", "unknown")
                sort_val = result["evaluation"].get("sort_value", "N/A")
                print(f"    {i}. {doc_name}: sort_value={sort_val}")

        # Apply result limit
        limit = self.extract_limit_from_query(query)
        limited_results = sorted_results[:limit] if limit > 0 else sorted_results

        # Format results
        return {
            "matching_documents": [
                {
                    "file_name": item["document"].get("file_name", ""),
                    "match_reason": item["evaluation"].get("match_reason", ""),
                    "match_score": item["evaluation"].get("match_score", ""),
                    "sort_value": item["evaluation"].get("sort_value", ""),
                    "document_data": item["document"]
                } for item in limited_results
            ],
            "total_matching": len(results),
            "limit_applied": limit,
            "explanation": f"Found {len(results)} matching documents, showing top {min(limit, len(limited_results))}"
        }

    def extract_limit_from_query(self, query):
        """Extract result limit from query."""
        limit_match = re.search(r'(?:top|return|limit|show)\s+(\d+)', query, re.IGNORECASE)
        if limit_match:
            return int(limit_match.group(1))
        else:
            return 3  # Default to 3

    def format_results_as_table(self, results):
        """Format the filtered results as a markdown table."""
        matching_docs = results.get("matching_documents", [])
        explanation = results.get("explanation", "")

        if not matching_docs:
            return "No matching documents found based on the specified criteria."

        # Determine which fields to show
        fields_to_show = ["Ink Coverage", "Media Coating", "Media Weight GSM"]

        # Create table header
        columns = ["Rank", "File Name"] + fields_to_show + ["Match Score", "Sort Value"]
        table = "| " + " | ".join(f"**{col}**" for col in columns) + " |\n"
        table += "|" + "|".join([":---------"] * len(columns)) + "|\n"

        # Add rows
        for rank, match in enumerate(matching_docs, 1):
            doc = match.get("document_data", {})

            row = [
                str(rank),
                match.get("file_name", f"Doc_{rank}")
            ]

            # Add field values
            for field in fields_to_show:
                row.append(str(doc.get(field, "N/A")))

            # Add match score and sort value
            row.append(match.get("match_score", "N/A"))
            row.append(str(match.get("sort_value", "N/A")))

            table += "| " + " | ".join(row) + " |\n"

        # Add explanation and match reasons
        if explanation:
            table += f"\n**Results Summary**: {explanation}\n"

        # Add detailed match explanations
        table += "\n**Match Details**:\n"
        for rank, match in enumerate(matching_docs, 1):
            table += f"{rank}. **{match.get('file_name', '')}**: {match.get('match_reason', 'N/A')}\n"

        return table

    def process_query(self, documents, query):
        """Process a natural language query using direct LLM document evaluation."""
        # Evaluate each document against the query
        results = self.find_matching_documents(documents, query)

        # Format the results
        formatted_results = self.format_results_as_table(results)

        return {
            "results": formatted_results,
            "raw_results": results
        }


def main():
    # Configure your folder path here
    folder_path = "/path/to/your/json/files"

    # Example query - customize as needed
    query = """Find documents that MUST match ALL of these criteria:
    1. "Ink Coverage" must be exactly "Heavy" 
    2. "Media Coating" must contain "Coated Inkjet"
    3. "Media Weight GSM" must be closest to 110
    Sort by how close the Media Weight is to 110 GSM.
    Return top 3 results only."""

    # Create filter and process documents
    filter_tool = ImprovedLLMFilter()
    documents = filter_tool.process_json_folder(folder_path)

    # Optional: List document fields for reference
    filter_tool.list_document_fields(documents)

    # Process the query
    results = filter_tool.process_query(documents, query)

    # Display results
    print(f"\n📊 Results:\n{results['results']}\n")

    # Report match count
    matching_docs = results['raw_results'].get('matching_documents', [])
    if matching_docs:
        print(f"✅ Found {len(matching_docs)} matching documents")
    else:
        print("❌ No matching documents found")


if __name__ == "__main__":
    main()
, r'\1', clean_response, count = 1)

# Remove markdown code blocks
clean_response = re.sub(r'```(?:json)?\s*(.*?)\s*```', r'\1', clean_response, re.DOTALL)

# Try to find a JSON object in the response
json_match = re.search(r'(\{.*\})', clean_response, re.DOTALL)
if json_match:
    json_str = json_match.group(1).strip()

    # Try parsing directly
try:
    return json.loads(json_str)
except json.JSONDecodeError:
    # Apply fixes to common JSON issues
    fixed_str = json_str

    # Fix unquoted keys (more comprehensive regex)
    fixed_str = re.sub(r'([{,]\s*)(\w+)(\s*:)', r'\1"\2"\3', fixed_str)

    # Convert single quotes to double quotes
    fixed_str = fixed_str.replace("'", '"')

    # Fix boolean values
    fixed_str = re.sub(r':\s*true\b', ': true', fixed_str, flags=re.IGNORECASE)
    fixed_str = re.sub(r':\s*false\b', ': false', fixed_str, flags=re.IGNORECASE)

    # Remove trailing commas
    fixed_str = re.sub(r',\s*}', '}', fixed_str)
    fixed_str = re.sub(r',\s*\]', ']', fixed_str)

    # Fix fraction-like match scores (e.g. 1/3 -> "1/3")
    fixed_str = re.sub(r'"match_score"\s*:\s*(\d+/\d+)', r'"match_score": "\1"', fixed_str)

    # Try again with fixes applied
    try:
        return json.loads(fixed_str)
    except json.JSONDecodeError as e:
        print(f"⚠️ JSON error after fixes: {e}\nAttempted to parse: {fixed_str}")

        # Last resort: try to construct a valid JSON from scratch
        # This is only for extreme cases where other fixes fail
        matches_match = re.search(r'"matches"\s*:\s*(true|false)', fixed_str, re.IGNORECASE)
        score_match = re.search(r'"match_score"\s*:\s*"?(\d+/\d+)"?', fixed_str)
        reason_match = re.search(r'"match_reason"\s*:\s*"([^"]*)"', fixed_str)
        sort_match = re.search(r'"sort_value"\s*:\s*(-?\d+(?:\.\d+)?)', fixed_str)

        if matches_match and sort_match:
            # If we have at least the matches and sort value, build a minimal valid JSON
            matches = matches_match.group(1).lower() == "true"
            score = score_match.group(1) if score_match else "0/3"
            reason = reason_match.group(1) if reason_match else "Unknown reason"
            sort_value = float(sort_match.group(1))

            result = {
                "matches": matches,
                "match_score": score,
                "match_reason": reason,
                "sort_value": sort_value
            }
            print(f"⚠️ Created fallback JSON: {result}")
            return result

        # If we can't extract the core fields, give up
        raise ValueError("Could not parse JSON after applying fixes")
else:
    raise ValueError("No JSON object found in LLM response")


def fallback_evaluation(self, document, query):
    """Create a fallback evaluation when LLM extraction fails.
    This is a fallback only - the LLM should be doing the actual sorting."""
    # Parse query to extract criteria
    ink_coverage_match = "Heavy" in query
    media_coating_match = "Coated Inkjet" in query

    # Extract target GSM weight from query
    gsm_match = re.search(r'(\d+)\s*GSM', query)
    target_gsm = int(gsm_match.group(1)) if gsm_match else 110  # Default to 110

    # Check document against criteria
    doc_ink = document.get("Ink Coverage", "")
    doc_coating = document.get("Media Coating", "")
    doc_gsm_str = document.get("Media Weight GSM", "0")
    # Convert GSM to integer for comparison
    try:
        doc_gsm = int(doc_gsm_str)
    except ValueError:
        doc_gsm = 0

    # Calculate matches
    matches = 0
    total_criteria = 3
    match_reasons = []

    if doc_ink == "Heavy":
        matches += 1
        match_reasons.append("Ink Coverage is Heavy")

    if "Coated Inkjet" in doc_coating:
        matches += 1
        match_reasons.append("Media Coating contains Coated Inkjet")

    # For GSM, calculate distance rather than exact match
    gsm_distance = abs(doc_gsm - target_gsm)

    # Even for GSM, we count it as a match if it's within a reasonable range
    # This is more lenient than the LLM's evaluation would be
    if gsm_distance <= 50:  # More lenient for fallback
        matches += 1
        match_reasons.append(f"Media Weight GSM ({doc_gsm}) is within range of target ({target_gsm})")

    # Determine overall match - the LLM would do this more intelligently
    # For fallback we're more lenient
    overall_match = matches >= 1  # At least one criterion must match

    # For sort value, use the raw GSM distance (LLM would be more sophisticated)
    # This preserves the LLM's role as the primary sorter when it works
    return {
        "matches": overall_match,
        "match_score": f"{matches}/{total_criteria} criteria matched",
        "match_reason": "; ".join(match_reasons) if match_reasons else "No criteria matched",
        "sort_value": gsm_distance  # Sort by GSM distance (lower is better)
    }


def find_matching_documents(self, documents, query):
    """Find documents matching the query, letting the LLM handle ranking."""
    print(f"\n🔍 Evaluating {len(documents)} documents with LLM-based sorting and matching")

    results = []

    # Process documents in batches to show progress
    for i, doc in enumerate(documents):
        if i % 5 == 0:
            print(f"  Progress: {i}/{len(documents)} documents evaluated")

        # Include only the fields relevant to the query
        simplified_doc = {
            "file_name": doc.get("file_name", "unknown"),
            "Ink Coverage": doc.get("Ink Coverage", ""),
            "Media Coating": doc.get("Media Coating", ""),
            "Media Weight GSM": doc.get("Media Weight GSM", ""),
            "Media Finish": doc.get("Media Finish", "")
        }

        # Get evaluation result with LLM determining the sort value
        evaluation = self.evaluate_document_with_llm(simplified_doc, query)

        # Only include if it matches
        if evaluation and evaluation.get("matches", False):
            results.append({
                "document": doc,
                "evaluation": evaluation
            })

    print(f"  ✅ LLM found {len(results)} matching documents")

    # Sort by sort_value provided by the LLM (convert to float to ensure sorting works)
    sorted_results = sorted(
        results,
        key=lambda x: float(x["evaluation"].get("sort_value", float('inf')))
    )

    # Log the top sorting values to verify LLM is doing the sorting
    if sorted_results:
        print("  📊 Top 3 LLM sort values:")
        for i, result in enumerate(sorted_results[:3], 1):
            doc_name = result["document"].get("file_name", "unknown")
            sort_val = result["evaluation"].get("sort_value", "N/A")
            print(f"    {i}. {doc_name}: sort_value={sort_val}")

    # Apply result limit
    limit = self.extract_limit_from_query(query)
    limited_results = sorted_results[:limit] if limit > 0 else sorted_results

    # Format results
    return {
        "matching_documents": [
            {
                "file_name": item["document"].get("file_name", ""),
                "match_reason": item["evaluation"].get("match_reason", ""),
                "match_score": item["evaluation"].get("match_score", ""),
                "sort_value": item["evaluation"].get("sort_value", ""),
                "document_data": item["document"]
            } for item in limited_results
        ],
        "total_matching": len(results),
        "limit_applied": limit,
        "explanation": f"Found {len(results)} matching documents, showing top {min(limit, len(limited_results))}"
    }


def extract_limit_from_query(self, query):
    """Extract result limit from query."""
    limit_match = re.search(r'(?:top|return|limit|show)\s+(\d+)', query, re.IGNORECASE)
    if limit_match:
        return int(limit_match.group(1))
    else:
        return 3  # Default to 3


def format_results_as_table(self, results):
    """Format the filtered results as a markdown table."""
    matching_docs = results.get("matching_documents", [])
    explanation = results.get("explanation", "")

    if not matching_docs:
        return "No matching documents found based on the specified criteria."

    # Determine which fields to show
    fields_to_show = ["Ink Coverage", "Media Coating", "Media Weight GSM"]

    # Create table header
    columns = ["Rank", "File Name"] + fields_to_show + ["Match Score", "Sort Value"]
    table = "| " + " | ".join(f"**{col}**" for col in columns) + " |\n"
    table += "|" + "|".join([":---------"] * len(columns)) + "|\n"

    # Add rows
    for rank, match in enumerate(matching_docs, 1):
        doc = match.get("document_data", {})

        row = [
            str(rank),
            match.get("file_name", f"Doc_{rank}")
        ]

        # Add field values
        for field in fields_to_show:
            row.append(str(doc.get(field, "N/A")))

        # Add match score and sort value
        row.append(match.get("match_score", "N/A"))
        row.append(str(match.get("sort_value", "N/A")))

        table += "| " + " | ".join(row) + " |\n"

    # Add explanation and match reasons
    if explanation:
        table += f"\n**Results Summary**: {explanation}\n"

    # Add detailed match explanations
    table += "\n**Match Details**:\n"
    for rank, match in enumerate(matching_docs, 1):
        table += f"{rank}. **{match.get('file_name', '')}**: {match.get('match_reason', 'N/A')}\n"

    return table


def process_query(self, documents, query):
    """Process a natural language query using direct LLM document evaluation."""
    # Evaluate each document against the query
    results = self.find_matching_documents(documents, query)

    # Format the results
    formatted_results = self.format_results_as_table(results)

    return {
        "results": formatted_results,
        "raw_results": results
    }


def main():
    # Configure your folder path here
    folder_path = "/path/to/your/json/files"

    # Example query - customize as needed
    query = """Find documents that MUST match ALL of these criteria:
    1. "Ink Coverage" must be exactly "Heavy" 
    2. "Media Coating" must contain "Coated Inkjet"
    3. "Media Weight GSM" must be closest to 110
    Sort by how close the Media Weight is to 110 GSM.
    Return top 3 results only."""

    # Create filter and process documents
    filter_tool = ImprovedLLMFilter()
    documents = filter_tool.process_json_folder(folder_path)

    # Optional: List document fields for reference
    filter_tool.list_document_fields(documents)

    # Process the query
    results = filter_tool.process_query(documents, query)

    # Display results
    print(f"\n📊 Results:\n{results['results']}\n")

    # Report match count
    matching_docs = results['raw_results'].get('matching_documents', [])
    if matching_docs:
        print(f"✅ Found {len(matching_docs)} matching documents")
    else:
        print("❌ No matching documents found")


if __name__ == "__main__":
    main()