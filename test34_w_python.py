import os
import json
import re
from PyPDF2 import PdfReader
from langchain_ollama import OllamaLLM


class RobustMinCriteriaFilter:
    def __init__(self, model_name="llama3.2"):
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

    def parse_query_with_llm(self, query):
        """Use LLM to parse a natural language query into structured criteria with robust error handling."""
        prompt = f"""
        Parse the following user query for document filtering:

        USER QUERY: {query}

        Extract the exact filtering criteria and sort specifications.

        Return ONLY a plain JSON object with this structure:
        {{
            "criteria": [
                {{
                    "field": "[exact field name]",
                    "operation": "equals|contains|greater_than|less_than|closest_to",
                    "value": "[exact value]"
                }}
            ],
            "sort": {{
                "field": "[field to sort by]",
                "operation": "closest_to",
                "value": "[target value]"
            }},
            "limit": [number of results to return],
            "min_criteria_match": [minimum number of criteria that must match]
        }}

        Make sure to:
        1. Use the EXACT field names as they appear in the documents
        2. Interpret the user's intent accurately
        3. Format as valid JSON only - no markdown, no comments, no explanation
        4. Set a default limit of 3 unless specified otherwise
        5. If the query mentions "at least X criteria", set min_criteria_match to X
        """

        # Get criteria parsing from LLM
        parse_response = self.model.invoke(prompt)
        print("\n🔍 LLM Parse Response:")
        print(parse_response)

        # Try to extract and fix JSON from the response
        try:
            # Remove any markdown code blocks or explanation text
            clean_response = re.sub(r'```(?:json)?\s*(.*?)\s*```', r'\1', parse_response, flags=re.DOTALL)

            # Extract the JSON object (anything that looks like it could be JSON)
            json_match = re.search(r'(\{.*\})', clean_response, re.DOTALL)

            if json_match:
                json_str = json_match.group(1)

                # Try to fix common JSON errors
                # 1. Ensure the JSON is properly terminated
                open_braces = json_str.count('{')
                close_braces = json_str.count('}')
                if open_braces > close_braces:
                    json_str += '}' * (open_braces - close_braces)

                # 2. Fix trailing commas
                json_str = re.sub(r',\s*}', '}', json_str)
                json_str = re.sub(r',\s*]', ']', json_str)

                # Parse the JSON
                filter_spec = json.loads(json_str)

                # Ensure required fields exist
                if "criteria" not in filter_spec:
                    filter_spec["criteria"] = []
                if "sort" not in filter_spec:
                    filter_spec["sort"] = {}
                if "limit" not in filter_spec:
                    filter_spec["limit"] = 3

                # Extract the min_criteria_match
                if "min_criteria_match" not in filter_spec:
                    # Check if query contains phrases like "at least X criteria"
                    min_match = re.search(r'at\s+least\s+(\d+)', query, re.IGNORECASE)
                    if min_match:
                        filter_spec["min_criteria_match"] = int(min_match.group(1))
                    else:
                        filter_spec["min_criteria_match"] = 1  # Default to 1

                return filter_spec
            else:
                print("⚠️ No JSON object found in response")

        except json.JSONDecodeError as e:
            print(f"⚠️ JSON decode error: {e}")
        except Exception as e:
            print(f"⚠️ Error parsing response: {e}")

        # If we get here, something went wrong - create a default spec
        # but try to extract minimum criteria from the query anyway
        min_match = re.search(r'at\s+least\s+(\d+)', query, re.IGNORECASE)
        min_criteria = int(min_match.group(1)) if min_match else 1

        # Fall back to a manually constructed filter spec based on the query
        return self._construct_fallback_filter_spec(query, min_criteria)

    def _construct_fallback_filter_spec(self, query, min_criteria=1):
        """Manually construct a filter spec as a fallback when parsing fails."""
        print("⚠️ Using fallback query parsing")
        filter_spec = {
            "criteria": [],
            "sort": {},
            "limit": 3,
            "min_criteria_match": min_criteria
        }

        # Extract field names from the query
        fields = {
            "Ink Coverage": re.search(r'Ink\s+Coverage\s+is\s+[\"\']?([^\"\']+)[\"\']?', query, re.IGNORECASE),
            "Media Coating": re.search(r'Media\s+Coating\s+(?:is|contains)\s+[\"\']?([^\"\']+)[\"\']?', query,
                                       re.IGNORECASE),
            "Media Weight GSM": re.search(r'Media\s+Weight\s+GSM\s+is\s+(?:closest\s+to|around)\s+(\d+)', query,
                                          re.IGNORECASE)
        }

        # Add criteria based on extracted fields
        for field, match in fields.items():
            if match:
                value = match.group(1).strip()
                if field == "Media Weight GSM":
                    filter_spec["criteria"].append({
                        "field": field,
                        "operation": "closest_to",
                        "value": int(value)
                    })

                    # Also set up sort if it mentions sorting by this field
                    if "sort" in query.lower() and field.lower() in query.lower():
                        filter_spec["sort"] = {
                            "field": field,
                            "operation": "closest_to",
                            "value": int(value)
                        }
                elif "contains" in query.lower() and field.lower() in query.lower():
                    filter_spec["criteria"].append({
                        "field": field,
                        "operation": "contains",
                        "value": value
                    })
                else:
                    filter_spec["criteria"].append({
                        "field": field,
                        "operation": "equals",
                        "value": value
                    })

        # Set limit based on query
        limit_match = re.search(r'(?:return|show|display)\s+(?:top\s+)?(\d+)', query, re.IGNORECASE)
        if limit_match:
            filter_spec["limit"] = int(limit_match.group(1))

        return filter_spec

    def min_criteria_filter(self, documents, filter_spec):
        """Apply filter with minimum criteria matching approach."""
        criteria = filter_spec.get("criteria", [])
        sort_spec = filter_spec.get("sort", {})
        limit = filter_spec.get("limit", 3)
        min_match = filter_spec.get("min_criteria_match", 1)

        print(f"🔍 Looking for documents matching at least {min_match} of {len(criteria)} criteria")

        # Track how many criteria each document matches
        scored_docs = []
        for doc in documents:
            matched_criteria = 0
            match_details = []

            for criterion in criteria:
                field = criterion.get("field")
                operation = criterion.get("operation")
                value = criterion.get("value")

                if not all([field, operation, value is not None]):
                    continue

                # Check if document matches this criterion
                if self.matches_criterion(doc, field, operation, value):
                    matched_criteria += 1
                    match_details.append(f"{field} matches '{value}'")

            # Add document if it matches minimum required criteria
            if matched_criteria >= min_match:
                scored_docs.append({
                    "doc": doc,
                    "matched_criteria": matched_criteria,
                    "total_criteria": len(criteria),
                    "match_details": match_details
                })

        # Sort documents
        if sort_spec and sort_spec.get("field"):
            field = sort_spec.get("field")
            operation = sort_spec.get("operation")
            value = sort_spec.get("value")

            if operation == "closest_to" and value is not None:
                # Sort by distance to target value, then by number of matches
                scored_docs.sort(key=lambda x: (
                    self.distance_to_value(x["doc"], field, value),
                    -x["matched_criteria"]  # Higher matches come first for tiebreaking
                ))
            else:
                # Sort by field value
                scored_docs.sort(key=lambda x: self.extract_sortable_value(x["doc"], field))
        else:
            # Sort by number of criteria matched (descending)
            scored_docs.sort(key=lambda x: -x["matched_criteria"])

        # Limit results if specified
        if limit > 0:
            scored_docs = scored_docs[:limit]

        return scored_docs

    def matches_criterion(self, doc, field, operation, value):
        """Check if a document matches a single criterion."""
        if field not in doc:
            return False

        doc_value = doc[field]

        # Handle different operations
        if operation == "equals":
            return str(doc_value).strip() == str(value).strip()

        elif operation == "contains":
            return str(value).lower() in str(doc_value).lower()

        elif operation == "greater_than":
            try:
                return float(doc_value) > float(value)
            except (ValueError, TypeError):
                return False

        elif operation == "less_than":
            try:
                return float(doc_value) < float(value)
            except (ValueError, TypeError):
                return False

        elif operation == "closest_to":
            # For closest_to, as long as the field exists and is numeric, we consider it a match
            try:
                float(doc_value)
                return True
            except (ValueError, TypeError):
                return False

        # Handle "preferred" as a synonym for contains/equals
        elif operation == "preferred":
            # First try exact match, then partial match
            if str(doc_value).strip() == str(value).strip():
                return True
            return str(value).lower() in str(doc_value).lower()

        return False

    def distance_to_value(self, doc, field, target_value):
        """Calculate distance between document field value and target value."""
        if field not in doc:
            return float('inf')

        try:
            doc_value = float(doc[field])
            target = float(target_value)
            return abs(doc_value - target)
        except (ValueError, TypeError):
            return float('inf')

    def extract_sortable_value(self, doc, field):
        """Extract a value that can be used for sorting."""
        if field not in doc:
            return None

        value = doc[field]

        # Try numeric conversion
        try:
            return float(value)
        except (ValueError, TypeError):
            return str(value)

    def format_results(self, scored_docs, filter_spec):
        """Format filtered documents as markdown table with match details."""
        if not scored_docs:
            return "No matching documents found based on the specified criteria."

        # Determine which fields to show
        fields_to_show = set()
        criteria = filter_spec.get("criteria", [])

        for criterion in criteria:
            field = criterion.get("field")
            if field:
                fields_to_show.add(field)

        sort_spec = filter_spec.get("sort", {})
        sort_field = sort_spec.get("field")
        if sort_field:
            fields_to_show.add(sort_field)

        # Create table header
        columns = ["Rank", "File Name"] + list(fields_to_show) + ["Match Score", "Reason"]
        table = "| " + " | ".join(f"**{col}**" for col in columns) + " |\n"
        table += "|" + "|".join([":---------"] * len(columns)) + "|\n"

        # Add rows
        for rank, scored_doc in enumerate(scored_docs, 1):
            doc = scored_doc["doc"]
            row = [str(rank)]

            # Add file name
            row.append(doc.get("file_name", f"Doc_{rank}"))

            # Add field values
            for field in fields_to_show:
                row.append(str(doc.get(field, "N/A")))

            # Add match score
            match_score = f"{scored_doc['matched_criteria']}/{scored_doc['total_criteria']}"
            row.append(match_score)

            # Add reason
            if sort_spec and sort_spec.get("field") and sort_spec.get("operation") == "closest_to":
                field = sort_spec.get("field")
                target = sort_spec.get("value")
                try:
                    doc_value = float(doc.get(field, 0))
                    target_value = float(target)
                    distance = abs(doc_value - target_value)
                    match_details = ", ".join(scored_doc.get("match_details", []))
                    reason = f"Distance to {target} {field}: {distance}; {match_details}"
                except (ValueError, TypeError):
                    reason = ", ".join(scored_doc.get("match_details", ["Matches criteria"]))
            else:
                reason = ", ".join(scored_doc.get("match_details", ["Matches criteria"]))

            row.append(reason)

            table += "| " + " | ".join(row) + " |\n"

        return table

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

    def process_query(self, documents, query):
        """Process a natural language query and return filtered and sorted documents."""
        # Parse the query into structured criteria
        filter_spec = self.parse_query_with_llm(query)
        print("\n🔍 Extracted Filter Specification:")
        print(json.dumps(filter_spec, indent=2))

        # Apply the filter specification to documents
        scored_docs = self.min_criteria_filter(documents, filter_spec)

        # Format the results
        results = self.format_results(scored_docs, filter_spec)

        return {
            "filter_spec": filter_spec,
            "scored_docs": scored_docs,
            "results": results
        }


def main():
    folder_path = "/Users/suriya/Documents/Github/Structured-Data-Similarity-Search-using-NLP/Dataset/ExtractedJSON/GT"

    # Example user query
    query = """Find documents that MUST match ALL of these criteria:
    1. "Ink Coverage" must be exactly "Heavy" 
    2. "Media Coating" must contain "Coated Inkjet"
    3. "Media Weight GSM" must be closest to 110
    Sort by how close the Media Weight is to 110 GSM.
    Return top 3 results only."""

    filter_tool = RobustMinCriteriaFilter()
    documents = filter_tool.process_json_folder(folder_path)

    # List document fields for debugging
    filter_tool.list_document_fields(documents)

    # Process the query
    results = filter_tool.process_query(documents, query)

    print(f"\n📊 Results:\n{results['results']}\n")

    # If there are matching documents, let's count them
    if results['scored_docs']:
        print(
            f"✅ Found {len(results['scored_docs'])} matching documents, showing top {results['filter_spec'].get('limit', 3)}")
    else:
        print("❌ No matching documents found")


if __name__ == "__main__":
    main()