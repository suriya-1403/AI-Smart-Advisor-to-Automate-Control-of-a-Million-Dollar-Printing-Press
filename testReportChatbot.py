from langchain.prompts import PromptTemplate
from lib import get_cached_components, import_or_split_pdf

from log import get_logger

import streamlit as st
import json
import os

# Initialize the logger
logger = get_logger(__name__)

# Load documents from folder (no changes needed here)
def load_documents_from_folder(folder_path):
    documents = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r') as file:
                documents.append({
                    "filename": filename,
                    "content": json.load(file)
                })
    return documents

# Extract text from documents (no changes needed here)
def extract_text_from_json(documents):
    text_data = []
    for doc in documents:
        combined_text = ""
        combined_text += str(doc['content'].get('event_id', '')) + " "
        combined_text += str(doc['content'].get('publish_date', '')) + " "
        combined_text += str(doc['content'].get('location', '')) + " "
        combined_text += str(doc['content'].get('hp_press_details', {})) + " "
        combined_text += str(doc['content'].get('print_application', {})) + " "
        combined_text += str(doc['content'].get('color_ink_details', {})) + " "
        combined_text += str(doc['content'].get('media_information', {})) + " "
        combined_text += str(doc['content'].get('print_process_settings', {})) + " "
        combined_text += str(doc['content'].get('project_status', {})) + " "
        combined_text += str(doc['content'].get('press_speed_settings', {})) + " "
        combined_text += str(doc['content'].get('finishing_details', {})) + " "
        combined_text += str(doc['content'].get('environmental_certifications', [])) + " "
        combined_text += str(doc['content'].get('regions_available', [])) + " "
        combined_text += str(doc['content'].get('contact_information', {})) + " "
        text_data.append(combined_text)
    return text_data


def get_information_with_llama3_2(query, documents, model, general_prompt):
    """
    Use Llama 3.2 to evaluate document relevance based on the input query and return explanations with document names.

    :param query: str
        The query entered by the user.
    :param documents: list
        List of document objects that includes filenames and content.
    :param model: OllamaLLM
        The Llama 3.2 model instance.
    :param general_prompt: PromptTemplate
        The general-purpose prompt template.

    :return: list
        A list of explanations including document names.
    """
    try:
        # Prepare a list to store explanations
        relevance_results = []

        # Loop through each document in documents
        for doc in documents:
            # Ensure doc is a dictionary
            if isinstance(doc, dict):
                doc_name = doc['filename']  # Access filename directly from the doc
                doc_text = extract_text_from_json([doc])  # Extract text for the individual document

                # Combine query and document text for Llama's prompt
                query_prompt = general_prompt.format(input_text=doc_text[0], query=query)

                # Query the Llama model with the prompt
                response = model(query_prompt)

                # Append the document name and reasoning to the results list
                relevance_results.append({
                    "document_name": doc_name,  # Use the actual filename
                    "explanation": response  # The reasoning for relevance
                })
            else:
                logger.warning(f"Expected a dictionary but got {type(doc)} instead.")

        return relevance_results

    except Exception as error:
        logger.exception("Error during document relevance evaluation with Llama 3.2.")
        raise error


def main():
    st.title("Document Search and Reasoning with Llama 3.2")

    # Load documents and extract text
    folder_path = 'Dataset/SamplePrintJSONChatbot/JSON/Data'
    documents = load_documents_from_folder(folder_path)
    document_texts = extract_text_from_json(documents)

    # Get query from user
    query = st.text_area("Enter your query", height=150)

    # Initialize components (model, prompt)
    components = get_cached_components()
    model = components["model"]
    general_prompt = components["general_prompt"]
    print("gp:",general_prompt)
    if query:
        # Use the Llama model to evaluate document relevance and get reasoning
        relevance_explanation = get_information_with_llama3_2(query, documents, model, general_prompt)

        # Display the query and relevance explanations
        st.subheader("Document Relevance and Justification")
        st.write(f"Query: {query}")

        # Display each document's relevance explanation
        for idx, explanation in enumerate(relevance_explanation):
            st.subheader(f"Document {idx + 1} Relevance Explanation:")
            st.text_area(f"Document {idx + 1} Explanation", value=explanation, height=300)

if __name__ == "__main__":
    main()