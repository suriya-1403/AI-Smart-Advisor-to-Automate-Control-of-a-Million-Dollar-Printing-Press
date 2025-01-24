from langchain.prompts import PromptTemplate
from lib import get_cached_components, import_or_split_pdf
from log import get_logger
from langchain_ollama import OllamaLLM
from pathlib import Path
import json

# Initialize the logger
logger = get_logger(__name__)

def extract_information_with_model(text, model, general_prompt):
    """
    Extract structured information from text using a given model.

    :param text: str
        The input text extracted from the PDF.
    :param model: Object
        The model instance to use for information extraction.
    :param general_prompt: PromptTemplate
        The general-purpose prompt template.
    :return: dict
        Extracted structured information.
    """
    try:
        # Combine the prompt with the input text
        query_prompt = general_prompt.format(input_text=text)

        # Query the model
        response = model(query_prompt)

        # Parse the response (assuming JSON-like structured output)
        return response
    except Exception as error:
        logger.exception(f"Error during information extraction with {model.__class__.__name__}.")
        return {"error": str(error)}

if __name__ == "__main__":
    # Load cached components
    components = get_cached_components()

    # Define models to use
    models = {
        # "Llama 3.2": components["model"],
        # "Falcon": components.get("model_falcon"),
        "Llama 3.2 vision": OllamaLLM(model="llama3.2-vision"),
        "Llama 3.2": OllamaLLM(model="llama3.2"),
        "Granite": OllamaLLM(model="granite3-dense"),
        "Falcon 3": OllamaLLM(model="falcon3"),
        "mistral 7b": OllamaLLM(model="mistral"),
        "Deep Seek 7b": OllamaLLM(model="deepseek-r1"),
    }

    # Define the prompts to use
    general_prompts = {
        "ExtractionPrompt": PromptTemplate(template="""
            Extract the following details from the document and provide them in JSON format:
            - Event Name
            - Location
            - Date
            - Publish Date
            - HP Press Model
            - Print Mode
            - Type of Print Application
            {input_text}
        """),
        "DetailedJsonStructure": PromptTemplate(template="""
            Extract structured details in nested JSON format, grouping logically:
            {{
                "EventDetails": {{"EventName": "", "Location": "", "Date": "", "PublishDate": ""}},
                "MediaSpecifications": {{
                    "MediaName": "",
                    "MediaType": "",
                    "MediaWeight": "",
                    "Certifications": []
                }},
                "Additional": {{"Images": [], "Comments": ""}}
            }}
            {input_text}
        """),
        "RoleSpecificPrompt": PromptTemplate(template="""
            Imagine you're processing this report for system automation. Extract key structured details in JSON format:
            - Group event details (name, location, date).
            - Capture all press-related parameters (model, ink type, settings).
            {input_text}
        """),
        "ErrorHandlingPrompt": PromptTemplate(template="""
            Extract details and include an errors section for missing or ambiguous data:
            - Event Details (Event Name, Location, Date, Publish Date)
            - Media Specs (Media Name, Type, Weight, Certifications)
            {input_text}
        """),
    }

    # Define the PDF path
    pdf_path = Path("/Users/suriya/Documents/Github/AI-Smart-Advisor-to-Automate-Control-of-a-Million-Dollar-Printing-Press/Dataset/Sample print reports- HP Confidential/sample.pdf")

    # Process the PDF and extract text
    try:
        pages = import_or_split_pdf(str(pdf_path))
        pdf_text = " ".join([page.page_content for page in pages])
    except Exception as e:
        logger.error(f"Failed to process the PDF: {e}")
        exit(1)

    # Iterate through models and prompts
    results = {}
    for model_name, model_instance in models.items():
        if model_instance is None:
            logger.warning(f"Skipping model: {model_name} (Not initialized)")
            continue

        results[model_name] = {}
        for prompt_name, prompt_template in general_prompts.items():
            logger.info(f"Running {model_name} with {prompt_name}")
            structured_data = extract_information_with_model(pdf_text, model_instance, prompt_template)
            results[model_name][prompt_name] = structured_data

            # Print the output
            print(f"\nModel: {model_name}, Prompt: {prompt_name}")
            print(json.dumps(structured_data, indent=4))
            print("-" * 50)

    # Save results to a JSON file for further analysis
    output_file = "benchmark_results-multimodel.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)
    logger.info(f"Benchmark results saved to {output_file}")
