from langchain.prompts import PromptTemplate
from lib import get_cached_components, import_or_split_pdf

from log import get_logger
from pydantic import ValidationError

# Initialize the logger
logger = get_logger(__name__)

def extract_information_with_llama3_2(text, model, general_prompt):
    """
    Extract structured information from text using Llama 3.2.

    :param text: str
        The input text extracted from the PDF.
    :param model: OllamaLLM
        The Llama 3.2 model instance.
    :param general_prompt: PromptTemplate
        The general-purpose prompt template.
    :return: dict
        Extracted structured information.
    """
    try:
        # Combine the prompt with the input text
        query_prompt = general_prompt.format(input_text=text)

        # Query the Llama model
        response = model(query_prompt)

        # Parse the response (assuming JSON-like structured output)
        return response
    except Exception as error:
        logger.exception("Error during information extraction with Llama 3.2.")
        raise error


if __name__ == "__main__":

    components = get_cached_components()
    model = components["model"]
    general_prompt = components["general_prompt"]

    pdf_path = "/Users/suriya/Documents/Github/AI-Smart-Advisor-to-Automate-Control-of-a-Million-Dollar-Printing-Press/Dataset/Sample print reports- HP Confidential/PWP at the Show Prints On Heavy Media Pixelle Speciality Solutions PIXELLE SUPERIOR GLOSS Inkjet Coated Gloss 266 gsm-180lb- 9pt Cover Oct 22.pdf"
    pages = import_or_split_pdf(pdf_path)

    pdf_text = " ".join([page.page_content for page in pages])

    extraction_prompt_template = """
        Extract the following details from the document and provide them in JSON format:
        - Event Name
        - Location
        - Date
        - Publish Date
        - HP Press Model
        - Print Mode
        - Type of Print Application
        - Average Ink Coverage Class
        - ICC Profiles Used
        - Post-Coat Applied
        - Short Job Description
        - Media Name
        - Media Type
        - Media Weight
        - Environmental Certifications
        - Dryer Settings Range
        - Print Speed Range
        - Ink Type
        - Contact Email
        - The Media Star For This Print Run
        - Print Event ID
        - Delivered to
        - Printed at the HP Location
        - Short Description of Job
        - Media Weight Class
        - Optimizer via Process
        - Moisturizer Status
        - Tension via Formula
        - Special Setup+Tension Comments
        - Printed Images Below
        - Recipe for Success on This Media

        Here is the document:
        {input_text}
        """
    detailed_json_structure = """
        Extract the following details from the input document and format them into well-organized nested JSON, grouping information logically:
        {{
          "EventDetails": {{"EventName": "", "Location": "", "Date": "", "PublishDate": ""}},
          "MediaSpecifications": {{
            "MediaName": "",
            "MediaType": "",
            "MediaWeight": "",
            "EnvironmentalCertifications": [],
            "StarMedia": ""
          }},
          "PrintSpecs": {{
            "PressModel": "",
            "PrintMode": "",
            "InkType": "",
            "DryerSettingsRange": "",
            "PrintSpeedRange": ""
          }},
          "Process": {{
            "Optimizer": "",
            "MoisturizerStatus": "",
            "Tension": "",
            "SetupComments": ""
          }},
          "Contact": {{"Email": ""}},
          "Additional": {{"PrintedImages": [], "RecipeForSuccess": ""}}
        }}
        Make sure all details are accurate and indicate missing or ambiguous values as "N/A". Input: {input_text}
        """

    role_specific = """
    Imagine you're designing a system for automated report processing. Your task is to extract key structured details in JSON format from the given document:

    Group event details (name, location, date).
    Capture all press-related parameters (model, ink type, dryer settings).
    Extract media-specific details (type, weight, certifications).
    Parse any process-related comments (optimizer, moisturizer, tension). If data is missing, mark it as 'N/A' and add any detected issues under a separate issues section. Use this as input: {input_text}"""

    error_handling = """
    Extract the following details from the document and format them in JSON. Handle missing or ambiguous data and include an errors section for issues. Key details to extract:
    
    Event Details: Event Name, Location, Date, Publish Date, Event ID.
    Media Specifications: Media Name, Type, Weight, Certifications, Star Media.
    Print and Press Details: Press Model, Print Mode, Ink Coverage, ICC Profiles, Speed Range, Dryer Settings.
    Process Parameters: Optimizer, Moisturizer, Tension, Setup Comments, Recipe for Success.
    Contact and Delivery: Delivered To, Printed At, Contact Email.
    Additional Info: Printed Images, Job Description.
    
    For checkbox fields, determine their state: If marked, label as "checked".If unmarked, label as "unchecked". If the status is unclear or missing, label as "missing" and include a note in an errors section.
    For missing fields, return "N/A" as the value and add a corresponding entry in an errors section describing the issue (e.g., 'Field not found in the document').
    {input_text}
    """

    extraction_prompt = PromptTemplate(template=error_handling)

    structured_data = extract_information_with_llama3_2(pdf_text, model, extraction_prompt)

    print("Extracted Information:")
    print(structured_data)