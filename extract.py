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
    extraction_prompt = PromptTemplate(template=extraction_prompt_template)

    structured_data = extract_information_with_llama3_2(pdf_text, model, extraction_prompt)

    print("Extracted Information:")
    print(structured_data)